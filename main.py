import base64
import io

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
from pydantic import BaseModel

app = FastAPI(title="Selfie Validator")

from fastapi.staticfiles import StaticFiles
from pathlib import Path

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thresholds (tunable) ─────────────────────────────────────────────────────
BLUR_THRESHOLD = 100.0   # Laplacian variance — below this = blurry
EAR_THRESHOLD = 0.22     # Eye Aspect Ratio — below this = eyes closed
SYMMETRY_THRESHOLD = 0.60  # Nose-to-face-edge ratio — below this = not facing forward

BASE_URL = "https://view-immigration-status.service.gov.uk"
JOB_TITLE = "Verification team"
ORGANISATION_NAME = "Aspora"
CHECK_REASON = "Personal finance (including bank and building society accounts, loans, credit cards and mortgages)"


# ── Selfie validation logic ──────────────────────────────────────────────────

def _validate_image_bytes(data: bytes) -> dict:
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clear = blur_score >= BLUR_THRESHOLD

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return {
                "passed": False,
                "reasons": ["No face detected"],
                "scores": {"blur": round(blur_score, 2)},
            }

        lm = results.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Eye Aspect Ratio (left eye)
        left_ear = np.linalg.norm(pt(159) - pt(145)) / np.linalg.norm(pt(33) - pt(133))
        # Eye Aspect Ratio (right eye)
        right_ear = np.linalg.norm(pt(386) - pt(374)) / np.linalg.norm(pt(362) - pt(263))
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_open = avg_ear >= EAR_THRESHOLD

        # Face symmetry (looking forward)
        nose = pt(4)
        left_edge = pt(234)
        right_edge = pt(454)
        d_left = np.linalg.norm(nose - left_edge)
        d_right = np.linalg.norm(nose - right_edge)
        symmetry = min(d_left, d_right) / max(d_left, d_right)
        looking_forward = symmetry >= SYMMETRY_THRESHOLD

    reasons = []
    if not is_clear:
        reasons.append("Image is blurry or out of focus")
    if not eyes_open:
        reasons.append("Eyes appear closed")
    if not looking_forward:
        reasons.append("Face is not looking straight at the camera")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "scores": {
            "blur": round(blur_score, 2),
            "eye_openness": round(avg_ear, 3),
            "symmetry": round(symmetry, 3),
        },
    }


# ── Sharecode photo retrieval (reuses visa_website logic) ────────────────────

class SharecodeError(Exception):
    """Structured error from the sharecode lookup flow."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


async def _fetch_photo_from_sharecode(share_code: str, date_of_birth: str) -> str:
    """Returns the photo as a data URI (data:image/jpeg;base64,...) or raises SharecodeError."""
    share_code = share_code.replace(" ", "").upper()

    if len(share_code) != 9 or not share_code.isalnum():
        raise SharecodeError("invalid_share_code", "Share code must be 9 characters, letters and numbers only (e.g. W9X3M2T4K).")

    parts = date_of_birth.split("-")
    if len(parts) != 3:
        raise SharecodeError("invalid_dob", "Date of birth must be in YYYY-MM-DD format.")
    year, month, day = parts

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            async def click_continue():
                await page.locator('button:visible:has-text("Continue")').click()
                await page.wait_for_load_state("networkidle", timeout=30000)

            try:
                await page.goto(f"{BASE_URL}/view/checker-details", wait_until="networkidle", timeout=15000)
            except Exception:
                raise SharecodeError("timeout", "Could not reach the UK immigration service. Please try again in a moment.")

            await page.fill('input[name="shareCode"]', share_code)
            await click_continue()

            await page.fill('input[name="dob-day"]', day.lstrip("0") or "0")
            await page.fill('input[name="dob-month"]', month.lstrip("0") or "0")
            await page.fill('input[name="dob-year"]', year)
            await click_continue()

            error_el = await page.query_selector(".govuk-error-summary")
            if error_el:
                raw = (await error_el.inner_text()).strip().lower()
                if "share code" in raw or "sharecode" in raw:
                    raise SharecodeError("invalid_share_code", "That share code was not recognised. Please check it and try again.")
                if "date of birth" in raw or "dob" in raw:
                    raise SharecodeError("dob_mismatch", "The date of birth does not match the share code. Please check both and try again.")
                raise SharecodeError("lookup_failed", "The share code and date of birth could not be verified. Please check your details.")

            await page.fill('input[name="jobTitle"]', JOB_TITLE)
            await page.fill('input[name="companyName"]', ORGANISATION_NAME)
            await click_continue()

            await page.locator(f'input[type="radio"][value="{CHECK_REASON}"]').click()
            await click_continue()

            error_el = await page.query_selector(".govuk-error-summary")
            if error_el:
                raise SharecodeError("lookup_failed", "Something went wrong retrieving the immigration record. Please try again.")

            photo_el = await page.query_selector("img#photo")
            if not photo_el:
                raise SharecodeError("no_photo", "No photo was found on this immigration record.")

            photo_src = await photo_el.get_attribute("src")
            return photo_src

        except SharecodeError:
            raise
        except Exception as e:
            raise SharecodeError("unexpected", "An unexpected error occurred. Please try again.") from e
        finally:
            await browser.close()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/validate-selfie")
async def validate_selfie(file: UploadFile = File(...)):
    """
    Validate a selfie image.
    Returns { passed, reasons, scores }.
    If passed=False, the caller should prompt for share_code + date_of_birth
    and call /api/sharecode-photo.
    """
    data = await file.read()
    try:
        result = _validate_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


class SharecodePhotoRequest(BaseModel):
    share_code: str
    date_of_birth: str  # YYYY-MM-DD


@app.post("/api/sharecode-photo")
async def sharecode_photo(payload: SharecodePhotoRequest):
    """
    Fetch the photo on record via the UK immigration status viewer.
    Returns { photo } where photo is a data URI.
    """
    try:
        photo = await _fetch_photo_from_sharecode(payload.share_code, payload.date_of_birth)
    except SharecodeError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "unexpected", "message": "An unexpected error occurred. Please try again."})
    return {"photo": photo}


@app.post("/api/process-selfie")
async def process_selfie(
    file: UploadFile = File(None),
    share_code: str = Form(None),
    date_of_birth: str = Form(None),
):
    """
    All-in-one endpoint.

    - If only `file` is provided: validates the selfie.
      - If it passes, returns { status: "ok", photo: <data-uri> }
      - If it fails, returns { status: "needs_sharecode", reasons: [...] }
    - If `file` + `share_code` + `date_of_birth` are all provided:
      validates the selfie first; if it fails, fetches the photo from
      the immigration status service and returns that instead.
    """
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")

    data = await file.read()
    try:
        validation = _validate_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if validation["passed"]:
        # Selfie is fine — return it as a data URI
        encoded = base64.b64encode(data).decode()
        mime = file.content_type or "image/jpeg"
        return {
            "status": "ok",
            "source": "selfie",
            "photo": f"data:{mime};base64,{encoded}",
            "scores": validation["scores"],
        }

    # Selfie failed
    if not share_code or not date_of_birth:
        return {
            "status": "needs_sharecode",
            "reasons": validation["reasons"],
            "scores": validation["scores"],
        }

    # Fetch from sharecode
    try:
        photo = await _fetch_photo_from_sharecode(share_code, date_of_birth)
    except SharecodeError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "unexpected", "message": "An unexpected error occurred. Please try again."})

    return {
        "status": "ok",
        "source": "sharecode",
        "photo": photo,
        "selfie_rejection_reasons": validation["reasons"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, app_dir="/Users/chandraerukulla/Documents/Dev/selfie_related")
