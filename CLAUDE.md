# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service does

Validates a selfie image for KYC compliance (blur, eyes open, looking forward). If the selfie fails, it falls back to fetching the photo on record from the UK immigration status viewer using the user's share code and date of birth — reusing the same Playwright flow built in `../visa_website/backend/main.py`.

## Running the service

```bash
pip install -r requirements.txt
playwright install chromium
uvicorn main:app --reload --port 8001
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/validate-selfie` | Validate a selfie (multipart `file`). Returns `{ passed, reasons, scores }`. |
| `POST /api/sharecode-photo` | Fetch photo from immigration status viewer. Body: `{ share_code, date_of_birth }` (YYYY-MM-DD). |
| `POST /api/process-selfie` | All-in-one. Multipart `file` + optional form fields `share_code` + `date_of_birth`. Returns `{ status: "ok" \| "needs_sharecode", photo, ... }`. |

## Validation logic

Three checks in `_validate_image_bytes()`:
- **Blur** — Laplacian variance of grayscale image. Threshold: `BLUR_THRESHOLD = 100.0`
- **Eyes open** — MediaPipe Face Mesh Eye Aspect Ratio (eyelid gap / eye width). Threshold: `EAR_THRESHOLD = 0.22`
- **Looking forward** — Nose-tip to left/right face-edge symmetry ratio. Threshold: `SYMMETRY_THRESHOLD = 0.75`

All three thresholds are module-level constants; adjust them without touching logic.

## Sharecode integration

`_fetch_photo_from_sharecode()` navigates `view-immigration-status.service.gov.uk` using Playwright, identical to `../visa_website/backend/main.py`. It fills in the checker's job title / org / reason (hardcoded as Aspora verification) and returns the `img#photo` data URI.

## Related projects

- `../visa_website/backend/main.py` — original immigration status checker (share code → full status + photo)
- `../share_code_generate_website/backend/main.py` — generates a share code via Playwright (passport number + DOB + OTP flow)
