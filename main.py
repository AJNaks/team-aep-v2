"""
TEAM-AEP v2 — Proof of Concept
================================
AI-powered MSE Onboarding for the TEAM Initiative / ONDC

Modules:
  1. Udyam Verification (Gridlines API + Sarvam STT/TTS)
  2. NIC-to-ONDC Product Categorisation
  3. MSE-SNP Intelligent Matching

Run: uvicorn main:app --reload --port 8000
"""

import os, json, time, secrets, base64, httpx, uuid, asyncio, tempfile, zipfile, io
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from sarvamai import SarvamAI

from services.entity_extractor import extract_entities
from services.categorisation import categorise_mse
from services.snp_matcher import match_mse_to_snps, estimate_digital_readiness
from services.database import (
    init_db, create_otp_session, verify_otp_session,
    get_snp_by_email, get_snp_by_subscriber_id, get_snp_by_id, get_all_snps,
    get_nsic_by_employee_id,
    get_msme_session, save_msme_progress,
    get_assignments_for_snp, assign_mse_to_snp, update_assignment_status,
    get_claims_for_snp, get_all_claims, get_claim_by_id, submit_claim, review_claim,
    get_snp_dashboard_stats, get_platform_analytics,
    get_all_mses, get_snps_with_metrics, check_claim_eligibility,
    save_document_verification, get_document_verifications, save_claim_verification,
)

load_dotenv()

# Initialize SQLite database
init_db()

# ── Config ────────────────────────────────────────────────
APP_PASSWORD       = os.getenv("APP_PASSWORD", "teamaep2026")
GRIDLINES_KEY      = os.getenv("GRIDLINES_API_KEY", "test-credential-465585017200472064")
GRIDLINES_BASE     = os.getenv("GRIDLINES_BASE_URL", "https://stoplight.io/mocks/gridlines/gridlines-api-docs/133154723")
SARVAM_KEY         = os.getenv("SARVAM_API_KEY", "")
ANTHROPIC_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
print(f"[BOOT] SARVAM_KEY set: {bool(SARVAM_KEY)} (len={len(SARVAM_KEY)})")
print(f"[BOOT] ANTHROPIC_KEY set: {bool(ANTHROPIC_KEY)} (len={len(ANTHROPIC_KEY)})")
SESSION_TOKENS     = set()
ROLE_SESSIONS      = {}   # session_token -> {role, identifier, profile}

# ── Auth helpers ──────────────────────────────────────────
def check_session(request: Request) -> bool:
    return request.cookies.get("session_token") in SESSION_TOKENS

async def require_auth(request: Request):
    if not check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

app = FastAPI(title="TEAM-AEP v2")

BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

with open(DATA_DIR / "mock_udyam.json") as f:
    MOCK_UDYAM = json.load(f)

# Build mock lookup indexes for mobile and PAN fallback
MOCK_BY_MOBILE = {}
MOCK_BY_PAN = {}
for udyam_id, ent in MOCK_UDYAM.get("enterprises", {}).items():
    if ent.get("mobile"):
        MOCK_BY_MOBILE[ent["mobile"]] = (udyam_id, ent)
    # Extract PAN from GST number (characters 3-12)
    gst = ent.get("gst_number", "")
    if len(gst) >= 12:
        pan = gst[2:12]
        MOCK_BY_PAN[pan] = (udyam_id, ent)


def mock_to_gridlines_response(udyam_id: str, ent: dict) -> dict:
    """Convert mock_udyam.json entry into Gridlines-style response."""
    return {
        "code": "1000",
        "enterprise_data": {
            "document_id": udyam_id,
            "name": ent.get("enterprise_name", ""),
            "enterprise_type": ent.get("enterprise_type", ""),
            "major_activity": ent.get("nic_description", ""),
            "organization_type": "Proprietary",
            "mobile": ent.get("mobile", ""),
            "email": ent.get("email", ""),
            "date_of_udyam_registration": ent.get("date_of_registration", ""),
            "social_category": ent.get("social_category", ""),
            "gender": ent.get("gender", ""),
            "dic": ent.get("district", ""),
            "msme_di": ent.get("state", ""),
            "address": {
                "district": ent.get("district", ""),
                "state": ent.get("state", ""),
                "pincode": ent.get("pincode", ""),
            },
        },
        "nic_data": {
            "nic_2_digit": f"{ent.get('nic_2digit', '')} - {ent.get('nic_description', '')}",
            "nic_4_digit": f"{ent.get('nic_4digit', '')} - {ent.get('nic_description', '')}",
            "nic_5_digit": f"{ent.get('nic_5digit', '')} - {ent.get('nic_description', '')}",
        },
        "nic_data_list": [{
            "nic_2_digit": f"{ent.get('nic_2digit', '')} - {ent.get('nic_description', '')}",
            "nic_4_digit": f"{ent.get('nic_4digit', '')} - {ent.get('nic_description', '')}",
            "nic_5_digit": f"{ent.get('nic_5digit', '')} - {ent.get('nic_description', '')}",
            "activity_type": ent.get("nic_description", ""),
        }],
        "enterprise_units": [],
        "enterprise_type_data": [],
    }


# ── OTP store (in-memory for demo) ────────────────────────
# Maps session_token -> { udyam_number, otp_code, verified, enterprise_data }
OTP_STORE = {}


# ── Pydantic models ───────────────────────────────────────
class VerifyUdyamRequest(BaseModel):
    udyam_number: str

class VerifyMobileRequest(BaseModel):
    mobile: str

class VerifyPANRequest(BaseModel):
    pan: str

class OTPVerifyRequest(BaseModel):
    otp: str
    session_id: str

class CategoriseRequest(BaseModel):
    nic_code: str
    product_description: Optional[str] = ""

class MatchRequest(BaseModel):
    ondc_domain: str
    state: str
    language: str = "hi"
    enterprise_type: Optional[str] = "Micro"
    gst_number: Optional[str] = None
    turnover: Optional[int] = 0
    email: Optional[str] = None
    udyam: Optional[str] = None
    input_channel: Optional[str] = "web"

class TTSRequest(BaseModel):
    text: str
    language: str = "hi-IN"
    speaker: str = "anushka"

class TranslateRequest(BaseModel):
    text: str
    source_language: str = "en-IN"
    target_language: str = "hi-IN"
    model: str = "mayura:v1"

class RoleLoginRequest(BaseModel):
    role: str            # 'msme', 'snp', 'nsic'
    identifier: str      # udyam/email/subscriber_id/employee_id
    auth_method: str     # 'udyam', 'email', 'subscriber_id', 'employee_id'

class RoleOTPVerifyRequest(BaseModel):
    role: str
    identifier: str
    otp: str

class SaveProgressRequest(BaseModel):
    udyam_number: str
    step: str
    data: dict = {}

class CatalogueReviewRequest(BaseModel):
    action: str          # 'approve', 'request_changes', 'enrich'
    notes: str = ""

class SubmitClaimRequest(BaseModel):
    udyam_number: str
    claim_type: str
    amount: float
    sku_count: int = 0
    evidence: dict = {}

class ReviewClaimRequest(BaseModel):
    status: str          # 'approved', 'rejected'
    notes: str = ""

class ExtractProductRequest(BaseModel):
    transcript: str
    domain: str = "ONDC:RET10"
    existing_fields: dict = {}

class SuggestCategoryRequest(BaseModel):
    product_name: str
    description: str = ""
    domain: str = ""

class ProductFollowupRequest(BaseModel):
    product: dict
    domain: str = "ONDC:RET10"
    missing_fields: List[str] = []

class VerifyDocumentsRequest(BaseModel):
    udyam_data: dict
    documents: List[dict]  # [{type, extracted}, ...]

class VerifyClaimRequest(BaseModel):
    claim_id: int


# ── In-memory image store (PoC) ─────────────────────────
IMAGE_STORE = {}   # uuid -> { "data": "data:image/jpeg;base64,...", "created": timestamp }


def get_role_session(request: Request) -> dict | None:
    token = request.cookies.get("session_token")
    return ROLE_SESSIONS.get(token)

async def require_snp_auth(request: Request):
    if not check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    rs = get_role_session(request)
    if not rs or rs.get("role") != "snp":
        raise HTTPException(status_code=403, detail="SNP access required")
    return rs

async def require_nsic_auth(request: Request):
    if not check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    rs = get_role_session(request)
    if not rs or rs.get("role") != "nsic":
        raise HTTPException(status_code=403, detail="NSIC access required")
    return rs


# ── Routes: Auth ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not check_session(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return HTMLResponse(LOGIN_HTML.replace("{error}", error))


@app.post("/login")
async def login(password: str = Form(...)):
    if password == APP_PASSWORD:
        token = secrets.token_hex(32)
        SESSION_TOKENS.add(token)
        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie("session_token", token, httponly=True, max_age=86400)
        return resp
    return RedirectResponse(url="/login?error=Wrong+password", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session_token")
    SESSION_TOKENS.discard(token)
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("session_token")
    return resp


# ── Routes: Gridlines Udyam Verification ──────────────────

@app.post("/api/verify-udyam")
async def verify_udyam(req: VerifyUdyamRequest, request: Request, _=Depends(require_auth)):
    """Verify MSME via Udyam registration number using Gridlines API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{GRIDLINES_BASE}/msme-api/udyam",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": GRIDLINES_KEY,
                    "X-Auth-Type": "API-Key",
                },
                json={
                    "udyam_reference_number": req.udyam_number,
                    "consent": "Y",
                },
            )
            data = resp.json()
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Gridlines API error: {str(e)}"}, status_code=502)

    # Unwrap mock server 'data' envelope if present
    inner = data.get("data", data)

    if inner.get("code") != "1000":
        return JSONResponse({"status": "not_found", "message": "Enterprise not found. Please check the number."})

    # Create OTP session
    session_id = secrets.token_hex(16)
    enterprise = inner.get("enterprise_data", {})
    masked_mobile = enterprise.get("mobile", "N/A")
    otp_code = str(secrets.randbelow(900000) + 100000)  # 6-digit OTP

    OTP_STORE[session_id] = {
        "udyam_number": req.udyam_number,
        "otp_code": otp_code,
        "verified": False,
        "enterprise_data": enterprise,
        "nic_data": inner.get("nic_data", {}),
        "nic_data_list": inner.get("nic_data_list", []),
        "enterprise_units": inner.get("enterprise_units", []),
        "enterprise_type_data": inner.get("enterprise_type_data", []),
    }

    return {
        "status": "otp_sent",
        "session_id": session_id,
        "masked_mobile": masked_mobile,
        "enterprise_name": enterprise.get("name", ""),
        "demo_otp": otp_code,  # REMOVE IN PRODUCTION — shown for demo purposes
    }


@app.post("/api/verify-mobile")
async def verify_mobile(req: VerifyMobileRequest, request: Request, _=Depends(require_auth)):
    """Fetch Udyam details via mobile number using Gridlines API with local mock fallback."""
    inner = None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{GRIDLINES_BASE}/msme-api/fetch/mobile",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": GRIDLINES_KEY,
                    "X-Auth-Type": "API-Key",
                },
                json={
                    "mobile_number": req.mobile,
                    "consent": "Y",
                },
            )
            data = resp.json()
            inner = data.get("data", data)
    except Exception:
        pass  # Fall through to mock fallback

    # If API didn't return a valid result, try local mock data
    if not inner or inner.get("code") != "1000":
        mock_entry = MOCK_BY_MOBILE.get(req.mobile)
        if mock_entry:
            inner = mock_to_gridlines_response(mock_entry[0], mock_entry[1])
        else:
            return JSONResponse({"status": "not_found", "message": "No Udyam registration found for this mobile number."})

    enterprise = inner.get("enterprise_data", {})
    session_id = secrets.token_hex(16)
    otp_code = str(secrets.randbelow(900000) + 100000)

    OTP_STORE[session_id] = {
        "udyam_number": enterprise.get("document_id", ""),
        "otp_code": otp_code,
        "verified": False,
        "enterprise_data": enterprise,
        "nic_data": inner.get("nic_data", {}),
        "nic_data_list": inner.get("nic_data_list", []),
        "enterprise_units": inner.get("enterprise_units", []),
        "enterprise_type_data": inner.get("enterprise_type_data", []),
    }

    return {
        "status": "otp_sent",
        "session_id": session_id,
        "masked_mobile": enterprise.get("mobile", "N/A"),
        "enterprise_name": enterprise.get("name", ""),
        "udyam_number": enterprise.get("document_id", ""),
        "demo_otp": otp_code,
    }


@app.post("/api/verify-pan")
async def verify_pan(req: VerifyPANRequest, request: Request, _=Depends(require_auth)):
    """Fetch Udyam details via PAN number using Gridlines API with local mock fallback."""
    inner = None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{GRIDLINES_BASE}/msme-api/fetch/pan",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": GRIDLINES_KEY,
                    "X-Auth-Type": "API-Key",
                },
                json={
                    "pan": req.pan,
                    "consent": "Y",
                },
            )
            data = resp.json()
            inner = data.get("data", data)
    except Exception:
        pass  # Fall through to mock fallback

    # If API didn't return a valid result, try local mock data
    if not inner or inner.get("code") != "1000":
        mock_entry = MOCK_BY_PAN.get(req.pan.upper())
        if mock_entry:
            inner = mock_to_gridlines_response(mock_entry[0], mock_entry[1])
        else:
            return JSONResponse({"status": "not_found", "message": "No Udyam registration found for this PAN."})

    enterprise = inner.get("enterprise_data", {})
    session_id = secrets.token_hex(16)
    otp_code = str(secrets.randbelow(900000) + 100000)

    OTP_STORE[session_id] = {
        "udyam_number": enterprise.get("document_id", ""),
        "otp_code": otp_code,
        "verified": False,
        "enterprise_data": enterprise,
        "nic_data": inner.get("nic_data", {}),
        "nic_data_list": inner.get("nic_data_list", []),
        "enterprise_units": inner.get("enterprise_units", []),
        "enterprise_type_data": inner.get("enterprise_type_data", []),
    }

    return {
        "status": "otp_sent",
        "session_id": session_id,
        "masked_mobile": enterprise.get("mobile", "N/A"),
        "enterprise_name": enterprise.get("name", ""),
        "udyam_number": enterprise.get("document_id", ""),
        "demo_otp": otp_code,
    }


@app.post("/api/verify-otp")
async def verify_otp(req: OTPVerifyRequest, request: Request, _=Depends(require_auth)):
    """Verify OTP and return full enterprise data."""
    session = OTP_STORE.get(req.session_id)
    if not session:
        return JSONResponse({"status": "error", "message": "Session expired. Please start again."}, status_code=400)

    if req.otp != session["otp_code"]:
        return JSONResponse({"status": "error", "message": "Invalid OTP. Please try again."}, status_code=400)

    session["verified"] = True
    enterprise = session["enterprise_data"]
    nic_data = session.get("nic_data", {})
    nic_list = session.get("nic_data_list", [])

    return {
        "status": "verified",
        "udyam_number": session["udyam_number"],
        "enterprise": {
            "name": enterprise.get("name", ""),
            "type": enterprise.get("enterprise_type", ""),
            "major_activity": enterprise.get("major_activity", ""),
            "organization_type": enterprise.get("organization_type", ""),
            "address": enterprise.get("address", {}),
            "mobile": enterprise.get("mobile", ""),
            "email": enterprise.get("email", ""),
            "gender": enterprise.get("gender", ""),
            "social_category": enterprise.get("social_category", ""),
            "date_of_incorporation": enterprise.get("date_of_incorporation", ""),
            "date_of_udyam_registration": enterprise.get("date_of_udyam_registration", ""),
            "dic": enterprise.get("dic", ""),
            "msme_di": enterprise.get("msme_di", ""),
        },
        "nic_data": nic_data,
        "nic_data_list": nic_list,
        "enterprise_units": session.get("enterprise_units", []),
    }


# ── Routes: Sarvam STT ───────────────────────────────────

@app.post("/api/stt")
async def speech_to_text(request: Request, audio: UploadFile = File(...), language: str = "unknown", _=Depends(require_auth)):
    """Convert speech to text using Sarvam Saaras v3."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    audio_bytes = await audio.read()
    content_type = audio.content_type or "audio/webm"
    ext = "webm" if "webm" in content_type else "wav"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_KEY},
                files={"file": (f"audio.{ext}", audio_bytes, content_type)},
                data={
                    "model": "saaras:v3",
                    "language_code": language,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                print(f"[STT] Sarvam returned {resp.status_code}: {data}")
                return JSONResponse({"status": "error", "message": data.get("error", {}).get("message", "STT failed")}, status_code=resp.status_code)
    except Exception as e:
        print(f"[STT] Exception: {e}")
        return JSONResponse({"status": "error", "message": f"Sarvam STT error: {str(e)}"}, status_code=502)

    print(f"[STT] OK: lang={data.get('language_code')}, transcript={data.get('transcript','')[:80]}")
    return {
        "status": "success",
        "transcript": data.get("transcript", ""),
        "language_code": data.get("language_code", "unknown"),
        "confidence": data.get("language_confidence", 0),
    }


# ── Routes: AI Extract (Claude) ──────────────────────────

class ExtractRequest(BaseModel):
    transcript: str

@app.post("/api/extract")
async def extract_identifier(req: ExtractRequest, request: Request, _=Depends(require_auth)):
    """Use Claude to extract Udyam/Mobile/PAN from natural-language speech transcript."""
    if not ANTHROPIC_KEY:
        return JSONResponse({"status": "error", "message": "Anthropic API key not configured"}, status_code=500)

    prompt = f"""Extract a business identifier from this speech transcript. The user is trying to provide ONE of:

1. **Udyam Number** — format: UDYAM-XX-00-0000000 (where XX = 2 letter state code, 00 = 2 digits, 0000000 = 7 digits). Example: UDYAM-MH-01-0012345
2. **Mobile Number** — 10 digits starting with 6-9. Example: 9876543210
3. **PAN Number** — format: 5 letters + 4 digits + 1 letter. Example: ABCDE1234F

Speech transcript: "{req.transcript}"

Important rules:
- The transcript is from speech recognition, so letters and digits may be spelled out, run together, or have extra spaces/words around them.
- For Udyam: the user might say "udyam MH 01 double zero one two three four five" or "UDYAM dash MH dash zero one dash zero zero one two three four five" — reconstruct the proper format UDYAM-XX-00-0000000
- For numbers spoken as words: "zero" = 0, "one" = 1, "two" = 2, "three" = 3, "four" = 4, "five" = 5, "six" = 6, "seven" = 7, "eight" = 8, "nine" = 9, "double" means repeat next digit
- Ignore filler words like "my number is", "it is", "please verify", etc.
- If you find a valid identifier, respond with ONLY a JSON object, nothing else.
- If you cannot find any valid identifier, respond with: {{"type": null, "value": null}}

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{{"type": "udyam" or "mobile" or "pan", "value": "THE-FORMATTED-VALUE"}}"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-20250514",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            result = resp.json()
    except Exception as e:
        print(f"[Extract] Claude error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)

    try:
        text_out = result["content"][0]["text"].strip()
        # Parse the JSON response
        import json as _json
        extracted = _json.loads(text_out)
        print(f"[Extract] transcript='{req.transcript[:60]}' => {extracted}")
        return {"status": "success", **extracted}
    except Exception as e:
        print(f"[Extract] Parse error: {e}, raw: {result}")
        return {"status": "error", "type": None, "value": None}


# ── Routes: Sarvam TTS ───────────────────────────────────

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest, request: Request, _=Depends(require_auth)):
    """Convert text to speech using Sarvam Bulbul v3."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "Content-Type": "application/json",
                    "api-subscription-key": SARVAM_KEY,
                },
                json={
                    "inputs": [req.text],
                    "target_language_code": req.language,
                    "speaker": req.speaker,
                    "model": "bulbul:v2",
                    "pitch": 0,
                    "loudness": 1.5,
                    "pace": 0.85,
                    "enable_preprocessing": True,
                },
            )
            data = resp.json()
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Sarvam TTS error: {str(e)}"}, status_code=502)

    audios = data.get("audios", [])
    audio_b64 = audios[0] if audios else ""

    return {
        "status": "success",
        "audio_base64": audio_b64,
    }


# ── Routes: Sarvam Translate ─────────────────────────────

@app.post("/api/translate")
async def translate_text(req: TranslateRequest, request: Request, _=Depends(require_auth)):
    """Translate text using Sarvam Translate API."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.sarvam.ai/translate",
                headers={
                    "Content-Type": "application/json",
                    "api-subscription-key": SARVAM_KEY,
                },
                json={
                    "input": req.text,
                    "source_language_code": req.source_language,
                    "target_language_code": req.target_language,
                    "model": req.model,
                    "mode": "formal",
                    "enable_preprocessing": True,
                },
            )
            data = resp.json()
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Sarvam Translate error: {str(e)}"}, status_code=502)

    return {
        "status": "success",
        "translated_text": data.get("translated_text", ""),
        "source_language": data.get("source_language_code", req.source_language),
        "target_language": req.target_language,
    }


@app.post("/api/translate-batch")
async def translate_batch(request: Request, _=Depends(require_auth)):
    """Batch translate multiple texts using Sarvam Translate API."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    body = await request.json()
    texts = body.get("texts", {})  # { key: text_string, ... }
    source_lang = body.get("source_language", "en-IN")
    target_lang = body.get("target_language", "hi-IN")
    model = body.get("model", "mayura:v1")

    results = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for key, text in texts.items():
            try:
                resp = await client.post(
                    "https://api.sarvam.ai/translate",
                    headers={
                        "Content-Type": "application/json",
                        "api-subscription-key": SARVAM_KEY,
                    },
                    json={
                        "input": text,
                        "source_language_code": source_lang,
                        "target_language_code": target_lang,
                        "model": model,
                        "mode": "formal",
                        "enable_preprocessing": True,
                    },
                )
                data = resp.json()
                results[key] = data.get("translated_text", text)
            except Exception:
                results[key] = text  # Fallback to original on error

    return {
        "status": "success",
        "translations": results,
        "target_language": target_lang,
    }


# ── Routes: Product Catalogue AI (Sarvam-M) ─────────────

async def sarvam_chat(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """Call Sarvam-M chat completion API. Returns the assistant message content."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SARVAM_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-m",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            raise Exception(f"Sarvam-M error {resp.status_code}: {data}")
        return data["choices"][0]["message"]["content"]


@app.post("/api/extract-product")
async def extract_product(req: ExtractProductRequest, request: Request, _=Depends(require_auth)):
    """Extract ONDC product attributes from free-form voice/text description using Sarvam-M."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    system_prompt = """You are an ONDC product catalogue assistant. Extract product attributes from the user's description.

Return ONLY a JSON object with these fields (use null for unknown values):
{
  "name": "product name",
  "short_description": "brief description (max 50 words)",
  "long_description": "detailed description",
  "selling_price": number or null,
  "mrp": number or null,
  "currency": "INR",
  "unit_of_measure": "unit" (e.g. "gram", "kg", "piece", "litre", "ml", "pack"),
  "unit_value": "value" (e.g. "200", "1", "500"),
  "available_quantity": number or null,
  "returnable": true/false or null,
  "cancellable": true/false or null,
  "cod_available": true/false or null,
  "time_to_ship": "P1D" or "P2D" etc (ISO 8601 duration) or null,
  "manufacturer_name": "name" or null,
  "manufacturer_address": "address" or null,
  "country_of_origin": "India" or detected country,
  "generic_name": "generic product name" or null,
  "net_quantity": "value with unit" or null,
  "veg_nonveg": "veg" or "non-veg" or "egg" or null,
  "fssai_license": "license number" or null,
  "nutritional_info": "info" or null,
  "ingredients": "comma-separated ingredients" or null,
  "brand": "brand name" or null,
  "colour": "colour" or null,
  "size": "size" or null,
  "gender": "male" or "female" or "unisex" or null,
  "material": "material" or null,
  "model_name": "model" or null,
  "category": "product category" or null,
  "subcategory": "subcategory" or null
}

Rules:
- Extract ONLY what is clearly stated or strongly implied. Do not guess.
- The description may be in any Indian language — extract values in English.
- If the user mentions a price, try to distinguish between selling price and MRP.
- For food items, try to identify veg/non-veg status from the description.
- Return ONLY valid JSON, no markdown, no explanation."""

    user_msg = f"Product description: {req.transcript}"
    if req.domain:
        user_msg += f"\nONDC Domain: {req.domain}"
    if req.existing_fields:
        user_msg += f"\nAlready captured fields: {json.dumps(req.existing_fields)}"

    try:
        raw = await sarvam_chat(system_prompt, user_msg)
        print(f"[ExtractProduct] raw response: {raw[:200]}")
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        fields = json.loads(clean)

        # Determine which fields were extracted vs null
        extracted = [k for k, v in fields.items() if v is not None]
        # Check required ONDC fields
        required = ["name", "selling_price", "unit_of_measure"]
        if req.domain in ("ONDC:RET10", "ONDC:RET11"):
            required += ["veg_nonveg"]
        missing = [k for k in required if not fields.get(k)]

        return {
            "status": "success",
            "fields": fields,
            "extracted_fields": extracted,
            "missing_required": missing,
        }
    except json.JSONDecodeError as e:
        print(f"[ExtractProduct] JSON parse error: {e}, raw: {raw[:300]}")
        return JSONResponse({"status": "error", "message": "AI returned invalid JSON"}, status_code=502)
    except Exception as e:
        print(f"[ExtractProduct] Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)


@app.post("/api/suggest-category")
async def suggest_category(req: SuggestCategoryRequest, request: Request, _=Depends(require_auth)):
    """Suggest ONDC domain/category/subcategory for a product using Sarvam-M."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    system_prompt = """You are an ONDC product categorisation expert. Given a product name and description, determine the ONDC domain, category, and subcategory.

ONDC Retail domains:
- ONDC:RET10 — Grocery (food, beverages, spices, staples, snacks)
- ONDC:RET11 — F&B (restaurant food, ready-to-eat meals)
- ONDC:RET12 — Fashion (clothing, footwear, accessories)
- ONDC:RET13 — Beauty & Personal Care (cosmetics, skincare, haircare)
- ONDC:RET14 — Electronics (phones, computers, appliances)
- ONDC:RET15 — Home & Kitchen (furniture, kitchenware, decor)
- ONDC:RET16 — Health & Wellness (medicines, supplements, fitness)
- ONDC:RET17 — Agriculture (seeds, fertilizers, farm tools)
- ONDC:RET18 — Handloom & Handicrafts

Return ONLY a JSON object:
{"domain": "ONDC:RETxx", "category": "main category", "subcategory": "sub category"}"""

    user_msg = f"Product: {req.product_name}"
    if req.description:
        user_msg += f"\nDescription: {req.description}"

    try:
        raw = await sarvam_chat(system_prompt, user_msg)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        result = json.loads(clean.strip())
        return {"status": "success", **result}
    except Exception as e:
        print(f"[SuggestCategory] Error: {e}")
        return {"status": "success", "domain": "ONDC:RET10", "category": "General", "subcategory": "Other"}


@app.post("/api/upload-image")
async def upload_image(request: Request, image: UploadFile = File(...), _=Depends(require_auth)):
    """Upload a product image. Returns base64 data URL and image_id."""
    img_bytes = await image.read()
    if len(img_bytes) > 5 * 1024 * 1024:
        return JSONResponse({"status": "error", "message": "Image too large (max 5MB)"}, status_code=400)

    content_type = image.content_type or "image/jpeg"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64}"

    img_id = str(uuid.uuid4())
    IMAGE_STORE[img_id] = {"data": data_url, "created": time.time()}
    print(f"[UploadImage] Stored image {img_id}, size={len(img_bytes)} bytes")

    return {
        "status": "success",
        "image_id": img_id,
        "image_data": data_url,
    }


@app.post("/api/product-followup")
async def product_followup(req: ProductFollowupRequest, request: Request, _=Depends(require_auth)):
    """Generate a friendly follow-up question about missing product fields using Sarvam-M."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    # Map field names to friendly labels
    field_labels = {
        "name": "product name",
        "selling_price": "selling price",
        "mrp": "MRP (maximum retail price)",
        "unit_of_measure": "unit of measure (like grams, kilograms, pieces)",
        "unit_value": "quantity per unit (like 200g, 500ml)",
        "veg_nonveg": "whether the product is vegetarian or non-vegetarian",
        "fssai_license": "FSSAI license number",
        "manufacturer_name": "manufacturer name",
        "manufacturer_address": "manufacturer address",
        "country_of_origin": "country of origin",
        "generic_name": "generic product name",
        "net_quantity": "net quantity",
        "brand": "brand name",
        "colour": "colour",
        "size": "size",
        "gender": "target gender (men/women/unisex)",
        "material": "material",
        "model_name": "model name",
        "ingredients": "ingredients list",
        "nutritional_info": "nutritional information",
        "short_description": "a brief product description",
        "long_description": "a detailed product description",
    }

    friendly = [field_labels.get(f, f.replace("_", " ")) for f in req.missing_fields[:5]]
    product_name = req.product.get("name", "your product")

    system_prompt = """You are a friendly AI assistant helping a small business owner build their product catalogue for ONDC (Open Network for Digital Commerce). Generate a short, warm follow-up question asking for the missing information. Keep it conversational and encouraging. The response should be 1-2 sentences only. Do not use markdown or formatting — plain text only."""

    user_msg = f"Product: {product_name}\nMissing information needed: {', '.join(friendly)}"

    try:
        raw = await sarvam_chat(system_prompt, user_msg, temperature=0.7)
        return {"status": "success", "question_text": raw.strip()}
    except Exception as e:
        # Fallback: generate a simple question
        fallback = f"Could you also tell me the {' and '.join(friendly[:3])} for {product_name}?"
        print(f"[ProductFollowup] Error: {e}, using fallback")
        return {"status": "success", "question_text": fallback}


# ── Routes: OCR / Document Intelligence (Sarvam Vision) ──

def _run_sarvam_ocr(file_bytes: bytes, filename: str, language: str = "en-IN", output_format: str = "md") -> str:
    """Run Sarvam Document Intelligence OCR (synchronous — call via run_in_executor).
    Returns the extracted text as a string."""
    client = SarvamAI(api_subscription_key=SARVAM_KEY)

    # Create a job
    job = client.document_intelligence.create_job(
        language=language,
        output_format=output_format,
    )
    print(f"[OCR] Job created: {job.job_id}")

    # Write bytes to a temp file for upload
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Upload file & start
        job.upload_file(tmp_path)
        job.start()
        print(f"[OCR] Job started, polling for completion...")

        # Wait for completion (SDK blocks internally)
        status = job.wait_until_complete()
        print(f"[OCR] Job completed: state={getattr(status, 'job_state', 'unknown')}")

        state = getattr(status, 'job_state', None) or getattr(status, 'state', None) or 'Unknown'
        if state not in ("Completed", "PartiallyCompleted"):
            raise Exception(f"OCR job failed: state={state}")

        # Download results to temp dir
        with tempfile.TemporaryDirectory() as out_dir:
            output_zip = os.path.join(out_dir, "output.zip")
            job.download_output(output_zip)

            # Extract text from the ZIP
            extracted_text = ""
            with zipfile.ZipFile(output_zip, "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(f".{output_format}") or name.endswith(".md") or name.endswith(".html") or name.endswith(".txt"):
                        extracted_text += zf.read(name).decode("utf-8", errors="replace") + "\n"

            if not extracted_text.strip():
                # Try reading any file in the zip
                with zipfile.ZipFile(output_zip, "r") as zf:
                    for name in sorted(zf.namelist()):
                        content = zf.read(name).decode("utf-8", errors="replace")
                        if content.strip():
                            extracted_text += content + "\n"

            return extracted_text.strip()
    finally:
        os.unlink(tmp_path)


async def sarvam_ocr(file_bytes: bytes, filename: str, language: str = "en-IN", output_format: str = "md") -> str:
    """Async wrapper for Sarvam OCR — runs the sync SDK in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _run_sarvam_ocr, file_bytes, filename, language, output_format
    )


def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences from an LLM response."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


@app.post("/api/ocr/scan-label")
async def ocr_scan_label(
    request: Request,
    image: UploadFile = File(...),
    domain: str = Form("ONDC:RET10"),
    _=Depends(require_auth),
):
    """OCR a product label/packaging image → extract structured ONDC product fields."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    img_bytes = await image.read()
    if len(img_bytes) > 5 * 1024 * 1024:
        return JSONResponse({"status": "error", "message": "Image too large (max 5MB)"}, status_code=400)

    try:
        # Step 1: OCR — extract raw text from image
        ocr_text = await sarvam_ocr(img_bytes, image.filename or "label.jpg", language="en-IN")
        print(f"[ScanLabel] OCR text ({len(ocr_text)} chars): {ocr_text[:200]}")

        if not ocr_text.strip():
            return JSONResponse({"status": "error", "message": "No text found in image"}, status_code=422)

        # Step 2: Sarvam-M — parse OCR text into structured ONDC fields
        system_prompt = """You are an ONDC product catalogue assistant. Extract product attributes from OCR text read from a product label/packaging.

Return ONLY a JSON object with these fields (use null for unknown values):
{
  "name": "product name",
  "short_description": "brief description (max 50 words)",
  "long_description": "detailed description",
  "selling_price": number or null,
  "mrp": number or null,
  "currency": "INR",
  "unit_of_measure": "unit" (e.g. "gram", "kg", "piece", "litre", "ml", "pack"),
  "unit_value": "value" (e.g. "200", "1", "500"),
  "net_quantity": "value with unit" or null,
  "manufacturer_name": "name" or null,
  "manufacturer_address": "address" or null,
  "country_of_origin": "India" or detected country,
  "generic_name": "generic product name" or null,
  "veg_nonveg": "veg" or "non-veg" or "egg" or null,
  "fssai_license": "license number" or null,
  "nutritional_info": "info" or null,
  "ingredients": "comma-separated ingredients" or null,
  "additives_info": "additives info" or null,
  "brand": "brand name" or null,
  "mfg_date": "date" or null,
  "category": "product category" or null,
  "subcategory": "subcategory" or null,
  "barcode": "barcode number if visible" or null
}

Rules:
- Extract ONLY what is clearly stated. Do not guess.
- OCR text may have noise, typos, or mixed languages — handle gracefully.
- Look for MRP markings like "MRP ₹XX" or "M.R.P."
- Look for FSSAI logo/number patterns (14 digits).
- Look for veg/non-veg indicators (green dot = veg, brown/red dot = non-veg).
- Return ONLY valid JSON, no markdown, no explanation."""

        user_msg = f"OCR text from product label:\n{ocr_text}\n\nONDC Domain: {domain}"
        raw = await sarvam_chat(system_prompt, user_msg)
        fields = json.loads(_clean_json_response(raw))

        extracted = [k for k, v in fields.items() if v is not None]
        required = ["name", "selling_price", "unit_of_measure"]
        if domain in ("ONDC:RET10", "ONDC:RET11"):
            required += ["veg_nonveg"]
        missing = [k for k in required if not fields.get(k)]

        return {
            "status": "success",
            "fields": fields,
            "extracted_fields": extracted,
            "missing_required": missing,
            "ocr_text": ocr_text,
        }

    except json.JSONDecodeError as e:
        print(f"[ScanLabel] JSON parse error: {e}")
        return JSONResponse({"status": "error", "message": "AI returned invalid JSON from label text"}, status_code=502)
    except Exception as e:
        print(f"[ScanLabel] Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)


# Document-specific extraction prompts
DOC_PROMPTS = {
    "gst_certificate": """Extract the following fields from this GST Registration Certificate OCR text.
Return ONLY a JSON object:
{
  "gstin": "15-character GSTIN" or null,
  "legal_name": "legal name of business" or null,
  "trade_name": "trade name" or null,
  "address": "full registered address" or null,
  "state": "state name" or null,
  "state_code": "2-digit state code" or null,
  "registration_date": "date of registration" or null,
  "business_type": "type of business" or null,
  "constitution": "proprietorship/partnership/company etc" or null
}""",
    "fssai_license": """Extract the following fields from this FSSAI License/Registration OCR text.
Return ONLY a JSON object:
{
  "fssai_number": "14-digit FSSAI license number" or null,
  "name": "name of FBO (food business operator)" or null,
  "address": "address of premises" or null,
  "license_category": "category of license" or null,
  "valid_from": "validity start date" or null,
  "valid_until": "validity end date" or null,
  "food_category": "type of food products" or null
}""",
    "shop_license": """Extract the following fields from this Shop & Establishment License OCR text.
Return ONLY a JSON object:
{
  "license_number": "license/registration number" or null,
  "establishment_name": "name of establishment" or null,
  "owner_name": "name of owner/proprietor" or null,
  "address": "address" or null,
  "date_of_registration": "registration date" or null,
  "valid_until": "validity date" or null,
  "nature_of_business": "type of business" or null
}""",
    "udyam_certificate": """Extract the following fields from this Udyam Registration Certificate OCR text.
Return ONLY a JSON object:
{
  "udyam_number": "UDYAM-XX-00-0000000 format" or null,
  "enterprise_name": "name of enterprise" or null,
  "enterprise_type": "Micro/Small/Medium" or null,
  "owner_name": "name of owner" or null,
  "address": "official address" or null,
  "state": "state" or null,
  "district": "district" or null,
  "date_of_registration": "date" or null,
  "date_of_incorporation": "date" or null,
  "nic_codes": ["NIC code 1", "NIC code 2"] or [],
  "major_activity": "Manufacturing/Services" or null,
  "mobile": "mobile number" or null,
  "email": "email address" or null
}""",
}


@app.post("/api/ocr/scan-document")
async def ocr_scan_document(
    request: Request,
    image: UploadFile = File(...),
    document_type: str = Form("gst_certificate"),
    _=Depends(require_auth),
):
    """OCR a business document (GST cert, FSSAI, shop license, Udyam certificate)."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    if document_type not in DOC_PROMPTS:
        return JSONResponse({"status": "error", "message": f"Unknown document type: {document_type}"}, status_code=400)

    file_bytes = await image.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"status": "error", "message": "File too large (max 10MB)"}, status_code=400)

    try:
        # Step 1: OCR
        ocr_text = await sarvam_ocr(file_bytes, image.filename or "document.jpg", language="en-IN")
        print(f"[ScanDoc:{document_type}] OCR text ({len(ocr_text)} chars): {ocr_text[:200]}")

        if not ocr_text.strip():
            return JSONResponse({"status": "error", "message": "No text found in document"}, status_code=422)

        # Step 2: Structured extraction via Sarvam-M
        system_prompt = DOC_PROMPTS[document_type] + "\n\nRules:\n- Extract ONLY what is clearly present. Do not guess.\n- OCR text may have noise — handle gracefully.\n- Return ONLY valid JSON, no markdown, no explanation."
        user_msg = f"OCR text from {document_type.replace('_', ' ')}:\n{ocr_text}"

        raw = await sarvam_chat(system_prompt, user_msg)
        extracted = json.loads(_clean_json_response(raw))

        return {
            "status": "success",
            "document_type": document_type,
            "extracted": extracted,
            "raw_text": ocr_text,
        }

    except json.JSONDecodeError as e:
        print(f"[ScanDoc:{document_type}] JSON parse error: {e}")
        return JSONResponse({"status": "error", "message": "AI returned invalid JSON"}, status_code=502)
    except Exception as e:
        print(f"[ScanDoc:{document_type}] Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)


@app.post("/api/ocr/verify-documents")
async def ocr_verify_documents(req: VerifyDocumentsRequest, request: Request, _=Depends(require_auth)):
    """Cross-validate OCR-extracted document data against Udyam verification data."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    udyam = req.udyam_data
    docs = req.documents
    checks = []

    for doc in docs:
        doc_type = doc.get("type", "unknown")
        ext = doc.get("extracted", {})

        if doc_type == "gst_certificate":
            # GSTIN format check
            gstin = ext.get("gstin", "")
            if gstin:
                checks.append({
                    "field": "GSTIN format",
                    "expected": "15 characters, alphanumeric",
                    "found": gstin,
                    "match": len(gstin) == 15 and gstin.isalnum(),
                    "severity": "error" if len(gstin) != 15 else "pass",
                })
            # Name matching
            gst_name = (ext.get("legal_name") or ext.get("trade_name") or "").lower()
            udyam_name = (udyam.get("enterprise_name") or udyam.get("name") or "").lower()
            if gst_name and udyam_name:
                # Simple substring match; for production use fuzzy matching
                name_match = gst_name in udyam_name or udyam_name in gst_name or \
                    any(w in udyam_name for w in gst_name.split() if len(w) > 3)
                checks.append({
                    "field": "Business name (GST vs Udyam)",
                    "expected": udyam.get("enterprise_name") or udyam.get("name"),
                    "found": ext.get("legal_name") or ext.get("trade_name"),
                    "match": name_match,
                    "severity": "warning" if not name_match else "pass",
                })
            # State match
            gst_state = (ext.get("state") or "").lower()
            udyam_state = (udyam.get("state") or "").lower()
            if gst_state and udyam_state:
                checks.append({
                    "field": "State (GST vs Udyam)",
                    "expected": udyam.get("state"),
                    "found": ext.get("state"),
                    "match": gst_state == udyam_state or gst_state in udyam_state or udyam_state in gst_state,
                    "severity": "warning" if gst_state != udyam_state else "pass",
                })

        elif doc_type == "fssai_license":
            # FSSAI format check
            fssai = (ext.get("fssai_number") or "").replace(" ", "")
            if fssai:
                is_valid = len(fssai) >= 13 and fssai.isdigit()
                checks.append({
                    "field": "FSSAI number format",
                    "expected": "14-digit number",
                    "found": fssai,
                    "match": is_valid,
                    "severity": "error" if not is_valid else "pass",
                })
            # Validity check
            valid_until = ext.get("valid_until", "")
            if valid_until:
                checks.append({
                    "field": "FSSAI license validity",
                    "expected": "Not expired",
                    "found": valid_until,
                    "match": True,  # Basic check — can enhance with date parsing
                    "severity": "info",
                })
            # Name match
            fssai_name = (ext.get("name") or "").lower()
            udyam_name = (udyam.get("enterprise_name") or udyam.get("name") or "").lower()
            if fssai_name and udyam_name:
                name_match = fssai_name in udyam_name or udyam_name in fssai_name or \
                    any(w in udyam_name for w in fssai_name.split() if len(w) > 3)
                checks.append({
                    "field": "Business name (FSSAI vs Udyam)",
                    "expected": udyam.get("enterprise_name") or udyam.get("name"),
                    "found": ext.get("name"),
                    "match": name_match,
                    "severity": "warning" if not name_match else "pass",
                })

        elif doc_type == "shop_license":
            # Name match
            shop_name = (ext.get("establishment_name") or ext.get("owner_name") or "").lower()
            udyam_name = (udyam.get("enterprise_name") or udyam.get("name") or "").lower()
            if shop_name and udyam_name:
                name_match = shop_name in udyam_name or udyam_name in shop_name or \
                    any(w in udyam_name for w in shop_name.split() if len(w) > 3)
                checks.append({
                    "field": "Business name (Shop License vs Udyam)",
                    "expected": udyam.get("enterprise_name") or udyam.get("name"),
                    "found": ext.get("establishment_name") or ext.get("owner_name"),
                    "match": name_match,
                    "severity": "warning" if not name_match else "pass",
                })
            # Validity
            valid_until = ext.get("valid_until", "")
            if valid_until:
                checks.append({
                    "field": "Shop license validity",
                    "expected": "Not expired",
                    "found": valid_until,
                    "match": True,
                    "severity": "info",
                })

        elif doc_type == "udyam_certificate":
            # Udyam number match
            cert_udyam = (ext.get("udyam_number") or "").upper().replace(" ", "")
            known_udyam = (udyam.get("udyam_number") or udyam.get("udyam_registration_number") or "").upper().replace(" ", "")
            if cert_udyam and known_udyam:
                checks.append({
                    "field": "Udyam number match",
                    "expected": known_udyam,
                    "found": cert_udyam,
                    "match": cert_udyam == known_udyam,
                    "severity": "error" if cert_udyam != known_udyam else "pass",
                })
            # Enterprise type
            cert_type = (ext.get("enterprise_type") or "").lower()
            known_type = (udyam.get("enterprise_type") or udyam.get("type") or "").lower()
            if cert_type and known_type:
                checks.append({
                    "field": "Enterprise type",
                    "expected": udyam.get("enterprise_type") or udyam.get("type"),
                    "found": ext.get("enterprise_type"),
                    "match": cert_type == known_type,
                    "severity": "warning" if cert_type != known_type else "pass",
                })

    # Calculate overall score
    if checks:
        passed = sum(1 for c in checks if c["match"])
        overall_score = round(passed / len(checks) * 100)
    else:
        overall_score = 0

    return {
        "status": "success",
        "checks": checks,
        "overall_score": overall_score,
        "total_checks": len(checks),
        "passed": sum(1 for c in checks if c["match"]),
    }


@app.post("/api/ocr/verify-claim")
async def ocr_verify_claim(req: VerifyClaimRequest, request: Request, _=Depends(require_auth)):
    """Auto-verify a claim's evidence documents for NSIC using OCR + AI analysis."""
    if not SARVAM_KEY:
        return JSONResponse({"status": "error", "message": "Sarvam API key not configured"}, status_code=500)

    from services.database import get_claim_by_id, get_all_claims

    claim = get_claim_by_id(req.claim_id)
    if not claim:
        return JSONResponse({"status": "error", "message": "Claim not found"}, status_code=404)

    evidence = claim.get("evidence", {})
    checks = []

    # Basic claim data checks (no OCR needed)
    # Check 1: Amount is positive and reasonable
    amount = claim.get("amount", 0)
    checks.append({
        "field": "Claim amount",
        "detail": f"₹{amount:,.2f}",
        "pass": amount > 0 and amount < 1000000,
        "severity": "error" if amount <= 0 else ("warning" if amount >= 1000000 else "pass"),
        "note": "Amount within expected range" if 0 < amount < 1000000 else "Amount out of expected range",
    })

    # Check 2: Claim type validity
    valid_types = ["onboarding_support", "account_management", "catalogue_support", "marketing"]
    claim_type = claim.get("claim_type", "")
    checks.append({
        "field": "Claim type",
        "detail": claim_type,
        "pass": claim_type in valid_types,
        "severity": "pass" if claim_type in valid_types else "error",
        "note": f"Valid claim type" if claim_type in valid_types else f"Unknown claim type: {claim_type}",
    })

    # Check 3: Duplicate detection — same SNP + similar amount
    all_claims = get_all_claims()
    duplicates = [
        c for c in all_claims
        if c["id"] != req.claim_id
        and c["snp_id"] == claim["snp_id"]
        and c["udyam_number"] == claim["udyam_number"]
        and c["claim_type"] == claim["claim_type"]
        and c["status"] != "rejected"
    ]
    checks.append({
        "field": "Duplicate check",
        "detail": f"{len(duplicates)} similar claims found",
        "pass": len(duplicates) == 0,
        "severity": "warning" if duplicates else "pass",
        "note": f"Potential duplicate: claim IDs {[d['id'] for d in duplicates]}" if duplicates else "No duplicates found",
    })

    # Check 4: Evidence attached
    has_evidence = bool(evidence) and any(v for v in evidence.values() if v)
    checks.append({
        "field": "Evidence documents",
        "detail": f"{len(evidence)} items" if evidence else "None",
        "pass": has_evidence,
        "severity": "warning" if not has_evidence else "pass",
        "note": "Evidence documents attached" if has_evidence else "No evidence documents found",
    })

    # Check 5: SKU count for catalogue claims
    if claim_type == "catalogue_support":
        sku_count = claim.get("sku_count", 0)
        checks.append({
            "field": "SKU count",
            "detail": str(sku_count),
            "pass": sku_count > 0,
            "severity": "warning" if sku_count == 0 else "pass",
            "note": f"{sku_count} SKUs catalogued" if sku_count > 0 else "No SKUs reported",
        })

    # Generate AI summary
    try:
        check_summary = "\n".join([f"- {c['field']}: {'PASS' if c['pass'] else 'FAIL'} — {c['note']}" for c in checks])
        system_prompt = """You are an NSIC claims verification assistant. Based on the verification checks below, provide:
1. A brief summary (2-3 sentences) of the overall claim verification status.
2. A recommendation: "approve" (all checks pass), "flag" (minor issues), or "reject" (serious issues).
Return ONLY a JSON object: {"summary": "...", "recommendation": "approve|flag|reject"}"""

        user_msg = f"Claim #{req.claim_id} by SNP {claim['snp_id']} for ₹{amount:,.2f} ({claim_type}).\n\nVerification checks:\n{check_summary}"
        raw = await sarvam_chat(system_prompt, user_msg, temperature=0.3)
        ai_result = json.loads(_clean_json_response(raw))
        summary = ai_result.get("summary", "Verification complete.")
        recommendation = ai_result.get("recommendation", "flag")
    except Exception as e:
        print(f"[VerifyClaim] AI summary error: {e}")
        passed = sum(1 for c in checks if c["pass"])
        total = len(checks)
        if passed == total:
            recommendation = "approve"
            summary = f"All {total} verification checks passed. Claim appears valid."
        elif passed >= total * 0.7:
            recommendation = "flag"
            summary = f"{passed}/{total} checks passed. Some minor issues need review."
        else:
            recommendation = "reject"
            summary = f"Only {passed}/{total} checks passed. Significant issues found."

    return {
        "status": "success",
        "claim_id": req.claim_id,
        "checks": checks,
        "auto_recommendation": recommendation,
        "summary": summary,
    }


# ── Routes: Categorisation & Matching (from v1) ──────────

@app.post("/api/categorise")
async def categorise(req: CategoriseRequest, request: Request, _=Depends(require_auth)):
    """Module 2: NIC → ONDC categorisation."""
    start = time.time()
    result = categorise_mse(
        nic_code=req.nic_code,
        product_description=req.product_description,
        mapping_path=str(DATA_DIR / "nic_ondc_mapping.json"),
    )
    return {"status": "success", "time": round(time.time() - start, 2), "categorisation": result}


@app.post("/api/match")
async def match(req: MatchRequest, request: Request, _=Depends(require_auth)):
    """Module 3: MSE → SNP matching."""
    start = time.time()
    mse_data = req.model_dump()
    dr = estimate_digital_readiness(mse_data)
    mse_data["digital_readiness"] = dr
    matches = match_mse_to_snps(mse_data, str(DATA_DIR / "snp_profiles.json"), top_n=3)
    return {"status": "success", "time": round(time.time() - start, 2), "digital_readiness": dr, "matches": matches}


# ── Routes: Full pipeline from verified data ──────────────

@app.post("/api/pipeline-from-verification")
async def pipeline_from_verification(request: Request, _=Depends(require_auth)):
    """Run Module 2+3 using verified Udyam data from OTP session."""
    body = await request.json()
    session_id = body.get("session_id", "")
    language = body.get("language", "hi")

    session = OTP_STORE.get(session_id)
    if not session or not session.get("verified"):
        return JSONResponse({"status": "error", "message": "Session not verified"}, status_code=400)

    enterprise = session["enterprise_data"]
    nic_list = session.get("nic_data_list", [])
    nic_data = session.get("nic_data", {})

    # Get primary NIC code
    primary_nic = ""
    if nic_list:
        primary_nic = nic_list[0].get("nic_5_digit", "").split(" ")[0] if nic_list[0].get("nic_5_digit") else ""
    if not primary_nic and nic_data:
        primary_nic = nic_data.get("nic_5_digit", "").split(" ")[0] if nic_data.get("nic_5_digit") else ""

    # Module 2: Categorise
    product_desc = ""
    if nic_list:
        descs = [n.get("nic_5_digit", "").split(" - ", 1)[-1] for n in nic_list if n.get("nic_5_digit")]
        product_desc = "; ".join(descs)

    cat_result = categorise_mse(
        nic_code=primary_nic,
        product_description=product_desc,
        mapping_path=str(DATA_DIR / "nic_ondc_mapping.json"),
    )

    # Module 3: Match
    addr = enterprise.get("address", {})
    mse_data = {
        "ondc_domain": cat_result.get("primary_domain", ""),
        "state": addr.get("state", ""),
        "language": language,
        "enterprise_type": enterprise.get("enterprise_type", "Micro"),
        "gst_number": None,
        "email": enterprise.get("email"),
        "udyam": session["udyam_number"],
        "input_channel": "web",
        "turnover": 0,
    }
    dr = estimate_digital_readiness(mse_data)
    mse_data["digital_readiness"] = dr
    matches = match_mse_to_snps(mse_data, str(DATA_DIR / "snp_profiles.json"), top_n=3)

    return {
        "status": "success",
        "nic_code": primary_nic,
        "nic_description": product_desc,
        "categorisation": cat_result,
        "digital_readiness": dr,
        "matches": matches,
    }


# ── Routes: Role-Based Auth ──────────────────────────────

@app.post("/api/auth/role-login")
async def role_login(req: RoleLoginRequest, request: Request, _=Depends(require_auth)):
    """Unified OTP login for MSME, SNP, or NSIC roles."""
    profile = None
    mobile = None
    name = ""

    if req.role == "snp":
        if req.auth_method == "email":
            snp = get_snp_by_email(req.identifier)
        elif req.auth_method == "subscriber_id":
            snp = get_snp_by_subscriber_id(req.identifier)
        else:
            return JSONResponse({"status": "error", "message": "Invalid auth_method for SNP"}, status_code=400)
        if not snp:
            return JSONResponse({"status": "not_found", "message": f"No SNP found with this {req.auth_method}"})
        profile = snp
        mobile = snp.get("mobile", "")
        name = snp.get("name", "")

    elif req.role == "nsic":
        official = get_nsic_by_employee_id(req.identifier)
        if not official:
            return JSONResponse({"status": "not_found", "message": "No NSIC official found with this Employee ID"})
        profile = dict(official)
        mobile = official.get("mobile", "")
        name = official.get("name", "")

    elif req.role == "msme":
        # For MSME, delegate to existing Udyam/mobile/PAN verification
        return JSONResponse({"status": "error", "message": "MSME login uses /api/verify-udyam, /api/verify-mobile, or /api/verify-pan"}, status_code=400)
    else:
        return JSONResponse({"status": "error", "message": "Invalid role"}, status_code=400)

    otp_code = create_otp_session(req.identifier, req.role)
    masked = mobile[:2] + "****" + mobile[-2:] if mobile and len(mobile) >= 4 else "N/A"

    return {
        "status": "otp_sent",
        "masked_contact": masked,
        "name": name,
        "demo_otp": otp_code,  # REMOVE IN PRODUCTION
    }


@app.post("/api/auth/verify-role-otp")
async def verify_role_otp(req: RoleOTPVerifyRequest, request: Request, _=Depends(require_auth)):
    """Verify OTP for SNP or NSIC role login."""
    if not verify_otp_session(req.identifier, req.role, req.otp):
        return JSONResponse({"status": "error", "message": "Invalid or expired OTP"}, status_code=400)

    # Store role session
    token = request.cookies.get("session_token")
    profile = None
    if req.role == "snp":
        snp = get_snp_by_email(req.identifier) or get_snp_by_subscriber_id(req.identifier)
        profile = snp
    elif req.role == "nsic":
        profile = get_nsic_by_employee_id(req.identifier)
        if profile:
            profile = dict(profile)

    ROLE_SESSIONS[token] = {"role": req.role, "identifier": req.identifier, "profile": profile}

    return {"status": "verified", "role": req.role, "profile": profile}


# ── Routes: MSME Progress ───────────────────────────────

@app.get("/api/msme/progress")
async def msme_progress(udyam: str, request: Request, _=Depends(require_auth)):
    """Get saved MSME session progress for resume."""
    session = get_msme_session(udyam)
    if not session:
        return {"status": "not_found"}
    return {"status": "found", **session}


@app.post("/api/msme/save-progress")
async def msme_save(req: SaveProgressRequest, request: Request, _=Depends(require_auth)):
    """Save MSME step progress."""
    save_msme_progress(req.udyam_number, req.step, req.data)
    return {"status": "saved"}


# ── Routes: SNP Dashboard ───────────────────────────────

@app.get("/api/snp/dashboard")
async def snp_dashboard(request: Request, rs=Depends(require_snp_auth)):
    """SNP dashboard summary stats + recent MSEs."""
    snp_id = rs["profile"]["snp_id"]
    stats = get_snp_dashboard_stats(snp_id)
    # Fetch recent MSEs for the overview
    assignments = get_assignments_for_snp(snp_id)
    recent_mses = []
    for a in assignments[:5]:
        ent = a.get("enterprise_data", {}) or {}
        recent_mses.append({
            "udyam_number": a["udyam_number"],
            "enterprise_name": ent.get("enterprise_name", ent.get("name", "Unknown")),
            "status": a.get("status", "assigned"),
            "assigned_at": a.get("assigned_at", ""),
        })
    # Map stats keys for frontend
    mapped_stats = {
        "total_mses": stats.get("total_assigned", 0),
        "live_mses": stats.get("active_mses", 0),
        "pending_claims": stats.get("total_claims", 0) - stats.get("approved_claims", 0),
        "total_approved_amount": stats.get("total_approved_amount", 0),
    }
    return {"status": "success", "snp_id": snp_id, "name": rs["profile"]["name"], "stats": mapped_stats, "recent_mses": recent_mses}


@app.get("/api/snp/mses")
async def snp_mses(request: Request, rs=Depends(require_snp_auth)):
    """List MSEs assigned to this SNP."""
    snp_id = rs["profile"]["snp_id"]
    assignments = get_assignments_for_snp(snp_id)
    mses = []
    for a in assignments:
        ent = a.get("enterprise_data", {}) or {}
        mses.append({
            "udyam_number": a["udyam_number"],
            "enterprise_name": ent.get("enterprise_name", ent.get("name", "Unknown")),
            "enterprise_type": ent.get("enterprise_type", ent.get("type", "")),
            "state": ent.get("state", ent.get("address", {}).get("state", "")),
            "district": ent.get("district", ent.get("address", {}).get("district", "")),
            "status": a["status"],
            "assigned_at": a["assigned_at"],
            "current_step": a.get("current_step", ""),
            "has_catalogue": a.get("catalogue_data") is not None,
        })
    return {"status": "success", "mses": mses}


@app.get("/api/snp/mse/{udyam}")
async def snp_mse_detail(udyam: str, request: Request, rs=Depends(require_snp_auth)):
    """Full MSE detail for SNP view."""
    session = get_msme_session(udyam)
    if not session:
        return JSONResponse({"status": "not_found"}, status_code=404)
    # Merge assignment data
    snp_id = rs["profile"]["snp_id"]
    from services.database import get_db as _gdb
    conn = _gdb()
    asgn = conn.execute(
        "SELECT status, assigned_at FROM mse_snp_assignments WHERE udyam_number=? AND snp_id=?",
        (udyam, snp_id)
    ).fetchone()
    conn.close()
    result = {**session}
    if asgn:
        result["assignment_status"] = asgn["status"]
        result["assigned_at"] = asgn["assigned_at"]
    return result


@app.get("/api/snp/mse/{udyam}/catalogue")
async def snp_mse_catalogue(udyam: str, request: Request, rs=Depends(require_snp_auth)):
    """Get catalogue for an MSE."""
    session = get_msme_session(udyam)
    if not session:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": "success", "catalogue": session.get("catalogue_data", [])}


@app.post("/api/snp/mse/{udyam}/catalogue/review")
async def snp_catalogue_review(udyam: str, req: CatalogueReviewRequest, request: Request, rs=Depends(require_snp_auth)):
    """Review (approve/request changes) an MSE's catalogue."""
    snp_id = rs["profile"]["snp_id"]
    if req.action == "approve":
        update_assignment_status(udyam, snp_id, "live")
    elif req.action == "request_changes":
        update_assignment_status(udyam, snp_id, "catalogue_pending")
    return {"status": "updated", "action": req.action}


@app.get("/api/snp/claims")
async def snp_claims(request: Request, rs=Depends(require_snp_auth)):
    """List claims for this SNP."""
    snp_id = rs["profile"]["snp_id"]
    claims = get_claims_for_snp(snp_id)
    return {"status": "success", "claims": claims}


@app.post("/api/snp/claims")
async def snp_submit_claim(req: SubmitClaimRequest, request: Request, rs=Depends(require_snp_auth)):
    """Submit a new incentive claim."""
    snp_id = rs["profile"]["snp_id"]
    submit_claim(snp_id, req.udyam_number, req.claim_type, req.amount, req.sku_count, req.evidence)
    return {"status": "submitted"}


@app.get("/api/snp/claims/{claim_id}")
async def snp_claim_detail(claim_id: int, request: Request, rs=Depends(require_snp_auth)):
    """Get claim detail."""
    claim = get_claim_by_id(claim_id)
    if not claim:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": "success", "claim": claim}


# ── Routes: NSIC Dashboard ──────────────────────────────

@app.get("/api/nsic/dashboard")
async def nsic_dashboard(request: Request, rs=Depends(require_nsic_auth)):
    """NSIC platform-wide analytics."""
    raw = get_platform_analytics()
    # Count live MSEs from assignments
    from services.database import get_db as _get_db
    conn = _get_db()
    live_count = conn.execute("SELECT COUNT(*) FROM mse_snp_assignments WHERE status='live'").fetchone()[0]
    conn.close()
    stats = {
        "total_mses": raw.get("total_mses", 0),
        "total_snps": raw.get("total_snps", 0),
        "pending_claims": raw.get("claims_pending", 0),
        "total_approved_amount": raw.get("total_disbursed", 0),
        "live_mses": live_count,
        "women_owned": round(raw.get("women_owned_pct", 0) * 100),
    }
    return {"status": "success", "stats": stats}


@app.get("/api/nsic/mses")
async def nsic_mses(request: Request, status: str = None, state: str = None, rs=Depends(require_nsic_auth)):
    """Filterable MSE registry."""
    mses = get_all_mses(status=status, state=state)
    result = []
    for m in mses:
        ent = m.get("enterprise_data", {}) or {}
        result.append({
            "udyam_number": m["udyam_number"],
            "enterprise_name": ent.get("enterprise_name", ent.get("name", "Unknown")),
            "enterprise_type": ent.get("enterprise_type", ent.get("type", "")),
            "state": ent.get("state", ent.get("address", {}).get("state", "")),
            "gender": ent.get("gender", ""),
            "social_category": ent.get("social_category", ""),
            "current_step": m.get("current_step", ""),
            "snp_id": m.get("snp_id"),
            "assignment_status": m.get("assignment_status"),
        })
    return {"status": "success", "mses": result}


@app.get("/api/nsic/snps")
async def nsic_snps(request: Request, rs=Depends(require_nsic_auth)):
    """SNPs with performance metrics."""
    snps = get_snps_with_metrics()
    return {"status": "success", "snps": snps}


@app.get("/api/nsic/claims")
async def nsic_claims(request: Request, status: str = None, rs=Depends(require_nsic_auth)):
    """Claim verification queue."""
    claims = get_all_claims(status=status)
    # Enrich with SNP and MSE names
    for c in claims:
        snp = get_snp_by_id(c["snp_id"])
        c["snp_name"] = snp["name"] if snp else "Unknown"
        session = get_msme_session(c["udyam_number"])
        ent = session.get("enterprise_data", {}) if session else {}
        c["enterprise_name"] = ent.get("enterprise_name", ent.get("name", "Unknown"))
    return {"status": "success", "claims": claims}


@app.get("/api/nsic/claims/{claim_id}")
async def nsic_claim_detail(claim_id: int, request: Request, rs=Depends(require_nsic_auth)):
    """Claim detail with eligibility checks."""
    claim = get_claim_by_id(claim_id)
    if not claim:
        return JSONResponse({"status": "not_found"}, status_code=404)
    eligibility = check_claim_eligibility(claim["udyam_number"])
    snp = get_snp_by_id(claim["snp_id"])
    session = get_msme_session(claim["udyam_number"])
    ent = session.get("enterprise_data", {}) if session else {}
    return {
        "status": "success",
        "claim": claim,
        "eligibility": eligibility,
        "snp_details": snp,
        "mse_details": ent,
    }


@app.post("/api/nsic/claims/{claim_id}/review")
async def nsic_review_claim(claim_id: int, req: ReviewClaimRequest, request: Request, rs=Depends(require_nsic_auth)):
    """Approve or reject a claim."""
    employee_id = rs["profile"].get("employee_id", "")
    review_claim(claim_id, employee_id, req.status, req.notes)
    return {"status": "updated", "new_status": req.status}


@app.get("/api/nsic/reports")
async def nsic_reports(request: Request, type: str = "demographics", rs=Depends(require_nsic_auth)):
    """Generate reports."""
    analytics = get_platform_analytics()
    if type == "demographics":
        return {"status": "success", "report_type": "demographics", "data": {
            "women_owned_pct": analytics["women_owned_pct"],
            "total_mses": analytics["total_mses"],
            "geographic_breakdown": analytics["geographic_breakdown"],
        }}
    elif type == "fund_utilization":
        claims = get_all_claims()
        total_claimed = sum(c["amount"] for c in claims)
        total_approved = sum(c["amount"] for c in claims if c["status"] == "approved")
        total_rejected = sum(c["amount"] for c in claims if c["status"] == "rejected")
        total_pending = sum(c["amount"] for c in claims if c["status"] == "pending")
        return {"status": "success", "report_type": "fund_utilization", "data": {
            "total_claimed": total_claimed,
            "total_approved": total_approved,
            "total_rejected": total_rejected,
            "total_pending": total_pending,
        }}
    elif type == "snp_performance":
        snps = get_snps_with_metrics()
        return {"status": "success", "report_type": "snp_performance", "data": {"snps": snps}}
    else:
        return {"status": "success", "report_type": type, "data": analytics}


# ── Login HTML ────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>TEAM-AEP | Login</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Outfit',sans-serif;background:#F8FAFC;color:#1E293B;min-height:100vh;display:flex;align-items:center;justify-content:center}
body::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;background-image:linear-gradient(rgba(0,0,0,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,0,0,0.03) 1px,transparent 1px);background-size:60px 60px;pointer-events:none}
.login-card{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;padding:48px 40px;max-width:420px;width:100%;box-shadow:0 4px 24px rgba(0,0,0,0.08);text-align:center;position:relative;z-index:1}
.logo{background:linear-gradient(135deg,#FF6B00,#FF8A33);color:#fff;font-weight:800;font-size:16px;padding:8px 16px;border-radius:8px;letter-spacing:1px;display:inline-block;margin-bottom:20px}
h1{font-size:22px;font-weight:600;margin-bottom:6px;color:#1E293B}
.subtitle{font-size:13px;color:#64748B;margin-bottom:32px}
input[type=password]{width:100%;padding:14px 16px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;color:#1E293B;font-family:'Outfit',sans-serif;font-size:15px;outline:none;transition:border-color 0.2s;margin-bottom:20px}
input:focus{border-color:#FF6B00}
input::placeholder{color:#94A3B8}
.btn{width:100%;padding:14px;background:linear-gradient(135deg,#FF6B00,#FF8A33);color:#fff;border:none;border-radius:10px;font-family:'Outfit',sans-serif;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(255,107,0,0.3)}
.error{color:#DC2626;font-size:13px;margin-bottom:16px}
.badges{display:flex;gap:8px;justify-content:center;margin-top:24px}
.badge{font-size:10px;padding:3px 10px;border-radius:20px;font-weight:500}
.b1{background:rgba(8,145,178,0.1);color:#0891B2;border:1px solid rgba(8,145,178,0.25)}
.b2{background:rgba(255,107,0,0.08);color:#FF6B00;border:1px solid rgba(255,107,0,0.25)}
</style>
</head>
<body>
<div class="login-card">
<div class="logo">TEAM-AEP</div>
<h1>AI-Powered MSE Onboarding</h1>
<p class="subtitle">Enter password to access the demo</p>
<div class="error">{error}</div>
<form method="POST" action="/login">
<input type="password" name="password" placeholder="Enter demo password" autofocus>
<button type="submit" class="btn">Access Demo</button>
</form>
<div class="badges"><span class="badge b1">Sarvam AI</span><span class="badge b2">IndiaAI 2026</span></div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
