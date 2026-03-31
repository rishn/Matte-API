"""
Photo Studio Backend - FastAPI Server
Zero-cost AI-powered background removal + photo editing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
import os
import uuid
import json
from supabase import create_client
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import traceback
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

# Import model handlers (will be implemented)
from models.u2net_handler import U2NetHandler
from models.sam_handler import SAMHandler
from utils.image_processing import (
    adjust_brightness_contrast,
    adjust_exposure,
    adjust_saturation,
    adjust_temperature_tint,
    apply_vignette,
    adjust_highlights_shadows,
    apply_filter_preset,
    composite_with_alpha,
    refine_mask_morphology,
    feather_mask
)

# DNS lookup for email domain validation (optional; requires dnspython)
try:
    import dns.resolver
except Exception:
    dns = None

app = FastAPI(title="Photo Studio API", version="2.6.4")

# Load environment variables from .env if present
load_dotenv()

def _init_firebase_admin_from_env():
    """Initialize firebase-admin using either:
    - FIREBASE_SERVICE_ACCOUNT_JSON (file path or JSON string or base64 JSON)
    - individual FIREBASE_* env vars (private key, client email, etc.)
    Returns True if initialized, False otherwise.
    """
    sa_env = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if sa_env:
        try:
            # If it's a JSON string
            if sa_env.strip().startswith('{'):
                sa_obj = json.loads(sa_env)
                cred = credentials.Certificate(sa_obj)
            # If it's a file path
            elif os.path.exists(sa_env):
                cred = credentials.Certificate(sa_env)
            else:
                # Try base64-decoded JSON
                try:
                    decoded = base64.b64decode(sa_env).decode('utf-8')
                    sa_obj = json.loads(decoded)
                    cred = credentials.Certificate(sa_obj)
                except Exception:
                    # last resort: try to parse as JSON directly
                    sa_obj = json.loads(sa_env)
                    cred = credentials.Certificate(sa_obj)

            firebase_admin.initialize_app(cred)
            print('Firebase admin initialized from FIREBASE_SERVICE_ACCOUNT_JSON')
            return True
        except Exception as e:
            print('Warning: Firebase admin init from FIREBASE_SERVICE_ACCOUNT_JSON failed', e)

    # Try per-attribute env vars
    private_key = os.environ.get('FIREBASE_PRIVATE_KEY')
    client_email = os.environ.get('FIREBASE_CLIENT_EMAIL')
    project_id = os.environ.get('FIREBASE_PROJECT_ID')
    if private_key and client_email:
        try:
            # replace escaped newlines
            private_key_fixed = private_key.replace('\\n', '\n')
            sa = {
                "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
                "project_id": project_id or os.environ.get('FIREBASE_PROJECT_ID', ''),
                "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": private_key_fixed,
                "client_email": client_email,
                "client_id": os.environ.get('FIREBASE_CLIENT_ID', ''),
                "auth_uri": os.environ.get('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ.get('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.environ.get('FIREBASE_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_X509_CERT_URL', ''),
                "universe_domain": os.environ.get('FIREBASE_UNIVERSE_DOMAIN', 'googleapis.com')
            }
            cred = credentials.Certificate(sa)
            firebase_admin.initialize_app(cred)
            print('Firebase admin initialized from individual FIREBASE_* env vars')
            return True
        except Exception as e:
            print('Warning: Firebase admin init from FIREBASE_* attributes failed', e)

    return False


# Initialize Firebase Admin SDK if service account provided
_init_firebase_admin_from_env()

# Supabase client (service role key required)
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
SUPABASE_BUCKET = os.environ.get('SUPABASE_BUCKET', 'projects')
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:4173", "https://matte-educify-app.web.app", "https://matte-educify-app.firebaseapp.com"],
    # Include Vite preview (`:4173`) and Firebase hosting domains used by the frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
u2net_handler = None
sam_handler = None


def get_u2net():
    """Lazy load U2Net model"""
    global u2net_handler
    if u2net_handler is None:
        u2net_handler = U2NetHandler()
    return u2net_handler


def get_sam():
    """Lazy load SAM model"""
    global sam_handler
    if sam_handler is None:
        sam_handler = SAMHandler()
    return sam_handler


# ==================== Request/Response Models ====================

class SegmentationRequest(BaseModel):
    image: str  # base64 encoded
    mode: str = "auto"  # auto, point, box
    points: Optional[List[Dict[str, float]]] = None  # [{x, y, label}]
    box: Optional[Dict[str, float]] = None  # {x1, y1, x2, y2}


class PhotoAdjustments(BaseModel):
    brightness: float = 0  # -100 to 100
    contrast: float = 0  # -100 to 100
    exposure: float = 0  # -2 to 2
    saturation: float = 0  # -100 to 100
    temperature: float = 0  # -100 to 100 (warm/cool)
    tint: float = 0  # -100 to 100 (green/magenta)
    highlights: float = 0  # -100 to 100
    shadows: float = 0  # -100 to 100
    vignette: float = 0  # 0 to 100
    sharpness: float = 0  # 0 to 100


class FilterPresetRequest(BaseModel):
    image: str  # base64 encoded
    preset: str  # vintage, cinematic, bw, warm, cool, etc.


class AdjustImageRequest(BaseModel):
    image: str  # base64 encoded
    adjustments: PhotoAdjustments


# ==================== Helper Functions ====================

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # Preserve alpha channel if present (e.g., PNG with transparency)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img


def encode_image_to_base64(img: np.ndarray, format: str = "png") -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode(f".{format}", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/{format};base64,{img_base64}"


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG"""
    # Ensure mask is uint8
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    _, buffer = cv2.imencode(".png", mask_uint8)
    mask_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{mask_base64}"


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Photo Studio API",
        "models": {
            "u2net": u2net_handler is not None,
            "sam": sam_handler is not None
        }
    }


@app.post('/api/upload')
async def upload_image(file: UploadFile = File(...), authorization: str | None = Header(None)):
    """Upload an image blob to Supabase Storage (protected bucket).
    Expects Authorization: Bearer <Firebase ID token>
    Returns storagePath and a signed download URL.
    """
    if supabase is None:
        raise HTTPException(status_code=503, detail='Supabase not configured')

    if not authorization:
        raise HTTPException(status_code=401, detail='Missing Authorization header')
    if not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Invalid Authorization header')

    id_token = authorization.split(' ', 1)[1]
    # verify token and extract uid
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        uid = decoded.get('uid')
    except Exception as e:
        raise HTTPException(status_code=401, detail=f'Invalid Firebase token: {e}')

    # Read content and upload under uid/namespace
    try:
        content = await file.read()
        filename = f"{uid}/{uuid.uuid4().hex}.png"

        # Upload to Supabase (private bucket)
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, content, {'content-type': file.content_type})

        # Helper to extract error from various response shapes
        def _extract_error(r):
            if r is None:
                return 'no response from Supabase'
            if isinstance(r, dict):
                return r.get('error')
            if hasattr(r, 'error'):
                return getattr(r, 'error')
            try:
                return r.get('error')
            except Exception:
                return None

        upload_err = _extract_error(res)
        if upload_err:
            raise HTTPException(status_code=500, detail=f"Supabase upload error: {upload_err}")

        # Create signed URL for download (24h)
        signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(filename, 60 * 60 * 24)

        def _extract_signed_url(s):
            if s is None:
                return None
            if isinstance(s, dict):
                return s.get('signedURL') or s.get('signed_url') or s.get('signedUrl')
            if hasattr(s, 'signedURL'):
                return getattr(s, 'signedURL')
            if hasattr(s, 'signed_url'):
                return getattr(s, 'signed_url')
            try:
                return s.get('signedURL') or s.get('signed_url')
            except Exception:
                return None

        signed_err = _extract_error(signed)
        if signed_err:
            raise HTTPException(status_code=500, detail=f"Supabase signed url error: {signed_err}")

        signed_url = _extract_signed_url(signed)
        return {'storagePath': filename, 'signedUrl': signed_url}
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        if os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'):
            raise HTTPException(status_code=500, detail='Upload failed (see server logs)')
        else:
            raise HTTPException(status_code=500, detail='Internal server error during upload')


@app.get('/api/check-email-domain')
async def check_email_domain(email: str = Query(..., description='Email address to validate')):
    """Check whether the email's domain appears to accept mail (MX records).
    Returns JSON: { ok: bool, reason?: str }
    Note: This is a best-effort check. Not finding MX records does not 100% prove
    the address is invalid, but it catches clearly bogus domains.
    """
    if not email or '@' not in email:
        raise HTTPException(status_code=400, detail='Invalid email')
    domain = email.split('@', 1)[1].strip().lower()
    if not domain:
        raise HTTPException(status_code=400, detail='Invalid email domain')

    # Try to resolve MX records if dnspython is available
    try:
        if dns is not None:
            try:
                answers = dns.resolver.resolve(domain, 'MX')
                if answers and len(answers) > 0:
                    return { 'ok': True }
            except Exception:
                # fallthrough to next checks
                pass

        # Fallback: try A/AAAA record lookup via socket by resolving domain
        import socket
        try:
            socket.gethostbyname(domain)
            return { 'ok': True }
        except Exception:
            return { 'ok': False, 'reason': 'no_mx_or_a_record' }
    except Exception as e:
        # Unexpected errors
        return { 'ok': False, 'reason': 'lookup_failed', 'detail': str(e) }


@app.get('/api/signed-url')
async def get_signed_url(path: str, authorization: str | None = Header(None)):
        """Return a signed URL for a stored object. Requires Authorization: Bearer <ID token>.
        Ensures the requesting user owns the object (path must start with uid/).
        """
        if supabase is None:
            raise HTTPException(status_code=503, detail='Supabase not configured')
        if not authorization or not authorization.lower().startswith('bearer '):
            raise HTTPException(status_code=401, detail='Missing Authorization header')
        id_token = authorization.split(' ', 1)[1]
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            uid = decoded.get('uid')
        except Exception as e:
            raise HTTPException(status_code=401, detail=f'Invalid Firebase token: {e}')

        if not path.startswith(f"{uid}/"):
            raise HTTPException(status_code=403, detail='Forbidden')

        try:
            def _is_prefix(p):
                return p.endswith('/') or '.' not in p.split('/')[-1]

            target_file = None
            if _is_prefix(path):
                # list objects under prefix
                listing = supabase.storage.from_(SUPABASE_BUCKET).list(path)
                files = []
                if isinstance(listing, dict):
                    files = listing.get('data') or listing.get('files') or []
                else:
                    files = getattr(listing, 'data', None) or listing
                if files:
                    first = files[0]
                    if isinstance(first, dict):
                        target_file = f"{path.rstrip('/')}/{first.get('name')}"
                    else:
                        target_file = f"{path.rstrip('/')}/{first}"
            else:
                target_file = path

            if not target_file:
                raise HTTPException(status_code=404, detail='No object found for given path/prefix')

            signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(target_file, 60 * 60 * 24)
            if isinstance(signed, dict):
                signed_url = signed.get('signedURL') or signed.get('signed_url') or signed.get('signedUrl')
            else:
                signed_url = getattr(signed, 'signedURL', None) or getattr(signed, 'signed_url', None)
            if not signed_url:
                raise HTTPException(status_code=500, detail='Failed to create signed URL')
            return {'signedUrl': signed_url, 'resolvedPath': target_file}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/delete')
async def delete_object(path: str | None = None, authorization: str | None = Header(None)):
        """Delete an object from Supabase storage. Body or query param 'path' required.
        Only allows deleting objects under the authenticated user's uid/ prefix.
        """
        if supabase is None:
            raise HTTPException(status_code=503, detail='Supabase not configured')
        if not path:
            raise HTTPException(status_code=400, detail='Missing path')
        if not authorization or not authorization.lower().startswith('bearer '):
            raise HTTPException(status_code=401, detail='Missing Authorization header')
        id_token = authorization.split(' ', 1)[1]
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            uid = decoded.get('uid')
        except Exception as e:
            raise HTTPException(status_code=401, detail=f'Invalid Firebase token: {e}')

        if not path.startswith(f"{uid}/"):
            raise HTTPException(status_code=403, detail='Forbidden')

        try:
            def _is_prefix(p):
                return p.endswith('/') or '.' not in p.split('/')[-1]

            paths_to_delete = []
            if _is_prefix(path):
                listing = supabase.storage.from_(SUPABASE_BUCKET).list(path)
                files = []
                if isinstance(listing, dict):
                    files = listing.get('data') or listing.get('files') or []
                else:
                    files = getattr(listing, 'data', None) or listing
                for f in files:
                    if isinstance(f, dict):
                        name = f.get('name')
                    else:
                        name = f
                    if name:
                        paths_to_delete.append(f"{path.rstrip('/')}/{name}")
            else:
                paths_to_delete = [path]

            if not paths_to_delete:
                raise HTTPException(status_code=404, detail='No object(s) found to delete')

            res = supabase.storage.from_(SUPABASE_BUCKET).remove(paths_to_delete)
            err = None
            if isinstance(res, dict):
                err = res.get('error')
            else:
                err = getattr(res, 'error', None)
            if err:
                raise HTTPException(status_code=500, detail=f'Supabase delete error: {err}')
            return {'success': True, 'deleted': paths_to_delete}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        return {'success': True, 'deleted': paths_to_delete}


@app.post("/api/segment/interactive")
async def interactive_segment(request: SegmentationRequest):
    """
    Interactive segmentation using SAM
    Supports point and box prompts for precise selection
    """
    try:
        # Decode image
        img = decode_base64_image(request.image)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get SAM handler
        handler = get_sam()
        
        # Prepare prompts
        point_coords = None
        point_labels = None
        box_coords = None
        
        if request.mode == "point" and request.points:
            point_coords = np.array([[p["x"], p["y"]] for p in request.points])
            point_labels = np.array([p.get("label", 1) for p in request.points])
        
        if request.mode == "box" and request.box:
            box_coords = np.array([
                request.box["x1"], request.box["y1"],
                request.box["x2"], request.box["y2"]
            ])
        
        # Predict with SAM
        mask = handler.predict(
            img,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords
        )
        
        # Refine and feather
        mask_refined = refine_mask_morphology(mask, kernel_size=3)
        if img.shape[2] == 4:
            _, _, _, a_exist = cv2.split(img)
            keep = (a_exist > 0).astype(np.uint8) * 255
            mask_refined = cv2.bitwise_and(mask_refined, keep)
        mask_feathered = feather_mask(mask_refined, feather_amount=5)
        
        # Create alpha composite; keep any existing transparency from the input
        if img.shape[2] == 4:
            b, g, r, a_existing = cv2.split(img)
            mask_u8 = (mask_feathered if mask_feathered.dtype == np.uint8 else np.clip(mask_feathered, 0, 255).astype(np.uint8))
            if len(mask_u8.shape) == 3:
                mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
            a_final = np.minimum(a_existing, mask_u8)
            result_rgba = cv2.merge([b, g, r, a_final])
        else:
            result_rgba = composite_with_alpha(img, mask_feathered)
        
        # Encode results
        result_base64 = encode_image_to_base64(result_rgba, format="png")
        mask_base64 = encode_mask_to_base64(mask_feathered)
        
        return {
            "success": True,
            "result": result_base64,
            "mask": mask_base64,
            "method": f"sam_{request.mode}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segment/auto")
async def auto_segment(file: UploadFile = File(...)):
    """
    Automatic segmentation using U2Net (multipart file upload)
    Expects form field `file` with an image. Returns mask + result PNG with alpha.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail='Empty file')

        # Decode image bytes into OpenCV image (preserve alpha if present)
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail='Invalid image data')

        handler = get_u2net()
        mask = handler.predict(img)

        # Refine and feather mask
        mask_refined = refine_mask_morphology(mask, kernel_size=3)
        if img.shape[2] == 4:
            _, _, _, a_exist = cv2.split(img)
            keep = (a_exist > 0).astype(np.uint8) * 255
            mask_refined = cv2.bitwise_and(mask_refined, keep)
        mask_feathered = feather_mask(mask_refined, feather_amount=5)

        # Create alpha composite; keep existing transparency if any
        if img.shape[2] == 4:
            b, g, r, a_existing = cv2.split(img)
            mask_u8 = (mask_feathered if mask_feathered.dtype == np.uint8 else np.clip(mask_feathered, 0, 255).astype(np.uint8))
            if len(mask_u8.shape) == 3:
                mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
            a_final = np.minimum(a_existing, mask_u8)
            result_rgba = cv2.merge([b, g, r, a_final])
        else:
            result_rgba = composite_with_alpha(img, mask_feathered)

        result_base64 = encode_image_to_base64(result_rgba, format="png")
        mask_base64 = encode_mask_to_base64(mask_feathered)

        return {
            "success": True,
            "result": result_base64,
            "mask": mask_base64,
            "method": "u2net_auto"
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/adjust")
async def adjust_image(request: AdjustImageRequest):
    """
    Apply photo adjustments to image
    Supports brightness, contrast, exposure, saturation, etc.
    """
    try:
        # Decode image
        img = decode_base64_image(request.image)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Apply adjustments in sequence
        # Preserve alpha if present
        has_alpha = (img.shape[2] == 4)
        if has_alpha:
            bgr = img[:, :, :3].copy()
            alpha_channel = img[:, :, 3].copy()
        else:
            bgr = img.copy()
        result = bgr
        adj = request.adjustments
        
        # Brightness & Contrast
        if adj.brightness != 0 or adj.contrast != 0:
            result = adjust_brightness_contrast(result, adj.brightness, adj.contrast)
        
        # Exposure
        if adj.exposure != 0:
            result = adjust_exposure(result, adj.exposure)
        
        # Saturation
        if adj.saturation != 0:
            result = adjust_saturation(result, adj.saturation)
        
        # Temperature & Tint
        if adj.temperature != 0 or adj.tint != 0:
            result = adjust_temperature_tint(result, adj.temperature, adj.tint)
        
        # Highlights & Shadows
        if adj.highlights != 0 or adj.shadows != 0:
            result = adjust_highlights_shadows(result, adj.highlights, adj.shadows)
        
        # Vignette
        if adj.vignette > 0:
            result = apply_vignette(result, adj.vignette)
        
        # Sharpness
        if adj.sharpness > 0:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * (adj.sharpness / 100)
            kernel[1, 1] = 8 + (adj.sharpness / 100)
            result = cv2.filter2D(result, -1, kernel / kernel.sum())
        
        # Encode result
        if has_alpha:
            # Merge processed BGR with original alpha
            b, g, r = cv2.split(result)
            rgba = cv2.merge([b, g, r, alpha_channel])
            result_base64 = encode_image_to_base64(rgba, format="png")
        else:
            result_base64 = encode_image_to_base64(result, format="png")
        
        return {
            "success": True,
            "result": result_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/filter")
async def apply_filter(request: FilterPresetRequest):
    """
    Apply preset filter to image
    Presets: vintage, cinematic, bw, warm, cool, dramatic, etc.
    """
    try:
        # Decode image
        img = decode_base64_image(request.image)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Apply preset filter
        has_alpha = (img.shape[2] == 4)
        if has_alpha:
            bgr = img[:, :, :3]
            alpha_channel = img[:, :, 3]
        else:
            bgr = img
        filtered_bgr = apply_filter_preset(bgr, request.preset)
        if has_alpha:
            b, g, r = cv2.split(filtered_bgr)
            result_rgba = cv2.merge([b, g, r, alpha_channel])
            result = result_rgba
        else:
            result = filtered_bgr
        
        # Encode result
        result_base64 = encode_image_to_base64(result, format="png")
        
        return {
            "success": True,
            "result": result_base64,
            "filter": request.preset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filters/list")
async def list_filters():
    """Get available filter presets"""
    return {
        "filters": [
            {"id": "vintage", "name": "Vintage", "description": "Warm tones, faded look"},
            {"id": "cinematic", "name": "Cinematic", "description": "Teal & orange, high contrast"},
            {"id": "bw", "name": "B&W", "description": "Classic monochrome"},
            {"id": "warm", "name": "Warm", "description": "Increase warmth and saturation"},
            {"id": "cool", "name": "Cool", "description": "Cool tones, blue tint"},
            {"id": "dramatic", "name": "Dramatic", "description": "High contrast, deep shadows"},
            {"id": "soft", "name": "Soft", "description": "Low contrast, pastel"},
            {"id": "vivid", "name": "Vivid", "description": "Boosted saturation"},
            {"id": "sepia", "name": "Sepia", "description": "Warm brown vintage"},
            {"id": "fade", "name": "Fade", "description": "Washed out, retro"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
