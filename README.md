# <img src="https://github.com/rishn/Matte/blob/main/assets/Matte.png" alt="Matte logo" width="22" height="22" /> Matte - API
<p align="center">
  <img src="https://github.com/rishn/Matte/blob/main/assets/Expanse.png" alt="Expanse" width="60%" height="60%" />
</p>

<p align="center">
  <img alt="React" src="https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black" />
  <img alt="Vite" src="https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white" />
  <img alt="Konva" src="https://img.shields.io/badge/Konva-793EF7?logo=konva&logoColor=white" />
  <img alt="Zustand" src="https://img.shields.io/badge/Zustand-000000?logo=data:image/png;base64,iVBORw0KGgo" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="SAM" src="https://img.shields.io/badge/Segment%20Anything-model?color=8b5cf6" />
  <img alt="U2Net" src="https://img.shields.io/badge/U2-Net-model?color=7c3aed" />
  <img alt="GitHub Actions" src="https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
  <img alt="Render" src="https://img.shields.io/badge/Render-20232A?logo=render&logoColor=white" />
  <img alt="Firebase" src="https://img.shields.io/badge/Firebase-FFCA28?logo=firebase&logoColor=black" />
  <img alt="Supabase" src="https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white" />
</p>

**Matte,** a professional-grade photo editing web application with **AI-powered background removal, segmentation** and comprehensive photo editing features. Built with zero-cost deployment in mind using open-source models. This repository contains the backend workspace of the application.

![Editor](https://github.com/rishn/Matte/blob/main/assets/Editor.png)

<p align="center">
  <a href="https://matte-educify-app.web.app" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Open%20Deployed%20Site-Visit-blue?style=for-the-badge" alt="Open deployed site" />
  </a>
</p>

## Demos

- Desktop Demo 

<p align="center">
  <a href="https://drive.google.com/file/d/1nsE5mgC_IrErSmwBttrUX9DTZVUw6cI1/view?usp=sharing" target="_blank" rel="noopener noreferrer">
    <img src="https://github.com/user-attachments/assets/8574a4a7-40a0-40f9-bde4-b95663f55779" alt="Desktop Demo" width="600" />
  </a>
</p>

## Features

### AI-Powered Tools
- **Automatic Background Removal**: U²-Net for fast, accurate salient object detection
- **Magic Wand Tool**: SAM (Segment Anything Model) for interactive segmentation with point/box prompts
- **Smart Masking**: Morphological refinement and feathering for professional results

### Photo Editing
- **Adjustments**: Brightness, Contrast, Exposure, Saturation
- **Color Grading**: Temperature, Tint controls
- **Tone Controls**: Highlights, Shadows
- **Effects**: Vignette, Sharpness
- **Preset Filters**: Vintage, Cinematic, B&W, Warm, Cool, Dramatic, Soft, Vivid, Sepia, Fade

### Photoshop-like Interface
- Professional dark theme UI
- Tool panel with selection, magic wand, eraser, move, and box select tools
- Real-time canvas editing with Konva
- Properties panel with collapsible sections
- Layers panel for organization

## Screenshots (Edited with Matte!)

- Projects Page
![Projects](https://github.com/rishn/Matte/blob/main/assets/Projects.png)

- Using Crop Tool
![Crop Tool](ahttps://github.com/rishn/Matte/blob/main/assets/CropTool.png)

- Sign Up Page
![Sign Up](https://github.com/rishn/Matte/blob/main/assets/SignUp.png)

## Zero-Cost Setup

This application is designed to run **completely free** using:
- **Local CPU inference** (no cloud GPU costs)
- **Open-source models** (U²-Net, SAM)
- **Self-hosted backend** (FastAPI)
- **Modern frontend** (React + Vite)
- **Optional authentication and verification** (Firebase Auth, Firestore)
- **Optional object storage** (Supabase Storage) for storing full-resolution project images and reducing Firestore document sizes

## Prerequisites

- **Python 3.10+** with pip
- **Node.js 16+** with npm
- **4GB+ RAM** (8GB+ recommended for SAM)
- **Git**

## Quick Start

### 1. Backend Setup

```powershell
# Clone the backend repo
git clone https://github.com/rishn/Matte-API.git

# Navigate to backend
cd PhotoStudio\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Frontend Setup

```powershell
# Clone the frontend repo
git clone https://github.com/rishn/Matte-UI.git

# Navigate to frontend
cd PhotoStudio\frontend

# Install dependencies
npm install
```

### 3. Download Model Weights

Model weight files (place them under `backend/models/weights` or where your handlers expect them):

**U²-Net Weights:**  (U2NetP) — recommended for CPU inference
```powershell
Visit: https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view
Download to: .\backend\models\weights\u2netp.pth
```

**SAM Weights (optional, for magic wand tool):** (SAM ViT-B) — optional for interactive segmentation
```powershell
# Download SAM ViT-B (smallest, fastest for CPU)
Visit: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
Download to: .\backend\models\weights\sam_vit_b_01ec64.pth
```

## 4. Run and Test

1. **API**
   ```powershell
   # Navigate to backend
   cd PhotoStudio\backend

   # Start backend server
   # The backend exposes API routes used by the frontend (upload, signed-url, segmentation, etc.).
   # You can run directly with Python or via Uvicorn:
   python app.py # (or) uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Backend will run at: **http://localhost:8000**

2. **UI**
   ```powershell
   # Open new terminal and navigate to frontend
   cd PhotoStudio\frontend

   # Start development server
   npm run dev
   ```

   Frontend will run at: **http://localhost:3000**

3. The application is ready to test.

## Usage Guide

### Basic Workflow

1. **Open Image**: Click "Open" button in header to load an image
2. **Auto Remove Background**: 
   - Tool: Select "Magic Wand" tool
   - Click on subject to segment automatically
   - Uses U²-Net for fast background removal

3. **Interactive Selection**:
   - Tool: "Box Select" for box-based segmentation
   - Click points on object for SAM-based selection
   - Green points = foreground, Red points = background

4. **Apply Adjustments**:
   - Open Properties panel
   - Use sliders for brightness, contrast, etc.
   - Changes apply in real-time

5. **Apply Filters**:
   - Choose from preset filters (Vintage, Cinematic, etc.)
   - Click to apply instantly

6. **Export**: Click "Export" to download edited image

### Tool Descriptions

| Tool | Icon | Function |
|------|------|----------|
| Remove Background | 🖼️➖ | Remove background using AI |
| Magic Wand | 🪄 | AI-powered object selection |
| Eraser | 🧽 | Remove parts of image |
| Pen | ✏️ | Add annotations and edits to images |
| Box Select | ⬜ | Rectangle selection for SAM |

## API Endpoints

### Background Removal
- `POST /api/segment/auto` - Automatic segmentation (U²-Net) (multipart form-data file upload)
- `POST /api/segment/interactive` - Interactive segmentation (SAM) (JSON body: `{ image, mode, points?, box? }`)

### Storage & Upload
- `POST /api/upload` - Upload an image blob to server-side storage (Supabase). Expects `Authorization: Bearer <Firebase ID token>` and returns a `storagePath` plus a signed download URL.
- `GET /api/signed-url?path=<storagePath>` - Resolve a stored path to a temporary signed download URL (requires auth).
- `POST /api/delete?path=<storagePath>` - Delete previously uploaded storage objects (requires auth).

### Misc
- `GET /api/check-email-domain?email=...` - (Optional) Best-effort check to verify the email's domain has MX/A records to prevent creating accounts for clearly bogus domains.

### Photo Editing
- `POST /api/adjust` - Apply adjustments (JSON body: `{ image, adjustments }`)
- `POST /api/filter` - Apply preset filter (JSON body: `{ image, preset }`)
- `GET /api/filters/list` - Get/list available filters


## Frontend dependencies (where to find them)

All Node dependencies for the UI are declared in `frontend/package.json`. Key packages used in the UI include:

- `react`, `react-dom` — React
- `vite`, `@vitejs/plugin-react` — dev server/build
- `konva`, `react-konva` — canvas and drawing
- `axios` — HTTP client
- `zustand` — state management
- `lucide-react` — icons
- `react-dropzone` — file drop handling

## Configuration

### GPU Acceleration (Optional)

If you have a CUDA-capable GPU:

```powershell
# Uninstall CPU PyTorch
pip uninstall torch torchvision

# Install GPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
Matte/
├── backend/
│   ├── app.py                         # FastAPI server (API routes + Firebase Admin)
|   ├── .env
│   ├── models/
│   │   ├── u2net_handler.py           # U²-Net wrapper
│   │   └── sam_handler.py             # SAM wrapper
│   ├── utils/
│   │   └── image_processing.py        # Photo editing functions
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── firebase.json                  # Firebase hosting / config (optional)
│   ├── .firebaserc                    # Firebase CLI project alias (optional)
│   ├── .env                           # Vite env (VITE_FIREBASE_*, VITE_API_URL, etc.)
│   ├── src/
│   │   ├── firebaseConfig.js          # Firebase client init used by the UI
│   │   ├── components/                # React components (Header, Canvas, Panels, etc.)
│   │   ├── services/                  # API client + save/upload helpers
│   │   ├── store/                     # Zustand state (editor, toasts, history)
│   │   ├── utils/                     # helpers (authErrors, adjustments, compositor)
│   │   └── App.jsx
│   └── package.json
│   └── public/
├── U-2-Net/                           # Cloned repo (gitignored)
└── segment-anything/                  # Cloned repo (gitignored)
```

## Troubleshooting

- Start the backend first so the frontend proxy can reach `/api`.
- If you change Python dependencies, re-create the venv and `pip install -r requirements.txt`.
- To test CORS or proxy issues, hit `http://localhost:8000/` (health endpoint) and `http://localhost:3000/` (frontend).

### Models Not Loading
- Ensure weights are downloaded to correct directories
- Check console logs for specific errors
- Models will fall back to traditional CV methods if unavailable

### Slow Performance
- Use lighter model variants (U2NetP, SAM ViT-B)
- Reduce input image size
- Consider ONNX export for production use

### CORS Errors
- Backend must be running on port 8000
- Frontend on port 3000
- Check CORS middleware configuration in `app.py`

### Debug / Development flags

- `DEBUG` (env var): when set to `1`, `true`, or `yes` (case-insensitive) the backend will surface more detailed error information and may re-raise certain exceptions to help debugging. For example, upload and signed-url endpoints include more descriptive errors in responses when `DEBUG` is enabled. Do not enable `DEBUG` in production.

   ```powershell
   # .env (backend)
   DEBUG=true #development
   ```

## Authentication and Storage (optional)
### Firebase — setup for Projects and Auth

If you enable Firebase integration for user login and cloud projects, create a Firebase project and set the following Vite environment variables in a local `.env` or `.env.local` file at the `frontend` folder root:

```powershell
# .env (frontend)
VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_auth_domain
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_FIREBASE_CONTINUE_URL=http://localhost:3000
VITE_API_URL=http://localhost:8000/api
```

Notes:
- The frontend scaffold uses **Email/Password** authentication via **Firebase Authentication** (with email verification).
- The Projects feature saves small, compressed thumbnails directly inside **Firestore** documents as base64 (`thumbnailBase64`) to avoid requiring Firebase Storage or paid plans. Small thumbnails (suggested max 256px, JPEG quality ~0.7) are also stored to stay well under Firestore's 1 MiB document size limit.

### Supabase (recommended for storing full-resolution images)

This project uses Supabase Storage to hold full-resolution project images to avoid saving large base64 blobs in Firestore documents (which can exceed Firestore's 1 MiB limit). If you enable Supabase, set the following environment variables for the backend (or your deployment):

```powershell
# .env (backend)
SUPABASE_URL=https://xyzcompany.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_BUCKET=projects
```

Notes:
- The backend uses the Supabase service role key to upload/delete objects on behalf of authenticated users and to create signed URLs. Keep this key secret (server-side only).
- The frontend will call backend endpoints (`/api/upload`, `/api/signed-url`, `/api/delete`) which perform token verification and map storage operations to the requesting user.

### Firebase Service Account (admin)

The backend uses the Firebase Admin SDK to verify Firebase ID tokens and perform privileged server-side operations (for example, `verify_id_token` to authenticate uploads, generate signed URLs, or check ownership of objects). The Admin SDK requires service-account credentials so the server can safely perform these admin-only checks.

Where to get the service account secrets

- Firebase Console: Project Settings → Service Accounts → Generate new private key (this downloads the JSON key file).
- Google Cloud Console: IAM & Admin → Service Accounts → Create/Manage keys → Download JSON key. Both sources produce the same JSON credentials the Admin SDK accepts.

You can supply the service account credentials in several ways; the backend supports multiple formats for convenience:

- Provide the full service account JSON via an environment variable:
   ```powershell
   # .env (backend)
   FIREBASE_SERVICE_ACCOUNT_JSON # the full JSON string (keep this secret on the server).
   ```

- Point to a downloaded JSON file:
   ```powershell
   # .env (backend)
   FIREBASE_SERVICE_ACCOUNT_JSON=C:\path\to\service-account.json
   ```

- Provide individual fields as environment variables (for ease during hosting):
   ```powershell
   # .env (backend)
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n---\n-----END PRIVATE KEY-----\n" # ensure line breaks are preserved; replace literal newlines with `\n` if necessary
   FIREBASE_CLIENT_EMAIL=your_adminsdk_username@your_project_id.iam.gserviceaccount.com
   FIREBASE_PRIVATE_KEY_ID=your_private_key
   FIREBASE_CLIENT_ID=your_client_id
   FIREBASE_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your_client_email
   ```

- Or export a downloaded JSON into an env var (beware of quoting/newlines):

   ```powershell
   $svc = Get-Content 'path\to\service-account.json' -Raw
   [Environment]::SetEnvironmentVariable('FIREBASE_SERVICE_ACCOUNT_JSON', $svc, 'User')
   ```

Keep the service account secret server-side — do not expose it to the frontend.

### Email Domain Validation (optional)

For additional protection against account abuse, the backend exposes a best-effort `GET /api/check-email-domain` endpoint which attempts to resolve MX records for the provided email domain (uses `dnspython` when available, otherwise falls back to an A/AAAA lookup). This helps block obviously bogus emails at signup.

To enable MX checks (recommended), install `dnspython` in your backend environment:

```powershell
cd PhotoStudio/backend
pip install dnspython
```

The frontend uses this endpoint before attempting signup; the check is best-effort and will not block signup on transient network/DNS errors by default.

## Deployment

- **Frontend — Firebase Hosting + GitHub Actions:** 
   - The frontend is hosted on Firebase Hosting. In CI the workflow creates a `.env.production` (or sets build-time secrets) containing `VITE_API_URL` and `VITE_USE_SAM` before running `npm run build` so Vite embeds the build-time flags. Changing any `VITE_` environment variables requires rebuilding the frontend. 
   - A sample GitHub Actions workflow has been set up at `frontend/.github/workflows/firebase-hosting.yml`, **to enable automatic build and deploy on repo updates.** The workflow requires the repository secrets `FIREBASE_TOKEN`, `VITE_API_URL`, and `VITE_USE_SAM` to be configured.

- **Backend — Render:** 
   - The backend is hosted as a Render Web Service (or similar). Server environment variables include `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `FIREBASE_*`, `USE_SAM`, and `MODEL_CACHE_DIR`. 
   - For reliable model availability, U2Net is baked into the image at build time when `BUILD_MODELS=true` and `U2NET_DOWNLOAD_URL` are used, or a persistent disk is attached and mounted at `/models` so weights persist across instances. The `docker-entrypoint.sh` script begins with a proper shebang (`#!/usr/bin/env bash`) on the first line and uses LF line endings; the image should be rebuilt after any entrypoint changes.


## Roadmap

- [X] Filters + custom adjustments utility
- [X] History/undo system
- [X] Multi-layer compositing
- [X] Authentication
- [X] Save and work with multiple projects
- [ ] Explore other AI features

## Acknowledgments

- [U²-Net](https://github.com/NathanUA/U-2-Net) by Xuebin Qin et al.
- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI


**Built using zero-cost, open-source tools**

## Notes

** *Subject segmentation (subject select) is disabled in hosted application, to deploy the remaining features smoothly using available free hosting services*