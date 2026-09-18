# IntellijSpace: MVGGT Web Demo

Upload multi-view images or a video to segment or remove objects and replace furniture in 3D scenes using natural-language instructions. Preview the results and download them as GLB files.

The frontend uses React, Vite, and React Three Fiber; the backend uses FastAPI.

## Directory Layout and Model Version

- backend/: HTTP API, task orchestration, instruction parsing, and geometry editing.
- frontend-react/: uploads, task status, Three.js previews, and interactive transforms.
- mvggt/: the model source snapshot used by this application.
- example/, glb/, image/: sample images, furniture GLB assets, and asset images.
- app.py: the standalone Gradio demo and its predict_remote API.
- backend_workspace/, models/, ckpts/: runtime data and locally prepared files; excluded from version control.

The application's mvggt/models/mvggt_training.py differs from the version at the repository root, and the application also includes mvggt/models/mvggt.py. The complete application model source is therefore kept here without replacing the repository's training model.

Run commands from this directory, or launch the backend using the path to run_backend.py, to ensure that the correct model version is imported.

## Installation

Run these commands from the repository root in PowerShell. Skip the environment creation step if the environment already exists:

    cd IntellijSpace
    conda create -n mvggt python=3.10 -y
    conda activate mvggt
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

Edit .env and set MVGGT_LLM_API_KEY. The default configuration uses Qwen to parse natural-language instructions and requires a valid API key.

For segmentation only, leave MVGGT_LLM_API_URL empty. The raw prompt will then be used directly as the segmentation target, without interpreting REMOVE or REPLACE instructions.

The actual .env file is ignored by Git. Do not commit credentials.

requirements_demo.txt preserves the original model and Gradio dependency versions. requirements.txt adds the web backend dependencies.

Local GPU inference requires a CUDA-enabled PyTorch build compatible with your driver. This project pins torch 2.5.1 and torchvision 0.20.1. Check the installation with:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

## Running the Backend and Frontend

Start the backend from this directory:

    conda activate mvggt
    python run_backend.py

- API documentation: http://127.0.0.1:8001/docs
- Health check: http://127.0.0.1:8001/api/v1/health

Open another terminal in this directory and start the frontend:

    cd frontend-react
    npm ci
    npm run dev

Open http://localhost:5173. Vite proxies /api/v1 requests to the backend on local port 8001.

Use Node.js 20.19+ within the 20.x series, or Node.js 22.12 and later, as required by the locked Vite version's engines field.

Run npm run build from frontend-react/ to generate production static files. The Vite development proxy is not part of the production build; configure an /api/v1 reverse proxy or set VITE_API_BASE for deployment. The frontend also accepts the api_base URL parameter and localStorage.MVGGT_API_BASE.

## Inference and Model Setup

- local: local PyTorch inference; requires sufficient GPU memory and complete model weights.
- hf_api: calls the configured Hugging Face Space; depends on remote availability, quotas, and API compatibility.
- auto: selects a backend based on the local GPU status.

The current backend requires the model-related Python dependencies even when using remote inference.

Local inference first checks MVGGT_LOCAL_MODEL_PATH. If the weights are missing, it downloads them from the configured Hugging Face repository into the model cache directory. The tokenizer first checks MVGGT_TOKENIZER_PATH and falls back to roberta-base. The model's text encoder may also need to download files from Hugging Face.

Model weights are not included. The first run may require network access and a lengthy download. When a local GPU is detected, the backend attempts to warm up the model at startup, which may trigger downloads. Warmup failures are logged as warnings.

The frontend supports switching between result, nomask, and mask outputs. Rotation and scaling of replacement assets are previewed in the browser; downloading calls the dynamic endpoint to export the transformed result.

## Standalone Gradio Demo

Run from this directory:

    python app.py

This entry point runs independently of FastAPI, loads the model at startup, and requires CUDA for inference. It retains the original demo's model download and path behavior and does not read the FastAPI .env configuration. Its predict_remote endpoint provides the corresponding remote inference API.

## Version Control and Runtime Limitations

Commit source code, dependency manifests, the frontend lockfile, and small sample assets. Exclude node_modules, dist, model weights, uploaded files, task outputs, Python caches, and credentials.

The application currently uses in-memory task records and background threads for demonstration purposes. Task queries cannot be restored after a backend restart.
