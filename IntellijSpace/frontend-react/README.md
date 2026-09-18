# MVGGT React Frontend (r3f)

This folder contains a React + Three.js (`@react-three/fiber`) implementation
that replaces the old static viewer for real-time interaction.

## Highlights

- Uses existing backend APIs (`/api/v1`)
- Supports task submit, status polling, and result retrieval
- Supports `result` / `nomask` / `mask` variant switching
- Real-time slider transform for replacement asset in frontend Three.js scene
- Calls backend `/dynamic` only when downloading to persist slider transform

## API base resolution

Priority:

1. `?api_base=...` in URL
2. `localStorage.MVGGT_API_BASE`
3. `VITE_API_BASE`
4. fallback `/api/v1` (via Vite proxy)

## Backend expectation

Run backend at port `8001`:

`uvicorn backend.main:app --host 0.0.0.0 --port 8001`

`vite.config.js` already proxies `/api/v1` to `http://127.0.0.1:8001`.
