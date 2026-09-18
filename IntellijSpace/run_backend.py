"""Start the application with its bundled model package taking precedence."""
import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001)