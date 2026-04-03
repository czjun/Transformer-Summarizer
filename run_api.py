import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import uvicorn


if __name__ == "__main__":
    uvicorn.run("summarizer_app.api:app", host="0.0.0.0", port=8888, reload=False)
