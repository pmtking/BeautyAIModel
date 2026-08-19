import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # ✅ space را حذف کنید
        port=8000,
        reload=True
    )