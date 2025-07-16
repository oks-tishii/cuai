import threading
import uvicorn
import webview
from api import app


def start_backend():
    uvicorn.run(app=app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    threading.Thread(target=start_backend, daemon=True).start()
    webview.create_window("中部大学AI課題プロジェクト", "http://127.0.0.1:8000")
    webview.start()
