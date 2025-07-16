import mimetypes
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

# MIMEタイプの追加：.jsファイルを正しく読み込ませる
mimetypes.add_type("application/javascript", ".js")

app = FastAPI()


# APIルートは静的ファイルの前に定義する
@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI!"}


# Reactのビルド済みファイルのパス
frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)
# 静的ファイルとしてReactビルド済みファイルをマウント
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
