import asyncio
import mimetypes
import tempfile
from typing import List
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from create_model import train_and_save
from inference import PatchCoreInference

# MIMEタイプの追加：.jsファイルを正しく読み込ませる
mimetypes.add_type("application/javascript", ".js")


detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    model_path = "./exports/patchcore_screw.pt"

    if not os.path.exists(model_path):
        train_and_save()

    detector = PatchCoreInference(model_path)

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/detect-anomaly-single")
async def detect_anomaly_single(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file.filename.split('.')[-1]}",  # type: ignore
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # 異常検出を実行（メインスレッドで実行）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            detector.detect_anomaly_single,  # type: ignore
            tmp_file_path,
            threshold,  # type: ignore
        )

        # 一時ファイルを削除
        os.unlink(tmp_file_path)

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        if "tmp_file_path" in locals():
            try:
                os.unlink(tmp_file_path)  # type: ignore
            except:  # noqa: E722
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-anomaly-batch")
async def detect_anomaly_batch(
    files: List[UploadFile] = File(...), threshold: float = 0.5
):
    try:
        tmp_file_paths = []

        # 一時ファイルに保存
        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{file.filename.split('.')[-1]}",  # type: ignore
            ) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_paths.append(tmp_file.name)

        # 異常検出を実行
        result = await detector.detect_anomalies_async(tmp_file_paths, threshold)  # type: ignore

        # 一時ファイルを削除
        for tmp_file_path in tmp_file_paths:
            try:
                os.unlink(tmp_file_path)
            except:  # noqa: E722
                pass

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        # エラー時も一時ファイルを削除
        for tmp_file_path in tmp_file_paths:  # type: ignore
            try:
                os.unlink(tmp_file_path)
            except:  # noqa: E722
                pass
        raise HTTPException(status_code=500, detail=str(e))


# Reactのビルド済みファイルのパス
frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

# 静的ファイルとしてReactビルド済みファイルをマウント
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
