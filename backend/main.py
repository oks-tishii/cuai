import threading
import time
import requests
import uvicorn
import webview
from api import app


def start_backend():
    uvicorn.run(app=app, host="127.0.0.1", port=8000, reload=False)


def wait_for_server(url: str, timeout=3600, interval=1):
    """指定URLにtimeout秒間、interval秒ごとにアクセスを試みる"""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("サーバーが起動しました！")
                return True
        except requests.exceptions.ConnectionError:
            pass  # サーバーがまだ起動していない状態
        if time.time() - start_time > timeout:
            print("サーバー起動待機タイムアウト")
            return False
        time.sleep(interval)


if __name__ == "__main__":
    threading.Thread(target=start_backend, daemon=True).start()
    # 起動確認（例：ルートにGET）
    server_up = wait_for_server("http://127.0.0.1:8000")

    if server_up:
        webview.create_window("中部大学AI課題プロジェクト", "http://127.0.0.1:8000")
        webview.start()
    else:
        print("サーバー起動に失敗したため、webviewを開きませんでした。")

    # detector = PatchCoreInference("./exports/patchcore_screw.pt")
    # # ディレクトリ内の全画像を処理
    # dir_result = detector.detect_anomalies_from_directory(
    #     "./data/raw/screw/test/good", threshold=0.5
    # )
    # print(f"ディレクトリ内画像数: {dir_result.total_images}")
