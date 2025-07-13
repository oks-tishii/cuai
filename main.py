import webview
import os
import subprocess
from app_ml.main import initialize_model, run_inference_on_image, train_and_save_model


class Api:
    """
    React(Javascript)とPythonを連携させるためのAPI
    """

    def __init__(self):
        print("Initializing model...")
        self.model = initialize_model()
        print("Model initialized.")

    def process_image(self, image_base64):
        """
        Base64エンコードされた画像データを受け取り、異常検知を実行する
        """
        try:
            header, encoded = image_base64.split(",", 1)
            result = run_inference_on_image(self.model, encoded)
            return result
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"error": str(e)}

    def retrain_model(self):
        """
        モデルの再学習を実行し、APIが保持するモデルを更新する
        """
        try:
            self.model = train_and_save_model()
            return {"status": "success", "message": "Model re-trained successfully."}
        except Exception as e:
            print(f"Error re-training model: {e}")
            return {"status": "error", "message": str(e)}


def run_vite_dev_server():
    """
    Vite開発サーバーをサブプロセスで起動する
    """
    vite_dir = os.path.join(os.path.dirname(__file__), "app_client")
    # Windowsではshell=Trueが必要
    proc = subprocess.Popen("npm run dev", shell=True, cwd=vite_dir)
    return proc


if __name__ == "__main__":
    # Viteサーバーを別スレッドで起動
    vite_process = run_vite_dev_server()

    api = Api()
    window = webview.create_window(
        "CUAI | Anomaly Detection",
        "http://localhost:5173",  # ViteサーバーのURL
        js_api=api,
        width=1200,
        height=800,
    )

    def on_closed():
        print("Window is closed, terminating Vite server.")
        vite_process.terminate()

    window.events.closed += on_closed

    webview.start(debug=True)
