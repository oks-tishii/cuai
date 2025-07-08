# 中部大学 AI プロジェクト課題

## VAE 画像異常検知アプリケーション

Flet GUI を備えた、変分オートエンコーダ（VAE）を用いた画像異常検知アプリケーションです。

## 開発環境の構築

1. **リポジトリをクローンする:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **`uv` をインストールする:**
   公式の指示に従って `uv` をインストールしてください: https://github.com/astral-sh/uv

3. **仮想環境を作成する:**

   ```bash
   uv venv
   ```

4. **依存関係をインストールする:**

   ```bash
   uv sync
   ```

5. **アプリケーションを実行する:**
   ```bash
   uv run flet run main.py
   ```

## フォルダ構成

```
├── _app_ml/            # ✨名称変更✨ コアMLバックエンドアプリケーションのソースコード (旧 ml_app/)
│   ├── __init__.py
│   ├── config/         # ✨変更✨ 個別アプリの設定（共通設定以外でml_app固有の設定があれば）
│   │   └── __init__.py
│   │   └── settings.py
│   ├── data/           # データローディング、前処理、データ拡張関連のモジュール
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── models/         # モデル定義関連のモジュール
│   │   ├── __init__.py
│   │   └── vae.py
│   ├── utils/          # 共通ユーティリティ関数やヘルパーモジュール
│   │   ├── __init__.py
│   │   └── common.py
│   └── main.py         # メインの学習スクリプト
├── _app_client/        # ✨名称変更✨ GUIアプリケーションのソースコード (旧 client_app/)
│   ├── __init__.py
│   ├── main_gui.py
│   ├── components/
│   ├── services/
│   └── config/         # ✨新規✨ 個別アプリの設定（client_app固有の設定があれば）
│       └── __init__.py
│       └── settings.py
├── _config/            # ✨名称変更✨ プロジェクト全体の共通設定ファイル (旧 config/)
│   └── __init__.py
│   └── project_settings.py # プロジェクト全体のパスや共有定数など
├── _data/              # データセット (旧 data/)
│   ├── processed/
│   └── raw/
├── _models/            # 学習済みモデルの保存先 (旧 models/)
├── _notebooks/         # 実験・分析用の Jupyter Notebook (旧 notebooks/)
├── _tests/             # テストスクリプト (旧 tests/)
├── .venv/              # 仮想環境のディレクトリ
├── .gitignore          # Git の追跡から除外するファイルを指定
├── pyproject.toml      # プロジェクトのメタデータと依存関係
└── README.md           # プロジェクトの説明
```
