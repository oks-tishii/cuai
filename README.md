# 中部大学 AI プロジェクト課題

## VAE 画像異常検知アプリケーション

Flet GUI を備えた、変分オートエンコーダ（VAE）を用いた画像異常検知アプリケーションです。

## 開発環境の構築

1. **リポジトリをクローンする:**

   ```bash
   git clone https://github.com/oks-tishii/cuai.git
   cd cuai
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
   uv run flet run app_client/main.py
   ```

## フォルダ構成

```
├── app_ml/            # コアMLバックエンドアプリケーションのソースコード
│   ├── __init__.py
│   ├── config/         #  個別アプリの設定
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
├── app_client/        # GUIアプリケーションのソースコード
│   ├── __init__.py
│   ├── main_gui.py
│   ├── components/
│   ├── services/
│   └── config/         # 個別アプリの設定
│       └── __init__.py
│       └── settings.py
├── config/            # プロジェクト全体の共通設定ファイル (旧 config/)
│   └── __init__.py
│   └── project_settings.py # プロジェクト全体のパスや共有定数など
├── data/              # データセット
│   ├── processed/
│   └── raw/
├── models/            # 学習済みモデルの保存先
├── notebooks/         # 実験・分析用の Jupyter Notebook
├── tests/             # テストスクリプト
├── .venv/              # 仮想環境のディレクトリ
├── .gitignore          # Git の追跡から除外するファイルを指定
├── pyproject.toml      # プロジェクトのメタデータと依存関係
└── README.md           # プロジェクトの説明
```
