# PatchCore 画像異常検知アプリケーション

## 概要

PatchCore を利用した画像異常検知アプリケーションです。FastAPI サーバー、React フロントエンド、そして PyWebview によるデスクトップ GUI で構成されています。

## アーキテクチャ

- **Backend:** FastAPI を使用して構築されており、PatchCore モデルを API として提供します。
- **Frontend:** React で構築されたウェブインターフェースです。
- **GUI:** PyWebview を使用して、React フロントエンドをネイティブなデスクトップウィンドウにラップします。

## 開発環境の構築

### Backend (FastAPI)

1. **ディレクトリを移動する:**

   ```bash
   cd backend
   ```

2. **`uv` をインストールする:**
   公式の指示に従って `uv` をインストールしてください: https://github.com/astral-sh/uv

3. **仮想環境を作成し、依存関係をインストールする:**

   ```bash
   uv venv
   uv sync
   ```

4. **サーバーを起動する:**
   ```bash
   uv run uvicorn main:app --reload
   ```

### Frontend (React)

1. **ディレクトリを移動する:**

   ```bash
   cd frontend
   ```

2. **依存関係をインストールする:**

   ```bash
   npm install
   ```

3. **開発サーバーを起動する:**
   ```bash
   npm run dev
   ```

## フォルダ構成

```
├── backend/            # FastAPIバックエンドアプリケーション
├   └── data/           # データセット
│      └── raw/
│         └── screw/
│   ├── api.py          # APIエンドポイント
│   ├── create_model.py # PatchCoreモデル作成
│   ├── dataset.py      # データセット関連
│   ├── main.py         # アプリケーションエントリーポイント
│   ├── models.py       # モデル関連
│   ├── pyproject.toml  # 依存関係定義
│   └── ...
├── frontend/           # Reactフロントエンドアプリケーション
│   ├── src/
│   │   ├── App.tsx     # メインのReactコンポーネント
│   │   └── main.tsx    # アプリケーションエントリーポイント
│   ├── package.json    # 依存関係定義
│   └── ...
├── .gitignore          # Git の追跡から除外するファイルを指定
└── README.md           # プロジェクトの説明
```
