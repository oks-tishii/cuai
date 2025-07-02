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
.
├── .venv
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
├── src
│   └── __init__.py
├── tests
│   └── __init__.py
├── .gitignore
├── pyproject.toml
└── README.md
```

- **`.venv/`**: 仮想環境のディレクトリ
- **`data/`**: データセット
  - **`raw/`**: 生データ
  - **`processed/`**: 加工済みデータ
- **`models/`**: 学習済みモデル
- **`notebooks/`**: 実験・分析用の Jupyter Notebook
- **`src/`**: アプリケーションのソースコード
- **`tests/`**: テストスクリプト
- **`.gitignore`**: Git の追跡から除外するファイルを指定
- **`pyproject.toml`**: プロジェクトのメタデータと依存関係
- **`README.md`**: このファイル
