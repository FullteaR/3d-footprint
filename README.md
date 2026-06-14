# 3d-footprint

GPXの移動軌跡と、その範囲の地形（国土地理院DEM）・建物（PLATEAU LOD2）から、
3Dプリント可能な **3MF / STL** を生成するWebアプリ。多色印刷対応。

設計の詳細は [DESIGN.md](./DESIGN.md) を参照。

## 起動（Docker 一発）

```bash
docker compose up --build
# → http://localhost:8000
```

DEM / PLATEAU のキャッシュは `./data` に永続化されます。

## ローカル開発（Dockerなし）

backend:

```bash
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

frontend（別ターミナル、`/api` は 8000 にプロキシ）:

```bash
cd frontend
npm install
npm run dev      # → http://localhost:5173
```

## ステータス

- **M1** 雛形＋Docker一発起動 ← 現在
- M2 DEM→地形 3MF/STL（単色）
- M3 GPX凸ライン
- M4 GLBプレビュー＋論理レイヤ多色
- M5 PLATEAU LOD2建物＋意味カテゴリ色（Optional）
