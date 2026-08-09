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

## データ出典・ライセンス

模型の生成とアプリ表示に以下のデータを利用しています。アプリを公開したり、
生成した模型・3MF/STLを頒布する際は、これらの**出典明記が必要**です。

- **標高・地形**: [地理院タイル（標高タイル）](https://maps.gsi.go.jp/development/ichiran.html)
  （国土地理院）—
  [国土地理院コンテンツ利用規約](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html)
  （政府標準利用規約2.0準拠・出典明記）。
  ※ DEMは測量成果のため、**印刷した模型の販売など**では別途
  測量成果の複製・使用承認が必要になる場合があります（要・最新規約確認）
- **建物・橋・土地利用**: [Project PLATEAU](https://www.mlit.go.jp/plateau/)
  （国土交通省）の CityGML（bldg / brid / luse）— 政府標準利用規約2.0（CC BY 4.0互換）
- **土地被覆（色の補完）**: [JAXA 高解像度土地利用土地被覆図](https://www.eorc.jaxa.jp/ALOS/jp/dataset/lulc_j.htm)
  v25.04（[JAXA Earth API](https://data.earth.jaxa.jp/) 経由）— 出典明記
- **地図表示**: [OpenStreetMap](https://www.openstreetmap.org/copyright)
  （© OpenStreetMap contributors, ODbL）— アプリの地図表示のみで、模型には含まれません

表記例:「出典: 国土地理院（地理院タイル）／国土交通省 Project PLATEAU／JAXA 高解像度土地利用土地被覆図」

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

## テスト

バックエンド（pytest / ランタイムと同じイメージで実行、ネットワーク不要）:

```bash
docker compose run --rm test
```

DEMと土地利用は合成配列に差し替えているので、ダウンロードは一切発生しません。
`backend/` はマウントされるため、テストを直したら再ビルドなしで再実行できます。

フロントエンド（vitest / jsdom）:

```bash
cd frontend && npm test
```

## ステータス

- **M1** 雛形＋Docker一発起動 ← 現在
- M2 DEM→地形 3MF/STL（単色）
- M3 GPX凸ライン
- M4 GLBプレビュー＋論理レイヤ多色
- M5 PLATEAU LOD2建物＋意味カテゴリ色（Optional）
