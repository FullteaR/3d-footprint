# 3d-footprint 設計

GPXの移動軌跡と、その範囲の地形（国土地理院DEM）・建物（PLATEAU LOD2）から、
3Dプリント可能な **3MF / STL** を生成するWebアプリ。多色印刷対応。

ローカルの強力なPCで自分用に動かす想定。`docker compose up` 一発で起動する。

---

## 全体像

```
GPX → 範囲決定 → 地形(DEM)・建物(PLATEAU)・軌跡 を生成
    → 色レイヤ別メッシュとして保持 → 3MF/STL 出力
```

計算の重心は **サーバ（Python）**。屋根つきメッシュ(LOD2)を印刷用ソリッドにする処理
（水密化・robust boolean union）はブラウザでは安定しないため、サーバで実施する。
クライアントは UI とプレビュー表示のみ。

```
[Client / React + Three.js]  …UIとプレビューのみ
  GPXドロップ・パラメータ調整・3Dプレビュー(サーバGLBを表示)・ダウンロード
        │  リクエスト(bbox, params)        ▲ GLB(プレビュー) / 3MF・STL(成果物)
        ▼                                  │
[Server / Python + FastAPI]   …重い処理 全部
  DEM取得・デコード        → 地形ソリッド
  PLATEAU LOD2取得         → 屋根つき建物ソリッド (Optional)
  GPX                      → 凸ライン
  → 色レイヤ別に保持 → 必要範囲のみ manifold union → 3MF / STL / GLB
```

---

## データソース

| 用途 | ソース | 形式 | 備考 |
|---|---|---|---|
| 地形 | 国土地理院 標高PNGタイル (`dem_png` 等) | XYZ RGBタイル | RGB符号化標高。サーバで取得・デコード・キャッシュ |
| 建物(屋根つき) | PLATEAU LOD2 | CityGML / 3D Tiles / FGB | Optional (M5)。意味属性・テクスチャあり |

### 国土地理院 標高タイルのデコード
`h_raw = R*65536 + G*256 + B`（u24）
- `h_raw == 2^23 (0x800000)` … 無効値（海など） → noData
- `h_raw <  2^23` … 標高 = `h_raw * 0.01` [m]
- `h_raw >  2^23` … 標高 = `(h_raw - 2^24) * 0.01` [m]（負標高）

### 座標・スケールの注意
- Webメルカトルのため緯度方向で1px実距離が変わる → 平面直角座標等へ変換しXYアスペクトを実距離で補正（南北の歪み防止）。`pyproj`。
- bboxが広いとタイル枚数が爆発 → 範囲・zoom上限のガードを設ける。

---

## 多色印刷

**重要**: STLは色を持てない。多色の本命は **3MF**（複数object＋material）。
単色用にSTLも残す。色ごと分割STL(zip)も同一前段から安価に出せる。

### 色の決め方（積み上げ）
1. **論理レイヤ色（MVP）**: 軌跡 / 地形 / 建物 / 底面 を別色。追加データ不要で即多色。
2. **PLATEAU意味カテゴリ色（後続）**: 水域 / 道路 / 建物用途 等で色分け。
- 標高バンド色・LOD2テクスチャ由来色は今回スコープ外。

### 仕組み
メッシュは**最後まで色レイヤ別に保持**（全部union して1枚にしない）。
色境界をまたがない範囲だけ union する。「レイヤ → パレット(4〜8色)」マッピングUIで割当。

```
colored_bodies: list[(mesh, layer_name, color)]
   ├─ exporter.three_mf(bodies)   → 1ファイルに複数object＋material（lib3mf 等）
   ├─ exporter.stl_multi(bodies)  → 色ごとSTL + 同一原点で zip（リセンタリング禁止）
   └─ exporter.stl_single(union)  → 単色用に全部 union
```
重い処理（色領域へのジオメトリ分割）は全フォーマット共通。出力はシリアライザ差し替えのみ。
→ exportはプラガブル。MVPは `stl_single` と `3mf` を動かし、`stl_multi` は口だけ用意。

---

## 3Dプリント要件
- **水密(manifold)**: 地形は四方に垂直スカート＋底面で閉じる。`trimesh` + `manifold3d`。
- **垂直強調**: 実スケールだと標高差・建物が潰れる。地形・建物に**同一倍率がデフォルト**
  （水平スケールが小さいと実高10mのビルが0.5mm以下になり印刷不能なため）。後で別倍率可。
- **軌跡**: 地表標高をバイリニア補間でサンプリングし、その高さ+オフセットに凸ライン。
  断面幅はノズル径以上（例1.2mm）を保証。
- **最小フィーチャ幅**: 細い建物・薄い軌跡は割れる → 最小幅ガード。
- **物理サイズfit**: 最大辺(例100mm)に収め、底面厚(例3mm)を付与。
- **ポリゴン数**: タイルzoom／グリッド間引きで制御。

---

## 技術スタック
- **Backend**: Python 3.12 + FastAPI + uvicorn
  - `numpy`/`scipy`（DEMグリッド・補間）, `shapely`（フットプリント）,
    `trimesh` + `manifold3d`（水密化・boolean）, `pyproj`（座標変換）,
    `pillow`（タイルデコード）, PLATEAU取込（`plateaukit` 等）,
    3MF出力（`lib3mf` もしくは薄い3MF XMLライタ）, GLB出力（trimesh）
- **Frontend**: React + TypeScript + Vite + Three.js（プレビュー専用）
- **配信**: 単一コンテナ。Viteビルド成果をFastAPIが `/` で静的配信、`/api/*` がAPI。
- **永続化**: DEM/PLATEAUキャッシュは Docker volume にマウント（再DL回避）。

> コンテナ内Pythonは scientific lib の wheel 互換性のため 3.12 を使う
> （ホストは 3.14 だが geo 系の wheel が揃わない可能性があるため）。

---

## API（MVP）
```
GET  /api/health
  → {status:"ok"}

POST /api/generate
  body: {
    gpx: <GPX文字列 or トラック点列>,
    bbox?: [minLon,minLat,maxLon,maxLat],   // 省略時はGPXから算出
    params: {
      sizeMm: number,            // 出力最大辺
      verticalScale: number,     // 垂直強調倍率
      baseThicknessMm: number,
      trackWidthMm: number,
      trackHeightMm: number,
      demZoom?: number,
      includeBuildings?: boolean,   // M5
      colors: { [layer]: hexColor }
    },
    format: "3mf" | "stl" | "glb"
  }
  → バイナリ（3MF/STL/GLB）

(将来) GET /api/elevation?bbox=...&zoom=...  → 標高グリッドJSON（デバッグ/再利用）
```

---

## リポジトリ構成
```
3d-footprint/
  DESIGN.md
  docker-compose.yml
  Dockerfile                 # multi-stage: frontend build → python runtime
  .dockerignore
  backend/
    requirements.txt
    app/
      main.py                # FastAPI: /api/* + 静的配信
      config.py
      api/routes.py          # health, generate(placeholder)
      core/                  # M2以降: terrain / track / buildings / export
  frontend/
    package.json
    vite.config.ts
    tsconfig.json
    index.html
    src/
      main.tsx
      App.tsx
```

---

## マイルストーン
1. **M1** 雛形＋Docker一発起動（FastAPI＋Vite静的配信、空アプリ） ← 現在
2. **M2** DEM→地形 3MF/STL（単色、印刷可能な最小成果）
3. **M3** GPX凸ライン合成
4. **M4** サーバGLBプレビュー＋パラメータ往復＋論理レイヤ多色(3MF)
5. **M5** PLATEAU LOD2建物＋意味カテゴリ色（Optional）
