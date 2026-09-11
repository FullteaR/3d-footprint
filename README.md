# 3d-footprint

GPXの移動軌跡と、その範囲の地形（国土地理院DEM）・建物（PLATEAU LOD2）から、
3Dプリント可能な **3MF / STL** を生成するWebアプリ。多色印刷対応。

設計の詳細は [DESIGN.md](./DESIGN.md) を参照。

## 起動（Docker 一発）

```bash
docker compose up --build
# → http://localhost:8000
```

ローカル起動では `127.0.0.1:8000` のみで待ち受けます。
DEM / PLATEAU のキャッシュはDockerの `cache` ボリューム、生成結果は `jobs`
ボリュームに保存します。既存の `./data` は変更しません。以前のキャッシュを
引き続き使う場合は、Composeの `cache:/app/data` を `./data:/app/data` に変更し、
コンテナのUID/GID `10001:10001` が読み書きできる権限を設定してください。

## ウェブ公開

Docker Compose **2.24.4以降**を使い、公開するホストへドメインのDNSを向け、
80/443番ポートを到達可能にしてから起動します。

```bash
PUBLIC_DOMAIN=your-domain.example docker compose \
  -f docker-compose.yml -f compose.production.yml up -d --build
```

CaddyがHTTPS証明書を取得・更新し、HTTPをHTTPSへ転送します。
公開構成ではAPIの8000番ポートをホストへ公開しません。
アプリは非root、読み取り専用ルート、標準でCPU 2コア・メモリ4 GiB、
プロセス数256の上限で実行します。ログは10 MB×3世代に制限します。

`/api/health` はジョブ監視スレッドが動いているかも確認します。
Dockerのhealthcheckとプロセス終了時の自動再起動を設定しています。
**unhealthy表示だけではDockerは再起動しません**。公開基盤側でこのURLの外形監視を
設定し、異常通知と必要な再起動を行ってください。初回生成時間や必要メモリは地域・
解像度で変わるため、実際の公開サーバーで代表的なGPXを使って確認してください。

## 生成ジョブと保存期間

「3Dモデルを作成する」で生成を受け付け、画面に処理段階を表示します。
キャンセルと時間超過では、解析用の子プロセスを含めて生成を終了します。
完成した同じ模型からGLB・3MF・STL・色別STL ZIPを保存し、ダウンロードでは
模型を再生成しません。設定を変更するとプレビューを古い結果として表示し、
再生成するまでダウンロードを無効にします。

元GPX/SVGは生成のため一時保存され、処理終了・キャンセル・失敗時に削除します。
再起動で中断された処理は失敗として扱い、入力も削除します。元ファイル全体が
サーバーへ送信される仕様です。完成した模型・警告は標準で1時間保存します。
ジョブIDを知る人は結果を取得できるため、IDを含むURLを共有しないでください。
標準構成はAPIのアクセスログを無効にし、ジョブURLをログへ残しません。

土地利用・建物などの取得失敗、解析失敗、提供範囲外、縮尺による省略は画面に表示します。
地形データの取得に失敗した場合は生成自体を失敗とします。

環境変数（Composeでは `.env` にも記載できます）:

| 変数 | 標準値 | 意味 |
| --- | --- | --- |
| `JOB_WORKERS` | `1` | 同時生成数 |
| `MAX_JOBS` | `8` | 待機・実行・保存済み結果を合わせた保持数。満杯時は429 |
| `JOB_TIMEOUT_SECONDS` | `900` | 待機を含む受付からの処理期限 |
| `JOB_TTL_SECONDS` | `3600` | 終了した結果の保存期間 |
| `MAX_JOB_BYTES` | `536870912` | 1ジョブの出力合計上限（512 MiB） |
| `CACHE_MAX_BYTES` | `5368709120` | データキャッシュ上限（5 GiB）。古いアクセスから削除 |
| `CACHE_TTL_SECONDS` | `2592000` | データキャッシュの有効期間（30日） |
| `ABSENT_TTL_SECONDS` | `3600` | 存在しないタイルの再確認までの時間 |
| `PARSE_PROCS` | `2` | 1生成あたりのPLATEAU解析並列数 |
| `APP_MEMORY_LIMIT` / `APP_CPUS` | `4g` / `2` | アプリコンテナの資源上限 |

キャッシュ容量とは別に、生成結果、書き込み中の一時ファイル、ログ、Dockerイメージの
ディスク余裕が必要です。キャッシュは原子的に置換し、複数プロセス間で容量調整を排他します。
ジョブ管理は**単一ホスト・Uvicorn 1ワーカー**用です。同じ `JOBS_DIR` を複数の
APIプロセスが開いた場合は起動を拒否します。複数ホストへの水平分散には外部キューと
共有オブジェクトストレージが別途必要です。

API（以前の同期レスポンスから変更）:

- `POST /api/generate`: 従来と同じmultipart入力、`202`でジョブIDを返す。
  `Idempotency-Key` にハイフンなしUUIDを指定すると同じIDの再送を重複実行しない。
- `GET /api/jobs/{id}`: 状態・処理段階・警告・保存期限。
- `GET /api/jobs/{id}/files/{fmt}`: 完成ファイル。`fmt` は `glb` / `3mf` / `stl` / `stl_multi`。
- `DELETE /api/jobs/{id}`: 待機・実行中はキャンセル、完成済みは保存結果を削除。

## データ出典・ライセンス

模型の生成とアプリ表示に以下のデータを利用しています。アプリを公開したり、
生成した模型・3MF/STLを頒布する際は、これらの**出典明記が必要**です。

- **標高・地形**: [地理院タイル（標高タイル）](https://maps.gsi.go.jp/development/ichiran.html)
  （国土地理院）—
  [国土地理院コンテンツ利用規約](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html)
  （政府標準利用規約2.0準拠・出典明記）。
  ※ DEMは測量成果のため、**印刷した模型の販売など**では別途
  測量成果の複製・使用承認が必要になる場合があります（要・最新規約確認）
- **建物・橋・道路・土地利用**: [Project PLATEAU](https://www.mlit.go.jp/plateau/)
  （国土交通省）の CityGML（bldg / brid / tran / luse）— 政府標準利用規約2.0（CC BY 4.0互換）
- **土地被覆（色の補完）**: [JAXA 高解像度土地利用土地被覆図](https://www.eorc.jaxa.jp/ALOS/jp/dataset/lulc_j.htm)
  v25.04（[JAXA Earth API](https://data.earth.jaxa.jp/) 経由）— 出典明記
- **地図表示**: [OpenStreetMap](https://www.openstreetmap.org/copyright)
  （© OpenStreetMap contributors, ODbL）— アプリの地図表示のみで、模型には含まれません

表記例:「出典: 国土地理院（地理院タイル）／国土交通省 Project PLATEAU／JAXA 高解像度土地利用土地被覆図」

## ローカル開発（Dockerなし）

backend:

```bash
cd backend
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=../data JOBS_DIR=../jobs OPENBLAS_NUM_THREADS=1 uvicorn app.main:app --reload --port 8000
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

依存の既知の脆弱性はCIの `pip-audit` と `npm audit` でも検査します。
