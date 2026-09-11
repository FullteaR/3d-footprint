// UI text in both languages. JA is the source of truth: every key is declared
// there and EN is typed against it, so a string can't be added to one language
// only. Values that depend on numbers are functions rather than placeholders.
// A *starred* run inside a hint renders bold (see rich() in App.tsx).
export type Lang = "ja" | "en";

const pad2 = (n: number) => String(n).padStart(2, "0");

const JA = {
  subtitle: "GPXの移動軌跡と地形から、3Dプリント用の立体地図をつくる",
  apiOk: "接続中",

  dropGpx: "GPXファイルをここにドロップ",
  dropGpxSub: "またはクリックして選択",
  dropReplace: "クリックまたはドロップで差し替え",

  secTime: "時間範囲",
  rangeStart: "開始",
  rangeEnd: "終了",
  selected: "選択",
  whole: "全体",
  resetRange: "全体に戻す",
  noTimeHint: "時刻情報が無いため、軌跡全体を使います。",
  // Length of the kept leg, in the reading the language expects.
  duration: (sec: number) => {
    const m = Math.round(sec / 60);
    return m >= 60 ? `${Math.floor(m / 60)}時間${pad2(m % 60)}分` : `${m}分`;
  },

  secSize: "大きさ・縮尺",
  extent: "範囲（実距離）",
  printSize: "印刷サイズ（最大辺 mm）",
  scale: "縮尺　1:",
  lockOn: "固定中（クリックで解除）",
  lockOff: "この項目を固定する",

  secModel: "モデル",
  verticalScale: "垂直強調",
  baseThickness: "底面厚（mm）",
  resolution: "解像度（詳細度）",
  resStandard: "標準（速い・粗い）",
  resHigh: "高",
  resMax: "最高（細かい・重い）",

  secColor: "色・土地利用",
  minColor: "色の最小サイズ",
  minColorOff: "なし",
  minColorHint:
    "印刷実寸でこれより小さい色の斑点・細い筋を消し、色の境界をなめらかな曲線にします。",

  secTrack: "軌跡",
  includeTrack: "軌跡を含める",
  trackWidth: "軌跡の幅（mm）",
  trackHeight: "軌跡の高さ（mm）",
  trackColor: "軌跡の色",

  secBuildings: "建物・橋",
  includeBuildings: "建物・橋 (PLATEAU LOD2)",
  buildingsHint: "PLATEAU整備済みの都市のみ。",
  buildingScale: "高さ強調",
  minFeature: "最小幅",
  minFeatureOff: "デフォルメなし",
  buildingColor: "建物・橋の色",

  secPlate: "銘板",
  includePlate: "銘板を付ける",
  plateSvg: "銘板デザインのSVGを選択",
  plateFit: (w: number, h: number) => `${w}×${h}mmの範囲に自動で収めます`,
  platePreview: "銘板プレビュー",
  plateHint: "文字はデザインツールで*アウトライン化*（パスに変換）してください。",
  platePlace: "軌跡を避けて配置",
  plateHandles: "地図の青い枠で ■サイズ・●回転・✥移動",
  plateWidth: "範囲の幅（mm）",
  plateDepth: "範囲の奥行き（mm）",
  plateRotation: "範囲の回転（°）",
  plateRelief: "凸の高さ（板から mm）",

  create: "3Dモデルを作成する",
  format: "フォーマット",
  fmt3mf: "3MF（多色・単一ファイル）",
  fmtStlMulti: "STL（多色・色ごと分割ZIP）",
  fmtStl: "STL（単色）",
  download: "表示中の模型をダウンロード",
  stNeedFile: "GPXファイルを選択してください",
  stGenerating: "3Dモデル生成中…",
  stDownloading: "ダウンロード中…",
  stDownloaded: "ダウンロードしました。",
  stCancelled: "生成をキャンセルしました。",
  cancel: "生成をキャンセル",
  previewStale: "設定が変わりました。再生成するとダウンロードできます。",
  resultExpired: "保存期限が切れました。再生成してください。",
  dataWarnings: "データについてのお知らせ",
  previewError: "3Dプレビューを表示できません。WebGL対応のブラウザで開き直してください。",
  errorPrefix: "エラー: ",
  gpxFile: "GPXファイル",
  svgFile: "銘板のSVG",
  tooLarge: (what: string, mb: number) => `${what}が大きすぎます（上限 ${mb} MB）`,

  mapTitle: "モデル化する範囲",
  shapeRect: "長方形",
  shapeSquare: "正方形",
  shapeHex: "正六角形",
  rotation: "回転",
  handles: "■サイズ・●回転・✥移動",
  handlesLocked: "●回転・✥移動（実距離を固定中）",
  center: "中心",
  fitTrack: "軌跡に合わせる",
  centerTrack: "軌跡の中心へ",
  mapEmpty: "GPXを選択すると地図に軌跡と範囲を表示",

  previewTitle: "3Dプレビュー",
  previewHelp: "ドラッグで回転・ホイールで拡大",
  previewEmpty: "「3Dモデルを作成する」を押すとここにプレビュー",
  busy: "生成中…",

  luWater: "水面",
  luForest: "森林・緑地",
  luFarm: "農地",
  luUrban: "市街地",
  luRoad: "道路",
  luBare: "空地・荒地",
  luWetland: "湿地",

  credits: "データ出典:",
  gsi: "国土地理院（地理院タイル）",
  plateau: "国土交通省 Project PLATEAU",
  jaxa: "JAXA 高解像度土地利用土地被覆図",
  creditSep: "・",
  osm: "地図表示 ©",
  osmSuffix: " contributors",
};

const EN: typeof JA = {
  subtitle: "Turn a GPX track and the terrain under it into a 3D-printable relief map",
  apiOk: "connected",

  dropGpx: "Drop a GPX file here",
  dropGpxSub: "or click to choose one",
  dropReplace: "Click or drop to replace",

  secTime: "Time range",
  rangeStart: "Start",
  rangeEnd: "End",
  selected: "Selected",
  whole: "whole",
  resetRange: "Back to the whole track",
  noTimeHint: "This GPX carries no timestamps, so the whole track is used.",
  duration: (sec: number) => {
    const m = Math.round(sec / 60);
    return m >= 60 ? `${Math.floor(m / 60)}h ${pad2(m % 60)}m` : `${m} min`;
  },

  secSize: "Size & scale",
  extent: "Extent (real distance)",
  printSize: "Print size (longest edge, mm)",
  scale: "Scale　1:",
  lockOn: "Locked (click to release)",
  lockOff: "Lock this value",

  secModel: "Model",
  verticalScale: "Vertical exaggeration",
  baseThickness: "Base thickness (mm)",
  resolution: "Resolution (detail)",
  resStandard: "Standard (fast, coarse)",
  resHigh: "High",
  resMax: "Highest (fine, slow)",

  secColor: "Colour & land use",
  minColor: "Smallest colour patch",
  minColorOff: "off",
  minColorHint:
    "Drops colour specks and thin streaks smaller than this at printed size, and smooths colour boundaries into curves.",

  secTrack: "Track",
  includeTrack: "Include the track",
  trackWidth: "Track width (mm)",
  trackHeight: "Track height (mm)",
  trackColor: "Track colour",

  secBuildings: "Buildings & bridges",
  includeBuildings: "Buildings & bridges (PLATEAU LOD2)",
  buildingsHint: "Cities covered by PLATEAU only.",
  buildingScale: "Height exaggeration",
  minFeature: "Smallest width",
  minFeatureOff: "no massing",
  buildingColor: "Building & bridge colour",

  secPlate: "Nameplate",
  includePlate: "Add a nameplate",
  plateSvg: "Choose an SVG for the nameplate",
  plateFit: (w: number, h: number) => `Fitted into a ${w}×${h}mm area`,
  platePreview: "Nameplate preview",
  plateHint: "Convert text to *outlines* (paths) in your design tool.",
  platePlace: "Avoid the track",
  plateHandles: "Blue frame: ■ size ● angle ✥ move",
  plateWidth: "Area width (mm)",
  plateDepth: "Area depth (mm)",
  plateRotation: "Area rotation (°)",
  plateRelief: "Relief height (above the plate, mm)",

  create: "Create the 3D model",
  format: "Format",
  fmt3mf: "3MF (multi-colour, one file)",
  fmtStlMulti: "STL (multi-colour, ZIP split per colour)",
  fmtStl: "STL (single colour)",
  download: "Download previewed model",
  stNeedFile: "Choose a GPX file first",
  stGenerating: "Generating the 3D model…",
  stDownloading: "Downloading…",
  stDownloaded: "Downloaded.",
  stCancelled: "Generation cancelled.",
  cancel: "Cancel generation",
  previewStale: "Settings have changed. Generate again before downloading.",
  resultExpired: "The saved model has expired. Generate it again.",
  dataWarnings: "Data availability",
  previewError: "The 3D preview could not be displayed. Try a browser with WebGL support.",
  errorPrefix: "Error: ",
  gpxFile: "The GPX file",
  svgFile: "The nameplate SVG",
  tooLarge: (what: string, mb: number) => `${what} is too large (limit ${mb} MB)`,

  mapTitle: "AREA TO MODEL",
  shapeRect: "Rectangle",
  shapeSquare: "Square",
  shapeHex: "Hexagon",
  rotation: "Rotation",
  handles: "■ resize, ● rotate, ✥ move",
  handlesLocked: "● rotate, ✥ move (extent locked)",
  center: "Centre",
  fitTrack: "Fit to the track",
  centerTrack: "Centre on the track",
  mapEmpty: "Choose a GPX file to see the track and the model extent",

  previewTitle: "3D PREVIEW",
  previewHelp: "Drag to orbit, wheel to zoom",
  previewEmpty: "Press “Create the 3D model” to preview it here",
  busy: "Generating…",

  luWater: "Water",
  luForest: "Forest & greenery",
  luFarm: "Farmland",
  luUrban: "Built-up",
  luRoad: "Roads",
  luBare: "Open & bare land",
  luWetland: "Wetland",

  credits: "Data sources:",
  gsi: "Geospatial Information Authority of Japan (GSI Tiles)",
  plateau: "MLIT Project PLATEAU",
  jaxa: "JAXA High-Resolution Land Use and Land Cover Map",
  creditSep: " · ",
  osm: "Map display ©",
  osmSuffix: " contributors",
};

export type Text = typeof JA;

export const TEXT: Record<Lang, Text> = { ja: JA, en: EN };

export const localeOf = (lang: Lang) => (lang === "ja" ? "ja-JP" : "en-US");

const STORE_KEY = "3dfp.lang";

// Last choice wins; otherwise a Japanese browser gets Japanese and everyone
// else gets English.
export function initialLang(): Lang {
  const saved = localStorage.getItem(STORE_KEY);
  if (saved === "ja" || saved === "en") return saved;
  return navigator.language?.toLowerCase().startsWith("ja") ? "ja" : "en";
}

export function rememberLang(lang: Lang) {
  localStorage.setItem(STORE_KEY, lang);
}


export function jobStageOf(lang: Lang, stage: string): string {
  const stages: Record<string, [string, string]> = {
    queued: ["順番待ち…", "Queued…"], starting: ["生成を準備中…", "Preparing…"],
    terrain: ["地形を取得中…", "Loading terrain…"], landuse: ["土地利用を処理中…", "Processing land use…"],
    structures: ["建物・橋・道路を処理中…", "Processing buildings, bridges and roads…"],
    model: ["模型を組み立て中…", "Building the model…"], exporting: ["ファイルを保存中…", "Saving model files…"],
    succeeded: ["生成完了", "Complete"],
  };
  return stages[stage]?.[lang === "ja" ? 0 : 1] ?? TEXT[lang].stGenerating;
}

export function warningOf(lang: Lang, warning: { source: string; reason: string }): string {
  const sources: Record<string, [string, string]> = {
    jaxa: ["JAXA土地被覆", "JAXA land cover"], landuse: ["PLATEAU土地利用", "PLATEAU land use"],
    buildings: ["建物", "Buildings"], bridges: ["橋", "Bridges"], roads: ["道路", "Roads"],
    plateau_catalog: ["PLATEAUデータ一覧", "PLATEAU catalog"],
  };
  const name = sources[warning.source]?.[lang === "ja" ? 0 : 1] ?? warning.source;
  const reason = warning.reason === "no_coverage"
    ? (lang === "ja" ? "この範囲の提供データがありません。" : "No data is available for this area.")
    : warning.reason === "scale"
    ? (lang === "ja" ? "この縮尺では省略されます。" : "Omitted at this scale.")
    : (lang === "ja" ? "一部の取得・解析に失敗しました。模型から欠けている可能性があります。" : "Some data could not be loaded or parsed and may be missing from the model.");
  return `${name}: ${reason}`;
}
