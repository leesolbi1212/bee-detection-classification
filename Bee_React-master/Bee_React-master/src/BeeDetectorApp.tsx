import React, { useEffect, useMemo, useRef, useState } from "react";

// ==========================
// 프론트엔드 (React + Tailwind)
// - FastAPI 서버와 연동 (감지/분류)
// - YOLOv3: /detect/image, /detect/video
// - ResNet18: /classify/crops (base64 배열로 분류)
// - 한국어 라벨 매핑 적용: LI/CA/AP/BI, AB/QB
// ==========================

// 서버 베이스 URL
const BASE_URL =
  (import.meta as any)?.env?.VITE_API_BASE_URL || "http://localhost:8000";

// ---- Class label mapping (server returns like "AP_QB") ----
const SPECIES_MAP: Record<string, string> = {
  LI: "이탈리안",
  CA: "카니올란",
  AP: "한봉",
  BI: "호박벌",
};
const LIFECYCLE_MAP: Record<string, string> = {
  AB: "일벌",
  QB: "여왕벌",
};
function prettyLabel(code?: string) {
  if (!code) return "";
  const [sp, lf] = code.split("_");
  const spK = SPECIES_MAP[sp] || sp || "";
  const lfK = LIFECYCLE_MAP[lf] || lf || "";
  return `${spK}${spK && lfK ? " • " : ""}${lfK}`;
}

// 유틸: 파일을 DataURL로 미리보기
function fileToDataURL(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result as string);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

// 유틸: 쿼리 문자열 생성
function buildQuery(params: Record<string, any>) {
  const usp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") return;
    usp.set(k, String(v));
  });
  return usp.toString();
}

// 타입들
interface DetectImageResponse {
  image_size: { width: number; height: number };
  num_dets: number;
  conf_thr: number;
  iou_thr: number;
  detections: {
    bbox: [number, number, number, number];
    score: number;
    class_id: number;
    class_name?: string;
  }[];
  crops: {
    bbox: [number, number, number, number];
    png_b64: string; // "iVBORw0..." (헤더 없음) 또는 "data:image/png;base64,..."
    cls?: { label: string; prob: number }; // 서버 분류 결과 병합
  }[];
}

interface DetectVideoFrame {
  frame_index: number;
  time_sec: number | null;
  num_dets: number;
  detections_preview: {
    bbox: [number, number, number, number];
    score: number;
    class_id: number;
  }[];
  crops?: { bbox: [number, number, number, number]; png_b64: string }[];
}

interface DetectVideoResponse {
  total_frames: number;
  fps: number;
  processed_frames: number;
  conf_thr: number;
  iou_thr: number;
  stride: number;
  results: DetectVideoFrame[];
}

export default function BeeDetectorApp() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  // 파라미터
  const [conf, setConf] = useState(0.5);
  const [iou, setIou] = useState(0.45);
  const [returnCrops, setReturnCrops] = useState(true);
  const [maxCrops, setMaxCrops] = useState(16);
  const [margin, setMargin] = useState(0.08);

  // 비디오 파라미터
  const [frameStride, setFrameStride] = useState(5);
  const [maxFrames, setMaxFrames] = useState(300);
  const [cropsPerFrame, setCropsPerFrame] = useState(4);

  // 서버 상태
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [classifying, setClassifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 결과
  const [imgResult, setImgResult] = useState<DetectImageResponse | null>(null);
  const [vidResult, setVidResult] = useState<DetectVideoResponse | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    fetch(`${BASE_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const onPickFile: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setImgResult(null);
    setVidResult(null);
    setError(null);
    if (f && mode === "image") {
      setPreview(await fileToDataURL(f));
    } else if (f && mode === "video") {
      setPreview(null); // 비디오는 프리뷰 생략
    } else {
      setPreview(null);
    }
  };

  async function handleSubmit() {
    if (!file) {
      setError("파일을 선택하세요.");
      return;
    }
    setLoading(true);
    setError(null);
    setImgResult(null);
    setVidResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);

      if (mode === "image") {
        const qs = buildQuery({
          conf_thr: conf,
          iou_thr: iou,
          return_crops: returnCrops ? 1 : 0,
          max_crops: maxCrops,
          margin_ratio: margin,
        });
        const resp = await fetch(`${BASE_URL}/detect/image?${qs}`, {
          method: "POST",
          body: fd,
        });
        if (!resp.ok) throw new Error(`이미지 감지 실패 (${resp.status})`);
        const data: DetectImageResponse = await resp.json();
        setImgResult(data);
      } else {
        const qs = buildQuery({
          conf_thr: conf,
          iou_thr: iou,
          frame_stride: frameStride,
          max_frames: maxFrames,
          return_crops: returnCrops ? 1 : 0,
          crops_per_frame: cropsPerFrame,
          margin_ratio: margin,
        });
        const resp = await fetch(`${BASE_URL}/detect/video?${qs}`, {
          method: "POST",
          body: fd,
        });
        if (!resp.ok) throw new Error(`비디오 감지 실패 (${resp.status})`);
        const data: DetectVideoResponse = await resp.json();
        setVidResult(data);
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  // 분류 버튼: 현재 이미지 감지 결과의 crops를 /classify/crops 로 보냄
  async function handleClassifyCrops() {
    if (!imgResult?.crops?.length) {
      setError("분류할 크롭이 없습니다. 먼저 감지를 실행하세요.");
      return;
    }
    setClassifying(true);
    setError(null);
    try {
      // 서버가 "data:image/png;base64,..." 또는 순수 base64 모두 처리 가능하도록 strip_header 옵션 전송
      const crops_b64 = imgResult.crops.map((c) => c.png_b64);
      const resp = await fetch(`${BASE_URL}/classify/crops`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crops_b64, strip_header: true }),
      });
      if (!resp.ok) throw new Error(`분류 실패 (${resp.status})`);
      const data: { results: { label: string; prob: number }[] } =
        await resp.json();

      // 순서대로 병합
      const merged: DetectImageResponse = {
        ...imgResult,
        crops: imgResult.crops.map((c, i) => ({
          ...c,
          cls: data.results?.[i]
            ? { label: data.results[i].label, prob: data.results[i].prob }
            : c.cls,
        })),
      };
      setImgResult(merged);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setClassifying(false);
    }
  }

  // 이미지 박스 오버레이 렌더링 스케일 계산
  const overlayBoxes = useMemo(() => {
    if (!imgResult || !imgRef.current)
      return [] as {
        left: number;
        top: number;
        width: number;
        height: number;
        score: number;
      }[];
    const naturalW = imgResult.image_size.width;
    const naturalH = imgResult.image_size.height;
    const dispW = imgRef.current.clientWidth || naturalW;
    const dispH = (naturalH * dispW) / naturalW;
    const sx = dispW / naturalW;
    const sy = dispH / naturalH;

    return imgResult.detections.map((d) => {
      const [x1, y1, x2, y2] = d.bbox;
      return {
        left: x1 * sx,
        top: y1 * sy,
        width: (x2 - x1) * sx,
        height: (y2 - y1) * sy,
        score: d.score,
      };
    });
  }, [imgResult, imgRef.current]);

  return (
    <div className="min-h-screen flex justify-center bg-neutral-50 text-neutral-900">
      <div className="w-full max-w-none p-6">
        <header className="flex items-center justify-between mb-6">
          <h1 className="text-2xl md:text-3xl font-bold">🐝 Bee Detector</h1>
          <div className="text-sm text-neutral-600">
            <span className="mr-2">API: {BASE_URL}</span>
            {health ? (
              <span className="px-2 py-1 rounded bg-emerald-100 text-emerald-700">
                {health?.device} / weights: {String(health?.weights_loaded)}
              </span>
            ) : (
              <span className="px-2 py-1 rounded bg-amber-100 text-amber-700">
                서버 연결 확인 필요
              </span>
            )}
          </div>
        </header>

        {/* 모드 스위치 */}
        <div className="flex gap-2 mb-4">
          <button
            className={`px-4 py-2 rounded-xl shadow ${
              mode === "image" ? "bg-blue-600 text-white" : "bg-white"
            }`}
            onClick={() => {
              setMode("image");
              setPreview(null);
              setImgResult(null);
              setVidResult(null);
            }}
          >
            이미지
          </button>
          <button
            className={`px-4 py-2 rounded-xl shadow ${
              mode === "video" ? "bg-blue-600 text-white" : "bg-white"
            }`}
            onClick={() => {
              setMode("video");
              setPreview(null);
              setImgResult(null);
              setVidResult(null);
            }}
          >
            비디오
          </button>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* 왼쪽: 업로드 & 파라미터 */}
          <div className="md:col-span-1 bg-white rounded-2xl shadow p-4 space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                파일 업로드
              </label>
              <input
                type="file"
                accept={mode === "image" ? "image/*" : "video/*"}
                onChange={onPickFile}
                className="block w-full"
              />
              <p className="text-xs text-neutral-500 mt-1">
                {mode === "image" ? "JPG/PNG 등 이미지" : "MP4/AVI 등 동영상"}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium">conf</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={conf}
                  onChange={(e) => setConf(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-neutral-600">
                  {conf.toFixed(2)}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">IoU</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={iou}
                  onChange={(e) => setIou(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-neutral-600">{iou.toFixed(2)}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="flex items-center gap-2 col-span-2">
                <input
                  type="checkbox"
                  checked={returnCrops}
                  onChange={(e) => setReturnCrops(e.target.checked)}
                />
                <span className="text-sm">크롭(Base64) 반환</span>
              </label>
              <div>
                <label className="text-sm font-medium">max_crops</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1"
                  min={1}
                  max={64}
                  value={maxCrops}
                  onChange={(e) =>
                    setMaxCrops(parseInt(e.target.value || "0") || 16)
                  }
                />
              </div>
              <div>
                <label className="text-sm font-medium">margin</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={margin}
                  onChange={(e) =>
                    setMargin(parseFloat(e.target.value || "0") || 0)
                  }
                />
              </div>
            </div>

            {/* 분류 버튼 (이미지 모드에서만 노출) */}
            {mode === "image" && (
              <button
                disabled={!imgResult?.crops?.length || classifying}
                onClick={handleClassifyCrops}
                className={`w-full py-2 rounded-xl font-semibold shadow ${
                  !imgResult?.crops?.length || classifying
                    ? "bg-neutral-300 text-neutral-600 cursor-not-allowed"
                    : "bg-emerald-600 text-white hover:bg-emerald-700"
                }`}
              >
                {classifying ? "분류 중..." : "분류 / 재분류"}
              </button>
            )}

            {/* 라벨 안내 */}
            <div className="bg-white rounded-2xl p-3 border text-xs text-neutral-700">
              <div className="font-semibold mb-1">라벨 안내</div>
              <div className="grid grid-cols-2 gap-1">
                <div>LI = 이탈리안</div>
                <div>CA = 카니올란</div>
                <div>AP = 한봉</div>
                <div>BI = 호박벌</div>
                <div>AB = 일벌</div>
                <div>QB = 여왕벌</div>
              </div>
              <div className="mt-1 text-[11px] text-neutral-500">
                예: <b>AP_QB</b> → <b>{prettyLabel("AP_QB")}</b>
              </div>
            </div>

            <button
              disabled={!file || loading}
              onClick={handleSubmit}
              className={`w-full py-2 rounded-xl text-white font-semibold shadow ${
                loading ? "bg-neutral-400" : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {loading
                ? "처리 중..."
                : mode === "image"
                ? "이미지 감지"
                : "비디오 감지"}
            </button>

            {error && <div className="text-sm text-red-600">⚠️ {error}</div>}
          </div>

          {/* 오른쪽: 결과 패널 */}
          <div className="md:col-span-2 space-y-6">
            {/* 이미지 모드 결과 */}
            {mode === "image" && (
              <div className="bg-white rounded-2xl shadow p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="font-semibold">이미지 결과</h2>
                  <button
                    className={`px-3 py-1.5 rounded-lg text-sm font-semibold ${
                      !imgResult?.crops?.length || classifying
                        ? "bg-neutral-200 text-neutral-500 cursor-not-allowed"
                        : "bg-emerald-600 text-white hover:bg-emerald-700"
                    }`}
                    onClick={handleClassifyCrops}
                    disabled={!imgResult?.crops?.length || classifying}
                    title="탐지된 크롭들에 대해 ResNet18 분류 실행"
                  >
                    {classifying ? "분류 중..." : "분류 / 재분류"}
                  </button>
                </div>

                {!preview && !imgResult && (
                  <div className="text-sm text-neutral-500">
                    이미지를 업로드하고 감지를 실행하세요.
                  </div>
                )}

                {preview && (
                  <div className="relative inline-block">
                    <img
                      ref={imgRef}
                      src={preview}
                      className="max-w-full h-auto rounded"
                    />
                    {/* 박스 오버레이 */}
                    {imgResult &&
                      overlayBoxes.map((b, idx) => (
                        <div
                          key={idx}
                          className="absolute border-2 border-emerald-400 rounded"
                          style={{
                            left: b.left,
                            top: b.top,
                            width: b.width,
                            height: b.height,
                          }}
                          title={`score=${b.score.toFixed(2)}`}
                        />
                      ))}
                  </div>
                )}

                {imgResult && (
                  <div className="mt-4">
                    <div className="text-sm text-neutral-700 mb-2">
                      탐지 수: <b>{imgResult.num_dets}</b> (conf≥
                      {imgResult.conf_thr}, IoU={imgResult.iou_thr})
                    </div>

                    {returnCrops && imgResult.crops?.length > 0 && (
                      <div>
                        <div className="font-medium mb-2">
                          크롭 미리보기 ({imgResult.crops.length})
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {imgResult.crops.map((c, i) => (
                            <div
                              key={i}
                              className="border rounded-xl overflow-hidden"
                            >
                              <img
                                src={
                                  c.png_b64.startsWith("data:")
                                    ? c.png_b64
                                    : `data:image/png;base64,${c.png_b64}`
                                }
                                className="w-full h-auto"
                              />
                              {c.cls && (
                                <div className="px-2 py-1 text-xs bg-neutral-50 border-t">
                                  {prettyLabel(c.cls.label)} •{" "}
                                  {(c.cls.prob * 100).toFixed(1)}%
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* 원시 검출 리스트 */}
                    {imgResult.detections?.length > 0 && (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-sm text-neutral-600">
                          원시 검출 JSON 보기
                        </summary>
                        <pre className="text-xs bg-neutral-50 p-2 rounded border overflow-auto max-h-64">
                          {JSON.stringify(imgResult, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* 비디오 모드 결과 */}
            {mode === "video" && (
              <div className="bg-white rounded-2xl shadow p-4">
                <h2 className="font-semibold mb-3">동영상 결과</h2>
                {!vidResult && (
                  <div className="text-sm text-neutral-500">
                    비디오를 업로드하고 감지를 실행하세요.
                  </div>
                )}
                {vidResult && (
                  <div className="space-y-3">
                    <div className="text-sm">
                      총 프레임: <b>{vidResult.total_frames}</b> / 처리:{" "}
                      <b>{vidResult.processed_frames}</b> / FPS:{" "}
                      {vidResult.fps?.toFixed?.(2) || "-"}
                    </div>
                    <div className="overflow-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="border-b bg-neutral-50">
                            <th className="text-left py-2 px-2">#</th>
                            <th className="text-left py-2 px-2">time(s)</th>
                            <th className="text-left py-2 px-2">detections</th>
                            {returnCrops && (
                              <th className="text-left py-2 px-2">crops</th>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {vidResult.results.slice(0, 100).map((fr, i) => (
                            <tr key={i} className="border-b">
                              <td className="py-2 px-2">{fr.frame_index}</td>
                              <td className="py-2 px-2">
                                {fr.time_sec?.toFixed?.(2) ?? "-"}
                              </td>
                              <td className="py-2 px-2">{fr.num_dets}</td>
                              {returnCrops && (
                                <td className="py-2 px-2">
                                  <div className="flex flex-wrap gap-2">
                                    {(fr.crops || [])
                                      .slice(0, 4)
                                      .map((c, j) => (
                                        <img
                                          key={j}
                                          src={
                                            c.png_b64.startsWith("data:")
                                              ? c.png_b64
                                              : `data:image/png;base64,${c.png_b64}`
                                          }
                                          className="w-16 h-16 object-cover rounded"
                                        />
                                      ))}
                                  </div>
                                </td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <details>
                      <summary className="cursor-pointer text-sm text-neutral-600">
                        원시 결과 JSON 보기
                      </summary>
                      <pre className="text-xs bg-neutral-50 p-2 rounded border overflow-auto max-h-64">
                        {JSON.stringify(vidResult, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <footer className="mt-10 text-xs text-neutral-500">
          <div>
            ※ 분류는 **분류 / 재분류** 버튼을 눌러 실행합니다. 서버에서 crop마다{" "}
            <code>{`{label, prob}`}</code>를 반환하면 자동 표시됩니다.
          </div>
        </footer>
      </div>
    </div>
  );
}
