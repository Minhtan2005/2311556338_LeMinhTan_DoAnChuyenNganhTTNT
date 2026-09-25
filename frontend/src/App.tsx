import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeVideo,
  getVideo,
  indexSemanticVideo,
  listDetections,
  listEvents,
  listVideos,
  mediaUrl,
  queryVideo,
  semanticKis,
  uploadVideo
} from "./api";
import type { DetectionItem, EventItem, KISResponse, QueryResponse, VideoItem } from "./types";

const sampleQuestions = [
  "Có bao nhiêu ô tô?",
  "Có người trong video không?",
  "Có bao nhiêu người?",
  "Có xe máy không?",
  "Có xe tải không?"
];

const sampleKisQueries = [
  "tìm ô tô",
  "tìm người",
  "xe màu đỏ",
  "tìm xe tải",
  "tìm xe máy"
];

export function App() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [events, setEvents] = useState<EventItem[]>([]);
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [detectionsComplete, setDetectionsComplete] = useState(false);
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [kisQuery, setKisQuery] = useState(sampleKisQueries[0]);
  const [kisTopK, setKisTopK] = useState(20);
  const [kisAnswer, setKisAnswer] = useState<KISResponse | null>(null);
  const [indexMessage, setIndexMessage] = useState<string | null>(null);
  const [pendingSeek, setPendingSeek] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const selectedVideo = useMemo(
    () => videos.find((video) => video.id === selectedVideoId) ?? null,
    [videos, selectedVideoId]
  );

  async function refreshVideos(preferId?: string) {
    const next = await listVideos();
    setVideos(next);
    if (preferId) {
      setSelectedVideoId(preferId);
    } else if (!selectedVideoId && next.length > 0) {
      setSelectedVideoId(next[0].id);
    }
  }

  useEffect(() => {
    refreshVideos().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedVideoId) {
      setEvents([]);
      setDetections([]);
      setDetectionsComplete(false);
      return;
    }

    let cancelled = false;
    let stopped = false;
    let interval: number | undefined;

    function stopPolling() {
      stopped = true;
      if (interval !== undefined) {
        window.clearInterval(interval);
      }
    }

    async function refreshSelected() {
      if (stopped) return;
      try {
        const video = await getVideo(selectedVideoId);
        if (cancelled) return;
        setVideos((items) => items.map((item) => (item.id === video.id ? video : item)));
        const terminal = video.status === "completed" || video.status === "failed";

        try {
          const [timeline, latestDetections] = await Promise.all([
            listEvents(selectedVideoId),
            listDetections(selectedVideoId)
          ]);
          if (cancelled) return;
          setEvents(timeline);
          setDetections(latestDetections);
          setDetectionsComplete(latestDetections.length >= video.detection_count);
        } catch (err) {
          if (!cancelled) setError(err instanceof Error ? err.message : "Không tải được dữ liệu phân tích");
        }

        if (terminal) {
          stopPolling();
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Không tải được dữ liệu video");
      }
    }

    refreshSelected();
    interval = window.setInterval(refreshSelected, 4000);
    if (stopped) {
      window.clearInterval(interval);
    }
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [selectedVideoId]);

  useEffect(() => {
    if (pendingSeek === null || !videoRef.current) return;
    videoRef.current.currentTime = Math.max(pendingSeek, 0);
    videoRef.current.play().catch(() => undefined);
    setPendingSeek(null);
  }, [selectedVideoId, pendingSeek]);

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    setKisAnswer(null);
    try {
      const video = await uploadVideo(file);
      await refreshVideos(video.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload thất bại");
    } finally {
      setBusy(false);
    }
  }

  async function handleQuery(event: FormEvent) {
    event.preventDefault();
    if (!selectedVideo) return;
    setBusy(true);
    setError(null);
    try {
      const response = await queryVideo(selectedVideo.id, question);
      setAnswer(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Truy vấn thất bại");
    } finally {
      setBusy(false);
    }
  }

  async function handleAnalyze() {
    if (!selectedVideo) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    setKisAnswer(null);
    setIndexMessage(null);
    try {
      const video = await analyzeVideo(selectedVideo.id);
      setVideos((items) => items.map((item) => (item.id === video.id ? video : item)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không bắt đầu phân tích được");
    } finally {
      setBusy(false);
    }
  }

  async function handleIndexSelected() {
    if (!selectedVideo) return;
    setBusy(true);
    setError(null);
    setIndexMessage(null);
    try {
      const result = await indexSemanticVideo(selectedVideo.id);
      setIndexMessage(
        `Đã index ${result.total_keyframes} keyframes, ${result.total_embeddings} embeddings (${result.model_name}, ${result.embedding_dimension}D).`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không index semantic được");
    } finally {
      setBusy(false);
    }
  }

  async function handleKis(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await semanticKis(kisQuery, kisTopK);
      setKisAnswer(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Semantic KIS thất bại");
    } finally {
      setBusy(false);
    }
  }

  function seekTo(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(seconds, 0);
    videoRef.current.play().catch(() => undefined);
  }

  function resultTimestamp(item: { timestamp?: number | null; frame_id: number; fps?: number | null }) {
    if (typeof item.timestamp === "number" && Number.isFinite(item.timestamp)) {
      return item.timestamp;
    }
    if (item.fps && item.fps > 0) {
      return item.frame_id / item.fps;
    }
    return 0;
  }

  return (
    <main className="app-shell">
      <section className="sidebar">
        <div className="brand">
          <span className="brand-mark">AI</span>
          <div>
            <h1>Traffic Video Assistant</h1>
            <p>Computer Vision + NLP tiếng Việt</p>
          </div>
        </div>

        <label className="upload-box">
          <input
            type="file"
            accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm"
            onChange={(event) => handleUpload(event.target.files?.[0] ?? null)}
            disabled={busy}
          />
          <span>Tải video giao thông</span>
        </label>

        <div className="video-list">
          {videos.map((video) => (
            <button
              className={`video-row ${video.id === selectedVideoId ? "active" : ""}`}
              key={video.id}
              onClick={() => {
                setSelectedVideoId(video.id);
                setAnswer(null);
              }}
            >
              <span>{video.original_filename}</span>
              <small data-status={video.status}>{statusLabel(video.status)}</small>
            </button>
          ))}
          {videos.length === 0 && <p className="empty">Chưa có video nào.</p>}
        </div>
      </section>

      <section className="workspace">
        {error && <div className="error-banner">{error}</div>}

        {selectedVideo ? (
          <>
            <div className="video-panel">
              <video ref={videoRef} src={mediaUrl(selectedVideo.media_url)} controls />
              <div className="meta-strip">
                <span>{statusLabel(selectedVideo.status)}</span>
                <span>{formatDuration(selectedVideo.duration_seconds)}</span>
                <span>{selectedVideo.total_frames ?? 0} frames</span>
                <span>{selectedVideo.fps ? `${selectedVideo.fps.toFixed(1)} FPS` : "FPS chưa có"}</span>
                <span>{selectedVideo.detection_count} detections</span>
                <span>{selectedVideo.track_count} tracks</span>
                <span>{selectedVideo.event_count} events</span>
                <span>YOLO: {selectedVideo.yolo_model_source}</span>
                <span>Semantic: {selectedVideo.semantic_model}</span>
              </div>
              {selectedVideo.status === "processing" && (
                <div className="progress">
                  <div style={{ width: `${Math.max(2, selectedVideo.progress_percent)}%` }} />
                  <span>{selectedVideo.progress_percent}%</span>
                </div>
              )}
              <div className="action-row">
                <button
                  className="analyze-button"
                  onClick={handleAnalyze}
                  disabled={busy || selectedVideo.status === "processing"}
                >
                  {selectedVideo.status === "processing" ? "Đang phân tích" : "Phân tích video"}
                </button>
                <button
                  className="secondary-button"
                  onClick={handleIndexSelected}
                  disabled={busy || selectedVideo.status !== "completed"}
                >
                  Index Semantic KIS
                </button>
              </div>
              {indexMessage && <p className="info-text">{indexMessage}</p>}
              {selectedVideo.error_message && <p className="error-text">{selectedVideo.error_message}</p>}
            </div>

            <div className="analysis-grid">
              <section className="query-panel">
                <form onSubmit={handleQuery}>
                  <label htmlFor="question">Q&A tiếng Việt</label>
                  <div className="query-line">
                    <input
                      id="question"
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                      placeholder="Ví dụ: Tìm đoạn có xe máy"
                    />
                    <button disabled={busy || !question.trim()}>{busy ? "Đang xử lý" : "Hỏi"}</button>
                  </div>
                </form>
                {busy && isLikelyOpenVocabQuery(question) && (
                  <p className="info-text">Dang tim doi tuong mo rong...</p>
                )}

                <div className="chips">
                  {sampleQuestions.map((item) => (
                    <button key={item} onClick={() => setQuestion(item)}>
                      {item}
                    </button>
                  ))}
                </div>

                {answer && (
                  <div className="answer">
                    <h2>Câu trả lời</h2>
                    <p>{answer.answer}</p>
                    <div className="debug-line">
                      {answer.nlp_provider && <span>NLP: {answer.nlp_provider}</span>}
                      {answer.task_type && <span>Task: {answer.task_type}</span>}
                      {answer.status && <span>Status: {answer.status}</span>}
                      {answer.model && <span>Model: {answer.model}</span>}
                      <span>Intent: {answer.debug.intent}</span>
                      <span>Entity: {JSON.stringify(answer.debug.entities)}</span>
                    </div>
                    {answer.parsed_query && (
                      <details className="parsed-query">
                        <summary>Parsed query</summary>
                        <pre>{JSON.stringify(answer.parsed_query, null, 2)}</pre>
                      </details>
                    )}
                    {answer.kis_results && answer.kis_results.length > 0 && (
                      <div className="kis-results compact-kis-results">
                        {answer.kis_results.map((item) => (
                          <article key={`query-${item.video_id}-${item.frame_id}`}>
                            <button
                              className="thumbnail-button"
                              onClick={() => {
                                setSelectedVideoId(item.video_id);
                                setPendingSeek(resultTimestamp(item));
                              }}
                            >
                              <img src={mediaUrl(item.image_url)} alt={`${item.video_name} frame ${item.frame_id}`} />
                            </button>
                            <div>
                              <strong>{item.video_name}</strong>
                              <p>{item.original_filename}</p>
                              <p>
                                {item.object_name ?? "object"} · Frame {item.frame_id} · {item.timestamp.toFixed(2)}s
                                {item.track_id !== null && item.track_id !== undefined && <> · Track #{item.track_id}</>}
                                {formatConfidence(item.confidence) && <> · Confidence {formatConfidence(item.confidence)}</>}
                                {" "}· score {item.score.toFixed(3)}
                              </p>
                              <code>{item.competition_output}</code>
                            </div>
                          </article>
                        ))}
                      </div>
                    )}
                    <div className="segments">
                      {answer.segments.map((segment) => {
                        const averageConfidence = segmentAverageConfidence(segment, detections, detectionsComplete);
                        return (
                          <article key={`${segment.label}-${segment.start_time}-${segment.end_time}`}>
                            <button className="time-button" onClick={() => seekTo(segment.start_time)}>
                              {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                            </button>
                            <span>{segment.evidence}</span>
                            {segment.detector && (
                              <small className="confidence-line">
                                Detector: {segment.detector}
                                {segment.frame_id !== null && segment.frame_id !== undefined && <> Â· Frame {segment.frame_id}</>}
                                {formatConfidence(segment.confidence) && <> Â· Confidence {formatConfidence(segment.confidence)}</>}
                              </small>
                            )}
                            {averageConfidence && (
                              <small className="confidence-line">Confidence TB: {averageConfidence}</small>
                            )}
                          </article>
                        );
                      })}
                    </div>
                  </div>
                )}
              </section>

              <section className="events-panel">
                <div className="section-title">
                  <h2>Event timeline</h2>
                  <span>{events.length}</span>
                </div>
                <div className="events-list">
                  {events.map((item) => (
                    <article key={item.id}>
                      <button className="time-button" onClick={() => seekTo(item.timestamp_start)}>
                        {item.timestamp_start.toFixed(1)}s
                      </button>
                      <div>
                        <strong>{eventLabel(item.event_type)}</strong>
                        <p>{item.description}</p>
                      </div>
                    </article>
                  ))}
                  {events.length === 0 && <p className="empty">Event sẽ xuất hiện sau khi pipeline phân tích video.</p>}
                </div>

                <div className="section-title detection-title">
                  <h2>Detections</h2>
                  <span>{detections.length}</span>
                </div>
                <div className="detection-list">
                  {detections.map((item) => (
                    <article key={item.id}>
                      <button className="time-button" onClick={() => seekTo(item.timestamp)}>
                        {item.timestamp.toFixed(1)}s
                      </button>
                      <div>
                        <strong>
                          {formatClassName(item.class_name)}
                          {formatConfidence(item.confidence) && <> · {formatConfidence(item.confidence)}</>}
                        </strong>
                        <p>
                          Track #{item.track_id ?? "N/A"} · Frame {item.frame_index}
                        </p>
                      </div>
                    </article>
                  ))}
                  {detections.length === 0 && <p className="empty">Detection sẽ xuất hiện sau khi phân tích video.</p>}
                </div>
              </section>
            </div>

            <section className="kis-panel">
              <div className="section-title">
                <h2>Search</h2>
                <span>{kisAnswer?.results.length ?? 0}</span>
              </div>
              {kisAnswer && <p className="model-line">Model: {kisAnswer.model} · {kisAnswer.model_version}</p>}
              <form onSubmit={handleKis}>
                <div className="query-line kis-query-line">
                  <input
                    value={kisQuery}
                    onChange={(event) => setKisQuery(event.target.value)}
                    placeholder="Ví dụ: tìm ô tô, tìm người, xe màu đỏ"
                  />
                  <input
                    className="topk-input"
                    type="number"
                    min={1}
                    max={100}
                    value={kisTopK}
                    onChange={(event) => setKisTopK(Number(event.target.value))}
                  />
                  <button disabled={busy || !kisQuery.trim()}>{busy ? "Đang tìm" : "Tìm"}</button>
                </div>
              </form>
              <div className="chips">
                {sampleKisQueries.map((item) => (
                  <button key={item} onClick={() => setKisQuery(item)}>
                    {item}
                  </button>
                ))}
              </div>
              {kisAnswer && (
                <>
                  {kisAnswer.message && <p className="info-text">{kisAnswer.message}</p>}
                  <div className="kis-results">
                    {kisAnswer.results.map((item) => (
                      <article key={`${item.video_id}-${item.frame_id}`}>
                        <button
                          className="thumbnail-button"
                          onClick={() => {
                            setSelectedVideoId(item.video_id);
                            setPendingSeek(resultTimestamp(item));
                          }}
                        >
                          <img src={mediaUrl(item.image_url)} alt={`${item.video_name} frame ${item.frame_id}`} />
                        </button>
                        <div>
                          <strong>{item.video_name}</strong>
                          <p>{item.original_filename}</p>
                          <p>
                            {item.object_name ?? "object"} · Frame {item.frame_id} · {item.timestamp.toFixed(2)}s
                            {item.track_id !== null && item.track_id !== undefined && <> · Track #{item.track_id}</>}
                            {formatConfidence(item.confidence) && <> · Confidence {formatConfidence(item.confidence)}</>}
                            {" "}· score {item.score.toFixed(3)}
                            {" "}· semantic {item.semantic_score.toFixed(3)}
                          </p>
                          <code>{item.competition_output}</code>
                        </div>
                      </article>
                    ))}
                    {kisAnswer.results.length === 0 && (
                      <p className="empty">
                        {kisAnswer.message ?? "Chưa có kết quả phù hợp hoặc video chưa được index."}
                      </p>
                    )}
                  </div>
                  {kisAnswer.competition_preview.length > 0 && (
                    <pre className="competition-preview">{kisAnswer.competition_preview.join("\n")}</pre>
                  )}
                </>
              )}
            </section>
          </>
        ) : (
          <div className="empty-state">
            <h2>Tải một video để bắt đầu phân tích</h2>
            <p>Hệ thống sẽ nhận diện đối tượng, tracking phương tiện, sinh event và cho phép truy vấn bằng tiếng Việt.</p>
          </div>
        )}
      </section>
    </main>
  );
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "Đang chờ",
    uploaded: "Đã upload",
    processing: "Đang phân tích",
    completed: "Hoàn thành",
    failed: "Lỗi"
  };
  return labels[status] ?? status;
}

function eventLabel(type: string): string {
  const labels: Record<string, string> = {
    vehicle_appears: "Phương tiện xuất hiện",
    object_appears: "Đối tượng xuất hiện",
    vehicle_leaves: "Phương tiện rời khung hình",
    object_leaves: "Đối tượng rời khung hình",
    wrong_way: "Đi ngược chiều",
    vehicle_stopped: "Dừng lại",
    traffic_congestion: "Ùn tắc",
    enter_restricted_area: "Vào vùng hạn chế"
  };
  return labels[type] ?? type;
}

function formatClassName(value: string): string {
  return value.replace(/_/g, " ");
}

function isLikelyOpenVocabQuery(value: string): boolean {
  const normalized = value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d");
  const primary = ["nguoi", "o to", "oto", "xe hoi", "xe may", "mo to", "xe dap", "xe buyt", "xe bus", "xe tai"];
  const open = ["cho", "meo", "xe cuu thuong", "xe cuu hoa", "traffic cone", "cai o", "vali", "ghe", "bien bao"];
  return open.some((item) => normalized.includes(item)) && !primary.some((item) => normalized.includes(item));
}

function formatConfidence(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(1)}%`;
}

function segmentAverageConfidence(
  segment: QueryResponse["segments"][number],
  detections: DetectionItem[],
  detectionsComplete: boolean
): string | null {
  if (!detectionsComplete) return null;
  const trackId = parseTrackId(segment.label) ?? parseTrackId(segment.evidence);
  if (trackId === null) return null;

  const matched = detections.filter(
    (item) =>
      item.track_id === trackId &&
      item.timestamp >= segment.start_time &&
      item.timestamp <= segment.end_time &&
      typeof item.confidence === "number" &&
      Number.isFinite(item.confidence)
  );
  if (matched.length === 0) return null;

  const average = matched.reduce((total, item) => total + Number(item.confidence), 0) / matched.length;
  return formatConfidence(average);
}

function parseTrackId(value: string): number | null {
  const match = value.match(/track\s*#\s*(\d+)/i);
  return match ? Number(match[1]) : null;
}

function formatDuration(value: number | null): string {
  if (!value) return "Thời lượng chưa có";
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
