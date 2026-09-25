import type { DetectionItem, EventItem, KISResponse, QueryResponse, SemanticIndexResponse, VideoItem } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export async function uploadVideo(file: File): Promise<VideoItem> {
  const form = new FormData();
  form.append("file", file);
  const payload = await request<{ video: VideoItem }>("/api/videos/upload", {
    method: "POST",
    body: form
  });
  return payload.video;
}

export async function listVideos(): Promise<VideoItem[]> {
  return request<VideoItem[]>("/api/videos");
}

export async function getVideo(id: string): Promise<VideoItem> {
  return request<VideoItem>(`/api/videos/${id}`);
}

export async function analyzeVideo(id: string): Promise<VideoItem> {
  const payload = await request<{ video: VideoItem }>(`/api/videos/${id}/analyze`, {
    method: "POST"
  });
  return payload.video;
}

export async function listEvents(id: string): Promise<EventItem[]> {
  return request<EventItem[]>(`/api/videos/${id}/events`);
}

export async function listDetections(id: string, limit = 2000): Promise<DetectionItem[]> {
  return request<DetectionItem[]>(`/api/videos/${id}/detections?limit=${limit}`);
}

export async function indexSemanticVideo(id: string): Promise<SemanticIndexResponse> {
  return request<SemanticIndexResponse>(`/api/semantic/index/${id}`, {
    method: "POST"
  });
}

export async function semanticKis(query: string, topK = 20): Promise<KISResponse> {
  return request<KISResponse>("/api/retrieval/kis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK })
  });
}

export async function queryVideo(videoId: string, question: string): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, question })
  });
}

export function mediaUrl(path: string): string {
  return `${API_BASE}${path}`;
}
