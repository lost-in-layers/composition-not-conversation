// Thin typed client for the local backend. Everything stays on 127.0.0.1.
import type {
  Bootstrap,
  DashboardStats,
  FinalizeResult,
  Progress,
  ReviewDraft,
  ReviewerSlot,
  ScenePayload,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok && response.status !== 422) {
    const detail = body && typeof body === "object" && "detail" in body ? String(body.detail) : response.statusText;
    throw new Error(detail);
  }
  return body as T;
}

export const api = {
  bootstrap: () => request<Bootstrap>("/api/bootstrap"),
  rebuildIndex: () =>
    request<{ scenes: number; subsets: Record<string, number> }>("/api/index/rebuild", { method: "POST" }),
  setReviewer: (reviewer: ReviewerSlot) =>
    request<{ reviewer: ReviewerSlot; reviewer_id: string }>("/api/reviewer", {
      method: "POST",
      body: JSON.stringify({ reviewer }),
    }),
  switchReviewer: () =>
    request<{ reviewer: null }>("/api/reviewer/switch", { method: "POST" }),
  setGuidePreference: (do_not_autoshow: boolean) =>
    request<{ guide: { do_not_autoshow: boolean } }>("/api/guide-preference", {
      method: "POST",
      body: JSON.stringify({ do_not_autoshow }),
    }),
  currentScene: () => request<{ scene_id: string | null }>("/api/queue/current"),
  getScene: (sceneId: string) => request<ScenePayload>(`/api/scene/${encodeURIComponent(sceneId)}`),
  saveDraft: (sceneId: string, draft: ReviewDraft) =>
    request<{
      saved: boolean;
      missing_fields: ScenePayload["missing_fields"];
      category_suggestions: ScenePayload["category_suggestions"];
    }>(`/api/scene/${encodeURIComponent(sceneId)}/draft`, { method: "POST", body: JSON.stringify({ draft }) }),
  deferScene: (sceneId: string) =>
    request<{ deferred: boolean; next_scene_id: string | null }>(
      `/api/scene/${encodeURIComponent(sceneId)}/defer`,
      { method: "POST" },
    ),
  undeferScene: (sceneId: string) =>
    request<{ deferred: boolean }>(`/api/scene/${encodeURIComponent(sceneId)}/undefer`, { method: "POST" }),
  finalize: (sceneId: string, decision: "accepted" | "rejected", review: ReviewDraft) =>
    request<FinalizeResult>(`/api/scene/${encodeURIComponent(sceneId)}/finalize`, {
      method: "POST",
      body: JSON.stringify({ decision, review }),
    }),
  undo: () => request<{ ok: boolean; scene_id?: string; problems?: string[] }>("/api/undo", { method: "POST" }),
  progress: () => request<Progress>("/api/progress"),
  dashboard: () => request<{ progress: Progress; dashboard: DashboardStats }>("/api/dashboard"),
  deferred: () => request<Array<{ scene_id: string; subset: string; n_layers: number }>>("/api/deferred"),
  exportNow: () => request<{ accepted: number; rejected: number; problems: string[] }>("/api/export", { method: "POST" }),
  backupNow: () => request<{ backup_path: string; directory: string }>("/api/backup", { method: "POST" }),
};

export function imageUrl(repoRelativePath: string): string {
  return `/api/image/${repoRelativePath.split("/").map(encodeURIComponent).join("/")}`;
}
