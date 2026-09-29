import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Dashboard } from "./components/Dashboard";
import { GuideModal, type GuideSection } from "./components/GuideModal";
import { Header } from "./components/Header";
import { ConfirmModal, KeyboardHelpModal, Modal, ReviewerPicker } from "./components/Modals";
import { ScenePromptTemplate } from "./components/ScenePromptTemplate";
import { StepBar } from "./components/StepBar";
import { Step1LayerQuality } from "./components/steps/Step1LayerQuality";
import { Step2Captions } from "./components/steps/Step2Captions";
import { Step3Geometry } from "./components/steps/Step3Geometry";
import { Step4Categories } from "./components/steps/Step4Categories";
import { Step5Final } from "./components/steps/Step5Final";
import { VisualPane, type ViewMode } from "./components/VisualPane";
import {
  acceptanceProblems,
  allStepStatuses,
  hydrateDraft,
  unresolvedBlocking,
} from "./draftUtils";
import type { Bootstrap, Progress, ReviewDraft, ScenePayload } from "./types";

type SaveStatus = "idle" | "unsaved" | "saving" | "saved" | "error";

export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);

  const [view, setView] = useState<"review" | "dashboard">("review");
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [payload, setPayload] = useState<ScenePayload | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [draft, setDraft] = useState<ReviewDraft | null>(null);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const [step, setStep] = useState(1);
  const [showSceneTemplate, setShowSceneTemplate] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("composite");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [showPicker, setShowPicker] = useState(false);

  const [guideOpen, setGuideOpen] = useState(false);
  const [guideSection, setGuideSection] = useState<GuideSection>("workflow");
  const [kbHelpOpen, setKbHelpOpen] = useState(false);
  const [confirm, setConfirm] = useState<null | { kind: "accept" | "reject" | "undo"; body: string }>(null);
  const [finalizeProblems, setFinalizeProblems] = useState<string[]>([]);
  const [toasts, setToasts] = useState<Array<{ id: number; text: string; kind: "ok" | "err" }>>([]);

  const dirtyRef = useRef(false);
  const saveTimerRef = useRef<number | null>(null);
  const durationBaseRef = useRef(0);
  const sceneStartRef = useRef(Date.now());
  const guideShownRef = useRef(false);
  const toastIdRef = useRef(1);

  const toast = useCallback((text: string, kind: "ok" | "err" = "ok") => {
    const id = toastIdRef.current++;
    setToasts((prev) => [...prev, { id, text, kind }]);
    window.setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4200);
  }, []);

  // ------------------------------------------------------------ bootstrap
  const reloadBootstrap = useCallback(async () => {
    try {
      const b = await api.bootstrap();
      setBoot(b);
      if (b.progress) setProgress(b.progress);
      return b;
    } catch (err) {
      setBootError(String(err));
      return null;
    }
  }, []);

  useEffect(() => {
    void reloadBootstrap();
  }, [reloadBootstrap]);

  const selectReviewer = useCallback(
    async (slot: "r1" | "r2") => {
      await api.setReviewer(slot);
      // Reset the loaded scene so the newly-active reviewer's queue drives it.
      setShowPicker(false);
      setSceneId(null);
      setPayload(null);
      setDraft(null);
      await reloadBootstrap();
    },
    [reloadBootstrap],
  );

  // Auto-open the Review Guide on first visit unless opted out.
  useEffect(() => {
    if (boot?.ready && !boot.guide.do_not_autoshow && !guideShownRef.current) {
      guideShownRef.current = true;
      setGuideSection("workflow");
      setGuideOpen(true);
    }
  }, [boot]);

  // ------------------------------------------------------------ scene load
  const currentDuration = useCallback(
    () => Math.round(durationBaseRef.current + (Date.now() - sceneStartRef.current) / 1000),
    [],
  );

  const loadScene = useCallback(async (id: string) => {
    setSceneError(null);
    try {
      const p = await api.getScene(id);
      setPayload(p);
      const d = hydrateDraft(p);
      setDraft(d);
      durationBaseRef.current = d.duration_seconds || 0;
      sceneStartRef.current = Date.now();
      dirtyRef.current = false;
      setSaveStatus("idle");
      setStep(Math.min(Math.max(d.step_reached || 1, 1), 5));
      setShowSceneTemplate(false);
      setSelectedLayerId(p.scene.layers[0]?.layer_id ?? null);
      setViewMode("composite");
      setFinalizeProblems([]);
      // Prefetch the next scene JSON + full image.
      if (p.next_scene_id) {
        void api
          .getScene(p.next_scene_id)
          .then((np) => {
            const img = new Image();
            img.src = `/api/image/${np.scene.full_path}`;
          })
          .catch(() => undefined);
      }
    } catch (err) {
      setPayload(null);
      setDraft(null);
      setSceneError(String(err));
    }
  }, []);

  useEffect(() => {
    if (boot?.ready && !sceneId) {
      if (boot.current_scene_id) setSceneId(boot.current_scene_id);
    }
  }, [boot, sceneId]);

  useEffect(() => {
    if (sceneId) void loadScene(sceneId);
  }, [sceneId, loadScene]);

  const readOnly = payload?.status === "accepted" || payload?.status === "rejected";

  // ------------------------------------------------------------ autosave
  const flushSave = useCallback(async () => {
    if (!sceneId || !draft || !dirtyRef.current || readOnly) return;
    setSaveStatus("saving");
    try {
      const body = { ...draft, duration_seconds: currentDuration() };
      const res = await api.saveDraft(sceneId, body);
      dirtyRef.current = false;
      setSaveStatus("saved");
      setPayload((prev) =>
        prev
          ? {
              ...prev,
              missing_fields: res.missing_fields,
              category_suggestions: res.category_suggestions ?? prev.category_suggestions,
              status: "draft",
            }
          : prev,
      );
    } catch (err) {
      setSaveStatus("error");
      toast(`Draft save failed: ${err}`, "err");
    }
  }, [sceneId, draft, readOnly, currentDuration, toast]);

  const updateDraft = useCallback(
    (updater: (d: ReviewDraft) => ReviewDraft) => {
      if (readOnly) return;
      setDraft((prev) => {
        if (!prev) return prev;
        const next = updater(prev);
        return next;
      });
      dirtyRef.current = true;
      setSaveStatus("unsaved");
    },
    [readOnly],
  );

  useEffect(() => {
    if (!dirtyRef.current) return;
    if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(() => void flushSave(), 900);
    return () => {
      if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current);
    };
  }, [draft, flushSave]);

  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => {
      if (dirtyRef.current) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  // Track the deepest step reached in the draft.
  useEffect(() => {
    if (!draft || readOnly) return;
    if (step > (draft.step_reached || 1)) {
      updateDraft((d) => ({ ...d, step_reached: step }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  // ------------------------------------------------------------ navigation
  const gotoScene = useCallback(
    async (id: string | null | undefined) => {
      if (!id) {
        toast("No further scene in that direction.");
        return;
      }
      await flushSave();
      setSceneId(id);
    },
    [flushSave, toast],
  );

  const resumeQueue = useCallback(async () => {
    await flushSave();
    const res = await api.currentScene();
    if (res.scene_id) setSceneId(res.scene_id);
    else toast("Queue exhausted — every scene has a final decision.");
  }, [flushSave, toast]);

  const jumpToScene = useCallback(
    async (id: string) => {
      try {
        await api.getScene(id);
        await gotoScene(id);
      } catch {
        toast(`Scene not found: ${id}`, "err");
      }
    },
    [gotoScene, toast],
  );

  // ------------------------------------------------------------ actions
  const deferScene = useCallback(async () => {
    if (!sceneId || readOnly) return;
    await flushSave();
    try {
      const res = await api.deferScene(sceneId);
      toast("Scene deferred.");
      if (res.next_scene_id) setSceneId(res.next_scene_id);
      void reloadBootstrap();
    } catch (err) {
      toast(String(err), "err");
    }
  }, [sceneId, readOnly, flushSave, toast, reloadBootstrap]);

  const requestFinalize = useCallback(
    (decision: "accept" | "reject") => {
      if (!payload || !draft) return;
      if (decision === "accept") {
        const problems = acceptanceProblems(
          payload.scene,
          draft,
          payload.category_suggestions,
          unresolvedBlocking(payload, draft),
        );
        if (problems.length > 0) {
          setFinalizeProblems(problems);
          setStep(5);
          return;
        }
        setConfirm({
          kind: "accept",
          body: `Accept scene ${payload.scene.scene_id} with ${draft.suitable_categories.length} suitable categor${draft.suitable_categories.length === 1 ? "y" : "ies"} (primary: ${draft.primary_category})? It will be written to accepted_scenes.json.`,
        });
      } else {
        setConfirm({
          kind: "reject",
          body: `Reject scene ${payload.scene.scene_id}? It will be written to rejected_scenes.json, then the next scene will open.`,
        });
      }
    },
    [payload, draft],
  );

  const runFinalize = useCallback(
    async (decision: "accepted" | "rejected") => {
      if (!sceneId || !draft) return;
      setConfirm(null);
      const body = { ...draft, duration_seconds: currentDuration() };
      try {
        const res = await api.finalize(sceneId, decision, body);
        if (!res.ok) {
          setFinalizeProblems(res.problems ?? ["Finalize failed."]);
          setStep(5);
          return;
        }
        setFinalizeProblems([]);
        if (res.progress) setProgress(res.progress);
        dirtyRef.current = false;
        toast(decision === "accepted" ? "Scene accepted ✓" : "Scene rejected ✓");
        if (res.next_scene_id) setSceneId(res.next_scene_id);
        else await loadScene(sceneId);
      } catch (err) {
        toast(String(err), "err");
      }
    },
    [sceneId, draft, currentDuration, toast, loadScene],
  );

  const undoLast = useCallback(async () => {
    setConfirm(null);
    try {
      const res = await api.undo();
      if (!res.ok) {
        toast(res.problems?.[0] ?? "Nothing to undo.", "err");
        return;
      }
      toast(`Reverted final decision on ${res.scene_id}; it is a draft again.`);
      void reloadBootstrap();
      if (res.scene_id) setSceneId(res.scene_id);
      if (res.scene_id === sceneId && res.scene_id) void loadScene(res.scene_id);
    } catch (err) {
      toast(String(err), "err");
    }
  }, [toast, reloadBootstrap, sceneId, loadScene]);

  const openGuide = useCallback((section: GuideSection) => {
    setGuideSection(section);
    setGuideOpen(true);
  }, []);

  // ------------------------------------------------------------ keyboard
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable);
      if (event.key === "Escape") {
        setKbHelpOpen(false);
        setConfirm(null);
        return;
      }
      if (typing) return;
      if (guideOpen || kbHelpOpen || confirm) return;
      if (event.key === "?") setKbHelpOpen(true);
      else if (event.key === "g") openGuide("workflow");
      else if (event.key >= "1" && event.key <= "5" && view === "review") {
        setShowSceneTemplate(false);
        setStep(Number(event.key));
      }
      else if (event.key === "[") void gotoScene(payload?.prev_scene_id);
      else if (event.key === "]") void gotoScene(payload?.next_scene_id);
      else if (event.key === "d" && view === "review") void deferScene();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [guideOpen, kbHelpOpen, confirm, view, payload, gotoScene, deferScene, openGuide]);

  // ------------------------------------------------------------ screens
  if (bootError) {
    return (
      <div className="center-screen">
        <h2>Backend unreachable</h2>
        <p className="muted">{bootError}</p>
        <p className="muted">Start it with <code className="mono">run.ps1</code> and reload.</p>
        <button className="primary" onClick={() => window.location.reload()}>Reload</button>
      </div>
    );
  }
  if (!boot) {
    return (
      <div className="center-screen">
        <div className="spin" />
        <p className="muted">Connecting to the local backend…</p>
      </div>
    );
  }
  if (!boot.index_ready) {
    return (
      <div className="center-screen">
        <h2>First run: build the local scene index</h2>
        <p className="muted" style={{ maxWidth: 560 }}>
          The scene records are streamed once into a local JSONL cache under{" "}
          <code className="mono">vendor/human-review/runtime/cache/</code>, restricted to the
          curated scenes under <code className="mono">data/selected_1200/</code> and with their
          images repointed there. The sources are never modified and review decisions are always
          preserved.
        </p>
        {boot.sources && (
          <table className="kv" style={{ maxWidth: 560 }}>
            <tbody>
              {Object.entries(boot.sources).map(([subset, info]) => (
                <tr key={subset}>
                  <td>{subset}</td>
                  <td>{info.exists ? `✓ ${info.path}` : "✗ source file missing"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {indexing ? (
          <>
            <div className="spin" />
            <p className="muted">Streaming and indexing scenes… (this can take a few minutes)</p>
          </>
        ) : (
          <button
            className="primary"
            onClick={async () => {
              setIndexing(true);
              try {
                const res = await api.rebuildIndex();
                toast(`Indexed ${res.scenes} scenes.`);
                await reloadBootstrap();
              } catch (err) {
                toast(String(err), "err");
              } finally {
                setIndexing(false);
              }
            }}
          >
            Build index now
          </button>
        )}
      </div>
    );
  }

  const needsReviewer = !boot.reviewer;

  // Reviewer selection is a required gate once the index is built.
  if (needsReviewer) {
    return (
      <div className="app-shell">
        <ReviewerPicker reviewers={boot.reviewers ?? []} onSelect={selectReviewer} />
      </div>
    );
  }

  const stepStatuses =
    payload && draft ? allStepStatuses(payload.scene, draft, payload.category_suggestions) : [];
  const firstThreeComplete = stepStatuses.slice(0, 3).every((status) => status.complete);

  return (
    <div className="app-shell">
      <Header
        progress={progress}
        payload={payload}
        saveStatus={saveStatus}
        reviewerId={boot.reviewer_id}
        onSwitchReviewer={() => setShowPicker(true)}
        view={view}
        onToggleView={() => setView((v) => (v === "review" ? "dashboard" : "review"))}
        onOpenGuide={() => openGuide("workflow")}
        onJump={jumpToScene}
        onResume={() => void resumeQueue()}
        onUndo={() => setConfirm({ kind: "undo", body: "Undo the most recent final decision? The scene returns to draft state and the exports are rewritten." })}
        onBackup={async () => {
          try {
            const res = await api.backupNow();
            toast(`State backed up to runtime/backups/${res.backup_path}`);
          } catch (err) {
            toast(String(err), "err");
          }
        }}
        onExport={async () => {
          try {
            const res = await api.exportNow();
            toast(`Exports rewritten: ${res.accepted} accepted, ${res.rejected} rejected.`);
          } catch (err) {
            toast(String(err), "err");
          }
        }}
        onKeyboardHelp={() => setKbHelpOpen(true)}
      />

      {boot.warnings.length > 0 && (
        <div className="warn-banner" style={{ margin: "6px 12px 0" }}>
          {boot.warnings.map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      )}

      {view === "dashboard" ? (
        <Dashboard />
      ) : sceneError ? (
        <div className="center-screen" style={{ height: "auto", flex: 1 }}>
          <h3>Scene failed to load</h3>
          <p className="muted">{sceneError}</p>
          <button className="primary" onClick={() => void resumeQueue()}>Resume queue</button>
        </div>
      ) : !payload || !draft ? (
        <div className="center-screen" style={{ height: "auto", flex: 1 }}>
          <div className="spin" />
          <p className="muted">Loading scene…</p>
        </div>
      ) : (
        <div className="main-split">
          <div className="visual-area">
            <VisualPane
              payload={payload}
              draft={draft}
              selectedLayerId={selectedLayerId}
              onSelectLayer={setSelectedLayerId}
              viewMode={viewMode}
              onViewMode={setViewMode}
              editableGeometry={step === 3 && !showSceneTemplate && !readOnly}
              onBboxChange={(layerId, bbox) =>
                updateDraft((d) => ({
                  ...d,
                  layer_reviews: {
                    ...d.layer_reviews,
                    [String(layerId)]: { ...d.layer_reviews[String(layerId)], bbox_override: bbox },
                  },
                }))
              }
              onCentroidChange={(layerId, xy) =>
                updateDraft((d) => ({
                  ...d,
                  layer_reviews: {
                    ...d.layer_reviews,
                    [String(layerId)]: { ...d.layer_reviews[String(layerId)], centroid_override: xy },
                  },
                }))
              }
            />
          </div>
          <div className="review-area">
            {readOnly && (
              <div className="warn-banner" style={{ margin: "8px 12px 0" }}>
                This scene is finalized ({payload.status}). Use <b>Undo last decision</b> in the header to edit it again
                (only the most recent decision can be undone).
              </div>
            )}
            <StepBar
              step={step}
              onStep={(target) => {
                setShowSceneTemplate(false);
                setStep(target);
              }}
              statuses={stepStatuses}
            />
            <div className="step-content">
              {showSceneTemplate ? (
                <ScenePromptTemplate scene={payload.scene} draft={draft} />
              ) : (
                <>
              {step === 1 && (
                <Step1LayerQuality
                  payload={payload}
                  draft={draft}
                  updateDraft={updateDraft}
                  readOnly={!!readOnly}
                  onSelectLayer={setSelectedLayerId}
                  selectedLayerId={selectedLayerId}
                  openGuide={openGuide}
                  onRejectNow={() => requestFinalize("reject")}
                />
              )}
              {step === 2 && (
                <Step2Captions
                  payload={payload}
                  draft={draft}
                  updateDraft={updateDraft}
                  readOnly={!!readOnly}
                  onSelectLayer={setSelectedLayerId}
                  openGuide={openGuide}
                />
              )}
              {step === 3 && (
                <Step3Geometry
                  payload={payload}
                  draft={draft}
                  updateDraft={updateDraft}
                  readOnly={!!readOnly}
                  selectedLayerId={selectedLayerId}
                  onSelectLayer={setSelectedLayerId}
                  openGuide={openGuide}
                />
              )}
              {step === 4 && (
                <Step4Categories
                  payload={payload}
                  draft={draft}
                  updateDraft={updateDraft}
                  readOnly={!!readOnly}
                  progress={progress}
                  openGuide={openGuide}
                />
              )}
              {step === 5 && (
                <Step5Final
                  payload={payload}
                  draft={draft}
                  updateDraft={updateDraft}
                  readOnly={!!readOnly}
                  problems={finalizeProblems}
                  stepStatuses={stepStatuses}
                  openGuide={openGuide}
                  onAccept={() => requestFinalize("accept")}
                />
              )}
                </>
              )}
            </div>
            <div className="step-footer">
              <button
                onClick={() => {
                  if (showSceneTemplate) {
                    setShowSceneTemplate(false);
                    setStep(3);
                  } else {
                    setStep((s) => Math.max(1, s - 1));
                  }
                }}
                disabled={!showSceneTemplate && step === 1}
              >
                ← Back
              </button>
              <button
                onClick={() => {
                  if (showSceneTemplate) {
                    setShowSceneTemplate(false);
                    setStep(4);
                  } else if (step === 3) {
                    setShowSceneTemplate(true);
                  } else {
                    setStep((s) => Math.min(5, s + 1));
                  }
                }}
                disabled={!showSceneTemplate && (step === 5 || (step === 3 && !firstThreeComplete))}
                title={step === 3 && !firstThreeComplete ? "Complete Steps 1–3 first" : undefined}
              >
                {showSceneTemplate ? "Continue to Categories →" : "Next step →"}
              </button>
              <span className="spacer" />
              <button onClick={() => void gotoScene(payload.prev_scene_id)} title="Previous unfinished scene ([)">
                ‹ Prev scene
              </button>
              <button onClick={() => void gotoScene(payload.next_scene_id)} title="Next unfinished scene (])">
                Next scene ›
              </button>
              <button onClick={() => void deferScene()} disabled={!!readOnly} title="Defer this scene (d)">
                Defer
              </button>
              <button onClick={() => void flushSave()} disabled={!!readOnly}>
                Save draft
              </button>
            </div>
          </div>
        </div>
      )}

      {guideOpen && (
        <GuideModal
          section={guideSection}
          doNotAutoshow={boot.guide.do_not_autoshow}
          onPreference={async (value) => {
            try {
              await api.setGuidePreference(value);
              setBoot((prev) => (prev ? { ...prev, guide: { do_not_autoshow: value } } : prev));
            } catch (err) {
              toast(String(err), "err");
            }
          }}
          onClose={() => setGuideOpen(false)}
        />
      )}
      {showPicker && (
        <ReviewerPicker
          reviewers={boot.reviewers ?? []}
          onSelect={selectReviewer}
          onClose={() => setShowPicker(false)}
        />
      )}
      {kbHelpOpen && <KeyboardHelpModal onClose={() => setKbHelpOpen(false)} />}
      {confirm && (
        <ConfirmModal
          title={
            confirm.kind === "accept" ? "Accept scene" : confirm.kind === "reject" ? "Reject scene" : "Undo last decision"
          }
          body={confirm.body}
          confirmLabel={confirm.kind === "accept" ? "Accept" : confirm.kind === "reject" ? "Reject" : "Undo"}
          danger={confirm.kind !== "accept"}
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            if (confirm.kind === "accept") void runFinalize("accepted");
            else if (confirm.kind === "reject") void runFinalize("rejected");
            else void undoLast();
          }}
        />
      )}

      <div style={{ position: "fixed", bottom: 14, right: 14, display: "flex", flexDirection: "column", gap: 6, zIndex: 90 }}>
        {toasts.map((t) => (
          <div
            key={t.id}
            className="card"
            style={{
              margin: 0,
              borderColor: t.kind === "err" ? "var(--bad)" : "var(--good)",
              maxWidth: 420,
              fontSize: 13,
            }}
          >
            {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}
