import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { imageUrl } from "../api";
import type { ReviewDraft, Scene, SceneLayer, ScenePayload } from "../types";
import { GeometryOverlay, layerColor } from "./GeometryOverlay";

export type ViewMode = "composite" | "background" | "layer" | "alpha" | "overlay" | "strip";

const MODE_LABELS: Array<[ViewMode, string]> = [
  ["composite", "Full image"],
  ["background", "Background"],
  ["layer", "Isolated layer"],
  ["alpha", "Alpha mask"],
  ["overlay", "Overlay"],
  ["strip", "Progressive strip"],
];

/** Canvas that renders the alpha channel of an RGBA layer as grayscale. */
function AlphaCanvas(props: { src: string; width: number; height: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      try {
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const px = data.data;
        for (let i = 0; i < px.length; i += 4) {
          const a = px[i + 3];
          px[i] = a;
          px[i + 1] = a;
          px[i + 2] = a;
          px[i + 3] = 255;
        }
        ctx.putImageData(data, 0, 0);
      } catch {
        /* canvas tainted — never expected for same-origin images */
      }
    };
    img.src = props.src;
  }, [props.src]);
  return <canvas ref={canvasRef} width={props.width} height={props.height} data-testid="alpha-canvas" />;
}

function MissingImage(props: { label: string }) {
  return (
    <div className="missing-image-placeholder" data-testid="missing-image">
      <span style={{ fontSize: 30 }}>🖼✕</span>
      <span>{props.label} is missing on disk</span>
      <span className="small-note">Exclude the layer or reject/defer the scene.</span>
    </div>
  );
}

export function VisualPane(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  selectedLayerId: number | null;
  onSelectLayer: (layerId: number | null) => void;
  viewMode: ViewMode;
  onViewMode: (mode: ViewMode) => void;
  editableGeometry: boolean;
  onBboxChange: (layerId: number, bbox: [number, number, number, number]) => void;
  onCentroidChange: (layerId: number, xy: [number, number]) => void;
}) {
  const { payload, draft, selectedLayerId, viewMode } = props;
  const scene = payload.scene;
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ scale: 1, tx: 0, ty: 0 });
  const [showGeometry, setShowGeometry] = useState(true);
  const panRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  const selectedLayer = scene.layers.find((l) => l.layer_id === selectedLayerId) ?? null;

  const fit = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const cw = el.clientWidth;
    const ch = el.clientHeight;
    if (cw === 0 || ch === 0 || !scene.width || !scene.height) return;
    const scale = Math.min(cw / scene.width, ch / scene.height) * 0.97;
    setTransform({
      scale,
      tx: (cw - scene.width * scale) / 2,
      ty: (ch - scene.height * scale) / 2,
    });
  }, [scene.width, scene.height]);

  useLayoutEffect(() => {
    fit();
  }, [fit, scene.scene_id, viewMode]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => fit());
    observer.observe(el);
    return () => observer.disconnect();
  }, [fit]);

  const onWheel = useCallback(
    (event: React.WheelEvent) => {
      event.preventDefault();
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      setTransform((prev) => {
        const factor = Math.pow(1.0018, -event.deltaY);
        const scale = Math.min(40, Math.max(0.05, prev.scale * factor));
        const ratio = scale / prev.scale;
        return {
          scale,
          tx: mx - (mx - prev.tx) * ratio,
          ty: my - (my - prev.ty) * ratio,
        };
      });
    },
    [],
  );

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      if (event.button !== 0) return;
      const target = event.target as HTMLElement;
      if (target.closest(".editable") || target.closest(".handle")) return;
      panRef.current = { x: event.clientX, y: event.clientY, tx: transform.tx, ty: transform.ty };
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    },
    [transform],
  );

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    const pan = panRef.current;
    if (!pan) return;
    setTransform((prev) => ({
      ...prev,
      tx: pan.tx + (event.clientX - pan.x),
      ty: pan.ty + (event.clientY - pan.y),
    }));
  }, []);

  const onPointerUp = useCallback(() => {
    panRef.current = null;
  }, []);

  const checks = payload.file_checks;
  const stripPath = `${scene.render_dir}/progressive_strip.png`;

  const renderStage = () => {
    if (viewMode === "strip") {
      if (checks.progressive_strip === false) return <MissingImage label="progressive_strip.png" />;
      return (
        <img
          src={imageUrl(stripPath)}
          alt="Progressive strip"
          style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", position: "absolute", inset: 0, margin: "auto" }}
        />
      );
    }
    const missingFull = checks.full === false;
    const missingBackground = checks.background === false;
    const missingSelected = selectedLayer ? checks[`layer_${selectedLayer.layer_id}`] === false : false;

    if (viewMode === "composite" && missingFull) return <MissingImage label="full.png" />;
    if (viewMode === "background" && missingBackground) return <MissingImage label="background layer_0.png" />;
    if ((viewMode === "layer" || viewMode === "alpha" || viewMode === "overlay") && !selectedLayer)
      return <div className="missing-image-placeholder">Select a foreground layer on the right.</div>;
    if ((viewMode === "layer" || viewMode === "alpha") && missingSelected)
      return <MissingImage label={`layer_${selectedLayer!.layer_id}.png`} />;

    return (
      <div
        className="viewer-stage"
        style={{
          transform: `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`,
          width: scene.width,
          height: scene.height,
        }}
      >
        {viewMode === "composite" && <img src={imageUrl(scene.full_path)} alt="Full render" width={scene.width} height={scene.height} />}
        {viewMode === "background" && scene.background && (
          <img src={imageUrl(scene.background.path)} alt="Background layer" width={scene.width} height={scene.height} />
        )}
        {viewMode === "layer" && selectedLayer && (
          <img src={imageUrl(selectedLayer.path)} alt={`Layer ${selectedLayer.layer_id}`} width={scene.width} height={scene.height} />
        )}
        {viewMode === "alpha" && selectedLayer && (
          <AlphaCanvas src={imageUrl(selectedLayer.path)} width={scene.width} height={scene.height} />
        )}
        {viewMode === "overlay" && selectedLayer && (
          <>
            {!missingFull && (
              <img src={imageUrl(scene.full_path)} alt="Full render (dimmed)" width={scene.width} height={scene.height} style={{ opacity: 0.28 }} />
            )}
            {!missingSelected && (
              <img
                className="stacked"
                src={imageUrl(selectedLayer.path)}
                alt={`Layer ${selectedLayer.layer_id} highlighted`}
                width={scene.width}
                height={scene.height}
              />
            )}
          </>
        )}
        {showGeometry && viewMode !== "alpha" && (
          <GeometryOverlay
            scene={scene}
            draft={draft}
            selectedLayerId={selectedLayerId}
            scale={transform.scale}
            editable={props.editableGeometry}
            onBboxChange={props.onBboxChange}
            onCentroidChange={props.onCentroidChange}
          />
        )}
      </div>
    );
  };

  return (
    <>
      <div className="viewer-panel">
        <div className="panel-head">
          <div className="mode-tabs" role="tablist" aria-label="View mode">
            {MODE_LABELS.map(([mode, label]) => (
              <button key={mode} className={viewMode === mode ? "active" : ""} onClick={() => props.onViewMode(mode)}>
                {label}
              </button>
            ))}
          </div>
          <button className="small" onClick={() => setShowGeometry((v) => !v)} title="Toggle geometry overlay">
            {showGeometry ? "◼ Geometry on" : "◻ Geometry off"}
          </button>
          <button className="small" onClick={fit} title="Fit image to view">
            Fit
          </button>
          <button
            className="small"
            onClick={() => {
              const el = containerRef.current;
              if (!el) return;
              setTransform({ scale: 1, tx: (el.clientWidth - scene.width) / 2, ty: (el.clientHeight - scene.height) / 2 });
            }}
            title="Zoom to 100%"
          >
            1:1
          </button>
          <span className="zoom-info">
            {scene.width}×{scene.height} · {(transform.scale * 100).toFixed(0)}%
          </span>
        </div>
        <div
          ref={containerRef}
          className={`viewer-body ${viewMode === "layer" || viewMode === "alpha" ? "checker" : ""}`}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          style={{ cursor: panRef.current ? "grabbing" : "grab", touchAction: "none" }}
          data-testid="viewer-body"
        >
          {renderStage()}
        </div>
      </div>

      <div className="layer-rail" data-testid="layer-rail">
        <ThumbCard
          title="Full image"
          path={scene.full_path}
          missing={checks.full === false}
          selected={viewMode === "composite"}
          onClick={() => props.onViewMode("composite")}
          subtitle={`${scene.width}×${scene.height} · ${scene.subset.toUpperCase()}`}
        />
        <ThumbCard
          title="Background"
          path={scene.background?.path ?? ""}
          missing={checks.background === false}
          selected={viewMode === "background"}
          onClick={() => props.onViewMode("background")}
          subtitle={(scene.background?.captions?.blip2 ?? "").slice(0, 46) || "z0"}
          zIndex={0}
        />
        {scene.layers.map((layer, index) => (
          <LayerCard
            key={layer.layer_id}
            layer={layer}
            color={layerColor(index)}
            payload={payload}
            draft={draft}
            selected={selectedLayerId === layer.layer_id && (viewMode === "layer" || viewMode === "alpha" || viewMode === "overlay")}
            onClick={() => {
              props.onSelectLayer(layer.layer_id);
              if (viewMode !== "layer" && viewMode !== "alpha" && viewMode !== "overlay") props.onViewMode("layer");
            }}
          />
        ))}
        <details className="drawer">
          <summary>Scene metadata (annotation · inventory · validation)</summary>
          <div className="drawer-body">
            <pre>{JSON.stringify({ annotation: scene.annotation, inventory: scene.inventory, validation: scene.validation }, null, 1)}</pre>
          </div>
        </details>
      </div>
    </>
  );
}

function ThumbCard(props: {
  title: string;
  path: string;
  missing: boolean;
  selected: boolean;
  onClick: () => void;
  subtitle?: string;
  zIndex?: number;
}) {
  return (
    <div className={`layer-card ${props.selected ? "selected" : ""}`} onClick={props.onClick}>
      <div className="layer-thumb">{props.missing ? <span title="missing">✕</span> : props.path && <img src={imageUrl(props.path)} alt={props.title} loading="lazy" />}</div>
      <div className="layer-meta">
        <div className="row1">
          {props.title}
          {props.zIndex != null && <span className="z-chip">z{props.zIndex}</span>}
          {props.missing && <span className="badge blocking">missing file</span>}
        </div>
        {props.subtitle && <div className="row2">{props.subtitle}</div>}
      </div>
    </div>
  );
}

function LayerCard(props: {
  layer: SceneLayer;
  color: string;
  payload: ScenePayload;
  draft: ReviewDraft;
  selected: boolean;
  onClick: () => void;
}) {
  const { layer, payload, draft } = props;
  const lr = draft.layer_reviews[String(layer.layer_id)];
  const missingCounts = payload.missing_fields.per_layer[String(layer.layer_id)];
  const fileMissing = payload.file_checks[`layer_${layer.layer_id}`] === false;
  const label = lr?.label || layer.label || lr?.caption || layer.captions?.blip2 || "(no caption)";
  const decision = lr?.decision;

  // Live missing-count: label/caption completion reflected immediately from the draft.
  let humanMissing = 0;
  if (decision !== "exclude") {
    if (!lr?.decision) humanMissing += 1;
    if (!lr?.label?.trim()) humanMissing += 1;
    if (!lr?.caption?.trim()) humanMissing += 1;
    if (lr?.decision && lr.decision !== "exclude" && !lr?.geometry_status) humanMissing += 1;
  }

  return (
    <div
      className={`layer-card ${props.selected ? "selected" : ""} ${decision === "exclude" ? "excluded" : ""}`}
      onClick={props.onClick}
      data-testid={`layer-card-${layer.layer_id}`}
    >
      <div className="layer-thumb" style={{ boxShadow: `inset 0 0 0 2px ${props.color}55` }}>
        {fileMissing ? <span title="missing">✕</span> : <img src={imageUrl(layer.path)} alt={`Layer ${layer.layer_id}`} loading="lazy" />}
      </div>
      <div className="layer-meta">
        <div className="row1">
          <span style={{ color: props.color }}>L{layer.layer_id}</span>
          <span className="z-chip">z{layer.z_index}</span>
          {decision ? (
            <span className={`badge ${decision}`}>
              {decision === "keep" ? "keep" : "excluded"}
            </span>
          ) : (
            <span className="badge pending">unreviewed</span>
          )}
          {fileMissing && <span className="badge blocking">no file</span>}
          {humanMissing > 0 && decision !== "exclude" && (
            <span className="badge missing" title="Fields the reviewer still has to complete">
              {humanMissing} missing
            </span>
          )}
          {humanMissing === 0 && decision && decision !== "exclude" && <span className="badge ok">complete</span>}
          {(missingCounts?.technical_ok ?? 0) > 0 && (
            <span className="badge tech" title="Optional technical fields that legitimately stay null">
              tech-null
            </span>
          )}
        </div>
        <div className="row2" title={label}>
          {label}
        </div>
      </div>
    </div>
  );
}
