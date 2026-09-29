import { useCallback, useRef } from "react";
import { clampBbox } from "../draftUtils";
import type { ReviewDraft, Scene } from "../types";

export const LAYER_COLORS = ["#4f8ef7", "#34c07c", "#e8b93c", "#e5604c", "#a78bfa", "#22d3ee", "#f472b6", "#a3e635"];

export function layerColor(index: number): string {
  return LAYER_COLORS[index % LAYER_COLORS.length];
}

type DragState =
  | { kind: "move"; layerId: number; startX: number; startY: number; bbox: [number, number, number, number] }
  | { kind: "resize"; layerId: number; corner: number; bbox: [number, number, number, number] }
  | { kind: "centroid"; layerId: number };

/** SVG overlay drawn in image-pixel coordinates on top of the image stage. */
export function GeometryOverlay(props: {
  scene: Scene;
  draft: ReviewDraft;
  selectedLayerId: number | null;
  scale: number;
  editable: boolean;
  onBboxChange?: (layerId: number, bbox: [number, number, number, number]) => void;
  onCentroidChange?: (layerId: number, xy: [number, number]) => void;
}) {
  const { scene, draft, selectedLayerId, scale, editable } = props;
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<DragState | null>(null);

  const toImageCoords = useCallback(
    (event: PointerEvent | React.PointerEvent): [number, number] => {
      const rect = svgRef.current!.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * scene.width;
      const y = ((event.clientY - rect.top) / rect.height) * scene.height;
      return [x, y];
    },
    [scene.width, scene.height],
  );

  const beginDrag = useCallback(
    (state: DragState, startEvent: React.PointerEvent) => {
      if (!editable) return;
      startEvent.preventDefault();
      startEvent.stopPropagation();
      dragRef.current = state;
      const start = toImageCoords(startEvent);

      const onMove = (event: PointerEvent) => {
        const drag = dragRef.current;
        if (!drag) return;
        const [x, y] = toImageCoords(event);
        if (drag.kind === "move") {
          const dx = x - start[0];
          const dy = y - start[1];
          const [x1, y1, x2, y2] = drag.bbox;
          const w = x2 - x1;
          const h = y2 - y1;
          let nx1 = Math.max(0, Math.min(x1 + dx, scene.width - w));
          let ny1 = Math.max(0, Math.min(y1 + dy, scene.height - h));
          props.onBboxChange?.(drag.layerId, clampBbox([nx1, ny1, nx1 + w, ny1 + h], scene.width, scene.height));
        } else if (drag.kind === "resize") {
          const [x1, y1, x2, y2] = drag.bbox;
          let next: [number, number, number, number] = [x1, y1, x2, y2];
          if (drag.corner === 0) next = [x, y, x2, y2];
          else if (drag.corner === 1) next = [x1, y, x2, y2].map((v, i) => (i === 2 ? x : v)) as [number, number, number, number];
          else if (drag.corner === 2) next = [x1, y1, x2, y].map((v, i) => (i === 0 ? x : v)) as [number, number, number, number];
          else next = [x1, y1, x, y];
          const fixed: [number, number, number, number] = [
            Math.min(next[0], next[2] - 1),
            Math.min(next[1], next[3] - 1),
            Math.max(next[2], next[0] + 1),
            Math.max(next[3], next[1] + 1),
          ];
          props.onBboxChange?.(drag.layerId, clampBbox(fixed, scene.width, scene.height));
        } else {
          props.onCentroidChange?.(drag.layerId, [
            Math.max(0, Math.min(Math.round(x * 10) / 10, scene.width)),
            Math.max(0, Math.min(Math.round(y * 10) / 10, scene.height)),
          ]);
        }
      };
      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [editable, toImageCoords, scene.width, scene.height, props],
  );

  const handleSize = 8 / Math.max(scale, 0.01);

  return (
    <svg
      ref={svgRef}
      className="geometry-svg"
      width={scene.width}
      height={scene.height}
      viewBox={`0 0 ${scene.width} ${scene.height}`}
      data-testid="geometry-overlay"
    >
      {scene.layers.map((layer, index) => {
        const lr = draft.layer_reviews[String(layer.layer_id)];
        if (lr?.decision === "exclude") return null;
        const color = layerColor(index);
        const original = layer.geometry?.bbox_xyxy;
        const override = lr?.bbox_override ?? null;
        const active: [number, number, number, number] | undefined = override ?? original;
        if (!active) return null;
        const [x1, y1, x2, y2] = active;
        const selected = layer.layer_id === selectedLayerId;
        const editThis = editable && selected;
        const centroid =
          lr?.centroid_override ?? layer.geometry?.support_centroid_xy ?? layer.geometry?.alpha_weighted_centroid_xy;
        const corners: Array<[number, number]> = [
          [x1, y1],
          [x2, y1],
          [x1, y2],
          [x2, y2],
        ];
        return (
          <g key={layer.layer_id} opacity={selected ? 1 : 0.55}>
            {override && original && (
              <rect
                className="bbox"
                x={original[0]}
                y={original[1]}
                width={original[2] - original[0]}
                height={original[3] - original[1]}
                stroke={color}
                strokeDasharray="5 4"
                opacity={0.5}
              />
            )}
            <rect
              className={`bbox ${editThis ? "editable" : ""}`}
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              stroke={color}
              strokeDasharray={override ? undefined : "0"}
              onPointerDown={
                editThis ? (e) => beginDrag({ kind: "move", layerId: layer.layer_id, startX: x1, startY: y1, bbox: [x1, y1, x2, y2] }, e) : undefined
              }
            />
            <text className="bbox-label" x={x1 + 3} y={Math.max(11, y1 - 4)} fill={color} fontSize={12 / Math.max(scale, 0.35)}>
              L{layer.layer_id} z{layer.z_index}
              {override ? " (edited)" : ""}
            </text>
            {centroid && (
              <circle
                className={`centroid ${editThis ? "editable" : ""}`}
                cx={centroid[0]}
                cy={centroid[1]}
                r={handleSize / 1.6}
                fill={color}
                stroke="#000"
                strokeWidth={1}
                onPointerDown={editThis ? (e) => beginDrag({ kind: "centroid", layerId: layer.layer_id }, e) : undefined}
              />
            )}
            {editThis &&
              corners.map(([cx, cy], corner) => (
                <rect
                  key={corner}
                  className="handle"
                  x={cx - handleSize / 2}
                  y={cy - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#fff"
                  stroke={color}
                  onPointerDown={(e) => beginDrag({ kind: "resize", layerId: layer.layer_id, corner, bbox: [x1, y1, x2, y2] }, e)}
                />
              ))}
          </g>
        );
      })}
    </svg>
  );
}
