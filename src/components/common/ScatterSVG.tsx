import { useMemo, useRef, useState, type ReactNode, type PointerEvent as RPointerEvent } from 'react';
import { classColor } from '../../lib/theme';
import { MarkerShape } from './Panels';

export interface Arrow {
  x: number;
  y: number;
  from?: [number, number];
  color?: string;
  label?: string;
  dashed?: boolean;
  width?: number;
  opacity?: number;
}
export interface Segment {
  from: [number, number];
  to: [number, number];
  color?: string;
  width?: number;
  dashed?: boolean;
  opacity?: number;
}
export interface ExtraPoint {
  x: number;
  y: number;
  color?: string;
  r?: number;
  label?: string;
  shape?: 'circle' | 'square' | 'diamond' | 'cross' | 'ring' | 'class';
  classIndex?: number;
  opacity?: number;
  labelPosition?: 'above' | 'below' | 'right';
}
export interface Ellipse {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  /** rotation angle of the rx axis in radians */
  angle: number;
  color?: string;
  fill?: boolean;
  dashed?: boolean;
  width?: number;
}
export interface InfiniteLine {
  angle: number;
  through?: [number, number];
  color?: string;
  dashed?: boolean;
  label?: string;
  width?: number;
  opacity?: number;
}
export interface DirectionHandle {
  angle: number;
  onChange?: (angle: number) => void;
  color?: string;
  label?: string;
  /** handle radius in data units (default: 40% of the smaller half-range) */
  radius?: number;
  through?: [number, number];
  showLine?: boolean;
  /** treat direction as an axis (w ≡ −w): keep angle in (−π/2, π/2] */
  axial?: boolean;
}

interface Props {
  points: number[][];
  labels?: number[];
  classNames?: string[];
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  equalAspect?: boolean;
  domain?: { x: [number, number]; y: [number, number] };
  /** extra fraction of range added on each side */
  padding?: number;
  vectors?: Arrow[];
  segments?: Segment[];
  extraPoints?: ExtraPoint[];
  ellipses?: Ellipse[];
  lines?: InfiniteLine[];
  direction?: DirectionHandle;
  highlight?: number[];
  selected?: number[];
  onPointClick?: (i: number) => void;
  pointRadius?: number;
  pointOpacity?: number;
  pointColor?: string;
  showGrid?: boolean;
  showZeroLines?: boolean;
  title?: string;
  caption?: ReactNode;
  hoverInfo?: (i: number) => string;
  /** render arbitrary SVG in data coordinates through the scale functions */
  render?: (sx: (x: number) => number, sy: (y: number) => number) => ReactNode;
  /** force the plotted domain to include these points */
  include?: number[][];
  className?: string;
}

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min;
  if (!(span > 0) || !Number.isFinite(span)) return [min];
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + 1e-9 * span; v += step) ticks.push(Math.abs(v) < 1e-12 ? 0 : v);
  return ticks;
}

function fmtTick(v: number): string {
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  const s = Number(v.toPrecision(4));
  return String(s);
}

/**
 * Interactive scatterplot in SVG. Designed for the geometric lessons: arrows,
 * projection segments, covariance ellipses, an infinite line and a draggable
 * direction handle can be layered on top of the observations.
 */
export function ScatterSVG(props: Props) {
  const {
    points,
    labels,
    classNames,
    width = 480,
    height = 400,
    xLabel,
    yLabel,
    equalAspect = true,
    padding = 0.1,
    vectors = [],
    segments = [],
    extraPoints = [],
    ellipses = [],
    lines = [],
    direction,
    highlight,
    selected,
    onPointClick,
    pointRadius = 4,
    pointOpacity = 0.85,
    pointColor,
    showGrid = true,
    showZeroLines = true,
    title,
    caption,
    hoverInfo,
    render,
    include,
    className,
  } = props;

  const margin = { l: 46, r: 14, t: 12, b: 38 };
  const innerW = width - margin.l - margin.r;
  const innerH = height - margin.t - margin.b;

  const dom = useMemo(() => {
    if (props.domain) return props.domain;
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    const consider = (x: number, y: number) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
    };
    for (const p of points) consider(p[0], p[1]);
    for (const p of include ?? []) consider(p[0], p[1]);
    for (const v of vectors) {
      const f = v.from ?? [0, 0];
      consider(f[0] + v.x, f[1] + v.y);
      consider(f[0], f[1]);
    }
    for (const e of extraPoints) consider(e.x, e.y);
    if (!Number.isFinite(xmin)) {
      xmin = -1;
      xmax = 1;
      ymin = -1;
      ymax = 1;
    }
    if (xmax - xmin < 1e-9) {
      xmin -= 1;
      xmax += 1;
    }
    if (ymax - ymin < 1e-9) {
      ymin -= 1;
      ymax += 1;
    }
    const px = (xmax - xmin) * padding;
    const py = (ymax - ymin) * padding;
    xmin -= px;
    xmax += px;
    ymin -= py;
    ymax += py;
    if (equalAspect) {
      const unitX = (xmax - xmin) / innerW;
      const unitY = (ymax - ymin) / innerH;
      const unit = Math.max(unitX, unitY);
      const cx = (xmin + xmax) / 2;
      const cy = (ymin + ymax) / 2;
      xmin = cx - (unit * innerW) / 2;
      xmax = cx + (unit * innerW) / 2;
      ymin = cy - (unit * innerH) / 2;
      ymax = cy + (unit * innerH) / 2;
    }
    return { x: [xmin, xmax] as [number, number], y: [ymin, ymax] as [number, number] };
  }, [points, include, vectors, extraPoints, props.domain, padding, equalAspect, innerW, innerH]);

  const sx = (x: number) => margin.l + ((x - dom.x[0]) / (dom.x[1] - dom.x[0])) * innerW;
  const sy = (y: number) => margin.t + ((dom.y[1] - y) / (dom.y[1] - dom.y[0])) * innerH;
  const invX = (px: number) => dom.x[0] + ((px - margin.l) / innerW) * (dom.x[1] - dom.x[0]);
  const invY = (py: number) => dom.y[1] - ((py - margin.t) / innerH) * (dom.y[1] - dom.y[0]);
  const unitsPerPx = (dom.x[1] - dom.x[0]) / innerW;

  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const toLocal = (e: RPointerEvent<SVGElement>) => {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return { px: (e.clientX - rect.left) * scaleX, py: (e.clientY - rect.top) * scaleY };
  };

  const dirThrough = direction?.through ?? [0, 0];
  const dirRadius = direction?.radius ?? Math.min(dom.x[1] - dom.x[0], dom.y[1] - dom.y[0]) * 0.4;

  const updateAngle = (e: RPointerEvent<SVGElement>) => {
    if (!direction?.onChange) return;
    const { px, py } = toLocal(e);
    const dx = invX(px) - dirThrough[0];
    const dy = invY(py) - dirThrough[1];
    let a = Math.atan2(dy, dx);
    if (direction.axial) {
      if (a > Math.PI / 2) a -= Math.PI;
      if (a <= -Math.PI / 2) a += Math.PI;
    }
    direction.onChange(a);
  };

  const onPointerMove = (e: RPointerEvent<SVGSVGElement>) => {
    if (dragging) {
      updateAngle(e);
      return;
    }
    if (points.length === 0 && extraPoints.length === 0) return;
    const { px, py } = toLocal(e);
    let best = -1;
    let bestD = 14 * 14;
    for (let i = 0; i < points.length; i++) {
      const dx = sx(points[i][0]) - px;
      const dy = sy(points[i][1]) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    setHover(best >= 0 ? best : null);
  };

  const xTicks = niceTicks(dom.x[0], dom.x[1], 6);
  const yTicks = niceTicks(dom.y[0], dom.y[1], 5);

  const clipId = useMemo(() => `clip${Math.random().toString(36).slice(2, 8)}`, []);

  const lineEndpoints = (angle: number, through: [number, number]) => {
    // intersect the line with the plotting rectangle: extend far, clip by clipPath
    const L = Math.hypot(dom.x[1] - dom.x[0], dom.y[1] - dom.y[0]) * 2;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return { x1: sx(through[0] - L * c), y1: sy(through[1] - L * s), x2: sx(through[0] + L * c), y2: sy(through[1] + L * s) };
  };

  return (
    <div className={className}>
      {title && <div className="plot-title">{title}</div>}
      <svg
        ref={svgRef}
        className="svgplot"
        viewBox={`0 0 ${width} ${height}`}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
        onPointerUp={() => setDragging(false)}
        onPointerDown={(e) => {
          if (direction?.onChange && (e.target as Element).getAttribute('data-role') !== 'point') {
            // clicking anywhere in the plot area re-orients the direction
            const { px, py } = toLocal(e);
            if (px > margin.l && px < width - margin.r && py > margin.t && py < height - margin.b) {
              setDragging(true);
              updateAngle(e);
            }
          }
        }}
        role="img"
        aria-label={title ?? 'scatterplot'}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={margin.l} y={margin.t} width={innerW} height={innerH} />
          </clipPath>
          <marker id={`${clipId}-arrow`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
          </marker>
        </defs>
        {/* grid */}
        {showGrid &&
          xTicks.map((t) => <line key={`gx${t}`} x1={sx(t)} x2={sx(t)} y1={margin.t} y2={margin.t + innerH} stroke={t === 0 && showZeroLines ? '#c3c2b7' : '#e1e0d9'} strokeWidth={1} />)}
        {showGrid &&
          yTicks.map((t) => <line key={`gy${t}`} y1={sy(t)} y2={sy(t)} x1={margin.l} x2={margin.l + innerW} stroke={t === 0 && showZeroLines ? '#c3c2b7' : '#e1e0d9'} strokeWidth={1} />)}
        {/* axes */}
        <line x1={margin.l} x2={margin.l + innerW} y1={margin.t + innerH} y2={margin.t + innerH} stroke="#c3c2b7" />
        <line x1={margin.l} x2={margin.l} y1={margin.t} y2={margin.t + innerH} stroke="#c3c2b7" />
        {xTicks.map((t) => (
          <text key={`tx${t}`} x={sx(t)} y={margin.t + innerH + 14} fontSize={10.5} fill="#898781" textAnchor="middle">
            {fmtTick(t)}
          </text>
        ))}
        {yTicks.map((t) => (
          <text key={`ty${t}`} x={margin.l - 6} y={sy(t) + 3.5} fontSize={10.5} fill="#898781" textAnchor="end">
            {fmtTick(t)}
          </text>
        ))}
        {xLabel && (
          <text x={margin.l + innerW / 2} y={height - 6} fontSize={12} fill="#52514e" textAnchor="middle">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={12} y={margin.t + innerH / 2} fontSize={12} fill="#52514e" textAnchor="middle" transform={`rotate(-90 12 ${margin.t + innerH / 2})`}>
            {yLabel}
          </text>
        )}
        <g clipPath={`url(#${clipId})`}>
          {/* ellipses */}
          {ellipses.map((el, i) => (
            <ellipse
              key={`el${i}`}
              cx={sx(el.cx)}
              cy={sy(el.cy)}
              rx={el.rx / unitsPerPx}
              ry={el.ry / unitsPerPx}
              transform={`rotate(${(-el.angle * 180) / Math.PI} ${sx(el.cx)} ${sy(el.cy)})`}
              fill={el.fill ? el.color ?? '#2a78d6' : 'none'}
              fillOpacity={el.fill ? 0.08 : 0}
              stroke={el.color ?? '#2a78d6'}
              strokeWidth={el.width ?? 1.5}
              strokeDasharray={el.dashed ? '5 4' : undefined}
            />
          ))}
          {/* infinite lines */}
          {lines.map((ln, i) => {
            const e = lineEndpoints(ln.angle, ln.through ?? [0, 0]);
            return <line key={`ln${i}`} {...e} stroke={ln.color ?? '#52514e'} strokeWidth={ln.width ?? 1.5} strokeDasharray={ln.dashed ? '6 4' : undefined} opacity={ln.opacity ?? 0.9} />;
          })}
          {direction && direction.showLine !== false && (
            <line {...lineEndpoints(direction.angle, dirThrough)} stroke={direction.color ?? '#0b0b0b'} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.7} />
          )}
          {/* segments (e.g. projections) */}
          {segments.map((s, i) => (
            <line
              key={`sg${i}`}
              x1={sx(s.from[0])}
              y1={sy(s.from[1])}
              x2={sx(s.to[0])}
              y2={sy(s.to[1])}
              stroke={s.color ?? '#898781'}
              strokeWidth={s.width ?? 1}
              strokeDasharray={s.dashed ? '3 3' : undefined}
              opacity={s.opacity ?? 0.5}
            />
          ))}
          {/* observations */}
          {points.map((p, i) => {
            const k = labels ? labels[i] : 0;
            const color = pointColor ?? (labels ? classColor(k) : '#2a78d6');
            const isHl = highlight?.includes(i);
            const isSel = selected?.includes(i);
            const dim = highlight && highlight.length > 0 && !isHl;
            const r = pointRadius * (isSel ? 1.8 : isHl ? 1.4 : 1);
            return (
              <g key={i} transform={`translate(${sx(p[0])} ${sy(p[1])})`} data-role="point" onClick={onPointClick ? () => onPointClick(i) : undefined} style={onPointClick ? { cursor: 'pointer' } : undefined}>
                {labels ? (
                  <MarkerShape k={k} r={r} fill={color} opacity={dim ? 0.18 : pointOpacity} stroke={isSel ? '#0b0b0b' : '#fcfcfb'} strokeWidth={isSel ? 2 : 1} />
                ) : (
                  <circle r={r} fill={color} opacity={dim ? 0.18 : pointOpacity} stroke={isSel ? '#0b0b0b' : '#fcfcfb'} strokeWidth={isSel ? 2 : 1} />
                )}
              </g>
            );
          })}
          {/* extra points */}
          {extraPoints.map((e, i) => {
            const X = sx(e.x);
            const Y = sy(e.y);
            const r = e.r ?? 5;
            const color = e.color ?? '#0b0b0b';
            let mark: ReactNode;
            switch (e.shape) {
              case 'square':
                mark = <rect x={-r} y={-r} width={2 * r} height={2 * r} fill={color} stroke="#fcfcfb" strokeWidth={1.2} />;
                break;
              case 'diamond':
                mark = <polygon points={`0,${-r * 1.3} ${r * 1.3},0 0,${r * 1.3} ${-r * 1.3},0`} fill={color} stroke="#fcfcfb" strokeWidth={1.2} />;
                break;
              case 'cross':
                mark = (
                  <g stroke={color} strokeWidth={2.2}>
                    <line x1={-r} x2={r} y1={-r} y2={r} />
                    <line x1={-r} x2={r} y1={r} y2={-r} />
                  </g>
                );
                break;
              case 'ring':
                mark = <circle r={r} fill="none" stroke={color} strokeWidth={2} />;
                break;
              case 'class':
                mark = <MarkerShape k={e.classIndex ?? 0} r={r} fill={color} stroke="#0b0b0b" strokeWidth={1.5} />;
                break;
              default:
                mark = <circle r={r} fill={color} stroke="#fcfcfb" strokeWidth={1.2} />;
            }
            const lp = e.labelPosition ?? 'above';
            return (
              <g key={`ep${i}`} transform={`translate(${X} ${Y})`} opacity={e.opacity ?? 1}>
                {mark}
                {e.label && (
                  <text x={lp === 'right' ? r + 4 : 0} y={lp === 'above' ? -r - 4 : lp === 'below' ? r + 12 : 4} fontSize={11} fill="#0b0b0b" textAnchor={lp === 'right' ? 'start' : 'middle'} fontWeight={600} stroke="#fcfcfb" strokeWidth={3} paintOrder="stroke">
                    {e.label}
                  </text>
                )}
              </g>
            );
          })}
          {/* vectors */}
          {vectors.map((v, i) => {
            const f = v.from ?? [0, 0];
            const x1 = sx(f[0]);
            const y1 = sy(f[1]);
            const x2 = sx(f[0] + v.x);
            const y2 = sy(f[1] + v.y);
            const color = v.color ?? '#0b0b0b';
            return (
              <g key={`v${i}`} opacity={v.opacity ?? 1}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={v.width ?? 2.2} strokeDasharray={v.dashed ? '6 4' : undefined} markerEnd={`url(#${clipId}-arrow)`} strokeLinecap="round" />
                {v.label && (
                  <text x={x2 + (x2 >= x1 ? 6 : -6)} y={y2 + (y2 >= y1 ? 12 : -6)} fontSize={11.5} fill="#0b0b0b" fontWeight={600} textAnchor={x2 >= x1 ? 'start' : 'end'} stroke="#fcfcfb" strokeWidth={3} paintOrder="stroke">
                    {v.label}
                  </text>
                )}
              </g>
            );
          })}
          {render && render(sx, sy)}
          {/* direction handle */}
          {direction && (
            <g>
              <line x1={sx(dirThrough[0])} y1={sy(dirThrough[1])} x2={sx(dirThrough[0] + dirRadius * Math.cos(direction.angle))} y2={sy(dirThrough[1] + dirRadius * Math.sin(direction.angle))} stroke={direction.color ?? '#0b0b0b'} strokeWidth={2.4} markerEnd={`url(#${clipId}-arrow)`} strokeLinecap="round" />
              {direction.onChange && (
                <circle
                  cx={sx(dirThrough[0] + dirRadius * Math.cos(direction.angle))}
                  cy={sy(dirThrough[1] + dirRadius * Math.sin(direction.angle))}
                  r={9}
                  fill={direction.color ?? '#0b0b0b'}
                  fillOpacity={0.15}
                  stroke={direction.color ?? '#0b0b0b'}
                  strokeWidth={1.5}
                  style={{ cursor: 'grab' }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    (e.target as Element).setPointerCapture?.(e.pointerId);
                    setDragging(true);
                  }}
                  onPointerUp={() => setDragging(false)}
                />
              )}
              {direction.label && (
                <text
                  x={sx(dirThrough[0] + dirRadius * 1.12 * Math.cos(direction.angle))}
                  y={sy(dirThrough[1] + dirRadius * 1.12 * Math.sin(direction.angle)) + 4}
                  fontSize={12}
                  fontWeight={600}
                  fill={direction.color ?? '#0b0b0b'}
                  textAnchor="middle"
                  stroke="#fcfcfb"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {direction.label}
                </text>
              )}
            </g>
          )}
        </g>
        {/* hover tooltip */}
        {hover !== null && points[hover] && (
          <Tooltip x={sx(points[hover][0])} y={sy(points[hover][1])} width={width} text={hoverInfo ? hoverInfo(hover) : defaultHover(hover, points[hover], labels, classNames)} />
        )}
      </svg>
      {caption && <div className="plot-caption">{caption}</div>}
    </div>
  );
}

function defaultHover(i: number, p: number[], labels?: number[], classNames?: string[]): string {
  const cls = labels ? ` · ${classNames ? classNames[labels[i]] : `class ${labels[i]}`}` : '';
  return `#${i + 1}  (${p[0].toFixed(2)}, ${p[1].toFixed(2)})${cls}`;
}

function Tooltip({ x, y, width, text }: { x: number; y: number; width: number; text: string }) {
  const lines = text.split('\n');
  const w = Math.max(...lines.map((l) => l.length)) * 6.2 + 14;
  const h = lines.length * 14 + 8;
  const left = x + 12 + w > width ? x - 12 - w : x + 12;
  const top = y - h / 2;
  return (
    <g pointerEvents="none">
      <rect x={left} y={top} width={w} height={h} rx={4} fill="#0b0b0b" opacity={0.92} />
      {lines.map((l, i) => (
        <text key={i} x={left + 7} y={top + 15 + i * 14} fontSize={11} fill="#fff" fontFamily="ui-monospace, Menlo, monospace">
          {l}
        </text>
      ))}
    </g>
  );
}
