import type { Matrix } from '../../lib/linalg';
import { fmt } from '../../lib/linalg';
import { diverging, sequential, rampColor } from '../../lib/theme';

interface Props {
  M: Matrix;
  title?: string;
  rowLabels?: string[];
  colLabels?: string[];
  digits?: number;
  /** colour cells by value: 'diverging' for signed entries, 'sequential' for non-negative magnitudes */
  heat?: 'none' | 'diverging' | 'sequential';
  /** highlighted columns / rows (indices) */
  highlightCols?: number[];
  highlightRows?: number[];
  /** dim columns / rows not in the highlighted set */
  dimOthers?: boolean;
  caption?: string;
  maxRows?: number;
  maxCols?: number;
  /** absolute value used to normalise heat colouring (defaults to max |entry|) */
  heatMax?: number;
  className?: string;
  compact?: boolean;
}

/** Numeric matrix with brackets, optional labels and value-driven heat colouring. */
export function MatrixView({ M, title, rowLabels, colLabels, digits = 2, heat = 'none', highlightCols, highlightRows, dimOthers, caption, maxRows = 12, maxCols = 12, heatMax, className, compact }: Props) {
  const n = M.length;
  const p = n ? M[0].length : 0;
  const rows = Math.min(n, maxRows);
  const cols = Math.min(p, maxCols);
  const truncR = n > rows;
  const truncC = p > cols;
  let vmax = heatMax ?? 0;
  if (heat !== 'none' && heatMax === undefined) for (const r of M) for (const x of r) if (Number.isFinite(x)) vmax = Math.max(vmax, Math.abs(x));
  const hasRL = !!rowLabels;
  const hasCL = !!colLabels;
  const gridCols = (hasRL ? 1 : 0) + cols + (truncC ? 1 : 0);
  const cellStyle = (x: number): React.CSSProperties => {
    if (heat === 'none' || vmax === 0 || !Number.isFinite(x)) return {};
    if (heat === 'diverging') {
      const t = 0.5 + (x / vmax) * 0.5;
      const bg = rampColor(diverging, t);
      const strong = Math.abs(x) / vmax > 0.6;
      return { background: bg, color: strong ? '#fff' : undefined };
    }
    const t = Math.abs(x) / vmax;
    return { background: rampColor(sequential, t), color: t > 0.55 ? '#fff' : undefined };
  };
  const isHL = (i: number, j: number) => (highlightCols?.includes(j) ?? false) || (highlightRows?.includes(i) ?? false);
  const isDim = (i: number, j: number) => !!dimOthers && ((highlightCols && !highlightCols.includes(j)) || (highlightRows && !highlightRows.includes(i)));
  return (
    <div className={`matrix-wrap ${className ?? ''}`}>
      {title && <div className="matrix-title">{title}</div>}
      <div className="matrix-scroll">
        <div className="matrix" style={{ gridTemplateColumns: `repeat(${gridCols}, auto)`, fontSize: compact ? 11.5 : undefined }}>
          <div className="bracket l" style={{ top: hasCL ? 22 : 0 }} />
          <div className="bracket r" style={{ top: hasCL ? 22 : 0 }} />
          {hasCL && (
            <>
              {hasRL && <div className="clabel" />}
              {Array.from({ length: cols }, (_, j) => (
                <div className="clabel" key={`c${j}`}>
                  {colLabels![j]}
                </div>
              ))}
              {truncC && <div className="clabel">…</div>}
            </>
          )}
          {Array.from({ length: rows }, (_, i) => (
            <MatrixRow key={i} i={i} />
          ))}
          {truncR && (
            <>
              {hasRL && <div className="rlabel">⋮</div>}
              {Array.from({ length: cols + (truncC ? 1 : 0) }, (_, j) => (
                <div className="cell muted" key={`t${j}`} style={{ textAlign: 'center' }}>
                  ⋮
                </div>
              ))}
            </>
          )}
        </div>
      </div>
      {(truncR || truncC) && (
        <div className="matrix-caption">
          showing {rows} × {cols} of {n} × {p}
        </div>
      )}
      {caption && <div className="matrix-caption">{caption}</div>}
    </div>
  );

  function MatrixRow({ i }: { i: number }) {
    return (
      <>
        {hasRL && <div className="rlabel">{rowLabels![i]}</div>}
        {Array.from({ length: cols }, (_, j) => (
          <div key={j} className={`cell ${isHL(i, j) ? 'hl' : ''} ${isDim(i, j) ? 'dim' : ''}`} style={cellStyle(M[i][j])} title={String(M[i][j])}>
            {fmt(M[i][j], digits)}
          </div>
        ))}
        {truncC && <div className="cell muted">…</div>}
      </>
    );
  }
}

/** Lay out matrices in an equation-like row: A = B · C · D */
export function MatrixEquation({ items }: { items: (React.ReactNode | string)[] }) {
  return (
    <div className="matrix-eq">
      {items.map((it, i) => (typeof it === 'string' ? <span className="op" key={i}>{it}</span> : <span key={i}>{it}</span>))}
    </div>
  );
}
