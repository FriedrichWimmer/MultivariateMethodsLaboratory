import type { ReactNode } from 'react';

export interface LineLegendItem {
  label: ReactNode;
  color: string;
  dashed?: boolean;
  kind?: 'line' | 'dot' | 'square';
}

/** Legend for non-class series drawn in a ScatterSVG (direction lines, reference marks). */
export function LineLegend({ items }: { items: LineLegendItem[] }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <span className="item" key={i}>
          <svg width="22" height="12" viewBox="0 0 22 12" aria-hidden>
            {it.kind === 'dot' ? (
              <circle cx={11} cy={6} r={4.5} fill={it.color} />
            ) : it.kind === 'square' ? (
              <rect x={6} y={1} width={10} height={10} fill={it.color} />
            ) : (
              <line x1={1} x2={21} y1={6} y2={6} stroke={it.color} strokeWidth={2.2} strokeDasharray={it.dashed ? '5 3' : undefined} />
            )}
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  );
}
