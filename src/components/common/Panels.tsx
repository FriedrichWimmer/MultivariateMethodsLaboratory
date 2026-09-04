import { useState, type ReactNode } from 'react';
import { classColor, classSymbol } from '../../lib/theme';

export function Section({ id, title, subtitle, children, right }: { id?: string; title: ReactNode; subtitle?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="section" id={id}>
      <div className="section-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <div className="subtitle">{subtitle}</div>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Card({ title, children, className, plane, style }: { title?: ReactNode; children: ReactNode; className?: string; plane?: boolean; style?: React.CSSProperties }) {
  return (
    <div className={`card ${plane ? 'plane' : ''} ${className ?? ''}`} style={style}>
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}

export type CalloutKind = 'info' | 'warning' | 'danger' | 'theorem' | 'definition' | 'good';

const calloutIcon: Record<CalloutKind, string> = { info: 'ℹ', warning: '⚠', danger: '✕', theorem: '§', definition: '≔', good: '✓' };

export function Callout({ kind = 'info', title, children }: { kind?: CalloutKind; title?: ReactNode; children: ReactNode }) {
  return (
    <div className={`callout ${kind}`} role={kind === 'warning' || kind === 'danger' ? 'alert' : undefined}>
      {title && (
        <div className="callout-title">
          <span aria-hidden>{calloutIcon[kind]}</span>
          {title}
        </div>
      )}
      <div>{children}</div>
    </div>
  );
}

export interface InterpretationItems {
  seeing: ReactNode;
  why: ReactNode;
  math: ReactNode;
  stats: ReactNode;
  careful: ReactNode;
}

/** The five-question interpretation panel that accompanies every visualisation. */
export function Interpretation({ items, defaultOpen = true, title = 'Interpretation' }: { items: InterpretationItems; defaultOpen?: boolean; title?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  const rows: [string, ReactNode][] = [
    ['What am I seeing?', items.seeing],
    ['Why does it look this way?', items.why],
    ['Mathematical reason', items.math],
    ['Statistical interpretation', items.stats],
    ['Be careful about', items.careful],
  ];
  return (
    <div className="interp">
      <div className="interp-head" onClick={() => setOpen((o) => !o)} role="button" aria-expanded={open}>
        <span>{title}</span>
        <span aria-hidden>{open ? '−' : '+'}</span>
      </div>
      {open && (
        <div className="interp-body">
          {rows.map(([q, a]) => (
            <Fragment2 key={q} q={q} a={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function Fragment2({ q, a }: { q: string; a: ReactNode }) {
  return (
    <>
      <div className="q">{q}</div>
      <div className="a">{a}</div>
    </>
  );
}

export interface DerivationStep {
  title: ReactNode;
  body: ReactNode;
  note?: ReactNode;
}

/** Step-by-step derivation revealed one step at a time. */
export function Derivation({ steps, title, initiallyRevealed = 1 }: { steps: DerivationStep[]; title?: ReactNode; initiallyRevealed?: number }) {
  const [shown, setShown] = useState(Math.min(initiallyRevealed, steps.length));
  return (
    <div className="derivation">
      {title && (
        <div className="step" style={{ gridTemplateColumns: '1fr', background: 'var(--plane)' }}>
          <div className="title">{title}</div>
        </div>
      )}
      {steps.slice(0, shown).map((s, i) => (
        <div className="step" key={i}>
          <div className="n">{i + 1}</div>
          <div>
            <div className="title">{s.title}</div>
            <div>{s.body}</div>
            {s.note && <div className="note">{s.note}</div>}
          </div>
        </div>
      ))}
      <div className="actions">
        <button className="btn small primary" type="button" disabled={shown >= steps.length} onClick={() => setShown((s) => Math.min(steps.length, s + 1))}>
          {shown >= steps.length ? 'All steps shown' : `Reveal step ${shown + 1} of ${steps.length}`}
        </button>
        <button className="btn small" type="button" onClick={() => setShown(steps.length)} disabled={shown >= steps.length}>
          Reveal all
        </button>
        <button className="btn small" type="button" onClick={() => setShown(Math.min(1, steps.length))} disabled={shown <= 1}>
          Reset
        </button>
      </div>
    </div>
  );
}

export function Accordion({ items, single = false, defaultOpen = [] }: { items: { title: ReactNode; body: ReactNode; id?: string }[]; single?: boolean; defaultOpen?: number[] }) {
  const [open, setOpen] = useState<Set<number>>(new Set(defaultOpen));
  const toggle = (i: number) =>
    setOpen((prev) => {
      const next = new Set(single ? [] : prev);
      if (prev.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  return (
    <div className="accordion">
      {items.map((it, i) => (
        <div className="accordion-item" key={i} id={it.id}>
          <button className="accordion-head" type="button" onClick={() => toggle(i)} aria-expanded={open.has(i)}>
            <span>{it.title}</span>
            <span aria-hidden className="muted">
              {open.has(i) ? '−' : '+'}
            </span>
          </button>
          {open.has(i) && <div className="accordion-body">{it.body}</div>}
        </div>
      ))}
    </div>
  );
}

export function StatTile({ label, value, note, title }: { label: ReactNode; value: ReactNode; note?: ReactNode; title?: string }) {
  return (
    <div className="stat" title={title}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function Badge({ method, children }: { method?: 'SVD' | 'PCA' | 'MDS' | 'LDA'; children?: ReactNode }) {
  return <span className={`badge ${method ? method.toLowerCase() : 'neutral'}`}>{children ?? method}</span>;
}

/** Class legend with colour + marker shape. */
export function ClassLegend({ classNames, extra }: { classNames: string[]; extra?: ReactNode }) {
  return (
    <div className="legend">
      {classNames.map((c, k) => (
        <span className="item" key={k}>
          <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden>
            <MarkerShape k={k} r={5} />
          </svg>
          {c}
        </span>
      ))}
      {extra}
    </div>
  );
}

/** SVG marker for class k centred at (0,0). */
export function MarkerShape({ k, r = 4, fill, opacity = 1, stroke = '#fcfcfb', strokeWidth = 1.2 }: { k: number; r?: number; fill?: string; opacity?: number; stroke?: string; strokeWidth?: number }) {
  const color = fill ?? classColor(k);
  const sym = classSymbol(k);
  const common = { fill: color, opacity, stroke, strokeWidth };
  switch (sym) {
    case 'square':
      return <rect x={-r} y={-r} width={2 * r} height={2 * r} {...common} />;
    case 'diamond':
      return <polygon points={`0,${-r * 1.3} ${r * 1.3},0 0,${r * 1.3} ${-r * 1.3},0`} {...common} />;
    case 'triangle-up':
      return <polygon points={`0,${-r * 1.3} ${r * 1.2},${r} ${-r * 1.2},${r}`} {...common} />;
    case 'cross':
      return <path d={`M${-r},${-r / 3} h${r * 2} v${r / 1.5} h${-r * 2} z M${-r / 3},${-r} h${r / 1.5} v${r * 2} h${-r / 1.5} z`} {...common} />;
    default:
      return <circle r={r} {...common} />;
  }
}
