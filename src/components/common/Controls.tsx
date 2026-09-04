import type { ReactNode } from 'react';

interface SliderProps {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  disabled?: boolean;
  hint?: string;
}

export function Slider({ label, value, min, max, step = 1, onChange, format, disabled, hint }: SliderProps) {
  const shown = format ? format(value) : Number.isInteger(step) ? String(value) : value.toFixed(Math.max(0, -Math.floor(Math.log10(step))));
  return (
    <div className="control" title={hint}>
      <div className="control-label">
        <span>{label}</span>
        <span className="val">{shown}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

interface SelectProps<T extends string> {
  label?: ReactNode;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  hint?: string;
}

export function Select<T extends string>({ label, value, options, onChange, hint }: SelectProps<T>) {
  return (
    <div className="control" title={hint}>
      {label && <label>{label}</label>}
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SegmentedProps<T extends string | number> {
  label?: ReactNode;
  value: T;
  options: { value: T; label: ReactNode; hint?: string }[];
  onChange: (v: T) => void;
  compact?: boolean;
}

export function Segmented<T extends string | number>({ label, value, options, onChange, compact }: SegmentedProps<T>) {
  return (
    <div className="control">
      {label && <label>{label}</label>}
      <div className={`segmented ${compact ? 'compact' : ''}`} role="tablist">
        {options.map((o) => (
          <button key={String(o.value)} className={o.value === value ? 'active' : ''} onClick={() => onChange(o.value)} title={o.hint} type="button">
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="toggle" title={hint}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function NumberField({ label, value, onChange, min, max, step = 1, hint }: { label: ReactNode; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; hint?: string }) {
  return (
    <div className="control" title={hint}>
      <label>{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </div>
  );
}

export function Button({ children, onClick, primary, small, disabled, title }: { children: ReactNode; onClick?: () => void; primary?: boolean; small?: boolean; disabled?: boolean; title?: string }) {
  return (
    <button type="button" className={`btn ${primary ? 'primary' : ''} ${small ? 'small' : ''}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}
