import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface Props {
  tex?: string;
  children?: string;
  display?: boolean;
  className?: string;
}

/** KaTeX-rendered mathematics. `<M tex="x^2" />` inline, `<MBlock tex="..."/>` display. */
export function M({ tex, children, display = false, className }: Props) {
  const src = tex ?? children ?? '';
  const html = useMemo(() => {
    try {
      return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: 'ignore', trust: true });
    } catch {
      return src;
    }
  }, [src, display]);
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

export function MBlock({ tex, children, className }: Props) {
  const src = tex ?? children ?? '';
  const html = useMemo(() => {
    try {
      return katex.renderToString(src, { displayMode: true, throwOnError: false, strict: 'ignore', trust: true });
    } catch {
      return src;
    }
  }, [src]);
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Render a numeric matrix as a KaTeX bmatrix (for small matrices in equations). */
export function texMatrix(A: number[][], digits = 2): string {
  const f = (x: number) => {
    if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
    const v = Math.abs(x) < 0.5 * Math.pow(10, -digits) ? 0 : x;
    return v.toFixed(digits);
  };
  return `\\begin{bmatrix} ${A.map((r) => r.map(f).join(' & ')).join(' \\\\ ')} \\end{bmatrix}`;
}

export function texVector(v: number[], digits = 2, column = true): string {
  const f = (x: number) => (Math.abs(x) < 0.5 * Math.pow(10, -digits) ? 0 : x).toFixed(digits);
  return column ? `\\begin{bmatrix} ${v.map(f).join(' \\\\ ')} \\end{bmatrix}` : `\\begin{bmatrix} ${v.map(f).join(' & ')} \\end{bmatrix}`;
}

export function texDiag(v: number[], digits = 2): string {
  const n = v.length;
  const rows: string[] = [];
  for (let i = 0; i < n; i++) rows.push(v.map((x, j) => (i === j ? x.toFixed(digits) : '0')).join(' & '));
  return `\\begin{bmatrix} ${rows.join(' \\\\ ')} \\end{bmatrix}`;
}
