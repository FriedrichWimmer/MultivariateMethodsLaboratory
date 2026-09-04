/**
 * Pure helpers shared by the "Four-way comparison" and "One dataset, four questions" tabs.
 * Everything here is deterministic and side-effect free so that it can live inside useMemo.
 */
import { fmt, type Matrix } from '../../../lib/linalg';
import type { Scaling } from '../../../lib/pca';

export type Method = 'SVD' | 'PCA' | 'MDS' | 'LDA';

export const methodName: Record<Method, string> = {
  SVD: 'Singular value decomposition',
  PCA: 'Principal component analysis',
  MDS: 'Classical multidimensional scaling',
  LDA: "Fisher's linear discriminant analysis",
};

/** First two columns of a matrix as n×2 points; a missing second column is zero-filled (1-D strip). */
export function firstTwoColumns(M: Matrix): number[][] {
  return M.map((r) => [r[0] ?? 0, r[1] ?? 0]);
}

const TEX_ESCAPES: Record<string, string> = {
  '\\': '\\textbackslash{}',
  '{': '\\{',
  '}': '\\}',
  $: '\\$',
  '&': '\\&',
  '#': '\\#',
  '%': '\\%',
  _: '\\_',
  '^': '\\^{}',
  '~': '\\~{}',
};

/** Escape a plain string for use inside \text{…}. */
export function texText(s: string): string {
  return s.replace(/[\\{}$&#%_^~]/g, (c) => TEX_ESCAPES[c] ?? c);
}

const SUBSCRIPT_DIGITS = '₀₁₂₃₄₅₆₇₈₉';

/** Variable name → TeX: "x1", "x_1" or "x₁" become x_{1}; a single letter stays math; anything else is \text{…}. */
export function texName(name: string): string {
  const plain = name.replace(/[₀-₉]/g, (c) => String(SUBSCRIPT_DIGITS.indexOf(c)));
  const m = /^([A-Za-z])_?(\d+)$/.exec(plain);
  if (m) return `${m[1]}_{${m[2]}}`;
  if (/^[A-Za-z]$/.test(plain)) return plain;
  return `\\text{${texText(plain)}}`;
}

/** Number formatted for TeX math mode (ASCII minus, \times 10^{e} for extreme magnitudes). */
export function texNum(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
  const v = Math.abs(x) < 0.5 * Math.pow(10, -digits) ? 0 : x;
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) {
    const [mant, exp] = v.toExponential(Math.max(1, digits - 1)).split('e');
    return `${mant}\\times 10^{${Number(exp)}}`;
  }
  return v.toFixed(digits);
}

/** A column vector with the variable names alongside its entries. */
export function texNamedVector(v: number[], names: string[], digits = 3): string {
  const vals = v.map((x) => texNum(x, digits)).join(' \\\\ ');
  const nm = v.map((_, j) => texName(names[j] ?? `x${j + 1}`)).join(' \\\\ ');
  return `\\begin{bmatrix} ${vals} \\end{bmatrix}\\;\\begin{array}{l} ${nm} \\end{array}`;
}

export function pct(x: number, digits = 1): string {
  if (!Number.isFinite(x)) return '—';
  return `${(100 * x).toFixed(digits)}%`;
}

/** Comma-separated list of numbers, truncated after `max` entries. */
export function fmtList(xs: number[], digits = 3, max = 8): string {
  const shown = xs.slice(0, max).map((x) => fmt(x, digits));
  return xs.length > max ? `${shown.join(', ')}, … (${xs.length} values)` : shown.join(', ');
}

export function sumSquares(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x * x;
  return s;
}

/** Σ_{j≤k} σ_j² / Σ_j σ_j² — the share of squared Frobenius norm captured by the first k terms. */
export function energyFraction(s: number[], k: number): number {
  const total = sumSquares(s);
  return total > 0 ? sumSquares(s.slice(0, Math.max(0, k))) / total : 0;
}

/** Smallest number of leading components whose cumulative share reaches `thr` (length of the list if never). */
export function kForThreshold(cumulative: number[], thr: number): number {
  const idx = cumulative.findIndex((c) => c >= thr - 1e-12);
  return idx < 0 ? cumulative.length : idx + 1;
}

/** Sample (Pearson) correlation between columns a and b of M. */
export function columnCorrelation(M: Matrix, a: number, b: number): number {
  const n = M.length;
  if (n < 2) return NaN;
  let ma = 0;
  let mb = 0;
  for (const r of M) {
    ma += r[a];
    mb += r[b];
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (const r of M) {
    const da = r[a] - ma;
    const db = r[b] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN;
}

export function scalingWord(s: Scaling): string {
  return s === 'none' ? 'raw (uncentred)' : s === 'center' ? 'centred' : 'standardised';
}

/** Default class names when the dataset carries labels but no names. */
export function classNamesFor(K: number, given?: string[]): string[] | undefined {
  if (K <= 0) return undefined;
  if (given && given.length >= K) return given;
  return Array.from({ length: K }, (_, k) => given?.[k] ?? `class ${k + 1}`);
}
