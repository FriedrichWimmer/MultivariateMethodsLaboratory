/**
 * Pure helpers for the LDA laboratory. Every function here is a plain function of
 * its arguments so that the section components can wrap calls in useMemo.
 */
import type { Data, Shape } from 'plotly.js';
import { colMeans, column, covariance, dot, inverse, norm, quadForm, symmetricEigen, symmetricPower, zeros, type EigenResult, type Matrix, type Vector } from '../../../lib/linalg';
import { fisherCriterion, lda, type LDAResult } from '../../../lib/lda';
import type { Dataset } from '../../../lib/datasets';
import { classColor } from '../../../lib/theme';

/** A labelled data matrix together with the names needed for display. */
export interface LabelledData {
  X: Matrix;
  y: number[];
  classNames: string[];
  variableNames: string[];
  name: string;
  /** true when the data is a lesson-local stand-in rather than the global dataset */
  local: boolean;
}

export function classLetter(k: number): string {
  return String.fromCharCode(65 + k);
}

export function defaultClassNames(K: number): string[] {
  return Array.from({ length: K }, (_, k) => `Class ${classLetter(k)}`);
}

export function fromDataset(ds: Dataset, local: boolean): LabelledData | null {
  if (!ds.y) return null;
  const K = new Set(ds.y).size;
  return { X: ds.X, y: ds.y, classNames: ds.classNames ?? defaultClassNames(K), variableNames: ds.variableNames, name: ds.name, local };
}

// ---------------------------------------------------------------------------
// Angles and formatting
// ---------------------------------------------------------------------------

/** Normalise an angle to the axial range (−π/2, π/2]: w and −w describe the same direction. */
export function axialAngle(a: number): number {
  let t = a;
  while (t > Math.PI / 2) t -= Math.PI;
  while (t <= -Math.PI / 2) t += Math.PI;
  return t;
}

export const deg = (r: number): number => (r * 180) / Math.PI;

export const unitFromAngle = (theta: number): Vector => [Math.cos(theta), Math.sin(theta)];

export function fmtDeg(r: number, digits = 1): string {
  return `${deg(r).toFixed(digits).replace('-', '−')}°`;
}

export function pct(x: number, digits = 1): string {
  return Number.isFinite(x) ? `${(100 * x).toFixed(digits)}%` : '—';
}

/** Axial angle (degrees, in [0°, 90°]) between two directions, ignoring sign. */
export function axialAngleBetween(u: Vector, v: Vector): number {
  const d = norm(u) * norm(v);
  if (d === 0) return NaN;
  const c = Math.min(1, Math.abs(dot(u, v)) / d);
  return deg(Math.acos(c));
}

// ---------------------------------------------------------------------------
// One-dimensional projections
// ---------------------------------------------------------------------------

export function mean(v: number[]): number {
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
}

/** Sample standard deviation (denominator n − 1). */
export function sd(v: number[]): number {
  const n = v.length;
  if (n < 2) return 0;
  const m = mean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / (n - 1));
}

export interface ClassProjection {
  /** the class label value */
  label: number;
  name: string;
  n: number;
  mean: number;
  sd: number;
  values: number[];
}

/** Per-class summaries of a one-dimensional projection z. `classes` are the distinct labels in LDA order. */
export function classProjections(z: number[], y: number[], classes: number[], classNames: string[]): ClassProjection[] {
  return classes.map((c) => {
    const values: number[] = [];
    for (let i = 0; i < z.length; i++) if (y[i] === c) values.push(z[i]);
    return { label: c, name: classNames[c] ?? `class ${c}`, n: values.length, mean: mean(values), sd: sd(values), values };
  });
}

/**
 * The simplest one-dimensional classification rule for two classes: a threshold halfway
 * between the two projected class means (equivalently, assign to the nearer projected mean).
 */
export function midpointThreshold(z: number[], y: number[], classes: number[]): { threshold: number; accuracy: number; correct: number; predictions: number[] } {
  const c0 = classes[0];
  const c1 = classes[1] ?? classes[0];
  const m0 = mean(z.filter((_, i) => y[i] === c0));
  const m1 = mean(z.filter((_, i) => y[i] === c1));
  const threshold = (m0 + m1) / 2;
  const predictions = z.map((v) => (Math.abs(v - m0) <= Math.abs(v - m1) ? c0 : c1));
  const correct = predictions.reduce((a, p, i) => a + (p === y[i] ? 1 : 0), 0);
  return { threshold, accuracy: z.length ? correct / z.length : NaN, correct, predictions };
}

/** Overlaid per-class Plotly histograms of a projection, with bins shared across classes. */
export function classHistograms(z: number[], y: number[], classes: number[], classNames: string[], bins = 28): Data[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of z) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !(hi > lo)) {
    lo = Number.isFinite(lo) ? lo - 1 : -1;
    hi = Number.isFinite(hi) ? hi + 1 : 1;
  }
  const size = (hi - lo) / bins;
  return classes.map((c) => ({
    x: z.filter((_, i) => y[i] === c),
    type: 'histogram',
    name: classNames[c] ?? `class ${c}`,
    opacity: 0.55,
    marker: { color: classColor(c) },
    xbins: { start: lo, end: hi + size, size },
    hovertemplate: `${classNames[c] ?? `class ${c}`}<br>z ≈ %{x}<br>count %{y}<extra></extra>`,
  }));
}

/** Dashed vertical lines at the projected class means, for Plotly `layout.shapes`. */
export function meanLineShapes(projs: ClassProjection[]): Partial<Shape>[] {
  return projs
    .filter((p) => Number.isFinite(p.mean))
    .map((p) => ({ type: 'line', xref: 'x', yref: 'paper', x0: p.mean, x1: p.mean, y0: 0, y1: 1, line: { color: classColor(p.label), width: 1.5, dash: 'dash' } })) as unknown as Partial<Shape>[];
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

/** Inverse of a symmetric positive (semi-)definite matrix; falls back to the pseudo-inverse. */
export function invertSymmetric(A: Matrix): Matrix {
  return inverse(A) ?? symmetricPower(A, -1).M;
}

/** Confusion counts C[a][b] = #(true class a, predicted class b), classes in the given order. */
export function confusionMatrix(y: number[], pred: number[], classes: number[]): Matrix {
  const K = classes.length;
  const C = zeros(K, K);
  for (let i = 0; i < y.length; i++) {
    const a = classes.indexOf(y[i]);
    const b = classes.indexOf(pred[i]);
    if (a >= 0 && b >= 0) C[a][b] += 1;
  }
  return C;
}

/** Numerical rank of a symmetric PSD matrix from its eigenvalues (relative tolerance). */
export function eigenRank(values: number[], rel = 1e-10): number {
  const lmax = Math.max(Math.abs(values[0] ?? 0), 1e-300);
  return values.filter((v) => v > rel * lmax).length;
}

/**
 * Default pair of variables for the 2-D lessons: the two coordinate axes with the largest
 * univariate Fisher ratio (S_B)_jj / (S_W)_jj, i.e. the largest between-class separation
 * relative to within-class spread when each variable is taken on its own.
 */
export function defaultPair(SB: Matrix, SW: Matrix): [number, number] {
  const p = SB.length;
  if (p < 2) return [0, 0];
  const ratio = Array.from({ length: p }, (_, j) => (SW[j][j] > 0 ? SB[j][j] / SW[j][j] : 0));
  const order = ratio.map((_, j) => j).sort((i, j) => ratio[j] - ratio[i] || i - j);
  return [order[0], order[1]];
}

// ---------------------------------------------------------------------------
// The two-dimensional working example shared by lessons 2 and 5
// ---------------------------------------------------------------------------

export interface TwoD {
  /** n × 2 matrix of the two chosen variables, centred at the grand mean */
  X2: Matrix;
  y: number[];
  classNames: string[];
  names: [string, string];
  pair: [number, number];
  /** LDA on the two variables */
  res: LDAResult;
  /** sample covariance of X2 and its eigendecomposition (PCA of the two variables) */
  cov: Matrix;
  covEigen: EigenResult;
  /** unit Fisher direction and first principal direction */
  w1: Vector;
  v1: Vector;
  /** their axial angles in (−π/2, π/2] */
  thetaLDA: number;
  thetaPCA: number;
  /** largest absolute coordinate, for scaling drawn arrows */
  halfRange: number;
  local: boolean;
  sourceName: string;
}

export function buildTwoD(src: LabelledData, pair: [number, number]): TwoD {
  const [a, b] = pair;
  const gm = colMeans(src.X);
  const X2 = src.X.map((r) => [r[a] - gm[a], r[b] - gm[b]]);
  const res = lda(X2, src.y);
  const cov = covariance(X2);
  const covEigen = symmetricEigen(cov);
  const v1 = column(covEigen.vectors, 0);
  const w1 = res.maxDims > 0 ? column(res.W, 0) : [1, 0];
  const thetaLDA = axialAngle(Math.atan2(w1[1], w1[0]));
  const thetaPCA = axialAngle(Math.atan2(v1[1], v1[0]));
  let halfRange = 0;
  for (const r of X2) halfRange = Math.max(halfRange, Math.abs(r[0]), Math.abs(r[1]));
  return {
    X2,
    y: src.y,
    classNames: src.classNames,
    names: [src.variableNames[a], src.variableNames[b]],
    pair,
    res,
    cov,
    covEigen,
    w1,
    v1,
    thetaLDA,
    thetaPCA,
    halfRange: halfRange || 1,
    local: src.local,
    sourceName: src.name,
  };
}

/** J(θ) and the variance wᵀSw along w(θ) = (cos θ, sin θ) for θ ∈ [−90°, 90°]. */
export function fisherCurve(SB: Matrix, SW: Matrix, cov: Matrix, stepDeg = 1): { thetaDeg: number[]; J: number[]; variance: number[] } {
  const thetaDeg: number[] = [];
  const J: number[] = [];
  const variance: number[] = [];
  for (let d = -90; d <= 90; d += stepDeg) {
    const w = unitFromAngle((d * Math.PI) / 180);
    thetaDeg.push(d);
    J.push(fisherCriterion(SB, SW, w).J);
    variance.push(quadForm(cov, w));
  }
  return { thetaDeg, J, variance };
}

/** Standard normal distribution function Φ(x) (Abramowitz–Stegun 7.1.26, |error| < 1.5·10⁻⁷). */
export function normalCdf(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-z * z);
  const c = 0.5 * (1 + erf);
  return x >= 0 ? c : 1 - c;
}
