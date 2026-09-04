/**
 * Small pure helpers shared by the PCA laboratory lessons. Everything here is
 * deterministic and free of React so it can be unit-tested if desired.
 */
import { symmetricEigen, type Matrix } from '../../../lib/linalg';
import type { Scaling } from '../../../lib/pca';

/** Percentage with fixed decimals; en dash for non-finite input. */
export function pct(x: number, digits = 1): string {
  return Number.isFinite(x) ? `${(100 * x).toFixed(digits)}%` : '–';
}

export const deg = (r: number): number => (r * 180) / Math.PI;
export const rad = (d: number): number => (d * Math.PI) / 180;

/** Reduce an angle to the axial range (−π/2, π/2]: a direction w and −w span the same line. */
export function axial(a: number): number {
  let t = a;
  while (t > Math.PI / 2) t -= Math.PI;
  while (t <= -Math.PI / 2) t += Math.PI;
  return t;
}

/** Number formatted for use inside TeX (ASCII minus; scientific notation for extreme magnitudes). */
export function texNum(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\mathrm{NaN}';
  const a = Math.abs(x);
  if (a === 0) return (0).toFixed(digits);
  if (a >= 1e6 || a < 0.5 * Math.pow(10, -digits)) {
    const [m, e] = x.toExponential(2).split('e');
    return `${m}\\times 10^{${Number(e)}}`;
  }
  return x.toFixed(digits);
}

/** Scientific notation in plain text, for round-off-sized gaps. */
export function sci(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : 'NaN';
  if (x === 0) return '0';
  return x.toExponential(digits).replace(/-/g, '−');
}

export const pcLabels = (r: number): string[] => Array.from({ length: r }, (_, j) => `PC${j + 1}`);

export interface Eigen2 {
  l1: number;
  l2: number;
  v1: [number, number];
  v2: [number, number];
  /** axial angles (−π/2, π/2] of v1 and v2 */
  theta1: number;
  theta2: number;
}

/** Eigendecomposition of a 2×2 symmetric matrix, with the pieces the geometry lesson needs. */
export function eigen2(S: Matrix): Eigen2 {
  const e = symmetricEigen(S);
  const v1: [number, number] = [e.vectors[0][0], e.vectors[1][0]];
  const v2: [number, number] = [e.vectors[0][1], e.vectors[1][1]];
  return {
    l1: Math.max(e.values[0], 0),
    l2: Math.max(e.values[1] ?? 0, 0),
    v1,
    v2,
    theta1: axial(Math.atan2(v1[1], v1[0])),
    theta2: axial(Math.atan2(v2[1], v2[0])),
  };
}

/** Broken-stick expectations b_j = (1/p) Σ_{i=j}^{p} 1/i for j = 1..p (returned 0-based). */
export function brokenStick(p: number): number[] {
  const out: number[] = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = j + 1; i <= p; i++) s += 1 / i;
    out[j] = s / p;
  }
  return out;
}

/**
 * Elbow heuristic: the scree point farthest from the chord joining the first and
 * last points, after rescaling both axes to [0, 1]. Returns a 1-based index.
 */
export function elbowIndex(values: number[]): number {
  const r = values.length;
  if (r <= 2) return 1;
  const span = values[0] - values[r - 1] || 1;
  let best = 0;
  let bestD = -1;
  for (let j = 0; j < r; j++) {
    const x = j / (r - 1);
    const y = (values[j] - values[r - 1]) / span;
    const d = Math.abs(x + y - 1) / Math.SQRT2;
    if (d > bestD) {
      bestD = d;
      best = j;
    }
  }
  return best + 1;
}

export function scalingName(s: Scaling): string {
  return s === 'none' ? 'raw (uncentred)' : s === 'center' ? 'mean-centred' : 'standardised';
}

/** TeX for the p × p matrix that the given preprocessing diagonalises. */
export function decomposedMatrixTex(s: Scaling): string {
  return s === 'none' ? '\\tfrac{1}{n-1}X^{T}X' : s === 'center' ? 'S=\\tfrac{1}{n-1}X_c^{T}X_c' : 'R=\\tfrac{1}{n-1}X_s^{T}X_s';
}

/** Euclidean norm of the difference of two vectors. */
export function residualNorm(x: number[], h: number[]): number {
  let s = 0;
  for (let j = 0; j < x.length; j++) s += (x[j] - h[j]) * (x[j] - h[j]);
  return Math.sqrt(s);
}
