/**
 * Pure helpers local to the SVD laboratory (no React). Everything here is a
 * small function that the lib does not provide: matrix-editor state, angle
 * bookkeeping for the 2×2 geometry, and the regression-side quantities
 * (variance factors, ridge filter factors) used in the "statistician" lesson.
 */
import { diagRect, matmul, transpose, type Matrix, type SVDResult } from '../../../lib/linalg';

/** Editable matrix state: strings, so that partially typed numbers ("-", "1.") survive re-renders. */
export type Cells = string[][];

export interface Preset {
  id: string;
  label: string;
  A: Matrix;
  hint: string;
}

export const presets: Preset[] = [
  { id: 'shear', label: 'Shear', A: [[1, 1], [0, 1]], hint: 'Non-normal matrix: the two rotation angles differ' },
  { id: 'rotation', label: 'Rotation-like', A: [[1, -1], [1, 1]], hint: 'A scaled rotation: equal singular values, the circle stays a circle' },
  { id: 'rank1', label: 'Rank-1', A: [[1, 2], [2, 4]], hint: 'Proportional columns: the second singular value vanishes' },
  { id: 'symmetric', label: 'Symmetric', A: [[3, 1], [1, 2]], hint: 'Symmetric positive definite: the SVD coincides with the eigendecomposition' },
  { id: 'worked', label: 'Worked example', A: [[3, 0], [4, 5]], hint: 'Singular values verifiable by hand (lesson 5)' },
  { id: 'tall', label: '3×2', A: [[1, 0], [0, 1], [1, 1]], hint: 'Three rows, two columns: rectangular Sigma' },
  { id: 'wide', label: '4×3', A: [[1, 2, 0], [0, 1, 1], [1, 0, 1], [2, 1, 1]], hint: 'Four rows, three columns' },
];

export function cellsFromMatrix(A: Matrix): Cells {
  return A.map((r) => r.map((x) => String(x)));
}

export function matrixFromCells(c: Cells): Matrix {
  return c.map((r) =>
    r.map((s) => {
      const v = Number(s);
      return s.trim() !== '' && Number.isFinite(v) ? v : 0;
    }),
  );
}

/** Resize, keeping existing entries; new entries follow the identity pattern so the matrix does not silently lose rank. */
export function resizeCells(c: Cells, n: number, p: number): Cells {
  return Array.from({ length: n }, (_, i) => Array.from({ length: p }, (_, j) => c[i]?.[j] ?? (i === j ? '1' : '0')));
}

export function presetIdOf(c: Cells): string {
  const A = matrixFromCells(c);
  const hit = presets.find((pr) => pr.A.length === A.length && pr.A[0].length === A[0].length && pr.A.every((row, i) => row.every((x, j) => x === A[i][j])));
  return hit ? hit.id : 'custom';
}

const SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉';
/** Unicode subscript for matrix labels (not for prose — prose uses KaTeX). */
export function subscript(j: number): string {
  return String(j)
    .split('')
    .map((ch) => SUBSCRIPTS[Number(ch)] ?? ch)
    .join('');
}

export const toDeg = (rad: number): number => (rad * 180) / Math.PI;
export const toRad = (deg: number): number => (deg * Math.PI) / 180;

export function fmtDeg(rad: number, digits = 1): string {
  return `${toDeg(rad).toFixed(digits).replace('-', '−')}°`;
}

export function fmtPct(x: number, digits = 1): string {
  return `${(100 * x).toFixed(digits).replace('-', '−')}%`;
}

/** Number formatted for use inside a TeX string (ASCII minus, \infty, scientific notation as a\times 10^{e}). */
export function texNum(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
  const a = Math.abs(x);
  if (a < 0.5 * Math.pow(10, -digits)) return (0).toFixed(digits);
  if (a >= 1e6 || a < 1e-3) {
    const [m, e] = x.toExponential(Math.max(digits - 1, 1)).split('e');
    return `${m} \\times 10^{${Number(e)}}`;
  }
  return x.toFixed(digits);
}

/** Smallest distance between two directions modulo π (a sign flip of a singular vector adds π to its angle). */
export function angleDiffModPi(a: number, b: number): number {
  const d = Math.abs(a - b) % Math.PI;
  return Math.min(d, Math.PI - d);
}

/** Orthogonal 2×2 matrix whose first column is (cos θ, sin θ); `reflect` negates the second column, giving det = −1. */
export function orthogonal2(theta: number, reflect: boolean): Matrix {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const d = reflect ? -1 : 1;
  return [
    [c, -d * s],
    [s, d * c],
  ];
}

/** U Σ Vᵀ for U (n×n), singular values s and V (p×p). */
export function composeSVD(U: Matrix, s: number[], V: Matrix): Matrix {
  return matmul(matmul(U, diagRect(s, U.length, V.length)), transpose(V));
}

/** An asymmetric "house" inside the unit disc, so that reflections are visible at a glance. */
export const housePolygon: [number, number][] = [
  [-0.5, -0.5],
  [0.5, -0.5],
  [0.5, 0.2],
  [0.3, 0.36],
  [0.3, 0.65],
  [0.18, 0.65],
  [0.18, 0.456],
  [0, 0.6],
  [-0.5, 0.2],
];

/** The rank-one term σ_j u_j v_jᵀ (0-based j) restricted to the first `rows` rows. */
export function rankOneBlock(res: { U: Matrix; s: number[]; V: Matrix }, j: number, rows: number): Matrix {
  const p = res.V.length;
  const out: Matrix = [];
  for (let i = 0; i < Math.min(rows, res.U.length); i++) {
    const c = res.U[i][j] * res.s[j];
    const row = new Array<number>(p);
    for (let t = 0; t < p; t++) row[t] = c * res.V[t][j];
    out.push(row);
  }
  return out;
}

/**
 * Diagonal of (XᵀX)⁺ = V Σ⁻² Vᵀ (summing over σ_t > tol) — the per-coefficient
 * variance factors of least squares — and the share of each due to the
 * smallest retained singular direction.
 */
export function coefficientVarianceFactors(res: SVDResult): { diag: number[]; lastShare: number[]; retained: number } {
  const p = res.V.length;
  const retained = res.s.filter((x) => x > res.tol).length;
  const diag = new Array<number>(p).fill(0);
  const last = new Array<number>(p).fill(0);
  for (let j = 0; j < p; j++)
    for (let t = 0; t < retained; t++) {
      const c = (res.V[j][t] * res.V[j][t]) / (res.s[t] * res.s[t]);
      diag[j] += c;
      if (t === retained - 1) last[j] = c;
    }
  return { diag, lastShare: diag.map((d, j) => (d > 0 ? last[j] / d : 0)), retained };
}

/** Ridge filter factors f_j = σ_j² / (σ_j² + λ). */
export function ridgeFilterFactors(s: number[], lambda: number): number[] {
  return s.map((x) => (x * x) / (x * x + lambda));
}

/** Σ_{j ≤ upto, σ_j > tol} σ_j⁻² — trace of the OLS (upto = r) or PCR (upto = k) coefficient covariance per unit error variance. */
export function sumInverseSquares(s: number[], upto: number, tol: number): number {
  let acc = 0;
  for (let j = 0; j < Math.min(upto, s.length); j++) if (s[j] > tol) acc += 1 / (s[j] * s[j]);
  return acc;
}
