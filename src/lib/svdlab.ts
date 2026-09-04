/**
 * Helpers specific to the SVD laboratory: geometric decomposition of a 2×2
 * matrix into rotation/reflection · scaling · rotation/reflection, and
 * low-rank approximation bookkeeping.
 */
import { svdFull, truncatedReconstruction, frobenius, sub, determinant, type Matrix, matmul, transpose } from './linalg';

export interface SVD2D {
  U: Matrix;
  V: Matrix;
  s: number[];
  Sigma: Matrix;
  /** angle of first right singular vector v₁ */
  thetaV: number;
  /** angle of first left singular vector u₁ */
  thetaU: number;
  reflectionU: boolean;
  reflectionV: boolean;
  rank: number;
  det: number;
  conditionNumber: number;
}

export function svd2d(A: Matrix): SVD2D {
  const f = svdFull(A);
  const thetaV = Math.atan2(f.V[1][0], f.V[0][0]);
  const thetaU = Math.atan2(f.U[1][0], f.U[0][0]);
  return {
    U: f.U,
    V: f.V,
    s: f.s,
    Sigma: f.Sigma,
    thetaV,
    thetaU,
    reflectionU: determinant(f.U) < 0,
    reflectionV: determinant(f.V) < 0,
    rank: f.rank,
    det: determinant(A),
    conditionNumber: f.s[1] > 0 ? f.s[0] / f.s[1] : Infinity,
  };
}

/** Apply a 2×2 matrix to a list of 2-D points. */
export function transformPoints(M: Matrix, pts: [number, number][]): [number, number][] {
  return pts.map(([x, y]) => [M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y]);
}

/** Unit circle sample points. */
export function unitCircle(m = 64): [number, number][] {
  return Array.from({ length: m + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / m;
    return [Math.cos(t), Math.sin(t)];
  });
}

export interface LowRankSummary {
  k: number;
  Xk: Matrix;
  error: number; // ‖X − X_k‖_F
  relativeError: number;
  /** √(Σ_{j>k} σ_j²) — the Eckart–Young value, should equal `error` */
  eckartYoung: number;
  energy: number; // Σ_{j≤k} σ_j² / Σ σ_j²
}

export function lowRankSummaries(X: Matrix, res: { U: Matrix; s: number[]; V: Matrix }): LowRankSummary[] {
  const total = res.s.reduce((a, b) => a + b * b, 0);
  const normX = frobenius(X);
  const out: LowRankSummary[] = [];
  for (let k = 0; k <= res.s.length; k++) {
    const Xk = truncatedReconstruction(res, k);
    const err = frobenius(sub(X, Xk));
    const tail = res.s.slice(k).reduce((a, b) => a + b * b, 0);
    const head = res.s.slice(0, k).reduce((a, b) => a + b * b, 0);
    out.push({ k, Xk, error: err, relativeError: normX > 0 ? err / normX : 0, eckartYoung: Math.sqrt(Math.max(tail, 0)), energy: total > 0 ? head / total : 0 });
  }
  return out;
}

/** Verify U Σ Vᵀ ≈ X and orthonormality; returns residual norms for the QC panel. */
export function svdChecks(X: Matrix, res: { U: Matrix; s: number[]; V: Matrix }) {
  const recon = truncatedReconstruction(res, res.s.length);
  const reconError = frobenius(sub(X, recon));
  const UtU = matmul(transpose(res.U), res.U);
  const VtV = matmul(transpose(res.V), res.V);
  const r = res.s.length;
  let uErr = 0;
  let vErr = 0;
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++) {
      uErr = Math.max(uErr, Math.abs(UtU[i][j] - (i === j ? 1 : 0)));
      vErr = Math.max(vErr, Math.abs(VtV[i][j] - (i === j ? 1 : 0)));
    }
  let ordered = true;
  for (let i = 1; i < r; i++) if (res.s[i] > res.s[i - 1] + 1e-12) ordered = false;
  return { reconError, uOrthoError: uErr, vOrthoError: vErr, ordered };
}
