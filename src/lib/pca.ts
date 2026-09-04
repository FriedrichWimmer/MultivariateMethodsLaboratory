/**
 * Principal component analysis, computed through the SVD of the centred
 * (optionally standardised) data matrix. Never forms and diagonalises S
 * explicitly for the actual computation — but S and its eigendecomposition
 * are also returned so the dashboard can show that both routes agree.
 */
import {
  svd,
  centerColumns,
  standardizeColumns,
  covariance,
  correlationFromCovariance,
  symmetricEigen,
  matmul,
  transpose,
  firstColumns,
  frobenius,
  sub,
  type Matrix,
  type Vector,
  type EigenResult,
  type SVDResult,
} from './linalg';

export type Scaling = 'none' | 'center' | 'standardize';

export interface PCAResult {
  n: number;
  p: number;
  scaling: Scaling;
  means: Vector;
  stds: Vector;
  /** the matrix actually decomposed (X, X_c or X_s) */
  Xc: Matrix;
  /** covariance (or correlation, when standardised) of the analysed matrix */
  S: Matrix;
  /** eigendecomposition of S — for comparison with the SVD route */
  eigen: EigenResult;
  svd: SVDResult;
  /** eigenvalues λ_j = σ_j²/(n−1), decreasing */
  eigenvalues: number[];
  singularValues: number[];
  /** loadings: columns are principal directions v_j (p × r) */
  V: Matrix;
  /** scores Z = X_c V = U Σ (n × r) */
  scores: Matrix;
  explained: number[];
  cumulative: number[];
  totalVariance: number;
  rank: number;
}

export function pca(X: Matrix, scaling: Scaling = 'center'): PCAResult {
  const n = X.length;
  const p = X[0].length;
  let Xc: Matrix;
  let means: Vector;
  let stds: Vector;
  if (scaling === 'standardize') {
    const r = standardizeColumns(X);
    Xc = r.Xs;
    means = r.means;
    stds = r.stds;
  } else if (scaling === 'center') {
    const r = centerColumns(X);
    Xc = r.Xc;
    means = r.means;
    stds = new Array(p).fill(1);
  } else {
    Xc = X.map((r) => r.slice());
    means = new Array(p).fill(0);
    stds = new Array(p).fill(1);
  }
  const S = scaling === 'standardize' ? correlationFromCovariance(covariance(X)) : scaling === 'center' ? covariance(X) : scaleGram(Xc, n);
  const eigen = symmetricEigen(S);
  const dec = svd(Xc);
  const denom = Math.max(n - 1, 1);
  const eigenvalues = dec.s.map((s) => (s * s) / denom);
  const total = eigenvalues.reduce((a, b) => a + b, 0);
  const explained = eigenvalues.map((l) => (total > 0 ? l / total : 0));
  const cumulative: number[] = [];
  explained.reduce((acc, e) => {
    cumulative.push(acc + e);
    return acc + e;
  }, 0);
  const scores = matmul(Xc, dec.V);
  return {
    n,
    p,
    scaling,
    means,
    stds,
    Xc,
    S,
    eigen,
    svd: dec,
    eigenvalues,
    singularValues: dec.s,
    V: dec.V,
    scores,
    explained,
    cumulative,
    totalVariance: total,
    rank: dec.rank,
  };
}

/** For uncentred analysis, the "second-moment" matrix XᵀX/(n−1). */
function scaleGram(X: Matrix, n: number): Matrix {
  const G = matmul(transpose(X), X);
  const d = Math.max(n - 1, 1);
  return G.map((r) => r.map((x) => x / d));
}

/** X̂_k = Z_k V_kᵀ (in the analysed scale) and mapped back to original units. */
export function pcaReconstruct(res: PCAResult, k: number): { analysed: Matrix; original: Matrix } {
  const kk = Math.max(0, Math.min(k, res.V[0].length));
  const Zk = firstColumns(res.scores, kk);
  const Vk = firstColumns(res.V, kk);
  const analysed = kk === 0 ? res.Xc.map((r) => r.map(() => 0)) : matmul(Zk, transpose(Vk));
  const original = analysed.map((r) => r.map((x, j) => x * res.stds[j] + res.means[j]));
  return { analysed, original };
}

/** Frobenius reconstruction error ‖X_c − X̂_k‖_F for k = 0..r. */
export function reconstructionErrors(res: PCAResult): number[] {
  const r = res.V[0].length;
  const out: number[] = [];
  for (let k = 0; k <= r; k++) out.push(frobenius(sub(res.Xc, pcaReconstruct(res, k).analysed)));
  return out;
}

/** Variance of the projection z = X_c w for a unit direction w: wᵀ S w. */
export function projectionVariance(S: Matrix, w: Vector): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) s += w[i] * S[i][j] * w[j];
  return s;
}

export function projectOnDirection(Xc: Matrix, w: Vector): number[] {
  return Xc.map((r) => r.reduce((acc, x, j) => acc + x * w[j], 0));
}

/** Sample variance of a vector. */
export function sampleVariance(z: number[]): number {
  const n = z.length;
  const m = z.reduce((a, b) => a + b, 0) / n;
  return z.reduce((a, b) => a + (b - m) * (b - m), 0) / Math.max(n - 1, 1);
}
