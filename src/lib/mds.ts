/**
 * Distance matrices and classical (Torgerson–Gower) multidimensional scaling.
 */
import { symmetricEigenAuto as symmetricEigen, zeros, svd, matmul, transpose, determinant, type Matrix, type EigenResult } from './linalg';

export type Metric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski3' | 'sqeuclidean';

export const metricLabels: Record<Metric, string> = {
  euclidean: 'Euclidean (L₂)',
  manhattan: 'Manhattan (L₁)',
  chebyshev: 'Chebyshev (L∞)',
  minkowski3: 'Minkowski (L₃)',
  sqeuclidean: 'Squared Euclidean',
};

export function pointDistance(a: number[], b: number[], metric: Metric): number {
  let s = 0;
  switch (metric) {
    case 'euclidean':
      for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
      return Math.sqrt(s);
    case 'sqeuclidean':
      for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
      return s;
    case 'manhattan':
      for (let j = 0; j < a.length; j++) s += Math.abs(a[j] - b[j]);
      return s;
    case 'chebyshev':
      for (let j = 0; j < a.length; j++) s = Math.max(s, Math.abs(a[j] - b[j]));
      return s;
    case 'minkowski3':
      for (let j = 0; j < a.length; j++) s += Math.abs(a[j] - b[j]) ** 3;
      return Math.cbrt(s);
  }
}

export function distanceMatrix(X: Matrix, metric: Metric = 'euclidean'): Matrix {
  const n = X.length;
  const D = zeros(n, n);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = pointDistance(X[i], X[j], metric);
      D[i][j] = d;
      D[j][i] = d;
    }
  return D;
}

export interface MDSResult {
  n: number;
  D: Matrix;
  /** squared distances D⁽²⁾ */
  D2: Matrix;
  /** double-centred matrix B = −½ J D⁽²⁾ J */
  B: Matrix;
  eigen: EigenResult;
  /** all eigenvalues of B, decreasing (may include negatives) */
  eigenvalues: number[];
  positive: number;
  negative: number;
  /** Σ|negative eigenvalues| / Σ|eigenvalues| — a measure of non-Euclideanity */
  negativeMass: number;
  /** dimension actually used (≤ k, limited by the number of positive eigenvalues) */
  k: number;
  /** coordinates X_k = V_k Λ_k^{1/2} (n × k) */
  coords: Matrix;
  /** distances in the configuration */
  Dhat: Matrix;
  stress1: number;
  /** proportion of positive-eigenvalue "variance" retained by the first k axes */
  explained: number[];
  cumulative: number[];
  strain: number;
}

/** J = I − (1/n) 1 1ᵀ applied on both sides: B = −½ J A J. */
export function doubleCenter(A: Matrix): Matrix {
  const n = A.length;
  const rowMeans = A.map((r) => r.reduce((a, b) => a + b, 0) / n);
  const colMeans = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) colMeans[j] += A[i][j] / n;
  const grand = rowMeans.reduce((a, b) => a + b, 0) / n;
  const B = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] = -0.5 * (A[i][j] - rowMeans[i] - colMeans[j] + grand);
  return B;
}

export function classicalMDS(D: Matrix, k = 2): MDSResult {
  const n = D.length;
  const D2 = D.map((r) => r.map((d) => d * d));
  const B = doubleCenter(D2);
  const eigen = symmetricEigen(B);
  const values = eigen.values;
  const tol = Math.max(Math.abs(values[0] ?? 0), 1e-300) * n * 1e-12;
  const positive = values.filter((v) => v > tol).length;
  const negative = values.filter((v) => v < -tol).length;
  const absSum = values.reduce((a, b) => a + Math.abs(b), 0);
  const negSum = values.filter((v) => v < 0).reduce((a, b) => a + Math.abs(b), 0);
  const kk = Math.max(0, Math.min(k, positive));
  const coords = zeros(n, kk);
  for (let j = 0; j < kk; j++) {
    const s = Math.sqrt(values[j]);
    for (let i = 0; i < n; i++) coords[i][j] = eigen.vectors[i][j] * s;
  }
  const Dhat = distanceMatrix(coords.length && kk > 0 ? coords : zeros(n, 1), 'euclidean');
  // Kruskal stress-1 relative to the input distances
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      num += (D[i][j] - Dhat[i][j]) ** 2;
      den += D[i][j] ** 2;
    }
  const stress1 = den > 0 ? Math.sqrt(num / den) : 0;
  const posSum = values.filter((v) => v > 0).reduce((a, b) => a + b, 0);
  const explained = values.map((v) => (v > 0 && posSum > 0 ? v / posSum : 0));
  const cumulative: number[] = [];
  explained.reduce((acc, e) => {
    cumulative.push(acc + e);
    return acc + e;
  }, 0);
  // strain: ‖B − X Xᵀ‖_F / ‖B‖_F
  let bn = 0;
  let sn = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let xx = 0;
      for (let t = 0; t < kk; t++) xx += coords[i][t] * coords[j][t];
      sn += (B[i][j] - xx) ** 2;
      bn += B[i][j] ** 2;
    }
  const strain = bn > 0 ? Math.sqrt(sn / bn) : 0;
  return {
    n,
    D,
    D2,
    B,
    eigen,
    eigenvalues: values,
    positive,
    negative,
    negativeMass: absSum > 0 ? negSum / absSum : 0,
    k: kk,
    coords,
    Dhat,
    stress1,
    explained,
    cumulative,
    strain,
  };
}

/**
 * Procrustes alignment of configuration A onto B (both n × k, same k):
 * finds orthogonal Q (rotation/reflection), translation t minimising ‖B − (A Q + 1tᵀ)‖_F.
 * Used to compare PCA scores and MDS coordinates, which agree up to such a transformation.
 */
export function procrustesAlign(A: Matrix, B: Matrix): { aligned: Matrix; residual: number; relative: number; Q: Matrix; reflection: boolean } {
  const n = A.length;
  const k = A[0]?.length ?? 0;
  const ma = new Array(k).fill(0);
  const mb = new Array(k).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      ma[j] += A[i][j] / n;
      mb[j] += B[i][j] / n;
    }
  const Ac = A.map((r) => r.map((x, j) => x - ma[j]));
  const Bc = B.map((r) => r.map((x, j) => x - mb[j]));
  // M = Acᵀ Bc, Q = U Vᵀ from SVD of M
  const M = zeros(k, k);
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) M[a][b] += Ac[i][a] * Bc[i][b];
  const dec = svd(M);
  const Q = matmul(dec.U, transpose(dec.V));
  const aligned = Ac.map((r) => {
    const out = new Array(k).fill(0);
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) out[b] += r[a] * Q[a][b];
    return out.map((x, j) => x + mb[j]);
  });
  let res = 0;
  let bn = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      res += (aligned[i][j] - B[i][j]) ** 2;
      bn += Bc[i][j] ** 2;
    }
  return { aligned, residual: Math.sqrt(res), relative: bn > 0 ? Math.sqrt(res / bn) : 0, Q, reflection: determinant(Q) < 0 };
}
