/**
 * Pure numerical helpers for the MDS laboratory.
 *
 * Everything that touches the n × n problem re-uses the eigendecomposition the
 * store has already computed (analysis.mds.eigen); nothing here repeats an
 * O(n³) decomposition of the full data. The only decompositions performed are
 * on the small worked example (m ≤ 5) and on 2 × 2 Procrustes cross-products.
 */
import { identity, matmul, scale, transpose, sub, maxAbs, centerColumns, symmetricEigen, svd, zeros, diag, type Matrix, type EigenResult } from '../../../lib/linalg';
import { distanceMatrix, doubleCenter, type Metric } from '../../../lib/mds';
import type { Dataset } from '../../../lib/datasets';

/**
 * X_k = V_k Λ_k^{1/2} from an existing eigendecomposition of B.
 * Non-positive eigenvalues yield zero columns: they carry no Euclidean coordinate.
 * Always returns exactly k columns so callers can rely on the shape.
 */
export function coordsFromEigen(eigen: EigenResult, k: number): Matrix {
  const n = eigen.vectors.length;
  const X = zeros(n, k);
  for (let j = 0; j < k && j < eigen.values.length; j++) {
    const lam = eigen.values[j];
    if (!(lam > 0)) continue;
    const s = Math.sqrt(lam);
    for (let i = 0; i < n; i++) X[i][j] = eigen.vectors[i][j] * s;
  }
  return X;
}

/** Kruskal stress-1 of the k-dimensional classical configuration against D, without materialising D̂. */
export function stressForK(D: Matrix, eigen: EigenResult, k: number): number {
  const n = D.length;
  const X = coordsFromEigen(eigen, k);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) {
        const d = X[i][t] - X[j][t];
        s += d * d;
      }
      const dh = Math.sqrt(s);
      num += (D[i][j] - dh) ** 2;
      den += D[i][j] ** 2;
    }
  return den > 0 ? Math.sqrt(num / den) : 0;
}

/**
 * strain(k) = ‖B − X_k X_kᵀ‖_F / ‖B‖_F.
 * Because X_k X_kᵀ = Σ_{j ≤ k, λ_j > 0} λ_j v_j v_jᵀ exactly, the residual is
 * Σ over the remaining eigenpairs of λ_j², whatever their signs.
 */
export function strainForK(values: number[], k: number): number {
  let num = 0;
  let den = 0;
  values.forEach((l, j) => {
    den += l * l;
    if (!(j < k && l > 0)) num += l * l;
  });
  return den > 0 ? Math.sqrt(num / den) : 0;
}

export function classNamesOf(dataset: Dataset): string[] | undefined {
  if (!dataset.y) return undefined;
  if (dataset.classNames) return dataset.classNames;
  const K = new Set(dataset.y).size;
  return Array.from({ length: K }, (_, k) => `class ${k + 1}`);
}

/** Stable ordering of observation indices by class label (identity when unlabelled). */
export function orderByClass(y: number[] | undefined, n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  if (!y) return idx;
  return idx.sort((a, b) => y[a] - y[b] || a - b);
}

export function permuteMatrix(D: Matrix, order: number[]): Matrix {
  return order.map((i) => order.map((j) => D[i][j]));
}

/** Positions (in the ordered arrangement) at which the class changes. */
export function classBoundaries(y: number[] | undefined, order: number[]): number[] {
  if (!y) return [];
  const b: number[] = [];
  for (let t = 1; t < order.length; t++) if (y[order[t]] !== y[order[t - 1]]) b.push(t);
  return b;
}

export interface DistanceSummary {
  pairs: number;
  mean: number;
  /** mean of d_ij² over pairs (for centred Euclidean data this equals 2·tr(S)) */
  meanSq: number;
  max: number;
  minNonzero: number;
  argmax: [number, number];
}

export function distanceSummary(D: Matrix): DistanceSummary {
  const n = D.length;
  let sum = 0;
  let sumSq = 0;
  let max = 0;
  let minNonzero = Infinity;
  let pairs = 0;
  let argmax: [number, number] = [0, Math.min(1, Math.max(0, n - 1))];
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = D[i][j];
      pairs++;
      sum += d;
      sumSq += d * d;
      if (d > max) {
        max = d;
        argmax = [i, j];
      }
      if (d > 0 && d < minNonzero) minNonzero = d;
    }
  return { pairs, mean: pairs ? sum / pairs : 0, meanSq: pairs ? sumSq / pairs : 0, max, minNonzero: Number.isFinite(minNonzero) ? minNonzero : 0, argmax };
}

export function withinBetween(D: Matrix, y: number[]): { within: number; between: number; nWithin: number; nBetween: number } {
  const n = D.length;
  let w = 0;
  let b = 0;
  let nw = 0;
  let nb = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      if (y[i] === y[j]) {
        w += D[i][j];
        nw++;
      } else {
        b += D[i][j];
        nb++;
      }
    }
  return { within: nw ? w / nw : 0, between: nb ? b / nb : 0, nWithin: nw, nBetween: nb };
}

/** Deterministically subsampled list of pairs (i < j), at most `maxPairs` of them. */
export function samplePairs(n: number, maxPairs: number): [number, number][] {
  const P = (n * (n - 1)) / 2;
  const stride = Math.max(1, Math.ceil(P / maxPairs));
  const out: [number, number][] = [];
  let c = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++, c++) if (c % stride === 0) out.push([i, j]);
  return out;
}

/** The pair (i < j) with the largest absolute distortion |d_ij − d̂_ij|. */
export function mostDistortedPair(D: Matrix, Dhat: Matrix): [number, number] | null {
  const n = D.length;
  if (n < 2) return null;
  let best: [number, number] = [0, 1];
  let bestErr = -1;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const e = Math.abs(D[i][j] - Dhat[i][j]);
      if (e > bestErr) {
        bestErr = e;
        best = [i, j];
      }
    }
  return best;
}

export interface ShepardStats {
  /** share of pairs with d̂_ij < d_ij */
  compressed: number;
  /** Pearson correlation between d_ij and d̂_ij over all pairs */
  correlation: number;
  maxAbsErr: number;
  meanAbsErr: number;
  maxD: number;
  maxDhat: number;
}

export function shepardStats(D: Matrix, Dhat: Matrix): ShepardStats {
  const n = D.length;
  let P = 0;
  let comp = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  let maxAbsErr = 0;
  let sumAbs = 0;
  let maxD = 0;
  let maxDhat = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const x = D[i][j];
      const yv = Dhat[i][j];
      P++;
      if (yv < x - 1e-12) comp++;
      sx += x;
      sy += yv;
      sxx += x * x;
      syy += yv * yv;
      sxy += x * yv;
      const e = Math.abs(x - yv);
      sumAbs += e;
      if (e > maxAbsErr) maxAbsErr = e;
      if (x > maxD) maxD = x;
      if (yv > maxDhat) maxDhat = yv;
    }
  const cov = sxy - (sx * sy) / Math.max(P, 1);
  const vx = sxx - (sx * sx) / Math.max(P, 1);
  const vy = syy - (sy * sy) / Math.max(P, 1);
  const correlation = vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : 0;
  return { compressed: P ? comp / P : 0, correlation, maxAbsErr, meanAbsErr: P ? sumAbs / P : 0, maxD, maxDhat };
}

/**
 * Similarity Procrustes: rotation/reflection Q, isotropic scale s and translation
 * minimising ‖B − (s·A_c Q + 1 tᵀ)‖_F. Complements the library's rigid
 * `procrustesAlign` (which has no scale) so that a residual can be split into
 * "different size" and "different shape".
 */
export function similarityProcrustes(A: Matrix, B: Matrix): { aligned: Matrix; relative: number; scale: number; reflection: boolean } {
  const n = A.length;
  const k = A[0]?.length ?? 0;
  if (n === 0 || k === 0) return { aligned: [], relative: 0, scale: 1, reflection: false };
  const ma = new Array(k).fill(0);
  const mb = new Array(k).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      ma[j] += A[i][j] / n;
      mb[j] += B[i][j] / n;
    }
  const Ac = A.map((r) => r.map((x, j) => x - ma[j]));
  const Bc = B.map((r) => r.map((x, j) => x - mb[j]));
  const M = matmul(transpose(Ac), Bc);
  const dec = svd(M);
  const Q = matmul(dec.U, transpose(dec.V));
  let normA2 = 0;
  for (const r of Ac) for (const x of r) normA2 += x * x;
  const traceSigma = dec.s.reduce((a, b) => a + b, 0);
  const s = normA2 > 0 ? traceSigma / normA2 : 1;
  const rotated = matmul(Ac, Q);
  const aligned = rotated.map((r) => r.map((x, j) => s * x + mb[j]));
  let res = 0;
  let bn = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      res += (aligned[i][j] - B[i][j]) ** 2;
      bn += Bc[i][j] ** 2;
    }
  const detQ = k === 2 ? Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0] : 1;
  return { aligned, relative: bn > 0 ? Math.sqrt(res / bn) : 0, scale: s, reflection: detQ < 0 };
}

/** Number formatting for use INSIDE TeX strings (ASCII minus, scientific notation for tiny/huge values). */
export function texNum(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
  const a = Math.abs(x);
  if (a !== 0 && (a < Math.pow(10, -digits) || a >= 1e6)) {
    const [m, e] = x.toExponential(2).split('e');
    return `${m}\\times 10^{${Number(e)}}`;
  }
  const s = x.toFixed(digits);
  return s === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : s;
}

export function pct(x: number, digits = 1): string {
  return `${(100 * x).toFixed(digits)}%`;
}

/** Everything the worked example of the transformation chain displays, for m ≤ 5 points. */
export interface WorkedExample {
  m: number;
  labels: string[];
  metric: Metric;
  /** coordinates used (m × p) and their centred version (about the m-point mean) */
  X: Matrix;
  Xc: Matrix;
  D: Matrix;
  D2: Matrix;
  J: Matrix;
  /** B via the library's double centring */
  B: Matrix;
  /** B via the explicit product −½ J D⁽²⁾ J */
  Bexplicit: Matrix;
  /** max |B − Bexplicit| */
  centringCheck: number;
  eigen: EigenResult;
  Lambda: Matrix;
  positive: number;
  negative: number;
  X2: Matrix;
  Dhat: Matrix;
  Ddiff: Matrix;
  maxDistErr: number;
  /** Gram matrix X_c X_cᵀ of the centred points and max |B − X_c X_cᵀ| */
  gram: Matrix;
  gramDiff: number;
  traceB: number;
}

export function workedExample(X: Matrix, metric: Metric, labels: string[]): WorkedExample {
  const m = X.length;
  const D = distanceMatrix(X, metric);
  const D2 = D.map((r) => r.map((d) => d * d));
  const J = identity(m).map((r) => r.map((x) => x - 1 / m));
  const B = doubleCenter(D2);
  const Bexplicit = scale(matmul(matmul(J, D2), J), -0.5);
  const centringCheck = maxAbs(sub(B, Bexplicit));
  const eigen = symmetricEigen(B);
  const tol = Math.max(Math.abs(eigen.values[0] ?? 0), 1e-300) * m * 1e-12;
  const positive = eigen.values.filter((v) => v > tol).length;
  const negative = eigen.values.filter((v) => v < -tol).length;
  const Lambda = diag(eigen.values);
  const X2 = coordsFromEigen(eigen, 2);
  const Dhat = distanceMatrix(X2, 'euclidean');
  const Ddiff = sub(D, Dhat);
  const maxDistErr = maxAbs(Ddiff);
  const Xc = centerColumns(X).Xc;
  const gram = matmul(Xc, transpose(Xc));
  const gramDiff = maxAbs(sub(B, gram));
  const traceB = eigen.values.reduce((s, v) => s + v, 0);
  return { m, labels, metric, X, Xc, D, D2, J, B, Bexplicit, centringCheck, eigen, Lambda, positive, negative, X2, Dhat, Ddiff, maxDistErr, gram, gramDiff, traceB };
}
