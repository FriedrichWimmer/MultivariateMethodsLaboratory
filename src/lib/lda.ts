/**
 * Fisher's linear discriminant analysis.
 *
 *   S_W = Σ_k Σ_{i∈k} (x_i − m_k)(x_i − m_k)ᵀ      within-class scatter
 *   S_B = Σ_k n_k (m_k − m)(m_k − m)ᵀ              between-class scatter
 *   S_T = S_W + S_B                                 total scatter = (n−1) S
 *
 * Solve the generalised eigenproblem S_B w = λ S_W w through symmetric
 * whitening: with S_W = Q D Qᵀ, let W = Q D^{−1/2}; then C = Wᵀ S_B W is
 * symmetric, C u = λ u, and w = W u. Directions are returned with unit norm.
 * If S_W is singular, either a ridge γI is added (regularised LDA) or the
 * problem is solved on the range of S_W (pseudo-inverse) and flagged.
 */
import { symmetricEigen, zeros, column, dot, normalize, type Matrix, type Vector, EPS, identity, add, scale } from './linalg';

export interface LDAResult {
  n: number;
  p: number;
  K: number;
  classes: number[];
  classSizes: number[];
  classMeans: Matrix; // K × p
  grandMean: Vector;
  SW: Matrix;
  SB: Matrix;
  ST: Matrix;
  /** eigenvalues of the generalised problem (Fisher ratios), decreasing; length min(K−1, rank) */
  eigenvalues: number[];
  /** unit-norm discriminant directions as columns (p × m) */
  W: Matrix;
  /** projected scores Z = X_c W (n × m) */
  scores: Matrix;
  /** maximum number of meaningful discriminants = min(K−1, rank(S_W)) */
  maxDims: number;
  swRank: number;
  swSingular: boolean;
  swCondition: number;
  regularization: number;
  /** eigenvalues of S_W (for diagnostics) */
  swEigenvalues: number[];
  /** eigenvalues of S_B (rank ≤ K−1) */
  sbEigenvalues: number[];
  /** proportion of the discriminant "trace" explained by each direction */
  explained: number[];
}

export function classStats(X: Matrix, y: number[]) {
  const n = X.length;
  const p = X[0].length;
  const classes = Array.from(new Set(y)).sort((a, b) => a - b);
  const K = classes.length;
  const sizes = classes.map((c) => y.filter((v) => v === c).length);
  const means: Matrix = classes.map(() => new Array(p).fill(0));
  const grand = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const k = classes.indexOf(y[i]);
    for (let j = 0; j < p; j++) {
      means[k][j] += X[i][j];
      grand[j] += X[i][j];
    }
  }
  for (let k = 0; k < K; k++) for (let j = 0; j < p; j++) means[k][j] /= sizes[k];
  for (let j = 0; j < p; j++) grand[j] /= n;
  return { n, p, classes, K, sizes, means, grand };
}

export function scatterMatrices(X: Matrix, y: number[]): { SW: Matrix; SB: Matrix; ST: Matrix } & ReturnType<typeof classStats> {
  const st = classStats(X, y);
  const { n, p, classes, K, sizes, means, grand } = st;
  const SW = zeros(p, p);
  const SB = zeros(p, p);
  for (let i = 0; i < n; i++) {
    const k = classes.indexOf(y[i]);
    const d = X[i].map((x, j) => x - means[k][j]);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) SW[a][b] += d[a] * d[b];
  }
  for (let k = 0; k < K; k++) {
    const d = means[k].map((m, j) => m - grand[j]);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) SB[a][b] += sizes[k] * d[a] * d[b];
  }
  const ST = SW.map((r, a) => r.map((x, b) => x + SB[a][b]));
  return { SW, SB, ST, ...st };
}

export interface LDAOptions {
  /** ridge added to S_W: S_W + γ·tr(S_W)/p · I (γ relative) */
  regularization?: number;
  /** relative tolerance for treating S_W eigenvalues as zero */
  tol?: number;
}

export function lda(X: Matrix, y: number[], opts: LDAOptions = {}): LDAResult {
  const { SW, SB, ST, n, p, classes, K, sizes, means, grand } = scatterMatrices(X, y);
  const gamma = opts.regularization ?? 0;
  const swEig = symmetricEigen(SW);
  const lmax = Math.max(Math.abs(swEig.values[0] ?? 0), 1e-300);
  const tol = (opts.tol ?? 1e-10) * lmax;
  const swRank = swEig.values.filter((v) => v > tol).length;
  const swSingular = swRank < p;
  const lmin = swEig.values[p - 1];
  const swCondition = lmin > tol ? lmax / lmin : Infinity;

  // regularised within-class scatter
  const ridge = gamma > 0 ? (gamma * (swEig.values.reduce((a, b) => a + b, 0) || 1)) / p : 0;
  const SWr = ridge > 0 ? add(SW, scale(identity(p), ridge)) : SW;
  const eigR = ridge > 0 ? symmetricEigen(SWr) : swEig;
  const tolR = (opts.tol ?? 1e-10) * Math.max(Math.abs(eigR.values[0] ?? 0), 1e-300);

  // whitening transform W = Q D^{-1/2} on the range of S_W
  const keep: number[] = [];
  for (let j = 0; j < p; j++) if (eigR.values[j] > tolR) keep.push(j);
  const r = keep.length;
  const Wh = zeros(p, r); // p × r
  for (let t = 0; t < r; t++) {
    const j = keep[t];
    const f = 1 / Math.sqrt(eigR.values[j]);
    for (let i = 0; i < p; i++) Wh[i][t] = eigR.vectors[i][j] * f;
  }
  // C = Whᵀ S_B Wh (r × r)
  const C = zeros(r, r);
  for (let a = 0; a < r; a++)
    for (let b = 0; b < r; b++) {
      let s = 0;
      for (let i = 0; i < p; i++) {
        let t = 0;
        for (let j = 0; j < p; j++) t += SB[i][j] * Wh[j][b];
        s += Wh[i][a] * t;
      }
      C[a][b] = s;
    }
  const cEig = r > 0 ? symmetricEigen(C) : { values: [], vectors: [], sweeps: 0 };
  const sbEig = symmetricEigen(SB);
  const sbTol = Math.max(Math.abs(sbEig.values[0] ?? 0), 1e-300) * 1e-10;
  const sbRank = sbEig.values.filter((v) => v > sbTol).length;
  const maxDims = Math.max(0, Math.min(K - 1, r, sbRank));

  const eigenvalues: number[] = [];
  const Wcols: Vector[] = [];
  for (let t = 0; t < maxDims; t++) {
    const u = column(cEig.vectors, t);
    const w = new Array(p).fill(0);
    for (let i = 0; i < p; i++) for (let a = 0; a < r; a++) w[i] += Wh[i][a] * u[a];
    const wn = normalize(w);
    // sign convention: largest |component| positive
    let best = 0;
    for (let i = 1; i < p; i++) if (Math.abs(wn[i]) > Math.abs(wn[best]) + 1e-12) best = i;
    const wf = wn[best] < 0 ? wn.map((x) => -x) : wn;
    Wcols.push(wf);
    eigenvalues.push(Math.max(cEig.values[t], 0));
  }
  const W = zeros(p, maxDims);
  for (let t = 0; t < maxDims; t++) for (let i = 0; i < p; i++) W[i][t] = Wcols[t][i];
  const scores = X.map((row) => Wcols.map((w) => dot(row.map((x, j) => x - grand[j]), w)));
  const totalEig = eigenvalues.reduce((a, b) => a + b, 0);
  return {
    n,
    p,
    K,
    classes,
    classSizes: sizes,
    classMeans: means,
    grandMean: grand,
    SW,
    SB,
    ST,
    eigenvalues,
    W,
    scores,
    maxDims,
    swRank,
    swSingular,
    swCondition,
    regularization: gamma,
    swEigenvalues: swEig.values,
    sbEigenvalues: sbEig.values,
    explained: eigenvalues.map((l) => (totalEig > 0 ? l / totalEig : 0)),
  };
}

/** Fisher criterion J(w) = wᵀS_B w / wᵀS_W w for an arbitrary direction. */
export function fisherCriterion(SB: Matrix, SW: Matrix, w: Vector): { between: number; within: number; J: number } {
  const between = quad(SB, w);
  const within = quad(SW, w);
  return { between, within, J: within > EPS ? between / within : Infinity };
}

function quad(A: Matrix, w: Vector): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) {
    let t = 0;
    for (let j = 0; j < w.length; j++) t += A[i][j] * w[j];
    s += w[i] * t;
  }
  return s;
}

/**
 * Gaussian generative LDA classifier with shared covariance Σ̂ = S_W/(n−K):
 * δ_k(x) = xᵀ Σ̂⁻¹ m_k − ½ m_kᵀ Σ̂⁻¹ m_k + log π_k. Returns predicted class indices.
 */
export function ldaClassify(res: LDAResult, X: Matrix, priors?: number[]): { predictions: number[]; accuracy?: (y: number[]) => number } {
  const { classMeans, SW, n, K, classSizes, classes } = res;
  const p = SW.length;
  const Sigma = SW.map((r) => r.map((x) => x / Math.max(n - K, 1)));
  const eig = symmetricEigen(Sigma);
  const tol = Math.max(Math.abs(eig.values[0] ?? 0), 1e-300) * 1e-10;
  // pseudo-inverse
  const inv = zeros(p, p);
  for (let t = 0; t < p; t++) {
    if (eig.values[t] <= tol) continue;
    const v = column(eig.vectors, t);
    const f = 1 / eig.values[t];
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) inv[i][j] += f * v[i] * v[j];
  }
  const pri = priors ?? classSizes.map((s) => s / n);
  const predictions = X.map((x) => {
    let best = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < K; k++) {
      const m = classMeans[k];
      const Sm = inv.map((r) => dot(r, m));
      const score = dot(x, Sm) - 0.5 * dot(m, Sm) + Math.log(pri[k]);
      if (score > bestScore) {
        bestScore = score;
        best = k;
      }
    }
    return classes[best];
  });
  return {
    predictions,
    accuracy: (y: number[]) => predictions.reduce((a, pr, i) => a + (pr === y[i] ? 1 : 0), 0) / y.length,
  };
}
