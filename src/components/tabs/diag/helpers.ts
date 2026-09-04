/**
 * Pure numerical helpers for the "Diagnostics & numerical stability" and
 * "What can go wrong?" laboratories. Everything is computed from live data.
 */
import {
  type Matrix,
  type Vector,
  correlation,
  inverse,
  determinant,
  symmetricEigen,
  symmetricPower,
  column,
  norm,
  dot,
  colMeans,
  covariance,
  zeros,
  quadForm,
  EPS,
} from '../../../lib/linalg';

// ---------------------------------------------------------------------------
// Angles and directions
// ---------------------------------------------------------------------------

/** Angle in degrees between two directions, ignoring sign: arccos(|u·v| / (‖u‖‖v‖)). */
export function angleBetweenDeg(u: Vector, v: Vector): number {
  const nu = norm(u);
  const nv = norm(v);
  if (nu === 0 || nv === 0) return NaN;
  const c = Math.min(1, Math.abs(dot(u, v)) / (nu * nv));
  return (Math.acos(c) * 180) / Math.PI;
}

/** Axial angle (degrees) of a 2-D direction, folded into (−90°, 90°] because w ≡ −w. */
export function axialAngleDeg(v: Vector): number {
  let a = (Math.atan2(v[1], v[0]) * 180) / Math.PI;
  while (a > 90) a -= 180;
  while (a <= -90) a += 180;
  return a;
}

/** Signed axial difference a − b (degrees) folded into (−90°, 90°]. */
export function axialDiffDeg(a: number, b: number): number {
  let d = a - b;
  while (d > 90) d -= 180;
  while (d <= -90) d += 180;
  return d;
}

// ---------------------------------------------------------------------------
// Conditioning, leverage, collinearity
// ---------------------------------------------------------------------------

/** Leverage of each observation: h_i = ‖u_i‖² for the rows u_i of U (thin SVD). Σ h_i = rank. */
export function leverages(U: Matrix): number[] {
  return U.map((row) => row.reduce((s, x) => s + x * x, 0));
}

/** κ₂ = σ₁/σ_r from a decreasing list of singular values; ∞ when σ_r is below the tolerance. */
export function kappaFromSingularValues(s: number[], tol = 0): number {
  if (s.length === 0) return NaN;
  const last = s[s.length - 1];
  return last > tol && last > 0 ? s[0] / last : Infinity;
}

export interface VIFResult {
  /** VIF_j = (R⁻¹)_jj = 1 / (1 − R²_j) */
  vif: number[];
  /** R²_j of the regression of x_j on the other variables */
  rSquared: number[];
  /** true when R was numerically singular and a pseudo-inverse was used */
  pseudo: boolean;
}

/** Variance-inflation factors from the inverse correlation matrix. */
export function varianceInflation(X: Matrix): VIFResult {
  const R = correlation(X);
  const p = R.length;
  const inv = inverse(R);
  let vif: number[];
  let pseudo = false;
  if (inv) {
    vif = inv.map((r, j) => r[j]);
  } else {
    pseudo = true;
    const eig = symmetricEigen(R);
    const tol = Math.max(eig.values[0], 1e-300) * p * EPS * 100;
    vif = new Array(p).fill(0);
    for (let t = 0; t < p; t++) {
      if (eig.values[t] <= tol) continue;
      const v = column(eig.vectors, t);
      for (let j = 0; j < p; j++) vif[j] += (v[j] * v[j]) / eig.values[t];
    }
  }
  const rSquared = vif.map((f) => (f > 0 ? 1 - 1 / f : 0));
  return { vif, rSquared, pseudo };
}

// ---------------------------------------------------------------------------
// Floating-point demonstrations
// ---------------------------------------------------------------------------

/** One-pass "textbook" variance (Σx²/n − x̄²)·n/(n−1): suffers catastrophic cancellation. */
export function naiveVariance(x: number[]): number {
  const n = x.length;
  let s = 0;
  let s2 = 0;
  for (const v of x) {
    s += v;
    s2 += v * v;
  }
  const mean = s / n;
  return (s2 / n - mean * mean) * (n / (n - 1));
}

/** Two-pass variance Σ(x − x̄)²/(n−1). */
export function twoPassVariance(x: number[]): number {
  const n = x.length;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  let s = 0;
  for (const v of x) s += (v - mean) * (v - mean);
  return s / (n - 1);
}

/** Welford's streaming update — one pass, numerically stable. */
export function welfordVariance(x: number[]): number {
  let n = 0;
  let mean = 0;
  let m2 = 0;
  for (const v of x) {
    n++;
    const d = v - mean;
    mean += d / n;
    m2 += d * (v - mean);
  }
  return m2 / (n - 1);
}

/** Läuchli matrix: columns (1, δ, 0) and (1, 0, δ); σ₁ = √(2+δ²), σ₂ = δ. */
export function laeuchli(delta: number): Matrix {
  return [
    [1, 1],
    [delta, 0],
    [0, delta],
  ];
}

// ---------------------------------------------------------------------------
// Small statistics
// ---------------------------------------------------------------------------

export function sampleSd(x: number[]): number {
  if (x.length < 2) return 0;
  return Math.sqrt(twoPassVariance(x));
}

/** Standard normal CDF (Abramowitz–Stegun 7.1.26; |error| < 7.5·10⁻⁸). */
export function normalCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const p = 1 - d * poly;
  return x >= 0 ? p : 1 - p;
}

export function pseudoInverse(A: Matrix): Matrix {
  return symmetricPower(A, -1).M;
}

export function relErr(approx: number, exact: number): number {
  if (!Number.isFinite(approx) || !Number.isFinite(exact)) return NaN;
  if (exact === 0) return approx === 0 ? 0 : Infinity;
  return Math.abs(approx - exact) / Math.abs(exact);
}

// ---------------------------------------------------------------------------
// Class-wise quantities
// ---------------------------------------------------------------------------

export interface ClassCovariances {
  classes: number[];
  sizes: number[];
  means: Matrix;
  covs: Matrix[];
}

export function classCovariances(X: Matrix, y: number[]): ClassCovariances {
  const classes = Array.from(new Set(y)).sort((a, b) => a - b);
  const p = X[0].length;
  const covs: Matrix[] = [];
  const means: Matrix = [];
  const sizes: number[] = [];
  for (const c of classes) {
    const rows = X.filter((_, i) => y[i] === c);
    sizes.push(rows.length);
    means.push(rows.length ? colMeans(rows) : new Array(p).fill(0));
    covs.push(rows.length > 1 ? covariance(rows) : zeros(p, p));
  }
  return { classes, sizes, means, covs };
}

/**
 * Resubstitution accuracy of Gaussian quadratic discriminant analysis with
 * class-specific covariance matrices and empirical priors. Returns null when a
 * class covariance cannot be inverted (too few observations or singular).
 */
export function qdaResubstitution(X: Matrix, y: number[]): { accuracy: number; perClass: number[] } | null {
  const { classes, sizes, means, covs } = classCovariances(X, y);
  const n = X.length;
  const p = X[0].length;
  const invs: Matrix[] = [];
  const logdets: number[] = [];
  for (let k = 0; k < classes.length; k++) {
    if (sizes[k] <= p) return null;
    const inv = inverse(covs[k]);
    const det = determinant(covs[k]);
    if (!inv || !(det > 0)) return null;
    invs.push(inv);
    logdets.push(Math.log(det));
  }
  let correct = 0;
  const perClassCorrect = classes.map(() => 0);
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < classes.length; k++) {
      const d = X[i].map((v, j) => v - means[k][j]);
      const score = -0.5 * logdets[k] - 0.5 * quadForm(invs[k], d) + Math.log(sizes[k] / n);
      if (score > bestScore) {
        bestScore = score;
        best = k;
      }
    }
    const trueK = classes.indexOf(y[i]);
    if (best === trueK) {
      correct++;
      perClassCorrect[trueK]++;
    }
  }
  return { accuracy: correct / n, perClass: perClassCorrect.map((c, k) => (sizes[k] > 0 ? c / sizes[k] : NaN)) };
}

/** Per-class recall of a prediction vector. */
export function perClassRecall(pred: number[], y: number[], classes: number[]): number[] {
  return classes.map((c) => {
    let tot = 0;
    let hit = 0;
    for (let i = 0; i < y.length; i++) {
      if (y[i] !== c) continue;
      tot++;
      if (pred[i] === c) hit++;
    }
    return tot > 0 ? hit / tot : NaN;
  });
}

/** Smallest pairwise Mahalanobis distance between class means, Δ_ab = √((m_a − m_b)ᵀ Σ⁻¹ (m_a − m_b)). */
export function minPairwiseMahalanobis(means: Matrix, SigmaInv: Matrix): { min: number; pair: [number, number] } {
  let best = Infinity;
  let pair: [number, number] = [0, 1];
  for (let a = 0; a < means.length; a++)
    for (let b = a + 1; b < means.length; b++) {
      const d = means[a].map((m, j) => m - means[b][j]);
      const q = Math.sqrt(Math.max(quadForm(SigmaInv, d), 0));
      if (q < best) {
        best = q;
        pair = [a, b];
      }
    }
  return { min: best, pair };
}

function sqDist(a: number[], b: number[]): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
  return s;
}

/**
 * Neighbourhood preservation: the average fraction of each point's k nearest
 * neighbours in configuration B that are also among its k nearest neighbours in A.
 * Values well below 1 indicate that B has created "false neighbours" (folding).
 */
export function knnOverlap(A: Matrix, B: Matrix, k = 7): number {
  const n = A.length;
  if (n <= k + 1) return NaN;
  const neighbours = (Mx: Matrix, i: number): Set<number> => {
    const d: { j: number; d: number }[] = [];
    for (let j = 0; j < n; j++) if (j !== i) d.push({ j, d: sqDist(Mx[j], Mx[i]) });
    d.sort((u, v) => u.d - v.d);
    return new Set(d.slice(0, k).map((e) => e.j));
  };
  let total = 0;
  for (let i = 0; i < n; i++) {
    const a = neighbours(A, i);
    const b = neighbours(B, i);
    let common = 0;
    for (const j of b) if (a.has(j)) common++;
    total += common / k;
  }
  return total / n;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** TeX scientific notation m × 10^e. */
export function texSci(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{undefined}';
  if (x === 0) return '0';
  let e = Math.floor(Math.log10(Math.abs(x)));
  let m = x / Math.pow(10, e);
  if (Math.abs(Number(m.toFixed(digits))) >= 10) {
    m /= 10;
    e += 1;
  }
  if (e === 0) return m.toFixed(digits);
  return `${m.toFixed(digits)}\\times 10^{${e}}`;
}

/** Plain-text scientific notation for tables and captions. */
export function fmtSci(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : 'undefined';
  if (x === 0) return '0';
  let e = Math.floor(Math.log10(Math.abs(x)));
  let m = x / Math.pow(10, e);
  if (Math.abs(Number(m.toFixed(digits))) >= 10) {
    m /= 10;
    e += 1;
  }
  if (e === 0) return m.toFixed(digits).replace('-', '−');
  return `${m.toFixed(digits).replace('-', '−')}×10^${e}`.replace('^-', '^−');
}

export function fmtDeg(x: number, digits = 1): string {
  return Number.isFinite(x) ? `${x.toFixed(digits).replace('-', '−')}°` : '—';
}

export function fmtPct(x: number, digits = 1): string {
  return Number.isFinite(x) ? `${(100 * x).toFixed(digits)}%` : '—';
}

/** log₁₀ of a condition number = decimal digits lost; ∞-safe. */
export function digitsLost(kappa: number): string {
  if (!Number.isFinite(kappa)) return 'all 16';
  return Math.min(16, Math.log10(kappa)).toFixed(1);
}
