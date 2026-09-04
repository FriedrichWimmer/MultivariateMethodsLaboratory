/**
 * Numerical linear algebra core.
 *
 * All matrices are row-major arrays of rows: A[i][j] is row i, column j.
 * Conventions (consistent with the dashboard notation):
 *   n = number of observations (rows), p = number of variables (columns).
 *
 * Algorithms are chosen for robustness on small/medium dense matrices:
 *   - Symmetric eigendecomposition: cyclic Jacobi rotations (backward stable,
 *     eigenvectors orthogonal to working precision).
 *   - SVD: one-sided Jacobi (Hestenes) orthogonalisation of columns – gives
 *     singular values with high relative accuracy and never forms X^T X.
 */

export type Matrix = number[][];
export type Vector = number[];

export const EPS = 2.220446049250313e-16;

// ---------------------------------------------------------------------------
// Construction & shape
// ---------------------------------------------------------------------------

export function zeros(n: number, p: number): Matrix {
  const A: Matrix = new Array(n);
  for (let i = 0; i < n; i++) A[i] = new Array(p).fill(0);
  return A;
}

export function identity(n: number): Matrix {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

export function shape(A: Matrix): [number, number] {
  return [A.length, A.length ? A[0].length : 0];
}

export function copy(A: Matrix): Matrix {
  return A.map((r) => r.slice());
}

export function diag(v: Vector): Matrix {
  const D = zeros(v.length, v.length);
  for (let i = 0; i < v.length; i++) D[i][i] = v[i];
  return D;
}

/** Rectangular diagonal matrix (n × p) with v on the leading diagonal. */
export function diagRect(v: Vector, n: number, p: number): Matrix {
  const D = zeros(n, p);
  for (let i = 0; i < Math.min(n, p, v.length); i++) D[i][i] = v[i];
  return D;
}

export function diagOf(A: Matrix): Vector {
  const m = Math.min(A.length, A[0]?.length ?? 0);
  const d = new Array(m);
  for (let i = 0; i < m; i++) d[i] = A[i][i];
  return d;
}

export function column(A: Matrix, j: number): Vector {
  return A.map((r) => r[j]);
}

export function columns(A: Matrix, js: number[]): Matrix {
  return A.map((r) => js.map((j) => r[j]));
}

export function firstColumns(A: Matrix, k: number): Matrix {
  return A.map((r) => r.slice(0, k));
}

export function fromColumns(cols: Vector[]): Matrix {
  if (cols.length === 0) return [];
  const n = cols[0].length;
  const A = zeros(n, cols.length);
  for (let j = 0; j < cols.length; j++) for (let i = 0; i < n; i++) A[i][j] = cols[j][i];
  return A;
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

export function transpose(A: Matrix): Matrix {
  const [n, p] = shape(A);
  const T = zeros(p, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) T[j][i] = A[i][j];
  return T;
}

export function matmul(A: Matrix, B: Matrix): Matrix {
  const [n, k] = shape(A);
  const [k2, p] = shape(B);
  if (k !== k2) throw new Error(`matmul: incompatible shapes (${n}×${k}) · (${k2}×${p})`);
  const C = zeros(n, p);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let t = 0; t < k; t++) {
      const a = Ai[t];
      if (a === 0) continue;
      const Bt = B[t];
      for (let j = 0; j < p; j++) Ci[j] += a * Bt[j];
    }
  }
  return C;
}

export function matvec(A: Matrix, v: Vector): Vector {
  return A.map((row) => {
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * v[j];
    return s;
  });
}

/** v^T A (returns a row vector as a Vector). */
export function vecmat(v: Vector, A: Matrix): Vector {
  const [n, p] = shape(A);
  const out = new Array(p).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) out[j] += v[i] * A[i][j];
  return out;
}

export function add(A: Matrix, B: Matrix): Matrix {
  return A.map((r, i) => r.map((x, j) => x + B[i][j]));
}

export function sub(A: Matrix, B: Matrix): Matrix {
  return A.map((r, i) => r.map((x, j) => x - B[i][j]));
}

export function scale(A: Matrix, c: number): Matrix {
  return A.map((r) => r.map((x) => x * c));
}

export function outer(u: Vector, v: Vector): Matrix {
  return u.map((a) => v.map((b) => a * b));
}

export function dot(u: Vector, v: Vector): number {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}

export function norm(v: Vector): number {
  // scaled to avoid overflow/underflow
  let scaleMax = 0;
  for (const x of v) scaleMax = Math.max(scaleMax, Math.abs(x));
  if (scaleMax === 0) return 0;
  let s = 0;
  for (const x of v) {
    const t = x / scaleMax;
    s += t * t;
  }
  return scaleMax * Math.sqrt(s);
}

export function normalize(v: Vector): Vector {
  const nv = norm(v);
  return nv === 0 ? v.slice() : v.map((x) => x / nv);
}

export function vadd(u: Vector, v: Vector): Vector {
  return u.map((x, i) => x + v[i]);
}

export function vsub(u: Vector, v: Vector): Vector {
  return u.map((x, i) => x - v[i]);
}

export function vscale(u: Vector, c: number): Vector {
  return u.map((x) => x * c);
}

export function frobenius(A: Matrix): number {
  let s = 0;
  for (const r of A) for (const x of r) s += x * x;
  return Math.sqrt(s);
}

export function trace(A: Matrix): number {
  let s = 0;
  for (let i = 0; i < Math.min(A.length, A[0]?.length ?? 0); i++) s += A[i][i];
  return s;
}

export function maxAbs(A: Matrix): number {
  let m = 0;
  for (const r of A) for (const x of r) m = Math.max(m, Math.abs(x));
  return m;
}

/** Quadratic form w^T A w. */
export function quadForm(A: Matrix, w: Vector): number {
  return dot(w, matvec(A, w));
}

// ---------------------------------------------------------------------------
// Descriptive statistics on the data matrix
// ---------------------------------------------------------------------------

export function colMeans(X: Matrix): Vector {
  const [n, p] = shape(X);
  const m = new Array(p).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] += X[i][j];
  return m.map((s) => s / n);
}

export function colVariances(X: Matrix, ddof = 1): Vector {
  const [n, p] = shape(X);
  const m = colMeans(X);
  const v = new Array(p).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < p; j++) {
      const d = X[i][j] - m[j];
      v[j] += d * d;
    }
  const denom = Math.max(n - ddof, 1);
  return v.map((s) => s / denom);
}

export function colStds(X: Matrix, ddof = 1): Vector {
  return colVariances(X, ddof).map(Math.sqrt);
}

/** X_c = X − 1 x̄ᵀ */
export function centerColumns(X: Matrix): { Xc: Matrix; means: Vector } {
  const means = colMeans(X);
  const Xc = X.map((r) => r.map((x, j) => x - means[j]));
  return { Xc, means };
}

/** Standardise to unit variance (after centering). Columns with zero variance are left centred. */
export function standardizeColumns(X: Matrix): { Xs: Matrix; means: Vector; stds: Vector } {
  const { Xc, means } = centerColumns(X);
  const stds = colStds(X);
  const Xs = Xc.map((r) => r.map((x, j) => (stds[j] > 0 ? x / stds[j] : x)));
  return { Xs, means, stds };
}

/** Sample covariance S = (1/(n−1)) X_cᵀ X_c of an (already centred or not) matrix. */
export function covariance(X: Matrix, ddof = 1): Matrix {
  const [n, p] = shape(X);
  const { Xc } = centerColumns(X);
  const S = zeros(p, p);
  for (let i = 0; i < n; i++) {
    const r = Xc[i];
    for (let a = 0; a < p; a++) {
      const ra = r[a];
      if (ra === 0) continue;
      for (let b = a; b < p; b++) S[a][b] += ra * r[b];
    }
  }
  const denom = Math.max(n - ddof, 1);
  for (let a = 0; a < p; a++)
    for (let b = a; b < p; b++) {
      S[a][b] /= denom;
      S[b][a] = S[a][b];
    }
  return S;
}

/** Cross-product matrix Aᵀ A (used for scatter matrices). */
export function gram(A: Matrix): Matrix {
  const [n, p] = shape(A);
  const G = zeros(p, p);
  for (let i = 0; i < n; i++) {
    const r = A[i];
    for (let a = 0; a < p; a++) {
      const ra = r[a];
      if (ra === 0) continue;
      for (let b = a; b < p; b++) G[a][b] += ra * r[b];
    }
  }
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
  return G;
}

export function correlationFromCovariance(S: Matrix): Matrix {
  const p = S.length;
  const R = zeros(p, p);
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      const d = Math.sqrt(S[a][a] * S[b][b]);
      R[a][b] = d > 0 ? S[a][b] / d : a === b ? 1 : 0;
    }
  return R;
}

export function correlation(X: Matrix): Matrix {
  return correlationFromCovariance(covariance(X));
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition – cyclic Jacobi
// ---------------------------------------------------------------------------

export interface EigenResult {
  /** eigenvalues sorted in decreasing order */
  values: number[];
  /** eigenvectors as COLUMNS of a p×p matrix, in the same order as `values` */
  vectors: Matrix;
  sweeps: number;
}

/**
 * Eigendecomposition A = V Λ Vᵀ of a real symmetric matrix using cyclic Jacobi
 * rotations. Eigenvalues are returned in decreasing order and the eigenvectors
 * are sign-normalised so the largest-magnitude component of each is positive
 * (eigenvectors are only defined up to sign – see the dashboard's notes).
 */
export function symmetricEigen(Ain: Matrix, opts: { maxSweeps?: number; tol?: number; signNormalize?: boolean } = {}): EigenResult {
  const n = Ain.length;
  if (n === 0) return { values: [], vectors: [], sweeps: 0 };
  const A = copy(Ain);
  // symmetrise defensively (average off-diagonal pairs)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const m = 0.5 * (A[i][j] + A[j][i]);
      A[i][j] = m;
      A[j][i] = m;
    }
  const V = identity(n);
  const maxSweeps = opts.maxSweeps ?? 100;
  const tol = opts.tol ?? 1e-14;
  let sweeps = 0;
  const frob = Math.max(frobenius(A), Number.MIN_VALUE);

  for (sweeps = 0; sweeps < maxSweeps; sweeps++) {
    // off-diagonal norm
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    off = Math.sqrt(2 * off);
    if (off <= tol * frob) break;

    for (let pIdx = 0; pIdx < n - 1; pIdx++) {
      for (let q = pIdx + 1; q < n; q++) {
        const apq = A[pIdx][q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[pIdx][pIdx];
        const aqq = A[q][q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // rotate rows/cols p,q of A: A ← Jᵀ A J
        for (let k = 0; k < n; k++) {
          const akp = A[k][pIdx];
          const akq = A[k][q];
          A[k][pIdx] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[pIdx][k];
          const aqk = A[q][k];
          A[pIdx][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        // enforce exact symmetry of the zeroed pair
        A[pIdx][q] = 0;
        A[q][pIdx] = 0;
        // accumulate V ← V J
        for (let k = 0; k < n; k++) {
          const vkp = V[k][pIdx];
          const vkq = V[k][q];
          V[k][pIdx] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values = diagOf(A);
  const order = values.map((_, i) => i).sort((a, b) => values[b] - values[a]);
  const sortedValues = order.map((i) => values[i]);
  let vectors = columns(V, order);
  if (opts.signNormalize !== false) vectors = signNormalizeColumns(vectors);
  return { values: sortedValues, vectors, sweeps };
}

/** Flip the sign of each column so its largest-magnitude entry is positive. */
export function signNormalizeColumns(V: Matrix): Matrix {
  const [n, p] = shape(V);
  const out = copy(V);
  for (let j = 0; j < p; j++) {
    let best = 0;
    let bestAbs = -1;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(V[i][j]);
      if (a > bestAbs + 1e-12) {
        bestAbs = a;
        best = i;
      }
    }
    if (V[best][j] < 0) for (let i = 0; i < n; i++) out[i][j] = -out[i][j];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Singular value decomposition – one-sided Jacobi (Hestenes)
// ---------------------------------------------------------------------------

export interface SVDResult {
  /** n × r, orthonormal columns (r = min(n, p)) */
  U: Matrix;
  /** singular values, decreasing, length r */
  s: number[];
  /** p × r, orthonormal columns */
  V: Matrix;
  /** numerical rank (number of singular values above tolerance) */
  rank: number;
  /** tolerance used for the rank decision */
  tol: number;
  sweeps: number;
}

/**
 * Thin SVD  X = U Σ Vᵀ  of an n × p matrix.
 *
 * Works on columns of X (when n ≥ p) or of Xᵀ (when n < p); the roles of U and
 * V are swapped back at the end. Singular values are sorted in decreasing
 * order. Columns of U and V corresponding to (numerically) zero singular values
 * are completed to an orthonormal set so that U and V always have orthonormal
 * columns.
 */
export function svd(X: Matrix, opts: { maxSweeps?: number; tol?: number } = {}): SVDResult {
  const [n, p] = shape(X);
  if (n === 0 || p === 0) return { U: [], s: [], V: [], rank: 0, tol: 0, sweeps: 0 };
  const transposed = n < p;
  const A = transposed ? transpose(X) : copy(X); // m × r with m ≥ r
  const m = A.length;
  const r = A[0].length;
  const V = identity(r);
  const maxSweeps = opts.maxSweeps ?? 80;
  const tol = opts.tol ?? 1e-15;
  let sweeps = 0;

  // column access helpers – store columns for cache friendliness
  const cols: Float64Array[] = [];
  for (let j = 0; j < r; j++) {
    const c = new Float64Array(m);
    for (let i = 0; i < m; i++) c[i] = A[i][j];
    cols.push(c);
  }
  const vcols: Float64Array[] = [];
  for (let j = 0; j < r; j++) {
    const c = new Float64Array(r);
    c[j] = 1;
    vcols.push(c);
  }

  for (sweeps = 0; sweeps < maxSweeps; sweeps++) {
    let rotated = false;
    for (let i = 0; i < r - 1; i++) {
      for (let j = i + 1; j < r; j++) {
        const ci = cols[i];
        const cj = cols[j];
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let k = 0; k < m; k++) {
          alpha += ci[k] * ci[k];
          beta += cj[k] * cj[k];
          gamma += ci[k] * cj[k];
        }
        if (alpha === 0 || beta === 0) continue;
        if (Math.abs(gamma) <= tol * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let k = 0; k < m; k++) {
          const a = ci[k];
          const b = cj[k];
          ci[k] = c * a - s * b;
          cj[k] = s * a + c * b;
        }
        const vi = vcols[i];
        const vj = vcols[j];
        for (let k = 0; k < r; k++) {
          const a = vi[k];
          const b = vj[k];
          vi[k] = c * a - s * b;
          vj[k] = s * a + c * b;
        }
      }
    }
    if (!rotated) break;
  }

  // singular values are the column norms
  const sv: number[] = cols.map((c) => {
    let s = 0;
    for (let k = 0; k < m; k++) s += c[k] * c[k];
    return Math.sqrt(s);
  });
  const order = sv.map((_, i) => i).sort((a, b) => sv[b] - sv[a]);
  const s = order.map((i) => sv[i]);
  const smax = s[0] ?? 0;
  const rankTol = Math.max(m, r) * EPS * smax * 10;
  let rank = 0;
  for (const v of s) if (v > rankTol) rank++;

  // Build U (m × r) and V (r × r) in sorted order
  const Ucols: Vector[] = [];
  const Vcols: Vector[] = [];
  for (let idx = 0; idx < r; idx++) {
    const j = order[idx];
    const c = cols[j];
    const sig = sv[j];
    Ucols.push(sig > rankTol ? Array.from(c, (x) => x / sig) : new Array(m).fill(0));
    Vcols.push(Array.from(vcols[j]));
  }
  // complete U columns for zero singular values to an orthonormal set
  const completedU = completeOrthonormalColumns(Ucols, rank);

  let Umat = fromColumns(completedU);
  let Vmat = signNormalizeColumnsPair(fromColumns(Vcols), Umat);
  // fix signs jointly so X = U Σ Vᵀ still holds: signNormalizeColumnsPair flips both
  Umat = Vmat.U;
  const Vfinal = Vmat.V;

  if (transposed) {
    return { U: Vfinal, s, V: Umat, rank, tol: rankTol, sweeps };
  }
  return { U: Umat, s, V: Vfinal, rank, tol: rankTol, sweeps };
}

/** Flip sign of column j in both U and V so that the largest |entry| of V[:,j] is positive. */
function signNormalizeColumnsPair(V: Matrix, U: Matrix): { U: Matrix; V: Matrix } {
  const [pv, r] = shape(V);
  const Uo = copy(U);
  const Vo = copy(V);
  for (let j = 0; j < r; j++) {
    let best = 0;
    let bestAbs = -1;
    for (let i = 0; i < pv; i++) {
      const a = Math.abs(V[i][j]);
      if (a > bestAbs + 1e-12) {
        bestAbs = a;
        best = i;
      }
    }
    if (V[best][j] < 0) {
      for (let i = 0; i < pv; i++) Vo[i][j] = -Vo[i][j];
      for (let i = 0; i < Uo.length; i++) Uo[i][j] = -Uo[i][j];
    }
  }
  return { U: Uo, V: Vo };
}

/**
 * Given a list of column vectors where the first `given` are orthonormal and the
 * rest may be zero/garbage, replace the rest by vectors completing an orthonormal
 * set (Gram–Schmidt against standard basis vectors).
 */
export function completeOrthonormalColumns(cols: Vector[], given: number): Vector[] {
  if (cols.length === 0) return cols;
  const m = cols[0].length;
  const out = cols.slice(0, given).map((c) => c.slice());
  let e = 0;
  while (out.length < cols.length && e < m) {
    const cand = new Array(m).fill(0);
    cand[e++] = 1;
    // modified Gram–Schmidt, twice for stability
    let v = cand;
    for (let pass = 0; pass < 2; pass++) {
      for (const q of out) {
        const proj = dot(q, v);
        v = v.map((x, i) => x - proj * q[i]);
      }
    }
    const nv = norm(v);
    if (nv > 1e-8) out.push(v.map((x) => x / nv));
  }
  while (out.length < cols.length) out.push(new Array(m).fill(0));
  return out;
}

/**
 * Full SVD of a square (or general) matrix: U is n×n, V is p×p and Σ is n×p
 * rectangular diagonal. Useful for the 2-D geometric decomposition demo.
 */
export function svdFull(X: Matrix): { U: Matrix; s: number[]; V: Matrix; Sigma: Matrix; rank: number } {
  const [n, p] = shape(X);
  const thin = svd(X);
  const r = thin.s.length;
  const Ucols = Array.from({ length: n }, (_, j) => (j < r ? column(thin.U, j) : new Array(n).fill(0)));
  const Vcols = Array.from({ length: p }, (_, j) => (j < r ? column(thin.V, j) : new Array(p).fill(0)));
  const Ufull = fromColumns(completeOrthonormalColumns(Ucols, r));
  const Vfull = fromColumns(completeOrthonormalColumns(Vcols, r));
  return { U: Ufull, s: thin.s, V: Vfull, Sigma: diagRect(thin.s, n, p), rank: thin.rank };
}

/** Rank-k truncation  X_k = U_k Σ_k V_kᵀ. */
export function truncatedReconstruction(res: { U: Matrix; s: number[]; V: Matrix }, k: number): Matrix {
  const n = res.U.length;
  const p = res.V.length;
  const kk = Math.max(0, Math.min(k, res.s.length));
  const X = zeros(n, p);
  for (let t = 0; t < kk; t++) {
    const sig = res.s[t];
    if (sig === 0) continue;
    for (let i = 0; i < n; i++) {
      const uis = res.U[i][t] * sig;
      if (uis === 0) continue;
      for (let j = 0; j < p; j++) X[i][j] += uis * res.V[j][t];
    }
  }
  return X;
}

export function numericalRank(X: Matrix, tolFactor = 10): number {
  return svd(X).rank;
}

/** 2-norm condition number σ_max / σ_min (Infinity if rank deficient). */
export function conditionNumber(X: Matrix): number {
  const { s } = svd(X);
  if (s.length === 0) return NaN;
  const smin = s[s.length - 1];
  return smin > 0 ? s[0] / smin : Infinity;
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------

/** Cholesky factor L (A = L Lᵀ); returns null if A is not numerically positive definite. */
export function cholesky(A: Matrix, jitter = 0): Matrix | null {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j] + (i === j ? jitter : 0);
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0) || !Number.isFinite(s)) return null;
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

/** Solve L y = b for lower-triangular L. */
export function forwardSubstitution(L: Matrix, b: Vector): Vector {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  return y;
}

/** Solve Lᵀ x = y for lower-triangular L. */
export function backSubstitutionT(L: Matrix, y: Vector): Vector {
  const n = L.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** General inverse via Gauss–Jordan with partial pivoting. Returns null if singular. */
export function inverse(A: Matrix): Matrix | null {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...identity(n)[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((r) => r.slice(n));
}

/** Determinant via LU with partial pivoting. */
export function determinant(A: Matrix): number {
  const n = A.length;
  const M = copy(A);
  let det = 1;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) === 0) return 0;
    if (piv !== col) {
      [M[col], M[piv]] = [M[piv], M[col]];
      det = -det;
    }
    det *= M[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let j = col; j < n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return det;
}

/**
 * Symmetric square root / inverse square root via eigendecomposition.
 * Returns A^{power} with eigenvalues below `tol` treated as zero (pseudo-inverse
 * semantics for negative powers). Also reports how many eigenvalues were dropped.
 */
export function symmetricPower(A: Matrix, power: number, tol?: number): { M: Matrix; dropped: number; eigen: EigenResult } {
  const eig = symmetricEigen(A);
  const lmax = Math.max(...eig.values.map(Math.abs), 0);
  const t = tol ?? Math.max(lmax * A.length * EPS * 100, 1e-300);
  const n = A.length;
  const M = zeros(n, n);
  let dropped = 0;
  for (let k = 0; k < n; k++) {
    const lam = eig.values[k];
    if (lam <= t) {
      dropped++;
      continue;
    }
    const f = Math.pow(lam, power);
    const v = column(eig.vectors, k);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] += f * v[i] * v[j];
  }
  return { M, dropped, eigen: eig };
}

// ---------------------------------------------------------------------------
// 2-D helpers (for the geometric SVD demo)
// ---------------------------------------------------------------------------

/** Angle (radians) of a 2-D column vector. */
export function angleOf(v: Vector): number {
  return Math.atan2(v[1], v[0]);
}

/** 2×2 rotation matrix by angle θ. */
export function rotation2(theta: number): Matrix {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s],
    [s, c],
  ];
}

/** Whether a square orthogonal matrix is a reflection (det < 0). */
export function isReflection(Q: Matrix): boolean {
  return determinant(Q) < 0;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

export function fmt(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : 'NaN';
  if (Math.abs(x) < 0.5 * Math.pow(10, -digits)) return (0).toFixed(digits);
  const a = Math.abs(x);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return x.toExponential(Math.max(digits - 1, 1)).replace('-', '−');
  return x.toFixed(digits).replace('-', '−');
}

export function fmtMatrix(A: Matrix, digits = 3): string[][] {
  return A.map((r) => r.map((x) => fmt(x, digits)));
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition – Householder tridiagonalisation + implicit QL
// (tred2 / tql2, after EISPACK/JAMA). O(n³) with a small constant; used for
// the n × n matrices of classical MDS where Jacobi would be slow.
// ---------------------------------------------------------------------------

export function symmetricEigenQL(Ain: Matrix, signNormalize = true): EigenResult {
  const n = Ain.length;
  if (n === 0) return { values: [], vectors: [], sweeps: 0 };
  const V: number[][] = Ain.map((r) => r.slice());
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const m = 0.5 * (V[i][j] + V[j][i]);
      V[i][j] = m;
      V[j][i] = m;
    }
  const d = new Array(n).fill(0);
  const e = new Array(n).fill(0);

  // ---- tred2 ----
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0;
    let h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) {
        d[j] = V[i - 1][j];
        V[i][j] = 0;
        V[j][i] = 0;
      }
    } else {
      for (let k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      let f = d[i - 1];
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
        d[j] = V[i - 1][j];
        V[i][j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i];
    V[i][i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
  }
  for (let j = 0; j < n; j++) {
    d[j] = V[n - 1][j];
    V[n - 1][j] = 0;
  }
  V[n - 1][n - 1] = 1;
  e[0] = 0;

  // ---- tql2 ----
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0;
  let tst1 = 0;
  const eps = Math.pow(2, -52);
  let totalIter = 0;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) {
      if (Math.abs(e[m]) <= eps * tst1) break;
      m++;
    }
    if (m > l) {
      let iter = 0;
      do {
        iter++;
        totalIter++;
        if (iter > 200) break;
        let g = d[l];
        let p = (d[l + 1] - g) / (2 * e[l]);
        let r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1;
        let c2 = c;
        let c3 = c;
        const el1 = e[l + 1];
        let s = 0;
        let s2 = 0;
        for (let i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = Math.hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            h = V[k][i + 1];
            V[k][i + 1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0;
  }

  const order = d.map((_, i) => i).sort((a, b) => d[b] - d[a]);
  const values = order.map((i) => d[i]);
  let vectors = columns(V, order);
  if (signNormalize) vectors = signNormalizeColumns(vectors);
  return { values, vectors, sweeps: totalIter };
}

/**
 * Dispatching symmetric eigensolver: Jacobi for small matrices (maximal
 * accuracy, used in all the small worked examples), Householder+QL for large.
 */
export function symmetricEigenAuto(A: Matrix): EigenResult {
  return A.length <= 14 ? symmetricEigen(A) : symmetricEigenQL(A);
}
