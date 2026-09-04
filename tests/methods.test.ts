import { describe, it, expect } from 'vitest';
import {
  svd,
  svdFull,
  symmetricEigen,
  matmul,
  transpose,
  frobenius,
  sub,
  identity,
  truncatedReconstruction,
  cholesky,
  inverse,
  covariance,
  determinant,
  conditionNumber,
  diag,
} from '../src/lib/linalg';
import { makeRNG } from '../src/lib/random';
import { generateDataset, defaultParams, datasetCatalog } from '../src/lib/datasets';
import { pca, pcaReconstruct, reconstructionErrors, projectionVariance, projectOnDirection, sampleVariance } from '../src/lib/pca';
import { distanceMatrix, classicalMDS, procrustesAlign, doubleCenter } from '../src/lib/mds';
import { lda, fisherCriterion, scatterMatrices, ldaClassify } from '../src/lib/lda';
import { svd2d, lowRankSummaries, svdChecks, transformPoints, unitCircle } from '../src/lib/svdlab';

const rng = makeRNG(7);
function randomMatrix(n: number, p: number) {
  return Array.from({ length: n }, () => Array.from({ length: p }, () => rng.normal()));
}
function maxAbsDiff(A: number[][], B: number[][]) {
  let m = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  return m;
}

describe('SVD', () => {
  it('reconstructs X = UΣVᵀ for tall, wide and square matrices', () => {
    for (const [n, p] of [[10, 4], [4, 10], [6, 6], [1, 5], [5, 1]] as [number, number][]) {
      const X = randomMatrix(n, p);
      const res = svd(X);
      const recon = truncatedReconstruction(res, res.s.length);
      expect(frobenius(sub(X, recon))).toBeLessThan(1e-10 * Math.max(1, frobenius(X)));
      expect(res.U.length).toBe(n);
      expect(res.V.length).toBe(p);
      expect(res.s.length).toBe(Math.min(n, p));
    }
  });
  it('orders singular values decreasingly and returns orthonormal U, V', () => {
    const X = randomMatrix(12, 5);
    const res = svd(X);
    for (let i = 1; i < res.s.length; i++) expect(res.s[i]).toBeLessThanOrEqual(res.s[i - 1] + 1e-12);
    const c = svdChecks(X, res);
    expect(c.uOrthoError).toBeLessThan(1e-10);
    expect(c.vOrthoError).toBeLessThan(1e-10);
    expect(c.ordered).toBe(true);
  });
  it('handles rank-deficient matrices and completes orthonormal bases', () => {
    // rank-2 matrix in 5 columns
    const A = randomMatrix(8, 2);
    const B = randomMatrix(2, 5);
    const X = matmul(A, B);
    const res = svd(X);
    expect(res.rank).toBe(2);
    expect(res.s[2]).toBeLessThan(1e-10);
    const c = svdChecks(X, res);
    expect(c.uOrthoError).toBeLessThan(1e-8);
    expect(c.vOrthoError).toBeLessThan(1e-8);
    const recon = truncatedReconstruction(res, 2);
    expect(frobenius(sub(X, recon))).toBeLessThan(1e-9);
  });
  it('handles the zero matrix and rank-1 outer products', () => {
    const Z = [
      [0, 0],
      [0, 0],
    ];
    const r = svd(Z);
    expect(r.rank).toBe(0);
    expect(r.s.every((s) => s === 0)).toBe(true);
    const u = [1, 2, 3];
    const v = [4, 5];
    const X = u.map((a) => v.map((b) => a * b));
    const r1 = svd(X);
    expect(r1.rank).toBe(1);
    expect(r1.s[0]).toBeCloseTo(Math.sqrt(14) * Math.sqrt(41), 10);
  });
  it('Eckart–Young: truncation error equals √(Σ_{j>k} σ_j²) and beats random rank-k approximations', () => {
    const X = randomMatrix(9, 6);
    const res = svd(X);
    const sums = lowRankSummaries(X, res);
    for (const s of sums) expect(Math.abs(s.error - s.eckartYoung)).toBeLessThan(1e-9);
    // random rank-2 approximation via projecting on a random 2-D column subspace
    const Q = svd(randomMatrix(6, 2)).U; // 6×2 orthonormal
    const P = matmul(Q, transpose(Q));
    const Xr = matmul(X, P);
    expect(frobenius(sub(X, Xr))).toBeGreaterThanOrEqual(sums[2].error - 1e-12);
  });
  it('full SVD of 2×2 gives orthogonal U, V and correct determinant relationships', () => {
    const A = [
      [3, 1],
      [1, 2],
    ];
    const g = svd2d(A);
    expect(maxAbsDiff(matmul(transpose(g.U), g.U), identity(2))).toBeLessThan(1e-12);
    expect(maxAbsDiff(matmul(transpose(g.V), g.V), identity(2))).toBeLessThan(1e-12);
    expect(g.s[0] * g.s[1]).toBeCloseTo(Math.abs(determinant(A)), 10);
    expect(maxAbsDiff(matmul(matmul(g.U, g.Sigma), transpose(g.V)), A)).toBeLessThan(1e-12);
    // known singular values of symmetric matrix = |eigenvalues|
    const eig = symmetricEigen(A);
    expect(g.s[0]).toBeCloseTo(eig.values[0], 10);
    expect(g.s[1]).toBeCloseTo(eig.values[1], 10);
    const pts = transformPoints(A, unitCircle(8));
    expect(pts.length).toBe(9);
    // singular full-rank deficient matrix
    const S = svd2d([
      [1, 2],
      [2, 4],
    ]);
    expect(S.rank).toBe(1);
    expect(S.s[1]).toBeLessThan(1e-12);
    expect(maxAbsDiff(matmul(transpose(S.U), S.U), identity(2))).toBeLessThan(1e-10);
    const F = svdFull([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(F.U.length).toBe(2);
    expect(F.V.length).toBe(3);
    expect(F.V[0].length).toBe(3);
    expect(maxAbsDiff(matmul(transpose(F.V), F.V), identity(3))).toBeLessThan(1e-10);
  });
});

describe('Symmetric eigendecomposition', () => {
  it('diagonalises a random symmetric matrix', () => {
    const M = randomMatrix(7, 7);
    const A = matmul(transpose(M), M);
    const e = symmetricEigen(A);
    const recon = matmul(matmul(e.vectors, diag(e.values)), transpose(e.vectors));
    expect(maxAbsDiff(recon, A)).toBeLessThan(1e-9);
    expect(maxAbsDiff(matmul(transpose(e.vectors), e.vectors), identity(7))).toBeLessThan(1e-10);
    for (let i = 1; i < e.values.length; i++) expect(e.values[i]).toBeLessThanOrEqual(e.values[i - 1] + 1e-12);
  });
  it('recovers known eigenvalues of an equicorrelation matrix', () => {
    const p = 5;
    const rho = 0.6;
    const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : rho)));
    const e = symmetricEigen(A);
    expect(e.values[0]).toBeCloseTo(1 + (p - 1) * rho, 10);
    for (let i = 1; i < p; i++) expect(e.values[i]).toBeCloseTo(1 - rho, 10);
  });
  it('handles indefinite matrices (negative eigenvalues)', () => {
    const A = [
      [0, 1],
      [1, 0],
    ];
    const e = symmetricEigen(A);
    expect(e.values[0]).toBeCloseTo(1, 12);
    expect(e.values[1]).toBeCloseTo(-1, 12);
  });
});

describe('Solvers', () => {
  it('Cholesky and inverse agree', () => {
    const M = randomMatrix(6, 4);
    const A = matmul(transpose(M), M);
    const L = cholesky(A)!;
    expect(L).not.toBeNull();
    expect(maxAbsDiff(matmul(L, transpose(L)), A)).toBeLessThan(1e-10);
    const Ai = inverse(A)!;
    expect(maxAbsDiff(matmul(A, Ai), identity(4))).toBeLessThan(1e-8);
    expect(cholesky([[1, 2], [2, 1]])).toBeNull();
    expect(inverse([[1, 2], [2, 4]])).toBeNull();
    expect(conditionNumber([[1, 2], [2, 4]])).toBe(Infinity);
  });
});

describe('PCA', () => {
  const ds = generateDataset('correlated', { ...defaultParams, n: 60, p: 5, correlation: 0.5 });
  const res = pca(ds.X, 'center');
  it('eigenvalues of S equal σ²/(n−1) and both routes agree', () => {
    for (let j = 0; j < res.p; j++) {
      expect(res.eigenvalues[j]).toBeCloseTo(res.eigen.values[j], 9);
      expect(res.eigenvalues[j]).toBeCloseTo((res.singularValues[j] ** 2) / (res.n - 1), 9);
    }
    // eigenvectors agree up to sign (both sign-normalised so they should agree exactly)
    expect(maxAbsDiff(res.V, res.eigen.vectors)).toBeLessThan(1e-7);
  });
  it('explained variance ratios sum to one and total variance equals trace(S)', () => {
    expect(res.explained.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(res.cumulative[res.p - 1]).toBeCloseTo(1, 10);
    const tr = res.S.reduce((a, r, i) => a + r[i], 0);
    expect(res.totalVariance).toBeCloseTo(tr, 9);
  });
  it('scores are uncorrelated with variances equal to the eigenvalues', () => {
    const Sz = covariance(res.scores);
    for (let i = 0; i < res.p; i++) {
      expect(Sz[i][i]).toBeCloseTo(res.eigenvalues[i], 9);
      for (let j = 0; j < res.p; j++) if (i !== j) expect(Math.abs(Sz[i][j])).toBeLessThan(1e-9);
    }
  });
  it('reconstruction with all components is exact and errors decrease with k', () => {
    const errs = reconstructionErrors(res);
    expect(errs[res.p]).toBeLessThan(1e-9);
    for (let k = 1; k <= res.p; k++) expect(errs[k]).toBeLessThanOrEqual(errs[k - 1] + 1e-12);
    // matches Eckart–Young value
    for (let k = 0; k <= res.p; k++) {
      const tail = res.singularValues.slice(k).reduce((a, b) => a + b * b, 0);
      expect(errs[k]).toBeCloseTo(Math.sqrt(tail), 8);
    }
    const back = pcaReconstruct(res, res.p).original;
    expect(maxAbsDiff(back, ds.X)).toBeLessThan(1e-9);
  });
  it('projection variance wᵀSw equals the sample variance of X_c w and is maximised by v₁', () => {
    const w = [0.3, -0.4, 0.5, 0.6, 0.2];
    const nw = Math.hypot(...w);
    const wu = w.map((x) => x / nw);
    const z = projectOnDirection(res.Xc, wu);
    expect(sampleVariance(z)).toBeCloseTo(projectionVariance(res.S, wu), 9);
    expect(projectionVariance(res.S, wu)).toBeLessThanOrEqual(res.eigenvalues[0] + 1e-12);
    const v1 = res.V.map((r) => r[0]);
    expect(projectionVariance(res.S, v1)).toBeCloseTo(res.eigenvalues[0], 9);
  });
  it('standardised PCA works on the correlation matrix with trace p', () => {
    const rs = pca(ds.X, 'standardize');
    expect(rs.totalVariance).toBeCloseTo(rs.p, 9);
    for (let i = 0; i < rs.p; i++) expect(rs.S[i][i]).toBeCloseTo(1, 9);
  });
  it('p > n gives at most n − 1 non-zero eigenvalues', () => {
    const wide = generateDataset('pGreaterN', { ...defaultParams, n: 8, p: 20 });
    const r = pca(wide.X, 'center');
    expect(r.rank).toBeLessThanOrEqual(wide.X.length - 1);
    expect(r.singularValues.length).toBe(Math.min(wide.X.length, wide.X[0].length));
  });
});

describe('Classical MDS', () => {
  const ds = generateDataset('clusters', { ...defaultParams, n: 40, p: 4, K: 3 });
  const D = distanceMatrix(ds.X, 'euclidean');
  const m = classicalMDS(D, 2);
  it('double centering matches −½ J D² J', () => {
    const n = D.length;
    const J = identity(n).map((r) => r.map((x) => x - 1 / n));
    const D2 = D.map((r) => r.map((d) => d * d));
    const B = matmul(matmul(J, D2), J).map((r) => r.map((x) => -0.5 * x));
    expect(maxAbsDiff(B, m.B)).toBeLessThan(1e-9);
    expect(maxAbsDiff(doubleCenter(D2), B)).toBeLessThan(1e-9);
  });
  it('Euclidean input yields B = X_c X_cᵀ, non-negative eigenvalues, and coordinates matching PCA scores up to rotation/reflection', () => {
    const p = pca(ds.X, 'center');
    const B2 = matmul(p.Xc, transpose(p.Xc));
    expect(maxAbsDiff(B2, m.B)).toBeLessThan(1e-8);
    expect(m.negative).toBe(0);
    // eigenvalues of B equal squared singular values of X_c
    for (let j = 0; j < 4; j++) expect(m.eigenvalues[j]).toBeCloseTo(p.singularValues[j] ** 2, 7);
    // coordinates reproduce Euclidean distances exactly when k = p
    const full = classicalMDS(D, 4);
    expect(maxAbsDiff(full.Dhat, D)).toBeLessThan(1e-8);
    expect(full.stress1).toBeLessThan(1e-9);
    // 2-D MDS coords vs 2-D PCA scores: identical up to sign flips (Procrustes residual ≈ 0)
    const Z2 = p.scores.map((r) => r.slice(0, 2));
    const al = procrustesAlign(m.coords, Z2);
    expect(al.relative).toBeLessThan(1e-7);
  });
  it('non-Euclidean metrics can produce negative eigenvalues and stress > 0', () => {
    const Dm = distanceMatrix(ds.X, 'manhattan');
    const mm = classicalMDS(Dm, 2);
    expect(mm.negative).toBeGreaterThan(0);
    expect(mm.stress1).toBeGreaterThan(0);
    expect(mm.coords.length).toBe(40);
    expect(mm.coords[0].length).toBe(2);
  });
  it('k is capped by the number of positive eigenvalues', () => {
    // three collinear points: only one positive eigenvalue
    const X = [[0], [1], [3]];
    const r = classicalMDS(distanceMatrix(X), 2);
    expect(r.k).toBe(1);
    expect(r.positive).toBe(1);
  });
});

describe('LDA', () => {
  const ds = generateDataset('iris', { ...defaultParams, n: 150 });
  const res = lda(ds.X, ds.y!);
  it('constructs S_W, S_B with S_W + S_B = (n−1) S and S_B of rank ≤ K−1', () => {
    const S = covariance(ds.X);
    const ST = S.map((r) => r.map((x) => x * (ds.X.length - 1)));
    expect(maxAbsDiff(res.ST, ST)).toBeLessThan(1e-7);
    const sbRank = res.sbEigenvalues.filter((v) => v > 1e-8 * res.sbEigenvalues[0]).length;
    expect(sbRank).toBeLessThanOrEqual(res.K - 1);
    // S_W equals the sum over classes of (n_k − 1) × class covariance
    const { SW } = scatterMatrices(ds.X, ds.y!);
    let SWcheck = SW.map((r) => r.map(() => 0));
    for (let k = 0; k < 3; k++) {
      const Xk = ds.X.filter((_, i) => ds.y![i] === k);
      const Sk = covariance(Xk);
      SWcheck = SWcheck.map((r, a) => r.map((x, b) => x + Sk[a][b] * (Xk.length - 1)));
    }
    expect(maxAbsDiff(SWcheck, SW)).toBeLessThan(1e-7);
  });
  it('number of discriminants ≤ K−1 and directions satisfy S_B w = λ S_W w', () => {
    expect(res.maxDims).toBeLessThanOrEqual(res.K - 1);
    expect(res.W[0].length).toBe(res.maxDims);
    for (let t = 0; t < res.maxDims; t++) {
      const w = res.W.map((r) => r[t]);
      const lhs = res.SB.map((r) => r.reduce((a, x, j) => a + x * w[j], 0));
      const rhs = res.SW.map((r) => r.reduce((a, x, j) => a + x * w[j], 0) * res.eigenvalues[t]);
      const scaleRef = Math.max(...lhs.map(Math.abs), 1e-12);
      expect(Math.max(...lhs.map((x, i) => Math.abs(x - rhs[i]))) / scaleRef).toBeLessThan(1e-7);
      // Fisher criterion of w equals the eigenvalue
      expect(fisherCriterion(res.SB, res.SW, w).J).toBeCloseTo(res.eigenvalues[t], 7);
    }
    for (let t = 1; t < res.maxDims; t++) expect(res.eigenvalues[t]).toBeLessThanOrEqual(res.eigenvalues[t - 1] + 1e-12);
  });
  it('the Fisher direction beats every other direction and PC1 on the PCA-vs-LDA dataset', () => {
    const d2 = generateDataset('pcaVsLda', { ...defaultParams, n: 200 });
    const r2 = lda(d2.X, d2.y!);
    const p2 = pca(d2.X, 'center');
    const w = r2.W.map((r) => r[0]);
    const v1 = p2.V.map((r) => r[0]);
    const Jw = fisherCriterion(r2.SB, r2.SW, w).J;
    const Jv = fisherCriterion(r2.SB, r2.SW, v1).J;
    expect(Jw).toBeGreaterThan(Jv * 5);
    for (let a = 0; a < Math.PI; a += 0.05) {
      const J = fisherCriterion(r2.SB, r2.SW, [Math.cos(a), Math.sin(a)]).J;
      expect(J).toBeLessThanOrEqual(Jw + 1e-9);
    }
    // LDA classifier is accurate here
    const cls = ldaClassify(r2, d2.X);
    expect(cls.accuracy!(d2.y!)).toBeGreaterThan(0.9);
  });
  it('flags singular S_W when p > n and recovers with regularisation', () => {
    const wide = generateDataset('pGreaterN', { ...defaultParams, n: 10, p: 16, K: 2 });
    const r = lda(wide.X, wide.y!);
    expect(r.swSingular).toBe(true);
    expect(r.swCondition).toBe(Infinity);
    const rr = lda(wide.X, wide.y!, { regularization: 0.1 });
    expect(rr.maxDims).toBe(1);
    expect(Number.isFinite(rr.eigenvalues[0])).toBe(true);
  });
  it('is invariant to rescaling a variable (scores change only by a constant factor)', () => {
    const d = generateDataset('scales', { ...defaultParams, n: 100, p: 3, scaleFactor: 100 });
    const a = lda(d.X, d.y!);
    const Xs = d.X.map((r) => r.map((x, j) => (j === 0 ? x / 100 : x)));
    const b = lda(Xs, d.y!);
    const za = a.scores.map((r) => r[0]);
    const zb = b.scores.map((r) => r[0]);
    const ratio = za[0] / zb[0];
    for (let i = 0; i < za.length; i++) expect(za[i] / zb[i]).toBeCloseTo(ratio, 6);
    expect(a.eigenvalues[0]).toBeCloseTo(b.eigenvalues[0], 8);
  });
});

describe('Datasets', () => {
  it('every generator produces finite data of the right shape and is reproducible', () => {
    for (const d of datasetCatalog) {
      const a = generateDataset(d.kind, defaultParams);
      const b = generateDataset(d.kind, defaultParams);
      expect(a.X.length).toBeGreaterThan(3);
      expect(a.X.every((r) => r.length === a.variableNames.length && r.every(Number.isFinite))).toBe(true);
      expect(JSON.stringify(a.X)).toBe(JSON.stringify(b.X));
      if (d.supervised) {
        expect(a.y).toBeDefined();
        expect(a.y!.length).toBe(a.X.length);
        expect(new Set(a.y).size).toBeGreaterThanOrEqual(2);
      }
      const c = generateDataset(d.kind, { ...defaultParams, seed: 99 });
      expect(JSON.stringify(c.X)).not.toBe(JSON.stringify(a.X));
    }
  });
});

describe('Householder + QL eigensolver', () => {
  it('agrees with Jacobi on random symmetric matrices of various sizes', async () => {
    const { symmetricEigenQL } = await import('../src/lib/linalg');
    for (const n of [2, 3, 5, 9, 20, 45]) {
      const M = randomMatrix(n + 2, n);
      const A = matmul(transpose(M), M);
      const a = symmetricEigen(A);
      const b = symmetricEigenQL(A);
      for (let i = 0; i < n; i++) expect(b.values[i]).toBeCloseTo(a.values[i], 8);
      const recon = matmul(matmul(b.vectors, diag(b.values)), transpose(b.vectors));
      expect(maxAbsDiff(recon, A)).toBeLessThan(1e-8 * Math.max(1, Math.abs(a.values[0])));
      expect(maxAbsDiff(matmul(transpose(b.vectors), b.vectors), identity(n))).toBeLessThan(1e-9);
    }
    // indefinite matrix with negative eigenvalues (as in non-Euclidean MDS)
    const B = randomMatrix(30, 30);
    const Bs = B.map((r, i) => r.map((x, j) => 0.5 * (x + B[j][i])));
    const a = symmetricEigen(Bs);
    const b = symmetricEigenQL(Bs);
    for (let i = 0; i < 30; i++) expect(b.values[i]).toBeCloseTo(a.values[i], 8);
  });
});
