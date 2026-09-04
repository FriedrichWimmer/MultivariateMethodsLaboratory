/**
 * Dataset generators for the laboratory. Every generator is deterministic
 * given its seed and returns a `Dataset` with the raw data matrix X (n × p),
 * optional class labels y ∈ {0,…,K−1}, and metadata for the interpretation panels.
 */
import { makeRNG, type RNG } from './random';
import { cholesky, symmetricEigen, matvec, type Matrix, type Vector } from './linalg';

export interface Dataset {
  id: string;
  name: string;
  X: Matrix;
  y?: number[];
  classNames?: string[];
  variableNames: string[];
  description: string;
  /** which pedagogical points the dataset is designed to make */
  notes: string[];
  /** true covariance used for simulation (if applicable) */
  trueCov?: Matrix;
  trueClassMeans?: Matrix;
}

export type DatasetKind =
  | 'gaussian2d'
  | 'correlated'
  | 'clusters'
  | 'iris'
  | 'separated'
  | 'overlapping'
  | 'scales'
  | 'outliers'
  | 'pcaVsLda'
  | 'pGreaterN'
  | 'manifold'
  | 'unequalCov'
  | 'imbalanced'
  | 'collinear'
  | 'uploaded';

export interface DatasetParams {
  n: number;
  p: number;
  K: number;
  correlation: number; // in (−1, 1)
  variance: number; // overall variance scale
  noise: number; // isotropic noise sd added to every variable
  separation: number; // distance between class means (in sd units)
  classProportions: number[]; // length K, sums to 1
  seed: number;
  scaleFactor: number; // multiplier for the "different scales" dataset
  outlierCount: number;
  outlierMagnitude: number;
}

export const defaultParams: DatasetParams = {
  n: 150,
  p: 4,
  K: 3,
  correlation: 0.7,
  variance: 1,
  noise: 0.15,
  separation: 3,
  classProportions: [1 / 3, 1 / 3, 1 / 3],
  seed: 42,
  scaleFactor: 100,
  outlierCount: 5,
  outlierMagnitude: 8,
};

export interface DatasetDescriptor {
  kind: DatasetKind;
  label: string;
  short: string;
  supervised: boolean;
  /** which params are meaningful for this generator */
  params: (keyof DatasetParams)[];
}

export const datasetCatalog: DatasetDescriptor[] = [
  { kind: 'gaussian2d', label: '2-D Gaussian cloud', short: 'One bivariate normal cloud', supervised: false, params: ['n', 'correlation', 'variance', 'noise', 'seed'] },
  { kind: 'correlated', label: 'Correlated Gaussian variables', short: 'p variables with a common correlation structure', supervised: false, params: ['n', 'p', 'correlation', 'variance', 'noise', 'seed'] },
  { kind: 'clusters', label: 'Multiple clusters', short: 'K spherical clusters with labels', supervised: true, params: ['n', 'p', 'K', 'separation', 'variance', 'noise', 'classProportions', 'seed'] },
  { kind: 'iris', label: 'Iris-like (4 variables, 3 species)', short: 'Simulated from Iris class means and covariances', supervised: true, params: ['n', 'noise', 'seed'] },
  { kind: 'separated', label: 'Clearly separated classes', short: 'Classes far apart relative to spread', supervised: true, params: ['n', 'p', 'K', 'separation', 'correlation', 'classProportions', 'seed'] },
  { kind: 'overlapping', label: 'Overlapping classes', short: 'Class means close relative to spread', supervised: true, params: ['n', 'p', 'K', 'separation', 'correlation', 'classProportions', 'seed'] },
  { kind: 'scales', label: 'Different variable scales', short: 'One variable multiplied by a large factor', supervised: true, params: ['n', 'p', 'correlation', 'scaleFactor', 'seed'] },
  { kind: 'outliers', label: 'Noise and outliers', short: 'Gaussian cloud plus a few extreme points', supervised: false, params: ['n', 'p', 'correlation', 'outlierCount', 'outlierMagnitude', 'seed'] },
  { kind: 'pcaVsLda', label: 'PCA vs LDA (high variance ≠ separation)', short: 'Classes elongated along one axis but separated along another', supervised: true, params: ['n', 'separation', 'variance', 'seed'] },
  { kind: 'pGreaterN', label: 'p > n (more variables than observations)', short: 'Wide matrix: rank-deficient covariance', supervised: true, params: ['n', 'p', 'K', 'separation', 'seed'] },
  { kind: 'manifold', label: 'Nonlinear manifold (curved cloud)', short: 'Points on an arc / S-curve in 3-D', supervised: false, params: ['n', 'noise', 'seed'] },
  { kind: 'unequalCov', label: 'Unequal class covariances', short: 'Classes with very different spread and orientation', supervised: true, params: ['n', 'separation', 'seed'] },
  { kind: 'imbalanced', label: 'Highly imbalanced classes', short: 'One class dominates the sample', supervised: true, params: ['n', 'separation', 'classProportions', 'seed'] },
  { kind: 'collinear', label: 'Highly correlated (near-collinear) variables', short: 'Variables that are almost linear combinations of each other', supervised: false, params: ['n', 'p', 'noise', 'seed'] },
];

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/** Multivariate normal sampler via Cholesky (falls back to eigen square root). */
export function mvnormSampler(cov: Matrix): (rng: RNG) => Vector {
  const p = cov.length;
  let L = cholesky(cov);
  if (!L) {
    // eigen square root for semi-definite covariance
    const eig = symmetricEigen(cov);
    L = eig.vectors.map((row) => row.map((v, j) => v * Math.sqrt(Math.max(eig.values[j], 0))));
  }
  const Lf = L;
  return (rng: RNG) => {
    const z = Array.from({ length: p }, () => rng.normal());
    return matvec(Lf, z);
  };
}

export function equicorrelation(p: number, rho: number, variance = 1): Matrix {
  const S: Matrix = [];
  for (let i = 0; i < p; i++) {
    S.push([]);
    for (let j = 0; j < p; j++) S[i].push(i === j ? variance : rho * variance);
  }
  return S;
}

/** AR(1)-style correlation: corr(i,j) = ρ^{|i−j|}. */
export function ar1Correlation(p: number, rho: number, variance = 1): Matrix {
  const S: Matrix = [];
  for (let i = 0; i < p; i++) {
    S.push([]);
    for (let j = 0; j < p; j++) S[i].push(variance * Math.pow(rho, Math.abs(i - j)));
  }
  return S;
}

function classSizes(n: number, proportions: number[]): number[] {
  const K = proportions.length;
  const total = proportions.reduce((a, b) => a + b, 0) || 1;
  const sizes = proportions.map((q) => Math.floor((n * q) / total));
  let rem = n - sizes.reduce((a, b) => a + b, 0);
  let i = 0;
  while (rem > 0) {
    sizes[i % K]++;
    rem--;
    i++;
  }
  // guarantee at least 2 per class if possible
  for (let k = 0; k < K; k++) {
    while (sizes[k] < 2 && n >= 2 * K) {
      const big = sizes.indexOf(Math.max(...sizes));
      sizes[big]--;
      sizes[k]++;
    }
  }
  return sizes;
}

function varNames(p: number, prefix = 'x'): string[] {
  return Array.from({ length: p }, (_, j) => `${prefix}${j + 1}`);
}

function classNames(K: number): string[] {
  return Array.from({ length: K }, (_, k) => `Class ${String.fromCharCode(65 + k)}`);
}

/** Add isotropic Gaussian noise of standard deviation sd. */
function addNoise(X: Matrix, sd: number, rng: RNG): Matrix {
  if (sd <= 0) return X;
  return X.map((r) => r.map((x) => x + sd * rng.normal()));
}

/** Class means placed on a regular simplex-like configuration scaled by `sep`. */
function simplexMeans(K: number, p: number, sep: number): Matrix {
  const means: Matrix = [];
  for (let k = 0; k < K; k++) {
    const m = new Array(p).fill(0);
    if (p === 1) m[0] = (k - (K - 1) / 2) * sep;
    else {
      const ang = (2 * Math.PI * k) / K;
      m[0] = (sep / 2) * Math.cos(ang) * (K === 2 ? 2 : 1.3);
      m[1] = (sep / 2) * Math.sin(ang) * (K === 2 ? 2 : 1.3);
      if (K === 2) {
        m[0] = (k === 0 ? -1 : 1) * (sep / 2);
        m[1] = 0;
      }
      for (let j = 2; j < p; j++) m[j] = 0.25 * sep * ((k % (j + 1)) / Math.max(1, j) - 0.5);
    }
    means.push(m);
  }
  return means;
}

function labelled(X: Matrix, means: Matrix, sizes: number[], covs: Matrix[], rng: RNG, noise = 0): { X: Matrix; y: number[] } {
  const rows: Matrix = [];
  const y: number[] = [];
  means.forEach((m, k) => {
    const sampler = mvnormSampler(covs[k] ?? covs[0]);
    for (let i = 0; i < sizes[k]; i++) {
      const z = sampler(rng);
      rows.push(m.map((mu, j) => mu + z[j] + (noise > 0 ? noise * rng.normal() : 0)));
      y.push(k);
    }
  });
  void X;
  return { X: rows, y };
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

export function generateDataset(kind: DatasetKind, params: DatasetParams): Dataset {
  const rng = makeRNG(params.seed);
  const n = Math.max(4, Math.round(params.n));
  const p = Math.max(1, Math.round(params.p));
  const K = Math.max(2, Math.round(params.K));
  const rho = Math.max(-0.99, Math.min(0.99, params.correlation));
  const v = Math.max(1e-6, params.variance);

  switch (kind) {
    case 'gaussian2d': {
      const cov = [
        [v, rho * v],
        [rho * v, v],
      ];
      const sampler = mvnormSampler(cov);
      let X: Matrix = Array.from({ length: n }, () => sampler(rng));
      X = addNoise(X, params.noise, rng);
      return {
        id: 'gaussian2d',
        name: '2-D Gaussian cloud',
        X,
        variableNames: ['x1', 'x2'],
        description: `n = ${n} observations from a bivariate normal with variance ${v} and correlation ρ = ${rho}.`,
        notes: [
          'The principal axes of the ellipse are the eigenvectors of the covariance matrix.',
          'With ρ → 0 the eigenvalues coincide and the first principal direction becomes arbitrary (eigenvalue multiplicity).',
        ],
        trueCov: cov,
      };
    }
    case 'correlated': {
      const cov = equicorrelation(p, rho, v);
      const sampler = mvnormSampler(cov);
      let X: Matrix = Array.from({ length: n }, () => sampler(rng));
      X = addNoise(X, params.noise, rng);
      return {
        id: 'correlated',
        name: 'Correlated Gaussian variables',
        X,
        variableNames: varNames(p),
        description: `n = ${n}, p = ${p}. Equicorrelation structure: every pair of variables has correlation ρ = ${rho}.`,
        notes: [
          `Under equicorrelation the largest eigenvalue is 1 + (p−1)ρ = ${(1 + (p - 1) * rho).toFixed(2)} (times the variance) with eigenvector ∝ (1,…,1); the other p−1 eigenvalues all equal 1−ρ.`,
          'Repeated eigenvalues mean the trailing principal directions are not uniquely determined.',
        ],
        trueCov: cov,
      };
    }
    case 'clusters':
    case 'separated':
    case 'overlapping': {
      const sep = kind === 'overlapping' ? Math.min(params.separation, 1.2) : kind === 'separated' ? Math.max(params.separation, 4) : params.separation;
      const props = params.classProportions.length === K ? params.classProportions : new Array(K).fill(1 / K);
      const sizes = classSizes(n, props);
      const means = simplexMeans(K, p, sep);
      const cov = kind === 'clusters' ? equicorrelation(p, 0, v) : ar1Correlation(p, rho, v);
      const { X, y } = labelled([], means, sizes, [cov], rng, kind === 'clusters' ? params.noise : 0);
      const perm = rng.shuffle(X.map((_, i) => i));
      return {
        id: kind,
        name: kind === 'clusters' ? 'Multiple clusters' : kind === 'separated' ? 'Clearly separated classes' : 'Overlapping classes',
        X: perm.map((i) => X[i]),
        y: perm.map((i) => y[i]),
        classNames: classNames(K),
        variableNames: varNames(p),
        description: `n = ${n}, p = ${p}, K = ${K} classes with common covariance; class means separated by about ${sep.toFixed(1)} standard deviations.`,
        notes:
          kind === 'overlapping'
            ? ['When class means are close relative to within-class spread, no linear projection can separate the classes well — LDA finds the best direction but the Fisher criterion stays small.']
            : ['With equal covariance matrices across classes the classical LDA assumptions hold; the between-class scatter S_B has rank at most K−1.'],
        trueCov: cov,
        trueClassMeans: means,
      };
    }
    case 'iris': {
      // Class means and within-class covariance matrices approximating Fisher's Iris data
      const means = [
        [5.006, 3.428, 1.462, 0.246],
        [5.936, 2.77, 4.26, 1.326],
        [6.588, 2.974, 5.552, 2.026],
      ];
      const covs: Matrix[] = [
        [
          [0.1242, 0.0992, 0.0164, 0.0103],
          [0.0992, 0.1437, 0.0117, 0.0093],
          [0.0164, 0.0117, 0.0302, 0.0061],
          [0.0103, 0.0093, 0.0061, 0.0111],
        ],
        [
          [0.2664, 0.0852, 0.183, 0.0558],
          [0.0852, 0.0985, 0.0827, 0.0412],
          [0.183, 0.0827, 0.2208, 0.0731],
          [0.0558, 0.0412, 0.0731, 0.0391],
        ],
        [
          [0.4043, 0.0938, 0.3033, 0.0491],
          [0.0938, 0.104, 0.0714, 0.0476],
          [0.3033, 0.0714, 0.3046, 0.0488],
          [0.0491, 0.0476, 0.0488, 0.0754],
        ],
      ];
      const sizes = classSizes(n, [1 / 3, 1 / 3, 1 / 3]);
      const { X, y } = labelled([], means, sizes, covs, rng, params.noise * 0.3);
      const perm = rng.shuffle(X.map((_, i) => i));
      return {
        id: 'iris',
        name: 'Iris-like flowers',
        X: perm.map((i) => X[i].map((x) => Math.round(x * 10) / 10)),
        y: perm.map((i) => y[i]),
        classNames: ['setosa', 'versicolor', 'virginica'],
        variableNames: ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'],
        description: `n = ${n} simulated flowers from three species, using the class means and within-class covariance matrices of Fisher's Iris data (values in cm, rounded to 0.1).`,
        notes: [
          'Setosa is linearly separable from the other two species; versicolor and virginica overlap slightly.',
          'Variables are on comparable scales, so covariance-PCA and correlation-PCA give similar, but not identical, answers.',
          'The class covariance matrices differ (setosa is much tighter) — a mild violation of the equal-covariance assumption of classical LDA.',
        ],
        trueClassMeans: means,
      };
    }
    case 'scales': {
      const pp = Math.max(2, p);
      const cov = ar1Correlation(pp, rho, 1);
      const sampler = mvnormSampler(cov);
      const sizes = classSizes(n, [0.5, 0.5]);
      const X: Matrix = [];
      const y: number[] = [];
      for (let k = 0; k < 2; k++)
        for (let i = 0; i < sizes[k]; i++) {
          const z = sampler(rng);
          z[1] += (k === 0 ? -1 : 1) * 1.2; // classes separated along x2 (the small-scale variable)
          X.push(z);
          y.push(k);
        }
      const factor = Math.max(1, params.scaleFactor);
      const Xs = X.map((r) => r.map((x, j) => (j === 0 ? x * factor : x)));
      const names = varNames(pp);
      names[0] = `x1 (×${factor})`;
      return {
        id: 'scales',
        name: 'Different variable scales',
        X: Xs,
        y,
        classNames: classNames(2),
        variableNames: names,
        description: `n = ${n}, p = ${pp}. Variable x1 has been multiplied by ${factor}, e.g. measuring it in grams instead of kilograms. Two classes differ along x2 only.`,
        notes: [
          'On the covariance matrix the first principal component is dominated by x1 simply because of its units.',
          'On the correlation matrix (standardised variables) every variable contributes on an equal footing.',
          'LDA is invariant to rescaling of individual variables: rescaling changes w but not the projected scores z = Xw (up to scale).',
        ],
      };
    }
    case 'outliers': {
      const pp = Math.max(2, p);
      const cov = ar1Correlation(pp, rho, 1);
      const sampler = mvnormSampler(cov);
      const X: Matrix = Array.from({ length: n }, () => sampler(rng));
      const m = Math.min(Math.max(0, Math.round(params.outlierCount)), Math.floor(n / 2));
      for (let i = 0; i < m; i++) {
        // outliers placed off the main axis of correlation
        const dir = new Array(pp).fill(0);
        dir[0] = 1;
        dir[1] = -1;
        const sign = i % 2 === 0 ? 1 : -1;
        X[i] = dir.map((d, j) => sign * d * params.outlierMagnitude + 0.3 * rng.normal() + (j > 1 ? 0.5 * rng.normal() : 0));
      }
      return {
        id: 'outliers',
        name: 'Noise and outliers',
        X,
        variableNames: varNames(pp),
        description: `n = ${n}, p = ${pp}: a correlated Gaussian cloud in which the first ${m} observations were replaced by outliers of magnitude ≈ ${params.outlierMagnitude}.`,
        notes: [
          'PCA and classical MDS are least-squares methods: a handful of extreme points can rotate the leading principal direction toward the outliers.',
          'Robust alternatives (e.g. PCA on a robust covariance estimate) are needed when outliers are present.',
        ],
      };
    }
    case 'pcaVsLda': {
      // Two classes: large variance along x1, small variance along x2, separation along x2.
      const sd1 = Math.sqrt(v) * 3.0;
      const sd2 = Math.sqrt(v) * 0.45;
      const sep = params.separation * sd2 * 1.0;
      const sizes = classSizes(n, [0.5, 0.5]);
      const X: Matrix = [];
      const y: number[] = [];
      // slight tilt so that the PCA direction is not exactly the x1 axis
      const tilt = 0.18;
      for (let k = 0; k < 2; k++)
        for (let i = 0; i < sizes[k]; i++) {
          const a = sd1 * rng.normal();
          const b = sd2 * rng.normal() + (k === 0 ? -sep / 2 : sep / 2);
          X.push([a * Math.cos(tilt) - b * Math.sin(tilt), a * Math.sin(tilt) + b * Math.cos(tilt)]);
          y.push(k);
        }
      const perm = rng.shuffle(X.map((_, i) => i));
      return {
        id: 'pcaVsLda',
        name: 'PCA vs LDA: variance is not separation',
        X: perm.map((i) => X[i]),
        y: perm.map((i) => y[i]),
        classNames: classNames(2),
        variableNames: ['x1', 'x2'],
        description: `n = ${n}. Both classes are strongly elongated along one direction (sd ≈ ${sd1.toFixed(2)}) and thin in the orthogonal direction (sd ≈ ${sd2.toFixed(2)}); the class means differ only along the thin direction.`,
        notes: [
          'PC1 follows the elongated, high-variance direction and projects the two classes on top of each other.',
          'The Fisher discriminant direction is (almost) orthogonal to PC1: low total variance but large between-class separation relative to within-class spread.',
        ],
      };
    }
    case 'pGreaterN': {
      const nn = Math.min(n, 12);
      const pp = Math.max(p, nn + 4);
      const sizes = classSizes(nn, new Array(K).fill(1 / K));
      const means = simplexMeans(K, pp, params.separation);
      const cov = equicorrelation(pp, 0.2, 1);
      const { X, y } = labelled([], means, sizes, [cov], rng);
      return {
        id: 'pGreaterN',
        name: 'p > n: wide data matrix',
        X,
        y,
        classNames: classNames(K),
        variableNames: varNames(pp),
        description: `n = ${nn} observations but p = ${pp} variables. The covariance matrix (p × p) has rank at most n − 1 = ${nn - 1}.`,
        notes: [
          'At most n − 1 eigenvalues of S are non-zero: the data lie in an (n−1)-dimensional affine subspace of ℝᵖ.',
          'The within-class scatter matrix S_W is singular (rank ≤ n − K), so the classical LDA problem is ill-posed without regularisation.',
          'The thin SVD of X_c has only min(n, p) = n singular values — it is far cheaper than a p × p eigendecomposition.',
        ],
      };
    }
    case 'manifold': {
      // points on an S-curve in 3-D
      const X: Matrix = [];
      const t: number[] = [];
      for (let i = 0; i < n; i++) {
        const u = 3 * Math.PI * (rng.uniform() - 0.5);
        const h = 2 * (rng.uniform() - 0.5);
        const x = Math.sin(u);
        const z = Math.sign(u) * (Math.cos(u) - 1);
        X.push([x + params.noise * rng.normal(), h + params.noise * rng.normal(), z + params.noise * rng.normal()]);
        t.push(u);
      }
      // colour groups by position along the curve to visualise unfolding
      const y = t.map((u) => (u < -Math.PI / 2 ? 0 : u < Math.PI / 2 ? 1 : 2));
      return {
        id: 'manifold',
        name: 'Nonlinear manifold (S-curve)',
        X,
        y,
        classNames: ['start of curve', 'middle', 'end of curve'],
        variableNames: ['x', 'y', 'z'],
        description: `n = ${n} points on a two-dimensional S-shaped sheet embedded in ℝ³ with noise sd ${params.noise}. Colour marks position along the curve (not real classes).`,
        notes: [
          'The intrinsic dimension is 2 but the sheet is curved, so no linear projection (PCA/classical MDS) unrolls it.',
          'Euclidean distances between distant points on the sheet are much shorter than geodesic distances — classical MDS with Euclidean input inherits the same limitation as PCA.',
        ],
      };
    }
    case 'unequalCov': {
      const sizes = classSizes(n, [0.5, 0.5]);
      const sep = params.separation;
      const covA = [
        [0.15, 0],
        [0, 0.15],
      ];
      const covB = [
        [3.0, 2.4],
        [2.4, 3.0],
      ];
      const { X, y } = labelled([], [[-sep / 2, 0], [sep / 2, 0]], sizes, [covA, covB], rng);
      const perm = rng.shuffle(X.map((_, i) => i));
      return {
        id: 'unequalCov',
        name: 'Unequal class covariances',
        X: perm.map((i) => X[i]),
        y: perm.map((i) => y[i]),
        classNames: classNames(2),
        variableNames: ['x1', 'x2'],
        description: `n = ${n}. Class A is a tight spherical cloud; class B is a wide, strongly correlated cloud. Classical LDA pools them into a single within-class covariance.`,
        notes: [
          'The pooled S_W is a compromise between two very different covariance matrices; the linear boundary is suboptimal.',
          'Quadratic discriminant analysis (QDA) allows class-specific covariances at the price of many more parameters.',
        ],
      };
    }
    case 'imbalanced': {
      const props = params.classProportions.length === 2 && params.classProportions[0] !== params.classProportions[1] ? params.classProportions : [0.92, 0.08];
      const sizes = classSizes(n, props);
      const cov = ar1Correlation(2, 0.5, 1);
      const { X, y } = labelled([], [[0, 0], [params.separation, params.separation * 0.4]], sizes, [cov], rng);
      const perm = rng.shuffle(X.map((_, i) => i));
      return {
        id: 'imbalanced',
        name: 'Highly imbalanced classes',
        X: perm.map((i) => X[i]),
        y: perm.map((i) => y[i]),
        classNames: ['Majority', 'Minority'],
        variableNames: ['x1', 'x2'],
        description: `n = ${n}: ${sizes[0]} observations in the majority class and ${sizes[1]} in the minority class.`,
        notes: [
          'S_B weights each class mean by its sample size, so the grand mean sits almost on the majority mean; the small class barely influences S_W.',
          'The Fisher direction is still well defined, but the minority mean is estimated from very few points and is highly variable.',
        ],
      };
    }
    case 'collinear': {
      const pp = Math.max(3, p);
      const X: Matrix = [];
      for (let i = 0; i < n; i++) {
        const a = rng.normal();
        const b = rng.normal();
        const row = [a, b];
        for (let j = 2; j < pp; j++) {
          // near-linear combinations of the first two
          const w1 = 0.6 + 0.3 * Math.cos(j);
          const w2 = 0.5 * Math.sin(j) + 0.4;
          row.push(w1 * a + w2 * b + Math.max(params.noise, 0) * 0.2 * rng.normal());
        }
        X.push(row);
      }
      return {
        id: 'collinear',
        name: 'Near-collinear variables',
        X,
        variableNames: varNames(pp),
        description: `n = ${n}, p = ${pp}: variables x3…x${pp} are (almost) linear combinations of x1 and x2 with tiny noise (sd ${(params.noise * 0.2).toFixed(3)}).`,
        notes: [
          'The covariance matrix is nearly singular: only two eigenvalues are appreciably non-zero.',
          'The condition number σ₁/σₚ explodes; explicitly forming and inverting S (as in regression or LDA) amplifies rounding error.',
        ],
      };
    }
    case 'uploaded':
    default:
      throw new Error(`generateDataset: unsupported kind ${kind}`);
  }
}

// ---------------------------------------------------------------------------
// CSV parsing for uploaded data
// ---------------------------------------------------------------------------

export function parseCSV(text: string, name = 'Uploaded data'): Dataset {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith('#'));
  if (lines.length < 3) throw new Error('Need at least a header and two data rows.');
  const delim = lines[0].includes('\t') ? '\t' : lines[0].includes(';') ? ';' : ',';
  const header = lines[0].split(delim).map((h) => h.trim().replace(/^"|"$/g, ''));
  const rows = lines.slice(1).map((l) => l.split(delim).map((c) => c.trim().replace(/^"|"$/g, '')));
  const ncol = header.length;
  const numeric: boolean[] = new Array(ncol).fill(true);
  for (const r of rows) for (let j = 0; j < ncol; j++) if (r[j] === undefined || r[j] === '' || Number.isNaN(Number(r[j]))) numeric[j] = false;
  const numCols = header.map((_, j) => j).filter((j) => numeric[j]);
  const labelCols = header.map((_, j) => j).filter((j) => !numeric[j]);
  if (numCols.length < 2) throw new Error('Need at least two numeric columns.');
  const X = rows.map((r) => numCols.map((j) => Number(r[j])));
  let y: number[] | undefined;
  let classNames: string[] | undefined;
  if (labelCols.length > 0) {
    const lc = labelCols[labelCols.length - 1];
    const labels = rows.map((r) => r[lc]);
    const uniq = Array.from(new Set(labels));
    if (uniq.length >= 2 && uniq.length <= 12) {
      classNames = uniq;
      y = labels.map((l) => uniq.indexOf(l));
    }
  }
  return {
    id: 'uploaded',
    name,
    X,
    y,
    classNames,
    variableNames: numCols.map((j) => header[j]),
    description: `${X.length} rows × ${numCols.length} numeric columns from ${name}${classNames ? `; labels from column "${header[labelCols[labelCols.length - 1]]}" (${classNames.length} classes)` : ''}.`,
    notes: ['Uploaded data: check the scales of the variables before interpreting a covariance-based PCA.'],
  };
}
