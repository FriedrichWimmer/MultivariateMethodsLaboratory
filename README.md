# Multivariate Methods Laboratory — SVD · PCA · MDS · LDA

An interactive, Master's-level teaching dashboard for four dimensionality-reduction methods:
singular value decomposition, principal component analysis, classical multidimensional scaling
and linear discriminant analysis. Every quantity shown is computed live in the browser from the
data the student generates or uploads — nothing is precomputed or illustrative only.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # numerical verification suite (vitest)
npm run build      # typecheck + production bundle in dist/
```

## Structure

| Path | Content |
|---|---|
| `src/lib/linalg.ts` | Dense linear algebra: cyclic Jacobi eigensolver, Householder + QL eigensolver, one-sided Jacobi SVD, Cholesky, inverse, determinant, orthonormal completion |
| `src/lib/pca.ts` | PCA through the SVD of the centred / standardised matrix; reconstruction; projection variance |
| `src/lib/mds.ts` | Distance matrices (L1, L2, L∞, L3, squared L2), double centring, classical MDS, stress, Procrustes alignment |
| `src/lib/lda.ts` | Scatter matrices S_W, S_B, S_T; generalised eigenproblem by symmetric whitening; regularisation; Fisher criterion; Gaussian LDA classifier |
| `src/lib/svdlab.ts` | 2-D geometric decomposition, low-rank summaries, Eckart–Young checks |
| `src/lib/datasets.ts` | Seeded generators (Gaussian clouds, clusters, Iris-like, scale differences, outliers, PCA-vs-LDA, p > n, manifold, unequal covariance, imbalance, collinearity) and CSV parsing |
| `src/state/store.tsx` | Global dataset, preprocessing (centre / standardise, metric, k) and the shared analysis used by every tab |
| `src/components/common/` | KaTeX math, Plotly wrapper, matrix display, interactive SVG scatter with draggable directions, controls, interpretation and derivation panels |
| `src/components/tabs/` | The thirteen laboratories (concept map, data, SVD, PCA, MDS, LDA, four-way comparison, unified view + derivations, diagnostics, failure modes, assessment, experiment, takeaway) |
| `tests/methods.test.ts` | Verification of the quality-control requirements (see below) |
| `docs/ui-api.md` | Component and library API used by the tab modules |

## Numerical quality control

`npm test` verifies, among other things, that

- the SVD reconstructs X = UΣVᵀ, orders singular values, returns orthonormal U and V, and handles rank-deficient and zero matrices;
- truncation error equals √(Σ_{j>k} σ_j²) (Eckart–Young) and beats random rank-k projections;
- PCA eigenvalues equal σ_j²/(n−1) from the SVD route and agree with the Jacobi eigendecomposition of S; explained-variance ratios sum to one; scores are uncorrelated with variances λ_j; reconstruction with all components is exact;
- classical MDS uses B = −½JD⁽²⁾J, recovers B = X_cX_cᵀ for Euclidean input, reproduces the PCA scores up to a rigid motion, and exposes negative eigenvalues for non-Euclidean metrics;
- LDA builds S_W and S_B with S_W + S_B = (n−1)S, returns at most K−1 directions satisfying S_Bw = λS_Ww, is invariant to variable rescaling, flags singular S_W for p > n and recovers with regularisation;
- every dataset generator is finite, correctly shaped and reproducible from its seed.
