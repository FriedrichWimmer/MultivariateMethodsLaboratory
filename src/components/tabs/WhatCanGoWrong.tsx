import { useMemo, useState, type ReactNode } from 'react';
import type { Data } from 'plotly.js';
import { generateDataset, defaultParams, type DatasetKind, type DatasetParams } from '../../lib/datasets';
import { pca } from '../../lib/pca';
import { distanceMatrix, classicalMDS } from '../../lib/mds';
import { lda, ldaClassify, fisherCriterion, type LDAResult } from '../../lib/lda';
import { conditionNumber, covariance, fmt, column, dot, colStds } from '../../lib/linalg';
import { Section, Card, Callout, Interpretation, StatTile, Badge, ClassLegend } from '../common/Panels';
import { ScatterSVG } from '../common/ScatterSVG';
import { Plot } from '../common/Plot';
import { Slider, Select, Button } from '../common/Controls';
import { M, MBlock } from '../common/Math';
import { categorical, classColor, methodColor, ink } from '../../lib/theme';
import { qdaResubstitution, perClassRecall } from './diag/helpers';

interface Control {
  key: keyof DatasetParams;
  label: string;
  min: number;
  max: number;
  step: number;
}
interface Scenario {
  id: string;
  kind: DatasetKind;
  title: string;
  short: string;
  base: Partial<DatasetParams>;
  controls: Control[];
}

const scenarios: Scenario[] = [
  { id: 'outliers', kind: 'outliers', title: '1 · Strong outliers', short: 'A few extreme points in a correlated cloud', base: { n: 120, p: 2, correlation: 0.8, outlierCount: 4, outlierMagnitude: 8 }, controls: [{ key: 'outlierCount', label: 'Number of outliers', min: 0, max: 15, step: 1 }, { key: 'outlierMagnitude', label: 'Outlier magnitude', min: 1, max: 20, step: 0.5 }] },
  { id: 'collinear', kind: 'collinear', title: '2 · Highly correlated variables', short: 'Variables that are almost linear combinations of two others', base: { n: 120, p: 5, noise: 0.1 }, controls: [{ key: 'noise', label: 'Noise around exact collinearity', min: 0, max: 2, step: 0.02 }, { key: 'p', label: 'Variables p', min: 3, max: 8, step: 1 }] },
  { id: 'scales', kind: 'scales', title: '3 · Radically different scales', short: 'One variable measured in tiny units', base: { n: 120, p: 3, correlation: 0.5, scaleFactor: 100 }, controls: [{ key: 'scaleFactor', label: 'Scale factor for x₁', min: 1, max: 1000, step: 1 }] },
  { id: 'pGreaterN', kind: 'pGreaterN', title: '4 · More variables than observations (p > n)', short: 'Wide matrix, rank-deficient covariance', base: { n: 10, p: 20, K: 2, separation: 3 }, controls: [{ key: 'n', label: 'Observations n', min: 6, max: 12, step: 1 }, { key: 'p', label: 'Variables p', min: 14, max: 40, step: 1 }] },
  { id: 'manifold', kind: 'manifold', title: '5 · Nonlinear manifold', short: 'A curved two-dimensional sheet in three dimensions', base: { n: 200, noise: 0.05 }, controls: [{ key: 'noise', label: 'Noise sd', min: 0, max: 0.5, step: 0.01 }] },
  { id: 'overlapping', kind: 'overlapping', title: '6 · Overlapping classes', short: 'Class means close relative to the spread', base: { n: 150, p: 2, K: 3, separation: 1.0, correlation: 0.3 }, controls: [{ key: 'separation', label: 'Class separation', min: 0, max: 1.2, step: 0.05 }] },
  { id: 'unequalCov', kind: 'unequalCov', title: '7 · Unequal class covariances', short: 'A tight class next to a wide, tilted one', base: { n: 200, separation: 3 }, controls: [{ key: 'separation', label: 'Distance between class means', min: 0, max: 6, step: 0.25 }] },
  { id: 'imbalanced', kind: 'imbalanced', title: '8 · Highly imbalanced classes', short: 'One class dominates the sample', base: { n: 200, separation: 2.5, classProportions: [0.92, 0.08] }, controls: [{ key: 'separation', label: 'Class separation', min: 0, max: 6, step: 0.25 }] },
];

interface Ctx {
  sc: Scenario;
  params: DatasetParams;
  ds: ReturnType<typeof generateDataset>;
  pcaRes: ReturnType<typeof pca>;
  mdsRes: ReturnType<typeof classicalMDS>;
  ldaRes: LDAResult | null;
  ldaError: string | null;
  kappaX: number;
  kappaS: number;
  extra: Record<string, number>;
}

function angleDeg(v: number[]): number {
  return (Math.atan2(v[1], v[0]) * 180) / Math.PI;
}
function axialDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % 180;
  if (d > 90) d = 180 - d;
  return d;
}

function analyse(sc: Scenario, params: DatasetParams): Ctx {
  const ds = generateDataset(sc.kind, params);
  const pcaRes = pca(ds.X, 'center');
  const D = distanceMatrix(pcaRes.Xc, 'euclidean');
  const mdsRes = classicalMDS(D, 2);
  let ldaRes: LDAResult | null = null;
  let ldaError: string | null = null;
  if (ds.y && sc.kind !== 'manifold') {
    try {
      ldaRes = lda(pcaRes.Xc, ds.y);
    } catch (e) {
      ldaError = (e as Error).message;
    }
  }
  const kappaX = conditionNumber(pcaRes.Xc);
  const kappaS = Number.isFinite(kappaX) ? kappaX * kappaX : Infinity;
  const extra: Record<string, number> = {};
  switch (sc.id) {
    case 'outliers': {
      const m = Math.min(Math.round(params.outlierCount), Math.floor(ds.X.length / 2));
      const clean = pca(ds.X.slice(m), 'center');
      extra.angleAll = angleDeg(column(pcaRes.V, 0));
      extra.angleClean = angleDeg(column(clean.V, 0));
      extra.shift = axialDiff(extra.angleAll, extra.angleClean);
      extra.pc1All = pcaRes.explained[0];
      extra.pc1Clean = clean.explained[0];
      extra.m = m;
      break;
    }
    case 'collinear': {
      extra.rank = pcaRes.rank;
      extra.smin = pcaRes.singularValues[pcaRes.singularValues.length - 1];
      extra.smax = pcaRes.singularValues[0];
      const vr = column(pcaRes.V, pcaRes.V[0].length - 1);
      extra.nullVar = Math.sqrt(Math.max(pcaRes.eigenvalues[pcaRes.eigenvalues.length - 1], 0));
      extra.vrMax = Math.max(...vr.map(Math.abs));
      break;
    }
    case 'scales': {
      const corr = pca(ds.X, 'standardize');
      extra.v11cov = Math.abs(pcaRes.V[0][0]);
      extra.v11cor = Math.abs(corr.V[0][0]);
      extra.pc1cov = pcaRes.explained[0];
      extra.pc1cor = corr.explained[0];
      const sds = colStds(ds.X);
      extra.sdRatio = Math.max(...sds) / Math.min(...sds);
      if (ldaRes) {
        extra.Jlda = ldaRes.eigenvalues[0] ?? 0;
        extra.Jpc1 = fisherCriterion(ldaRes.SB, ldaRes.SW, column(pcaRes.V, 0)).J;
      }
      break;
    }
    case 'pGreaterN': {
      extra.rank = pcaRes.rank;
      extra.nonzero = pcaRes.eigenvalues.filter((l) => l > 1e-10 * pcaRes.eigenvalues[0]).length;
      extra.swRank = ldaRes?.swRank ?? NaN;
      break;
    }
    case 'manifold': {
      extra.pc12 = (pcaRes.explained[0] ?? 0) + (pcaRes.explained[1] ?? 0);
      extra.stress = mdsRes.stress1;
      // Euclidean vs along-curve distance between the two ends of the S-curve (labelled 0 and 2)
      const ends0 = ds.X.filter((_, i) => ds.y![i] === 0);
      const ends2 = ds.X.filter((_, i) => ds.y![i] === 2);
      const c0 = ends0.length ? ends0.reduce((a, r) => a.map((x, j) => x + r[j] / ends0.length), [0, 0, 0]) : [0, 0, 0];
      const c2 = ends2.length ? ends2.reduce((a, r) => a.map((x, j) => x + r[j] / ends2.length), [0, 0, 0]) : [0, 0, 0];
      extra.endDist = Math.hypot(c0[0] - c2[0], c0[1] - c2[1], c0[2] - c2[2]);
      extra.arcLength = 3 * Math.PI * 0.75; // approximate length of the S-curve between the end groups
      break;
    }
    case 'overlapping': {
      if (ldaRes) {
        extra.J = ldaRes.eigenvalues[0] ?? 0;
        const cls = ldaClassify(ldaRes, pcaRes.Xc);
        extra.acc = cls.accuracy!(ds.y!);
        extra.chance = Math.max(...ldaRes.classSizes) / ds.X.length;
      }
      break;
    }
    case 'unequalCov': {
      if (ldaRes && ds.y) {
        const X0 = ds.X.filter((_, i) => ds.y![i] === 0);
        const X1 = ds.X.filter((_, i) => ds.y![i] === 1);
        const S0 = covariance(X0);
        const S1 = covariance(X1);
        extra.tr0 = S0[0][0] + S0[1][1];
        extra.tr1 = S1[0][0] + S1[1][1];
        const cls = ldaClassify(ldaRes, pcaRes.Xc);
        extra.accLDA = cls.accuracy!(ds.y);
        const q = qdaResubstitution(pcaRes.Xc, ds.y);
        extra.accQDA = q ? q.accuracy : NaN;
        const rec = perClassRecall(cls.predictions, ds.y, ldaRes.classes);
        extra.rec0 = rec[0];
        extra.rec1 = rec[1];
      }
      break;
    }
    case 'imbalanced': {
      if (ldaRes && ds.y) {
        extra.n0 = ldaRes.classSizes[0];
        extra.n1 = ldaRes.classSizes[1];
        const g = ldaRes.grandMean;
        const d0 = Math.hypot(...ldaRes.classMeans[0].map((m, j) => m - g[j]));
        const d1 = Math.hypot(...ldaRes.classMeans[1].map((m, j) => m - g[j]));
        extra.d0 = d0;
        extra.d1 = d1;
        const cls = ldaClassify(ldaRes, pcaRes.Xc);
        extra.acc = cls.accuracy!(ds.y);
        const rec = perClassRecall(cls.predictions, ds.y, ldaRes.classes);
        extra.rec0 = rec[0];
        extra.rec1 = rec[1];
        const eq = ldaClassify(ldaRes, pcaRes.Xc, [0.5, 0.5]);
        const recEq = perClassRecall(eq.predictions, ds.y, ldaRes.classes);
        extra.rec1eq = recEq[1];
        extra.rec0eq = recEq[0];
        extra.J = ldaRes.eigenvalues[0] ?? 0;
      }
      break;
    }
  }
  return { sc, params, ds, pcaRes, mdsRes, ldaRes, ldaError, kappaX, kappaS, extra };
}

function explanation(c: Ctx): { observed: ReactNode; math: ReactNode; stats: ReactNode; remedy: ReactNode } {
  const e = c.extra;
  switch (c.sc.id) {
    case 'outliers':
      return {
        observed: (
          <>
            With the {e.m} outliers present, PC1 points at {fmt(e.angleAll, 1)}°; without them it points at {fmt(e.angleClean, 1)}° — a rotation of {fmt(e.shift, 1)}°. PC1 explains {(e.pc1All * 100).toFixed(1)} % of the variance with the outliers and {(e.pc1Clean * 100).toFixed(1)} % without. The MDS map shows the same rotation because it is the same projection.
          </>
        ),
        math: (
          <>
            PCA maximises <M tex="\sum_i (w^Tx_i)^2" />: each observation enters with its <i>squared</i> distance from the mean, so a point at distance {fmt(c.params.outlierMagnitude, 1)} counts like {fmt(c.params.outlierMagnitude ** 2, 0)} points at distance 1. The eigenvector of <M tex="S" /> is pulled toward <M tex="x_{\text{out}}x_{\text{out}}^T" />, a rank-one perturbation of size <M tex="\|x_{\text{out}}\|^2/(n-1)" />.
          </>
        ),
        stats: <>The sample covariance has breakdown point 0: a single sufficiently extreme observation can move the leading principal direction anywhere. Components estimated from contaminated data describe the outliers, not the population; "variance explained" figures become meaningless.</>,
        remedy: <>Inspect leverage (row norms of <M tex="U" />) and robust distances first; use a robust covariance estimate (MCD, S-estimators) or spherical/spatial-sign PCA; report results with and without the flagged points.</>,
      };
    case 'collinear':
      return {
        observed: (
          <>
            Numerical rank {e.rank} of {c.ds.X[0].length}; singular values range from σ₁ = {fmt(e.smax, 3)} down to σ_p = {fmt(e.smin, 5)}, so κ₂(X_c) = {c.kappaX === Infinity ? '∞' : fmt(c.kappaX, 0)} and κ₂(S) = κ₂(X_c)² = {c.kappaS === Infinity ? '∞' : fmt(c.kappaS, 0)}. The trailing right singular vector (largest entry {fmt(e.vrMax, 3)}) is a combination of the variables with standard deviation only {fmt(e.nullVar, 4)}.
          </>
        ),
        math: (
          <>
            Near-linear dependence among columns makes <M tex="X_c^TX_c" /> nearly singular. Any quantity involving <M tex="S^{-1}" /> — regression coefficients, Mahalanobis distances, <M tex="S_W^{-1}S_B" /> — is amplified by <M tex="1/\sigma_p^2" />. The leading components themselves are fine; the trailing ones are pure noise with arbitrary directions.
          </>
        ),
        stats: <>Interpreting individual loadings is unreliable: with strongly correlated variables many different loading vectors give nearly the same component. Confidence regions for the trailing eigenvectors are essentially the whole subspace. Regression on these variables has inflated coefficient variances (VIF).</>,
        remedy: <>Work in the low-rank space (principal component regression), drop or merge redundant variables, or regularise (ridge, shrinkage of <M tex="S" />). Always compute through the SVD of <M tex="X_c" /> rather than through <M tex="S^{-1}" />.</>,
      };
    case 'scales':
      return {
        observed: (
          <>
            The standard deviations differ by a factor of {fmt(e.sdRatio, 0)}. On the covariance matrix |v₁₁| = {fmt(e.v11cov, 3)} and PC1 explains {(e.pc1cov * 100).toFixed(1)} % — the component is x₁. On the correlation matrix |v₁₁| = {fmt(e.v11cor, 3)} and PC1 explains {(e.pc1cor * 100).toFixed(1)} %.{' '}
            {Number.isFinite(e.Jlda) && (
              <>
                The Fisher ratio along PC1 is J = {fmt(e.Jpc1, 3)} against J = {fmt(e.Jlda, 3)} for the LDA direction: the class information sits in the small-scale variable.
              </>
            )}
          </>
        ),
        math: (
          <>
            Multiplying a variable by <M tex="c" /> multiplies its variance by <M tex="c^2" /> and its covariances by <M tex="c" />; for <M tex="c = {c.params.scaleFactor}" /> the entry <M tex="s_{11}" /> dominates <M tex="S" /> and its eigenvector tends to <M tex="e_1" />. PCA is not invariant to diagonal rescaling; LDA is (rescaling <M tex="x_1" /> rescales <M tex="w_1" /> inversely and leaves <M tex="J" /> unchanged).
          </>
        ),
        stats: <>"Variance explained" is then a statement about units, not about structure. Euclidean distances, and hence classical MDS, inherit the same distortion: the map is a one-dimensional ruler for x₁.</>,
        remedy: <>Standardise (correlation PCA) when units differ or are arbitrary; keep the covariance when the variables share meaningful units. Report which choice was made — the two analyses answer different questions.</>,
      };
    case 'pGreaterN':
      return {
        observed: (
          <>
            n = {c.ds.X.length}, p = {c.ds.X[0].length}: the covariance has only {e.nonzero} non-zero eigenvalues (numerical rank {e.rank} ≤ n − 1 = {c.ds.X.length - 1}). {c.ldaRes ? <>S_W has rank {e.swRank} &lt; p and is singular (κ = ∞); the unregularised LDA solution is computed on the range of S_W only.</> : <>LDA could not be computed: {c.ldaError}.</>}
          </>
        ),
        math: (
          <>
            The rows of <M tex="X_c" /> sum to zero, so they span at most an <M tex="(n-1)" />-dimensional subspace of <M tex="\mathbb R^p" /> and <M tex="\operatorname{rank}(S) \le n-1" />. Likewise <M tex="\operatorname{rank}(S_W) \le n-K" />. The thin SVD has only <M tex="\min(n,p) = n" /> singular values; a <M tex="p\times p" /> eigendecomposition would compute <M tex="p - n + 1" /> exact zeros.
          </>
        ),
        stats: <>With p &gt; n every configuration of class labels can be separated perfectly by some linear rule: LDA "accuracy" on the training data is 100 % and meaningless. Eigenvalues of S are heavily biased (the largest too large, the rest too small).</>,
        remedy: <>Regularised or shrinkage LDA (<M tex="S_W + \gamma I" />), PCA-then-LDA, sparse or penalised methods; always validate on held-out data. Use the thin SVD, never the p × p covariance.</>,
      };
    case 'manifold':
      return {
        observed: (
          <>
            PC1 and PC2 together explain {(e.pc12 * 100).toFixed(1)} % of the variance and the two-dimensional MDS map has stress {fmt(e.stress, 4)}, yet neither unrolls the sheet: the two ends of the S (different colours) are folded near each other. Their Euclidean centre distance is {fmt(e.endDist, 2)} while the path along the sheet is about {fmt(e.arcLength, 1)}.
          </>
        ),
        math: <>PCA and classical MDS find the best <i>linear</i> subspace; a curved sheet has intrinsic dimension 2 but no two-dimensional linear subspace contains it. Classical MDS with Euclidean input equals PCA (Gower), so it inherits exactly the same limitation: Euclidean distances between points on opposite folds are short even though the geodesic distance is long.</>,
        stats: <>Low reconstruction error and low stress do not certify that the representation preserves neighbourhoods or the intrinsic geometry. Clusters that appear in the map may be folds.</>,
        remedy: <>Use geodesic (graph) distances in MDS — Isomap — or kernel PCA, local linear embedding, t-SNE/UMAP for neighbourhood preservation; check with the residual variance as a function of dimension. Mentioned here only as contrast: they are not linear methods.</>,
      };
    case 'overlapping':
      return {
        observed: (
          <>
            {c.ldaRes ? <>The largest Fisher ratio is λ₁ = {fmt(e.J, 3)} and the Gaussian LDA classifier reaches {(e.acc * 100).toFixed(1)} % training accuracy against a {(e.chance * 100).toFixed(1)} % majority-class baseline. The projected class histograms overlap heavily.</> : 'LDA unavailable.'}
          </>
        ),
        math: <>The Fisher criterion is bounded by the separation of the class means in Mahalanobis units: for two classes <M tex="\lambda_1 = \frac{n_1n_2}{n}(m_1-m_2)^TS_W^{-1}(m_1-m_2)" />. When the means are close relative to the within-class spread no direction can make the projected classes distinct — the optimum exists but is small.</>,
        stats: <>LDA always returns a direction; a small λ₁ signals that it is not useful. With K = 3 the second discriminant may carry as little as a few percent of the separation. Training accuracy overstates performance; cross-validate.</>,
        remedy: <>Check the Fisher eigenvalues and a permutation test of the class labels; collect more discriminating variables; consider that the classes may simply not be linearly separable in these variables.</>,
      };
    case 'unequalCov':
      return {
        observed: (
          <>
            {c.ldaRes ? <>Class A has total variance tr(S_A) = {fmt(e.tr0, 2)}, class B has tr(S_B) = {fmt(e.tr1, 2)}, a ratio of {fmt(e.tr1 / e.tr0, 1)}. Pooled LDA reaches {(e.accLDA * 100).toFixed(1)} % (recall {(e.rec0 * 100).toFixed(0)} % for A, {(e.rec1 * 100).toFixed(0)} % for B) while QDA with class-specific covariances reaches {Number.isFinite(e.accQDA) ? (e.accQDA * 100).toFixed(1) : '—'} %.</> : 'LDA unavailable.'}
          </>
        ),
        math: <>Classical LDA pools <M tex="S_W = \sum_k (n_k-1)\hat\Sigma_k" /> and assumes a common <M tex="\Sigma" />. If <M tex="\Sigma_A \ne \Sigma_B" /> the Bayes boundary is quadratic, <M tex="x^T(\Sigma_A^{-1}-\Sigma_B^{-1})x + \dots = 0" />, and no linear rule is optimal. The Fisher direction is still well defined but its optimality argument (maximising a ratio of quadratic forms) no longer corresponds to minimising error.</>,
        stats: <>The linear boundary is biased toward the wide class: points of the tight class that lie inside the wide cloud are misclassified. Estimated posterior probabilities are miscalibrated.</>,
        remedy: <>Quadratic discriminant analysis when n_k is large enough to estimate each Σ_k (p(p+1)/2 parameters per class), regularised discriminant analysis between LDA and QDA, or a variance-stabilising transformation.</>,
      };
    case 'imbalanced':
      return {
        observed: (
          <>
            {c.ldaRes ? <>Class sizes {e.n0} and {e.n1}. The grand mean lies {fmt(e.d0, 3)} from the majority mean and {fmt(e.d1, 3)} from the minority mean. LDA with sample priors: accuracy {(e.acc * 100).toFixed(1)} %, but minority recall only {(e.rec1 * 100).toFixed(0)} %; with equal priors the minority recall becomes {(e.rec1eq * 100).toFixed(0)} % (majority recall {(e.rec0eq * 100).toFixed(0)} %). λ₁ = {fmt(e.J, 3)}.</> : 'LDA unavailable.'}
          </>
        ),
        math: <>In <M tex="S_B = \sum_k n_k(m_k-m)(m_k-m)^T" /> each class is weighted by <M tex="n_k" /> and <M tex="m" /> is the pooled mean, so with <M tex="n_1 \gg n_2" /> the grand mean sits almost on <M tex="m_1" />. For two classes the Fisher direction is nevertheless <M tex="S_W^{-1}(m_1-m_2)" /> regardless of the sizes — but the classifier's threshold moves with <M tex="\log(\pi_1/\pi_2)" />.</>,
        stats: <>The minority mean is estimated from very few points and has variance <M tex="\Sigma/n_2" />; the within-class scatter is essentially the majority class's scatter. High overall accuracy hides a near-useless minority classification.</>,
        remedy: <>Set priors deliberately (equal priors or cost-weighted), evaluate per-class recall and balanced accuracy instead of overall accuracy, and use resampling or stratified cross-validation.</>,
      };
    default:
      return { observed: null, math: null, stats: null, remedy: null };
  }
}

export default function WhatCanGoWrong() {
  const [scId, setScId] = useState<string>('outliers');
  const sc = scenarios.find((s) => s.id === scId) ?? scenarios[0];
  const [overrides, setOverrides] = useState<Record<string, Partial<DatasetParams>>>({});
  const [seed, setSeed] = useState(11);
  const params = useMemo<DatasetParams>(() => ({ ...defaultParams, ...sc.base, ...(overrides[sc.id] ?? {}), seed }), [sc, overrides, seed]);
  const c = useMemo(() => analyse(sc, params), [sc, params]);
  const ex = useMemo(() => explanation(c), [c]);
  const { ds, pcaRes, mdsRes, ldaRes } = c;
  const classNames = ds.classNames;

  const pcaPts = useMemo(() => pcaRes.scores.map((r) => [r[0] ?? 0, r[1] ?? 0]), [pcaRes.scores]);
  const mdsPts = useMemo(() => mdsRes.coords.map((r) => [r[0] ?? 0, r[1] ?? 0]), [mdsRes.coords]);
  const svdData: Data[] = [{ x: pcaRes.singularValues.map((_, i) => i + 1), y: pcaRes.singularValues, type: 'bar', marker: { color: categorical[0] }, hovertemplate: 'σ_%{x} = %{y:.4g}<extra></extra>' }];

  const ldaPanel = (() => {
    if (!ds.y || sc.kind === 'manifold') return <Callout kind="info" title="No class labels">LDA is supervised: without labels there is nothing to separate.</Callout>;
    if (!ldaRes) return <Callout kind="danger" title="LDA failed">{c.ldaError}</Callout>;
    if (ldaRes.maxDims >= 2) {
      const pts = ldaRes.scores.map((r) => [r[0], r[1]]);
      return <ScatterSVG points={pts} labels={ds.y} classNames={classNames} width={420} height={300} xLabel={`LD1 (λ₁ = ${fmt(ldaRes.eigenvalues[0], 2)})`} yLabel={`LD2 (λ₂ = ${fmt(ldaRes.eigenvalues[1], 2)})`} pointRadius={3.5} />;
    }
    const z = ldaRes.scores.map((r) => r[0]);
    const traces: Data[] = ldaRes.classes.map((cl, k) => ({ x: z.filter((_, i) => ds.y![i] === cl), type: 'histogram', name: classNames?.[cl] ?? `class ${cl}`, opacity: 0.6, marker: { color: classColor(k) }, nbinsx: 30 }));
    return <Plot data={traces} layout={{ barmode: 'overlay', showlegend: true, xaxis: { title: { text: `LD1 score (λ₁ = ${fmt(ldaRes.eigenvalues[0], 3)})` } }, yaxis: { title: { text: 'count' } } }} height={300} />;
  })();

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>What can go wrong?</h2>
          <div className="lede">Eight deliberately difficult datasets. Watch how SVD, PCA, MDS and LDA respond, then read why: observed behaviour → mathematical reason → statistical implication → possible remedy.</div>
        </div>
      </div>

      <Section id="wrong-picker" title="Choose a failure mode" subtitle={sc.short}>
        <div className="grid side">
          <div className="stack">
            <Select label="Scenario" value={scId} options={scenarios.map((s) => ({ value: s.id, label: s.title }))} onChange={setScId} />
            {sc.controls.map((ctl) => (
              <Slider key={ctl.key} label={ctl.label} value={params[ctl.key] as number} min={ctl.min} max={ctl.max} step={ctl.step} onChange={(v) => setOverrides((o) => ({ ...o, [sc.id]: { ...(o[sc.id] ?? {}), [ctl.key]: v } }))} />
            ))}
            <div className="row">
              <Slider label="Random seed" value={seed} min={1} max={999} step={1} onChange={setSeed} />
              <Button small onClick={() => setSeed(Math.floor(Math.random() * 999) + 1)}>
                New seed
              </Button>
            </div>
          </div>
          <div className="stack">
            <Card title="Dataset">
              <div>
                <b>{ds.name}</b> — n = {ds.X.length}, p = {ds.X[0].length}
                {ds.y ? `, ${new Set(ds.y).size} groups` : ''}
              </div>
              <div className="small secondary" style={{ marginTop: 4 }}>
                {ds.description}
              </div>
              <ul className="small" style={{ marginTop: 6 }}>
                {ds.notes.map((nt, i) => (
                  <li key={i}>{nt}</li>
                ))}
              </ul>
            </Card>
            <div className="stats">
              <StatTile label="κ₂(X_c) = σ₁/σ_r" value={c.kappaX === Infinity ? '∞' : fmt(c.kappaX, 1)} />
              <StatTile label="numerical rank" value={`${pcaRes.rank} / ${Math.min(ds.X.length, ds.X[0].length)}`} />
              <StatTile label="PC1 variance share" value={`${(pcaRes.explained[0] * 100).toFixed(1)} %`} />
              <StatTile label="MDS stress-1 (k = 2)" value={fmt(mdsRes.stress1, 4)} note={`${mdsRes.negative} negative eigenvalues`} />
              <StatTile label="LDA λ₁" value={ldaRes ? fmt(ldaRes.eigenvalues[0] ?? 0, 3) : '—'} note={ldaRes ? (ldaRes.swSingular ? 'S_W singular' : `${ldaRes.maxDims} direction${ldaRes.maxDims === 1 ? '' : 's'}`) : 'no labels'} />
            </div>
          </div>
        </div>
      </Section>

      <Section id="wrong-panels" title="Four methods, one difficult dataset" subtitle="Small multiples on the same generated data; colours mark the classes (or, for the manifold, the position along the curve).">
        {classNames && ds.y && <ClassLegend classNames={classNames} />}
        <div className="grid c4" style={{ marginTop: 8 }}>
          <Card title={<Badge method="SVD" />}>
            <Plot data={svdData} layout={{ xaxis: { title: { text: 'j' }, dtick: 1 }, yaxis: { title: { text: 'σ_j' }, rangemode: 'tozero' } }} height={300} title={`Singular values of X_c (rank ${pcaRes.rank})`} />
          </Card>
          <Card title={<Badge method="PCA" />}>
            <ScatterSVG points={pcaPts} labels={sc.kind === 'manifold' || ds.y ? ds.y : undefined} classNames={classNames} width={420} height={300} xLabel={`PC1 (${(pcaRes.explained[0] * 100).toFixed(1)} %)`} yLabel={`PC2 (${((pcaRes.explained[1] ?? 0) * 100).toFixed(1)} %)`} pointRadius={3.5} />
          </Card>
          <Card title={<Badge method="MDS" />}>
            <ScatterSVG points={mdsPts} labels={sc.kind === 'manifold' || ds.y ? ds.y : undefined} classNames={classNames} width={420} height={300} xLabel="MDS 1 (Euclidean)" yLabel="MDS 2" pointRadius={3.5} caption={`stress-1 = ${fmt(mdsRes.stress1, 4)}; ${mdsRes.negative} negative eigenvalue${mdsRes.negative === 1 ? '' : 's'}`} />
          </Card>
          <Card title={<Badge method="LDA" />}>{ldaPanel}</Card>
        </div>
      </Section>

      <Section id="wrong-explain" title={`Explanation: ${sc.title.replace(/^\d+ · /, '')}`} subtitle="Observed behaviour → mathematical reason → statistical implication → possible remedy, with the numbers from the current simulation.">
        <div className="grid c4">
          <Card title="Observed behaviour" plane>
            {ex.observed}
          </Card>
          <Card title="Mathematical reason" plane>
            {ex.math}
          </Card>
          <Card title="Statistical implication" plane>
            {ex.stats}
          </Card>
          <Card title="Possible remedy" plane>
            {ex.remedy}
          </Card>
        </div>
        <Interpretation
          defaultOpen={false}
          items={{
            seeing: <>The four panels above analyse one and the same matrix; the explanation panel quantifies the failure for the current slider settings (seed {seed}).</>,
            why: <>Each method optimises a criterion — Frobenius error, variance, strain, the Fisher ratio — that is a sum of squares over observations or pairs. Every failure mode here corresponds to a situation where that sum is dominated by a few terms (outliers, one large-scale variable, one large class) or where the criterion's assumptions (linearity, invertibility, common covariance) are violated.</>,
            math: (
              <>
                <MBlock tex={String.raw`\text{PCA: } \max_{\|w\|=1}\sum_i (w^Tx_i)^2\qquad \text{MDS: } \min \|B - XX^T\|_F\qquad \text{LDA: } \max_w \frac{w^TS_Bw}{w^TS_Ww}`} />
              </>
            ),
            stats: <>Diagnostics precede interpretation: leverage and robust distances, condition numbers, scale ratios, class sizes and class covariances should be examined before any of the four low-dimensional pictures is trusted.</>,
            careful: <>All datasets here are simulated with known structure; on real data the failure modes usually occur together (scale differences and outliers, correlation and p ≈ n), and no single diagnostic reveals all of them.</>,
          }}
        />
        <div className="small muted">Method colours: SVD {methodColor.SVD}, PCA {methodColor.PCA}, MDS {methodColor.MDS}, LDA {methodColor.LDA}; text ink {ink.primary}.</div>
      </Section>
    </div>
  );
}

// unused-import guard for tree-shaking friendliness
void dot;
