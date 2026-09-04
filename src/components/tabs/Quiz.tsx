import { useMemo, useState, type ReactNode } from 'react';
import type { Data } from 'plotly.js';
import { generateDataset, defaultParams } from '../../lib/datasets';
import { pca } from '../../lib/pca';
import { distanceMatrix, classicalMDS } from '../../lib/mds';
import { lda, fisherCriterion } from '../../lib/lda';
import { column, fmt } from '../../lib/linalg';
import { Section, Card, Callout, StatTile, Badge } from '../common/Panels';
import { ScatterSVG } from '../common/ScatterSVG';
import { Plot } from '../common/Plot';
import { Button } from '../common/Controls';
import { M, MBlock } from '../common/Math';
import { categorical, classColor, methodColor, ink } from '../../lib/theme';

interface MCQ {
  id: string;
  topic: string;
  prompt: ReactNode;
  options: ReactNode[];
  correct: number;
  explanation: ReactNode;
}

const questions: MCQ[] = [
  {
    id: 'scaling',
    topic: 'PCA',
    prompt: <>Why does multiplying one variable by a constant <M tex="c" /> change the principal components?</>,
    options: [
      <>Because the covariance matrix changes non-uniformly: the variable's variance is multiplied by <M tex="c^2" /> and its covariances by <M tex="c" />, so the eigenvectors of <M tex="S" /> rotate.</>,
      <>Because PCA is computed on the raw (uncentred) data and the mean of that variable changes.</>,
      <>It does not — PCA is invariant to any invertible linear transformation of the variables.</>,
      <>Because the eigenvalues are multiplied by <M tex="c" /> while the eigenvectors stay fixed, which reorders the components.</>,
    ],
    correct: 0,
    explanation: (
      <>
        Rescaling by <M tex="D = \mathrm{diag}(c,1,\dots,1)" /> maps <M tex="S \mapsto DSD" />; this is a congruence, not a similarity, so eigenvectors are not preserved. PCA is invariant to <i>orthogonal</i> transformations (rotations), not to diagonal ones — that is the whole reason the covariance/correlation choice matters. Centring is unaffected by scale, and eigenvalues are not simply multiplied by <M tex="c" />.
      </>
    ),
  },
  {
    id: 'k1',
    topic: 'LDA',
    prompt: <>Why can LDA produce at most <M tex="K-1" /> meaningful discriminant directions for <M tex="K" /> classes?</>,
    options: [
      <>Because <M tex="S_W" /> has rank <M tex="K-1" />.</>,
      <>Because <M tex="S_B = \sum_k n_k(m_k-m)(m_k-m)^T" /> is a sum of <M tex="K" /> rank-one matrices whose vectors satisfy <M tex="\sum_k n_k(m_k - m) = 0" />, so <M tex="\operatorname{rank}(S_B) \le K-1" /> and the generalised eigenproblem has at most <M tex="K-1" /> non-zero eigenvalues.</>,
      <>Because there are only <M tex="K-1" /> independent class labels once the labels are one-hot encoded and centred, which limits <M tex="S_W" />.</>,
      <>Because the discriminant scores must be uncorrelated and there are only <M tex="K-1" /> degrees of freedom in the class means after standardising.</>,
    ],
    correct: 1,
    explanation: (
      <>
        The rank bound comes from <M tex="S_B" />, not <M tex="S_W" /> (which typically has full rank <M tex="p" />). The <M tex="K" /> mean deviations are linearly dependent through the weighted sum being zero, so they span at most <M tex="K-1" /> dimensions, and <M tex="S_B w = \lambda S_W w" /> can have at most <M tex="\operatorname{rank}(S_B)" /> non-zero eigenvalues. Option 3 mentions the right count but attributes it to the wrong matrix.
      </>
    ),
  },
  {
    id: 'eckart',
    topic: 'SVD',
    prompt: <>Why is the truncated SVD <M tex="X_k = U_k\Sigma_kV_k^T" /> the optimal rank-<M tex="k" /> approximation in the Frobenius norm?</>,
    options: [
      <>Because <M tex="U" /> and <M tex="V" /> are orthogonal, the Frobenius norm of any error is <M tex="\|\Sigma - U^TBV\|_F" />, and among rank-<M tex="k" /> matrices this is minimised by keeping the <M tex="k" /> largest diagonal entries — the Eckart–Young theorem, with error <M tex="\sqrt{\sum_{j>k}\sigma_j^2}" />.</>,
      <>Because the singular values are sorted, so any other choice of <M tex="k" /> columns would have larger reconstruction error; the result holds for the Frobenius norm but fails for the spectral norm.</>,
      <>Because <M tex="X_k" /> is the projection of <M tex="X" /> onto its first <M tex="k" /> rows, which by construction captures the largest entries.</>,
      <>It is optimal only when <M tex="X" /> is square and symmetric; for rectangular matrices a different factorisation is needed.</>,
    ],
    correct: 0,
    explanation: (
      <>
        Orthogonal invariance of <M tex="\|\cdot\|_F" /> reduces the problem to approximating the diagonal matrix <M tex="\Sigma" />; Weyl's inequalities show no rank-<M tex="k" /> matrix can do better than dropping the smallest singular values. The theorem holds for <i>every</i> unitarily invariant norm, including the spectral norm (with error <M tex="\sigma_{k+1}" />), so option 2's restriction is false; option 3 confuses rows with singular directions; option 4 is false — the SVD exists for any real matrix.
      </>
    ),
  },
  {
    id: 'mdspca',
    topic: 'MDS',
    prompt: <>When can classical MDS reproduce the PCA coordinates of a dataset?</>,
    options: [
      <>Always: classical MDS and PCA are the same algorithm with different names.</>,
      <>When the distance matrix contains Euclidean distances between the rows of the same centred (or standardised) matrix used for PCA; the configurations then agree up to rotation, reflection and translation.</>,
      <>When the number of retained dimensions equals <M tex="p" />, regardless of the metric.</>,
      <>Only when the data are exactly Gaussian, because then distances and variances carry the same information.</>,
    ],
    correct: 1,
    explanation: (
      <>
        For Euclidean distances <M tex="B = -\tfrac12JD^{(2)}J = X_cX_c^T = U\Sigma^2U^T" />, so the MDS coordinates <M tex="U_k\Sigma_k" /> equal the PCA scores <M tex="Z_k" /> up to sign flips and rotations within repeated-eigenvalue subspaces. With Manhattan or other non-Euclidean input the matrices differ and <M tex="B" /> may be indefinite, so option 3 is wrong; no distributional assumption is involved, so option 4 is wrong.
      </>
    ),
  },
  {
    id: 'pcafail',
    topic: 'PCA vs LDA',
    prompt: <>Why can PCA fail to find the directions that separate known classes?</>,
    options: [
      <>Because PCA uses the wrong eigen-solver for labelled data.</>,
      <>Because PCA maximises total variance, which is dominated by within-class spread when that spread is larger than the between-class separation; it never sees the labels and has no reason to prefer a low-variance direction along which the means differ.</>,
      <>Because class separation is a nonlinear property that no linear projection can capture.</>,
      <>Because PCA requires standardised variables and class differences are removed by standardisation.</>,
    ],
    correct: 1,
    explanation: (
      <>
        Total scatter decomposes as <M tex="S_T = S_W + S_B" />. PCA maximises <M tex="w^TS_Tw" />, LDA maximises <M tex="w^TS_Bw/w^TS_Ww" />. If the classes are elongated in a direction orthogonal to the mean difference, <M tex="S_W" /> dominates and PC1 aligns with within-class spread. Linear separation is perfectly possible — LDA finds it — so option 3 is false; standardisation is unrelated.
      </>
    ),
  },
  {
    id: 'negeig',
    topic: 'MDS',
    prompt: <>What does a negative eigenvalue of <M tex="B = -\tfrac12JD^{(2)}J" /> in classical MDS imply?</>,
    options: [
      <>A numerical error: <M tex="B" /> is always positive semi-definite and the eigenvalue should be set to zero.</>,
      <>That the dissimilarities are not Euclidean distances of any point configuration: no set of points in any Euclidean space, of any dimension, reproduces <M tex="D" /> exactly.</>,
      <>That the configuration must be reflected to make the eigenvalue positive.</>,
      <>That the data have more variables than observations.</>,
    ],
    correct: 1,
    explanation: (
      <>
        By the Schoenberg/Young–Householder theorem, <M tex="D" /> is Euclidean-embeddable if and only if <M tex="B" /> is positive semi-definite. Negative eigenvalues therefore certify non-Euclideanity (e.g. Manhattan distances, perceptual dissimilarities); their total magnitude measures how far from Euclidean the input is. Rounding can create tiny negatives, but large ones are real. Reflections do not change eigenvalues.
      </>
    ),
  },
  {
    id: 'swsing',
    topic: 'LDA',
    prompt: <>What happens when the within-class scatter matrix <M tex="S_W" /> is singular?</>,
    options: [
      <>Nothing changes: LDA depends only on <M tex="S_B" />.</>,
      <>The generalised eigenproblem <M tex="S_Bw = \lambda S_Ww" /> is ill-posed: directions in the null space of <M tex="S_W" /> have zero within-class variance, so the Fisher ratio is unbounded and the solution is not unique; regularisation (<M tex="S_W + \gamma I" />), a pseudo-inverse restricted to the range of <M tex="S_W" />, or prior dimension reduction is required.</>,
      <>LDA reduces to PCA because the whitening step becomes the identity.</>,
      <>The number of discriminant directions increases to <M tex="p" />.</>,
    ],
    correct: 1,
    explanation: (
      <>
        <M tex="S_W" /> is singular whenever <M tex="p > n - K" /> or variables are exactly collinear. Any <M tex="w" /> with <M tex="S_Ww = 0" /> and <M tex="S_Bw \ne 0" /> gives <M tex="J(w) = \infty" />: perfect training separation with no generalisation. The laboratory's implementation solves on the range of <M tex="S_W" /> and flags the singularity; regularised LDA adds a ridge.
      </>
    ),
  },
  {
    id: 'svdnum',
    topic: 'Numerics',
    prompt: <>Why is computing PCA through the SVD of <M tex="X_c" /> numerically preferable to eigen-decomposing <M tex="X_c^TX_c" />?</>,
    options: [
      <>Because the SVD is always faster.</>,
      <>Because forming <M tex="X_c^TX_c" /> squares the condition number (<M tex="\kappa(X_c^TX_c) = \kappa(X_c)^2" />) and singular values below about <M tex="\sqrt{\varepsilon}\,\sigma_1" /> are lost to rounding before the eigensolver even starts, whereas a good SVD works on <M tex="X_c" /> directly with backward error proportional to <M tex="\varepsilon\|X_c\|" />.</>,
      <>Because eigenvalues of <M tex="X_c^TX_c" /> can be negative while singular values cannot.</>,
      <>Because the eigenvectors of <M tex="X_c^TX_c" /> are not orthogonal.</>,
    ],
    correct: 1,
    explanation: (
      <>
        A singular value <M tex="\sigma_p = 10^{-9}\sigma_1" /> corresponds to an eigenvalue <M tex="10^{-18}\sigma_1^2" />, which is below the double-precision resolution of the entry <M tex="\sigma_1^2" /> — it is simply gone. Exact eigenvalues of <M tex="X_c^TX_c" /> are non-negative and its eigenvectors orthogonal, so options 3 and 4 misdiagnose the problem; the SVD is not always faster.
      </>
    ),
  },
  {
    id: 'lambda',
    topic: 'PCA ↔ SVD',
    prompt: <>If <M tex="X_c = U\Sigma V^T" /> with <M tex="n" /> observations, the eigenvalues of the sample covariance <M tex="S" /> are</>,
    options: [<M tex="\lambda_j = \sigma_j" />, <M tex="\lambda_j = \sigma_j^2/(n-1)" />, <M tex="\lambda_j = \sigma_j^2" />, <M tex="\lambda_j = \sqrt{\sigma_j}/(n-1)" />],
    correct: 1,
    explanation: (
      <>
        <M tex="S = \frac{1}{n-1}X_c^TX_c = \frac{1}{n-1}V\Sigma^2V^T" />, so the eigenvectors are the right singular vectors and <M tex="\lambda_j = \sigma_j^2/(n-1)" />. The factor <M tex="1/(n-1)" /> is the same one that turns the scatter matrix into the unbiased covariance; the eigenvalues of <M tex="X_cX_c^T" /> (the MDS matrix <M tex="B" />) are <M tex="\sigma_j^2" /> without it.
      </>
    ),
  },
  {
    id: 'J',
    topic: 'MDS',
    prompt: <>What does the centring matrix <M tex="J = I - \tfrac1n\mathbf 1\mathbf 1^T" /> do in <M tex="B = -\tfrac12JD^{(2)}J" />?</>,
    options: [
      <>It converts distances to similarities by subtracting them from the maximum distance.</>,
      <>It removes the row and column means of <M tex="D^{(2)}" />, killing the terms <M tex="\|x_i\|^2" /> and <M tex="\|x_j\|^2" /> in <M tex="d_{ij}^2 = \|x_i\|^2+\|x_j\|^2-2x_i^Tx_j" /> and leaving the Gram matrix of the centred configuration.</>,
      <>It normalises the distances to unit variance so that all pairs are weighted equally.</>,
      <>It makes <M tex="B" /> positive definite.</>,
    ],
    correct: 1,
    explanation: (
      <>
        <M tex="J" /> is the orthogonal projector onto the complement of <M tex="\mathbf 1" />; it annihilates the rank-one terms <M tex="c\mathbf 1^T" /> and <M tex="\mathbf 1c^T" /> because <M tex="J\mathbf 1 = 0" />, and fixes <M tex="X_cX_c^T" /> because the columns of <M tex="X_c" /> have zero mean. It fixes the translation indeterminacy — it does not change positive-(semi)definiteness.
      </>
    ),
  },
  {
    id: 'fisher-vs-gauss',
    topic: 'LDA',
    prompt: <>What is the difference between Fisher's linear discriminant and Gaussian LDA classification?</>,
    options: [
      <>None — they are two names for the same procedure.</>,
      <>Fisher's discriminant is a projection defined by maximising <M tex="w^TS_Bw/w^TS_Ww" /> without distributional assumptions; Gaussian LDA is a classifier derived from Bayes' rule under <M tex="x\mid k \sim N(\mu_k,\Sigma)" /> with common <M tex="\Sigma" />. They coincide in direction for two classes (<M tex="w \propto S_W^{-1}(m_1-m_2)" />), but the classifier adds thresholds, priors and posterior probabilities.</>,
      <>Fisher's discriminant assumes Gaussian classes, whereas Gaussian LDA is distribution-free.</>,
      <>Fisher's discriminant allows class-specific covariances; Gaussian LDA requires a common covariance.</>,
    ],
    correct: 1,
    explanation: <>The generative model is what Gaussian LDA adds: it produces decision boundaries <M tex="\delta_k(x) = \delta_l(x)" /> and calibrated posteriors, and it is optimal only under its assumptions. Fisher's criterion is purely a scatter-ratio optimisation. Class-specific covariances lead to QDA, not to either form of LDA.</>,
  },
  {
    id: 'invariance',
    topic: 'PCA',
    prompt: <>PCA is invariant to which transformations of the data?</>,
    options: [
      <>Rotations of the variable space (orthogonal transformations): components rotate accordingly, eigenvalues and scores are unchanged. It is not invariant to rescaling individual variables.</>,
      <>Any invertible linear transformation, like LDA.</>,
      <>Rescaling of individual variables but not rotations.</>,
      <>No transformation at all — PCA is coordinate-dependent in every respect.</>,
    ],
    correct: 0,
    explanation: <>Under <M tex="X_c \mapsto X_cQ" /> with <M tex="Q" /> orthogonal, <M tex="S \mapsto Q^TSQ" /> is a similarity transformation: eigenvalues and scores are preserved, loadings become <M tex="Q^TV" />. Diagonal scaling is a congruence and changes everything — the point of the scaling laboratory. LDA is invariant to all invertible linear maps because <M tex="J(w)" /> is a ratio of quadratic forms transformed identically.</>,
  },
  {
    id: 'sign',
    topic: 'Interpretation',
    prompt: <>Two software packages return opposite signs for the loadings of PC1 on the same data. Which statement is correct?</>,
    options: [
      <>One of the packages has a bug.</>,
      <>Both are correct: eigenvectors and singular vectors are defined only up to sign (and, for repeated eigenvalues, up to rotation within the eigenspace); scores flip sign accordingly and all variances, distances and reconstructions are identical.</>,
      <>The package returning positive loadings is right because variance must be positive.</>,
      <>The difference means the two packages used different centring conventions.</>,
    ],
    correct: 1,
    explanation: <>If <M tex="Sv = \lambda v" /> then <M tex="S(-v) = \lambda(-v)" />. Any sign convention (largest component positive, first component positive, …) is a display choice. Compare configurations after Procrustes alignment, never by raw coordinates.</>,
  },
  {
    id: 'varexp',
    topic: 'Interpretation',
    prompt: <>"PC1 explains 85 % of the variance." Which conclusion is justified?</>,
    options: [
      <>PC1 is the most important variable for predicting any outcome of interest.</>,
      <>Projecting onto PC1 loses 15 % of the total variance (equivalently, the rank-one reconstruction has relative squared error 0.15) — nothing more; relevance for prediction or for class separation is a separate question that requires labels or an outcome.</>,
      <>The data are one-dimensional and the other components are noise.</>,
      <>The variables are all measured on the same scale.</>,
    ],
    correct: 1,
    explanation: <>Variance explained is <M tex="\lambda_1/\operatorname{tr}(S)" />, a statement about dispersion under the chosen scaling. Dominant variance can be irrelevant (or be an artefact of units); small components can carry the class separation, as the PCA-versus-LDA laboratory shows.</>,
  },
];

// ---------------------------------------------------------------------------
// Predict-then-reveal scenarios (computed live)
// ---------------------------------------------------------------------------

interface Scenario {
  id: string;
  title: string;
  setup: ReactNode;
  question: ReactNode;
  options: ReactNode[];
  compute: () => { correct: number; result: ReactNode; chart: ReactNode };
}

function buildScenarios(): Scenario[] {
  return [
    {
      id: 'scale',
      title: 'Rescaling a variable',
      setup: <>Three correlated Gaussian variables (n = 150, ρ = 0.5). We multiply <M tex="x_1" /> by 100 and recompute PCA on the covariance matrix.</>,
      question: <>What happens to <M tex="|v_{11}|" />, the loading of <M tex="x_1" /> on PC1?</>,
      options: [<>It rises to almost 1: PC1 becomes essentially <M tex="x_1" />.</>, <>It is unchanged: PCA is scale-invariant.</>, <>It drops to almost 0: the large variable is down-weighted.</>, <>PC1 and PC2 swap but the loadings keep their magnitudes.</>],
      compute: () => {
        const ds = generateDataset('correlated', { ...defaultParams, n: 150, p: 3, correlation: 0.5, seed: 21 });
        const before = pca(ds.X, 'center');
        const Xs = ds.X.map((r) => r.map((x, j) => (j === 0 ? 100 * x : x)));
        const after = pca(Xs, 'center');
        const names = ['x1', 'x2', 'x3'];
        const data: Data[] = [
          { x: names, y: column(before.V, 0).map(Math.abs), type: 'bar', name: 'original scale', marker: { color: categorical[0] } },
          { x: names, y: column(after.V, 0).map(Math.abs), type: 'bar', name: 'x1 × 100', marker: { color: categorical[1] } },
        ];
        return {
          correct: 0,
          result: (
            <>
              |v₁₁| = {fmt(Math.abs(before.V[0][0]), 3)} before and {fmt(Math.abs(after.V[0][0]), 4)} after; PC1's variance share rises from {(before.explained[0] * 100).toFixed(1)} % to {(after.explained[0] * 100).toFixed(2)} %. The covariance matrix is now dominated by <M tex="s_{11}" />, which is 10 000 times larger than before.
            </>
          ),
          chart: <Plot data={data} layout={{ barmode: 'group', showlegend: true, yaxis: { title: { text: '|loading on PC1|' }, range: [0, 1.05] } }} height={260} />,
        };
      },
    },
    {
      id: 'pcalda',
      title: 'Variance versus separation',
      setup: <>Two classes (n = 200), both strongly elongated along one direction, with means that differ only along the thin orthogonal direction.</>,
      question: <>Which direction separates the classes better, measured by the Fisher ratio <M tex="J(w) = w^TS_Bw/w^TS_Ww" />?</>,
      options: [<>PC1 — it captures most of the variance, and variance is information.</>, <>The Fisher direction, by a large factor; PC1 mixes the classes almost completely.</>, <>Both about equally — they are nearly parallel.</>, <>Neither — J is scale-free, so all directions have the same J.</>],
      compute: () => {
        const ds = generateDataset('pcaVsLda', { ...defaultParams, n: 200, separation: 3, seed: 5 });
        const p = pca(ds.X, 'center');
        const l = lda(p.Xc, ds.y!);
        const v1 = column(p.V, 0);
        const w1 = column(l.W, 0);
        const Jp = fisherCriterion(l.SB, l.SW, v1).J;
        const Jl = fisherCriterion(l.SB, l.SW, w1).J;
        const L = Math.sqrt(p.eigenvalues[0]) * 1.5;
        return {
          correct: 1,
          result: (
            <>
              J(PC1) = {fmt(Jp, 3)} versus J(w_LDA) = {fmt(Jl, 3)} — a factor of {fmt(Jl / Math.max(Jp, 1e-12), 1)}. PC1 carries {(p.explained[0] * 100).toFixed(1)} % of the variance but almost none of the separation; the angle between the two directions is {fmt((Math.acos(Math.min(1, Math.abs(v1[0] * w1[0] + v1[1] * w1[1]))) * 180) / Math.PI, 1)}°.
            </>
          ),
          chart: (
            <ScatterSVG
              points={p.Xc}
              labels={ds.y}
              classNames={ds.classNames}
              width={520}
              height={300}
              vectors={[
                { x: v1[0] * L, y: v1[1] * L, color: methodColor.PCA, label: 'PC1' },
                { x: w1[0] * L * 0.4, y: w1[1] * L * 0.4, color: methodColor.LDA, label: 'LDA' },
              ]}
              pointRadius={3.5}
            />
          ),
        };
      },
    },
    {
      id: 'manhattan',
      title: 'Non-Euclidean input to classical MDS',
      setup: <>A two-dimensional Gaussian cloud (n = 80, ρ = 0.6). We compute the <b>Manhattan</b> distance matrix and apply classical MDS.</>,
      question: <>Are all eigenvalues of <M tex="B = -\tfrac12JD^{(2)}J" /> non-negative?</>,
      options: [<>Yes — B is a double-centred matrix and is always positive semi-definite.</>, <>No — some eigenvalues are negative because Manhattan distances of a planar cloud are not Euclidean distances of any configuration.</>, <>All eigenvalues are negative because the sign in front of ½ flips them.</>, <>B cannot be formed: double centring requires Euclidean distances.</>],
      compute: () => {
        const ds = generateDataset('gaussian2d', { ...defaultParams, n: 80, correlation: 0.6, seed: 8 });
        const m = classicalMDS(distanceMatrix(ds.X, 'manhattan'), 2);
        const vals = m.eigenvalues.slice(0, 20);
        const data: Data[] = [
          { x: vals.map((_, i) => i + 1), y: vals.map((v) => Math.max(v, 0)), type: 'bar', name: 'positive', marker: { color: categorical[0] } },
          { x: vals.map((_, i) => i + 1), y: vals.map((v) => Math.min(v, 0)), type: 'bar', name: 'negative', marker: { color: '#e34948' } },
        ];
        return {
          correct: m.negative > 0 ? 1 : 0,
          result: (
            <>
              {m.negative} of {m.n} eigenvalues are negative; the most negative is {fmt(Math.min(...m.eigenvalues), 3)} and the negative eigenvalues carry {(m.negativeMass * 100).toFixed(2)} % of the absolute eigenvalue mass. Stress-1 of the two-dimensional map is {fmt(m.stress1, 4)}.
            </>
          ),
          chart: <Plot data={data} layout={{ barmode: 'relative', showlegend: true, xaxis: { title: { text: 'index j' } }, yaxis: { title: { text: 'λ_j(B)' } } }} height={260} />,
        };
      },
    },
    {
      id: 'pgn',
      title: 'More variables than observations',
      setup: <>Two classes with n = 10 observations and p = 20 variables.</>,
      question: <>How many non-zero eigenvalues does the covariance matrix have, and is <M tex="S_W" /> invertible?</>,
      options: [<>20 non-zero eigenvalues; S_W is invertible.</>, <>At most n − 1 = 9 non-zero eigenvalues; S_W (rank ≤ n − K = 8) is singular.</>, <>Exactly K − 1 = 1 non-zero eigenvalue; S_W is invertible.</>, <>Exactly n = 10 non-zero eigenvalues; S_W is invertible.</>],
      compute: () => {
        const ds = generateDataset('pGreaterN', { ...defaultParams, n: 10, p: 20, K: 2, seed: 4 });
        const p = pca(ds.X, 'center');
        const l = lda(p.Xc, ds.y!);
        const nonzero = p.eigenvalues.filter((v) => v > 1e-10 * p.eigenvalues[0]).length;
        const data: Data[] = [{ x: p.eigenvalues.map((_, i) => i + 1), y: p.eigenvalues, type: 'bar', marker: { color: categorical[0] }, name: 'λ_j' }];
        return {
          correct: 1,
          result: (
            <>
              {nonzero} non-zero eigenvalues out of min(n, p) = {p.eigenvalues.length} computed (rank {p.rank}); S_W has rank {l.swRank} &lt; p = {l.p}, so it is singular (κ = ∞) and the laboratory's LDA solves on its range and flags the problem.
            </>
          ),
          chart: <Plot data={data} layout={{ xaxis: { title: { text: 'j' }, dtick: 1 }, yaxis: { title: { text: 'eigenvalue of S' } } }} height={260} />,
        };
      },
    },
    {
      id: 'outlier',
      title: 'One outlier',
      setup: <>A correlated 2-D Gaussian cloud (n = 100, ρ = 0.8) whose PC1 lies near the diagonal. We add a single point at (8, −8), far out along the anti-diagonal.</>,
      question: <>What happens to the direction of PC1?</>,
      options: [<>Essentially unchanged: one point in 101 is negligible.</>, <>It rotates noticeably toward the outlier (tens of degrees), because each point enters PCA with its squared distance.</>, <>It flips exactly to the anti-diagonal.</>, <>PC1 becomes undefined because the covariance matrix is singular.</>],
      compute: () => {
        const ds = generateDataset('gaussian2d', { ...defaultParams, n: 100, correlation: 0.8, seed: 13 });
        const before = pca(ds.X, 'center');
        const X2 = [...ds.X, [8, -8]];
        const after = pca(X2, 'center');
        const a0 = (Math.atan2(before.V[1][0], before.V[0][0]) * 180) / Math.PI;
        const a1 = (Math.atan2(after.V[1][0], after.V[0][0]) * 180) / Math.PI;
        let shift = Math.abs(a0 - a1) % 180;
        if (shift > 90) shift = 180 - shift;
        const L = 4;
        return {
          correct: 1,
          result: (
            <>
              PC1 points at {fmt(a0, 1)}° without the outlier and {fmt(a1, 1)}° with it — a rotation of {fmt(shift, 1)}°. The outlier alone contributes {fmt(128 / (X2.length - 1), 2)} to the total variance of {fmt(after.totalVariance, 2)}; PC1's share moves from {(before.explained[0] * 100).toFixed(1)} % to {(after.explained[0] * 100).toFixed(1)} %.
            </>
          ),
          chart: (
            <ScatterSVG
              points={X2}
              width={520}
              height={320}
              highlight={[X2.length - 1]}
              vectors={[
                { x: before.V[0][0] * L, y: before.V[1][0] * L, color: methodColor.PCA, label: 'PC1 without outlier' },
                { x: after.V[0][0] * L, y: after.V[1][0] * L, color: '#e34948', label: 'PC1 with outlier' },
              ]}
              pointRadius={3.5}
            />
          ),
        };
      },
    },
  ];
}

export default function Quiz() {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [predictions, setPredictions] = useState<Record<string, number>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const scenarios = useMemo(buildScenarios, []);
  const results = useMemo(() => Object.fromEntries(scenarios.map((s) => [s.id, s.compute()])), [scenarios]);

  const answered = Object.keys(answers).length;
  const correct = questions.filter((q) => answers[q.id] === q.correct).length;
  const predRevealed = scenarios.filter((s) => revealed[s.id]).length;
  const predRight = scenarios.filter((s) => revealed[s.id] && predictions[s.id] === results[s.id].correct).length;

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>Assessment mode</h2>
          <div className="lede">Questions that test reasoning rather than memory, followed by experiments in which you must commit to a prediction before the computation is revealed.</div>
        </div>
        <div className="chip">
          <span className="dot" />
          <span>
            <b>{correct}</b> / {answered} correct · predictions <b>{predRight}</b> / {predRevealed}
          </span>
        </div>
      </div>

      <Section id="quiz-mc" title={`Part A · ${questions.length} reasoning questions`} subtitle="Choose one option; the explanation states why it is right and why each alternative is wrong." right={<Button small onClick={() => setAnswers({})}>Reset answers</Button>}>
        <div className="stack" style={{ gap: 14 }}>
          {questions.map((q, qi) => {
            const chosen = answers[q.id];
            return (
              <Card key={q.id}>
                <div className="row between">
                  <div>
                    <span className="badge neutral">{q.topic}</span>{' '}
                    <b>
                      {qi + 1}. {q.prompt}
                    </b>
                  </div>
                </div>
                <div style={{ marginTop: 8 }}>
                  {q.options.map((opt, oi) => {
                    const cls = chosen === undefined ? '' : oi === q.correct ? 'correct' : oi === chosen ? 'wrong' : '';
                    return (
                      <button key={oi} type="button" className={`quiz-option ${cls} ${chosen === oi ? 'selected' : ''}`} onClick={() => chosen === undefined && setAnswers((a) => ({ ...a, [q.id]: oi }))} disabled={chosen !== undefined}>
                        <span className="muted" style={{ marginRight: 8 }}>
                          {String.fromCharCode(97 + oi)})
                        </span>
                        {opt}
                      </button>
                    );
                  })}
                </div>
                {chosen !== undefined && (
                  <Callout kind={chosen === q.correct ? 'good' : 'warning'} title={chosen === q.correct ? 'Correct' : `Not quite — the answer is (${String.fromCharCode(97 + q.correct)})`}>
                    {q.explanation}
                  </Callout>
                )}
              </Card>
            );
          })}
        </div>
      </Section>

      <Section id="quiz-predict" title="Part B · Predict, then reveal" subtitle="Each scenario is computed live from a seeded simulation. Commit to a prediction, then reveal the actual numbers.">
        <div className="stack" style={{ gap: 14 }}>
          {scenarios.map((s, si) => {
            const pred = predictions[s.id];
            const rev = revealed[s.id];
            const res = results[s.id];
            return (
              <Card key={s.id}>
                <b>
                  {si + 1}. {s.title}
                </b>
                <div className="secondary" style={{ marginTop: 4 }}>
                  {s.setup}
                </div>
                <div style={{ marginTop: 6 }}>{s.question}</div>
                <div style={{ marginTop: 6 }}>
                  {s.options.map((opt, oi) => {
                    const cls = rev ? (oi === res.correct ? 'correct' : oi === pred ? 'wrong' : '') : pred === oi ? 'selected' : '';
                    return (
                      <button key={oi} type="button" className={`quiz-option ${cls}`} disabled={rev} onClick={() => setPredictions((p) => ({ ...p, [s.id]: oi }))}>
                        <span className="muted" style={{ marginRight: 8 }}>
                          {String.fromCharCode(97 + oi)})
                        </span>
                        {opt}
                      </button>
                    );
                  })}
                </div>
                <div className="row" style={{ marginTop: 6 }}>
                  <Button primary small disabled={pred === undefined || rev} onClick={() => setRevealed((r) => ({ ...r, [s.id]: true }))}>
                    {rev ? 'Revealed' : pred === undefined ? 'Choose a prediction first' : 'Reveal the computation'}
                  </Button>
                  {rev && (
                    <Button small onClick={() => { setRevealed((r) => ({ ...r, [s.id]: false })); setPredictions((p) => { const q = { ...p }; delete q[s.id]; return q; }); }}>
                      Try again
                    </Button>
                  )}
                </div>
                {rev && (
                  <div className="grid c2" style={{ marginTop: 10 }}>
                    <Callout kind={pred === res.correct ? 'good' : 'warning'} title={pred === res.correct ? 'Your prediction was right' : `Your prediction was wrong — (${String.fromCharCode(97 + res.correct)}) is what happens`}>
                      {res.result}
                    </Callout>
                    <div>{res.chart}</div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </Section>

      <Section id="quiz-score" title="Score">
        <div className="stats">
          <StatTile label="Questions answered" value={`${answered} / ${questions.length}`} />
          <StatTile label="Correct" value={correct} note={answered ? `${((100 * correct) / answered).toFixed(0)} % of answered` : 'no answers yet'} />
          <StatTile label="Predictions revealed" value={`${predRevealed} / ${scenarios.length}`} />
          <StatTile label="Predictions right" value={predRight} />
        </div>
        <Callout kind="info" title="How to use the score">
          A wrong answer with a clear explanation is worth more than a lucky right one. Each explanation links to a laboratory where the effect can be manipulated: scaling (<Badge method="PCA" /> lesson 5), the K − 1 bound and singular <M tex="S_W" /> (<Badge method="LDA" /> lesson 4), Eckart–Young (<Badge method="SVD" /> lesson 3), Euclidean equivalence and negative eigenvalues (<Badge method="MDS" /> lessons 3–4).
        </Callout>
        <MBlock tex={String.raw`\text{Remember: } \quad S_T = S_W + S_B,\qquad \lambda_j = \sigma_j^2/(n-1),\qquad B = X_cX_c^T \text{ (Euclidean)},\qquad \operatorname{rank}(S_B)\le K-1.`} />
        <div className="small muted">Marker colours in the charts follow the fixed categorical palette ({classColor(0)}, {classColor(1)}); text stays in ink {ink.primary}.</div>
      </Section>
    </div>
  );
}
