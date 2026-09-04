import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { column, fmt, trace } from '../../../lib/linalg';
import { fisherCriterion, lda, ldaClassify } from '../../../lib/lda';
import { defaultParams, generateDataset } from '../../../lib/datasets';
import { ink, methodColor, neutralMark } from '../../../lib/theme';
import { useStore } from '../../../state/store';
import { M, MBlock } from '../../common/Math';
import { Button, Slider } from '../../common/Controls';
import { Accordion, Badge, Callout, Card, Interpretation, Section, StatTile } from '../../common/Panels';
import { Plot } from '../../common/Plot';
import { eigenRank, pct } from './helpers';

/** Lesson 4 — statistical constraints: rank of S_B, singular S_W, regularisation, assumptions. */
export default function ConstraintsSection() {
  const { navigate } = useStore();
  const [K, setK] = useState(3);
  const [gamma, setGamma] = useState(0);

  // (a) rank of S_B and the number of discriminants
  const dsK = useMemo(() => generateDataset('clusters', { ...defaultParams, n: 200, p: 5, K, separation: 3, seed: 5 }), [K]);
  const yK = useMemo(() => dsK.y ?? [], [dsK]);
  const resK = useMemo(() => lda(dsK.X, yK), [dsK, yK]);
  const sbRank = eigenRank(resK.sbEigenvalues);
  const sbTol = 1e-10 * Math.max(resK.sbEigenvalues[0] ?? 0, 1e-300);
  const sbData = useMemo<Data[]>(
    () => [
      {
        x: resK.sbEigenvalues.map((_, j) => `μ${j + 1}`),
        y: resK.sbEigenvalues.map((v) => Math.max(v, 0)),
        type: 'bar',
        name: 'eigenvalues of S_B',
        marker: { color: resK.sbEigenvalues.map((v) => (v > sbTol ? methodColor.LDA : ink.axis)) },
        hovertemplate: '%{x}: %{y:.4g}<extra></extra>',
      },
    ],
    [resK, sbTol],
  );
  const ldaEigData = useMemo<Data[]>(
    () => [
      {
        x: resK.eigenvalues.map((_, j) => `λ${j + 1}`),
        y: resK.eigenvalues,
        type: 'bar',
        name: 'generalised eigenvalues',
        marker: { color: methodColor.LDA },
        hovertemplate: '%{x}: %{y:.4g}<extra></extra>',
      },
    ],
    [resK],
  );
  const barLayout = useMemo<Record<string, unknown>>(() => ({ xaxis: { title: { text: 'index' } }, yaxis: { title: { text: 'value' }, rangemode: 'tozero' } }), []);

  // (b) singular S_W when p > n, and regularisation
  const dsWide = useMemo(() => generateDataset('pGreaterN', { ...defaultParams, n: 12, p: 20, K: 2 }), []);
  const yWide = useMemo(() => dsWide.y ?? [], [dsWide]);
  const res0 = useMemo(() => lda(dsWide.X, yWide), [dsWide, yWide]);
  const resR = useMemo(() => (gamma > 0 ? lda(dsWide.X, yWide, { regularization: gamma }) : res0), [dsWide, yWide, gamma, res0]);
  const pW = res0.p;
  const nW = res0.n;
  const KW = res0.K;
  const ridge = (gamma * trace(res0.SW)) / pW;
  const w0 = useMemo(() => (res0.maxDims > 0 ? column(res0.W, 0) : new Array(pW).fill(0)), [res0, pW]);
  const wR = useMemo(() => (resR.maxDims > 0 ? column(resR.W, 0) : new Array(pW).fill(0)), [resR, pW]);
  const fcRaw = useMemo(() => fisherCriterion(res0.SB, res0.SW, wR), [res0, wR]);
  const condR = gamma > 0 ? (res0.swEigenvalues[0] + ridge) / (Math.max(res0.swEigenvalues[pW - 1] ?? 0, 0) + ridge) : res0.swCondition;
  const accR = useMemo(() => {
    const cls = ldaClassify(resR, dsWide.X);
    return cls.accuracy ? cls.accuracy(yWide) : NaN;
  }, [resR, dsWide, yWide]);
  const swTol = 1e-10 * Math.max(res0.swEigenvalues[0] ?? 0, 1e-300);
  const swData = useMemo<Data[]>(
    () => [
      {
        x: res0.swEigenvalues.map((_, j) => `${j + 1}`),
        y: res0.swEigenvalues.map((v) => Math.max(v, 0)),
        type: 'bar',
        name: 'eigenvalues of S_W',
        marker: { color: res0.swEigenvalues.map((v) => (v > swTol ? methodColor.LDA : ink.axis)) },
        hovertemplate: 'eigenvalue %{x}: %{y:.4g}<extra></extra>',
      },
    ],
    [res0, swTol],
  );
  const swLayout = useMemo<Record<string, unknown>>(
    () => ({
      xaxis: { title: { text: 'eigenvalue index (p = 20)' } },
      yaxis: { title: { text: 'eigenvalue of S_W' }, rangemode: 'tozero' },
      shapes: gamma > 0 ? [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: ridge, y1: ridge, line: { color: ink.primary, width: 1.5, dash: 'dash' } }] : [],
      annotations: gamma > 0 ? [{ xref: 'paper', yref: 'y', x: 0.99, y: ridge, text: `ridge γ·tr(S_W)/p = ${fmt(ridge, 2)}`, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11, color: ink.secondary } }] : [],
    }),
    [gamma, ridge],
  );
  const wData = useMemo<Data[]>(
    () => [
      { x: dsWide.variableNames, y: w0, type: 'bar', name: 'γ = 0 (pseudo-inverse on the range of S_W)', marker: { color: neutralMark }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
      { x: dsWide.variableNames, y: wR, type: 'bar', name: `γ = ${gamma.toFixed(2)}`, marker: { color: methodColor.LDA }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
    ],
    [dsWide, w0, wR, gamma],
  );
  const wLayout = useMemo<Record<string, unknown>>(() => ({ barmode: 'group', showlegend: true, xaxis: { title: { text: 'variable' } }, yaxis: { title: { text: 'coefficient of w₁ (unit norm)' } } }), []);
  const wChange = useMemo(() => Math.sqrt(w0.reduce((a, v, j) => a + (v - wR[j]) * (v - wR[j]), 0)), [w0, wR]);

  return (
    <Section id="lda-constraints" title="4 · Statistical constraints" subtitle="How many discriminants can exist, what happens when S_W is singular, and which assumptions the method carries" right={<Badge method="LDA" />}>
      <Card title="(a) At most K − 1 meaningful discriminant directions">
        <div className="grid side">
          <div className="stack">
            <Slider label="Number of classes K (local clusters data, n = 200, p = 5)" value={K} min={2} max={5} step={1} onChange={setK} />
            <div className="stats">
              <StatTile label={<M tex="K - 1" />} value={K - 1} note="upper bound on rank S_B" />
              <StatTile label={<M tex="\operatorname{rank} S_B" />} value={sbRank} note={`of p = ${resK.p} eigenvalues, ${resK.p - sbRank} are numerically zero`} />
              <StatTile label={<M tex="m" />} value={resK.maxDims} note="discriminant directions returned" />
              <StatTile label={<M tex="\operatorname{rank} S_W" />} value={resK.swRank} note={`n − K = ${resK.n - K} ≥ p = ${resK.p}`} />
            </div>
            <div className="prose small">
              <p>
                <M tex="S_B = \sum_{k=1}^{K} n_k (m_k - m)(m_k - m)^{\mathsf T}" /> is a sum of <M tex="K" /> rank-one matrices, so <M tex="\operatorname{rank} S_B \le K" />. But the vectors are
                linearly dependent, because <M tex="\sum_k n_k (m_k - m) = \sum_k n_k m_k - n\,m = 0" />: any one of them is a combination of the others. Hence
              </p>
              <MBlock tex="\operatorname{rank} S_B \le K - 1 ." />
              <p>
                The generalised eigenproblem <M tex="S_B w = \lambda S_W w" /> can therefore have at most <M tex="K-1" /> non-zero eigenvalues; every further direction has{' '}
                <M tex="w^{\mathsf T} S_B w = 0" /> and carries no information about the class means. With <M tex="K = 2" /> there is exactly one discriminant direction, whatever <M tex="p" /> is.
              </p>
            </div>
          </div>
          <div className="grid c2">
            <Plot data={sbData} layout={barLayout} height={280} title={`Eigenvalues of S_B (p = ${resK.p}): ${sbRank} non-zero for K = ${K}`} />
            <Plot data={ldaEigData} layout={barLayout} height={280} title={`Generalised eigenvalues λ_j, m = ${resK.maxDims}`} />
          </div>
        </div>
        <Interpretation
          items={{
            seeing: (
              <>
                Left: the {resK.p} eigenvalues of <M tex="S_B" /> for <M tex={`K = ${K}`} /> classes in <M tex="p = 5" /> dimensions; {sbRank} are positive (
                {resK.sbEigenvalues
                  .slice(0, sbRank)
                  .map((v) => fmt(v, 1))
                  .join(', ')}
                ) and the remaining {resK.p - sbRank} are zero to machine precision (largest {fmt(resK.sbEigenvalues[sbRank] ?? 0, 2)}). Right: the <M tex={`m = ${resK.maxDims}`} /> Fisher
                ratios <M tex={`(${resK.eigenvalues.map((v) => fmt(v, 2)).join(',\\ ')})`} />.
              </>
            ),
            why: (
              <>
                The <M tex={`K = ${K}`} /> centred class means <M tex="m_k - m" /> span at most a <M tex={`${K - 1}`} />-dimensional subspace because their size-weighted sum vanishes. Moving the
                slider adds one rank-one term and one non-zero eigenvalue at a time; it never produces more than <M tex="K - 1" /> of them, even though <M tex="p = 5" /> would allow five.
              </>
            ),
            math: (
              <>
                <M tex="\operatorname{rank}(A + B) \le \operatorname{rank} A + \operatorname{rank} B" /> gives <M tex="\operatorname{rank} S_B \le K" />; the constraint{' '}
                <M tex="\sum_k n_k (m_k - m) = 0" /> removes one dimension. The generalised eigenvalues are the eigenvalues of <M tex="S_W^{-1/2} S_B S_W^{-1/2}" />, a congruence
                transformation of <M tex="S_B" />, which preserves rank: exactly <M tex="\operatorname{rank} S_B" /> of them are non-zero.
              </>
            ),
            stats: (
              <>
                LDA as dimension reduction can never produce more than <M tex="K-1" /> axes; for a binary problem it is a one-dimensional score, however many variables were measured. The
                trailing eigenvalues (here {resK.eigenvalues.length > 1 ? fmt(resK.eigenvalues[resK.eigenvalues.length - 1], 2) : fmt(resK.eigenvalues[0] ?? 0, 2)} for the last
                direction) tell you how much separation the later axes still add.
              </>
            ),
            careful: (
              <>
                "At most <M tex="K-1" />" is an upper bound: if class means are collinear the rank drops further. Numerical zero is a tolerance decision (relative <M tex="10^{-10}" /> here);
                the laboratory's <M tex="m" /> is <M tex="\min(K-1, \operatorname{rank} S_W, \operatorname{rank} S_B)" />.
              </>
            ),
          }}
        />
      </Card>

      <Card title="(b) Singular S_W when p > n − K, and regularised LDA">
        <div className="grid side">
          <div className="stack">
            <div className="small secondary">
              Local data: <M tex={`n = ${nW}`} /> observations, <M tex={`p = ${pW}`} /> variables, <M tex={`K = ${KW}`} /> classes. The pooled deviations <M tex="x_i - m_k" /> span at most{' '}
              <M tex={`n - K = ${nW - KW}`} /> dimensions, so <M tex="S_W" /> has rank <M tex={`\\le ${nW - KW}`} /> and is singular in <M tex="\mathbb{R}^{20}" />.
            </div>
            <Slider label="Regularisation γ" value={gamma} min={0} max={1} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} />
            <div className="stats">
              <StatTile label={<M tex="\operatorname{rank} S_W" />} value={res0.swRank} note={`n − K = ${nW - KW}; p = ${pW}`} />
              <StatTile label={<M tex="\kappa(S_W)" />} value={fmt(res0.swCondition, 1)} note={res0.swSingular ? 'singular: flagged by lda()' : 'invertible'} />
              <StatTile label={<M tex="\gamma\,\operatorname{tr}(S_W)/p" />} value={fmt(ridge, 3)} note="ridge added to the diagonal" />
              <StatTile label={<M tex="\kappa(S_W + \text{ridge}\cdot I)" />} value={fmt(condR, 1)} note={gamma > 0 ? 'regularised problem is well posed' : 'no regularisation'} />
              <StatTile label={<M tex="\lambda_1(\gamma)" />} value={fmt(resR.eigenvalues[0] ?? 0, 2)} note="regularised Fisher ratio" />
              <StatTile label={<M tex="J(w_1)" />} value={fmt(fcRaw.J, 2)} note={`raw matrices: ${fmt(fcRaw.between, 2)} / ${fmt(fcRaw.within, 4)}`} />
              <StatTile label="training accuracy" value={pct(accR)} note={`Gaussian LDA rule on the ${nW} training points`} />
              <StatTile label={<M tex="\|w_1(\gamma) - w_1(0)\|" />} value={fmt(wChange, 3)} note="shift of the unit direction" />
            </div>
          </div>
          <div className="grid c2">
            <Plot data={swData} layout={swLayout} height={300} title={`Eigenvalues of S_W: ${res0.swRank} positive, ${pW - res0.swRank} zero`} />
            <Plot data={wData} layout={wLayout} height={300} title="Leading discriminant direction w₁ without and with regularisation" />
          </div>
        </div>
        <MBlock tex="\text{Regularised LDA:}\qquad S_B w = \lambda\,\bigl(S_W + \gamma\,\tfrac{\operatorname{tr} S_W}{p}\, I\bigr)\, w, \qquad \gamma \ge 0 ." />
        <Interpretation
          items={{
            seeing: (
              <>
                Left: of the <M tex={`p = ${pW}`} /> eigenvalues of <M tex="S_W" /> only {res0.swRank} are positive; the other {pW - res0.swRank} are exactly the directions in which the{' '}
                {nW} training points show no within-class variation at all. Right: the unit direction <M tex="w_1" /> with <M tex="\gamma = 0" /> (grey) and with the current{' '}
                <M tex={`\\gamma = ${gamma.toFixed(2)}`} /> (orange); the two differ by <M tex={`${fmt(wChange, 3)}`} /> in Euclidean norm.
              </>
            ),
            why: (
              <>
                With <M tex="p = 20 > n - K = 10" /> the matrix <M tex="S_W" /> cannot be invertible: <M tex={`\\kappa(S_W) = ${fmt(res0.swCondition, 1)}`} />. Any direction in its null
                space has <M tex="w^{\mathsf T} S_W w = 0" />, so <M tex="J(w) = \infty" /> as soon as <M tex="w^{\mathsf T} S_B w > 0" /> — the training classes are perfectly separable by
                construction, not by insight. The current <M tex="w_1" /> attains <M tex={`w_1^{\\mathsf T} S_W w_1 = ${fmt(fcRaw.within, 4)}`} /> against{' '}
                <M tex={`w_1^{\\mathsf T} S_B w_1 = ${fmt(fcRaw.between, 2)}`} /> on the raw matrices, and the training accuracy is {pct(accR)}.
              </>
            ),
            math: (
              <>
                Adding <M tex="\gamma\,\operatorname{tr}(S_W)/p" /> to every eigenvalue makes <M tex="S_W + \text{ridge}\cdot I" /> positive definite with condition number{' '}
                <M tex={`${fmt(condR, 1)}`} />; the generalised eigenproblem is then well posed and its leading eigenvalue is <M tex={`\\lambda_1(\\gamma) = ${fmt(resR.eigenvalues[0] ?? 0, 2)}`} />.
                At <M tex="\gamma = 0" /> the laboratory instead solves on the range of <M tex="S_W" /> (a pseudo-inverse), which silently discards the null-space directions. As{' '}
                <M tex="\gamma \to \infty" /> the ridge dominates and <M tex="w_1 \to (m_1 - m_2)/\|m_1 - m_2\|" />, the naive mean-difference direction.
              </>
            ),
            stats: (
              <>
                Regularisation trades bias for variance: the shrunken <M tex="\hat\Sigma_\gamma = (1-\alpha)\hat\Sigma + \alpha\,\tfrac{\operatorname{tr}\hat\Sigma}{p} I" /> form is the
                Ledoit–Wolf / Friedman (RDA) estimator; <M tex="\gamma" /> should be chosen by cross-validation, never by looking at the training accuracy, which here is {pct(accR)} for
                every <M tex="\gamma" />. With <M tex="p/n = " /> {fmt(pW / nW, 2)} the sample covariance is so noisy that the unregularised direction is essentially an artefact of the
                sample.
              </>
            ),
            careful: (
              <>
                Singular <M tex="S_W" /> also arises with <M tex="p \le n" /> when variables are collinear (dummy coding of a factor with all levels, sums of other variables). A finite but
                large <M tex="\kappa(S_W)" /> is nearly as dangerous: <M tex="S_W^{-1}" /> then amplifies small sampling errors in <M tex="m_k" /> into large swings of <M tex="w" />.
              </>
            ),
          }}
        />
      </Card>

      <Accordion
        items={[
          {
            title: 'Full-rank requirement: n − K ≥ p',
            body: (
              <div className="prose small">
                <p>
                  <M tex="S_W = \sum_k \sum_{i\in k}(x_i - m_k)(x_i - m_k)^{\mathsf T}" /> is a sum of <M tex="n" /> rank-one terms, but each class contributes only <M tex="n_k - 1" />{' '}
                  independent deviations (they sum to zero), so <M tex="\operatorname{rank} S_W \le n - K" />. Invertibility therefore needs <M tex="n - K \ge p" /> and, additionally, no
                  exact linear relation among the variables. In practice one wants <M tex="n - K" /> to exceed <M tex="p" /> by a comfortable margin: with <M tex="n - K" /> only slightly
                  above <M tex="p" />, <M tex="S_W" /> is invertible but extremely ill-conditioned.
                </p>
              </div>
            ),
          },
          {
            title: 'Singular S_W: p > n, or collinear variables',
            body: (
              <div className="prose small">
                <p>
                  When <M tex="S_W" /> is singular the criterion <M tex="J(w)" /> is unbounded on the null space of <M tex="S_W" />, so the maximisation problem has no meaningful solution: any
                  direction that separates the training means and has zero within-class spread gives <M tex="J = \infty" /> and perfect training separation. This is the high-dimensional
                  analogue of fitting a line through two points. Remedies: reduce dimension first (PCA to <M tex="k < n - K" /> components, then LDA), regularise (next item), or use a
                  method that never inverts <M tex="S_W" /> (diagonal LDA, nearest-centroid rules).
                </p>
              </div>
            ),
          },
          {
            title: 'Multicollinearity: ill-conditioning inflates w',
            body: (
              <div className="prose small">
                <p>
                  Near-collinear variables make <M tex="S_W" /> nearly singular. Since <M tex="w_1 \propto S_W^{-1}(m_1 - m_2)" /> for two classes, the components of <M tex="w" /> along
                  the small eigenvectors of <M tex="S_W" /> are divided by tiny eigenvalues: the coefficients become huge, of opposite sign, and unstable across samples, even though the
                  projected scores <M tex="z = X_c w" /> may still be sensible. Interpreting individual coefficients of <M tex="w" /> as "importance" is then meaningless. The condition number
                  <M tex="\kappa(S_W) = \lambda_{\max}/\lambda_{\min}" /> reported by the laboratory is the diagnostic; the active dataset shows{' '}
                  <M tex={`\\kappa(S_W) = ${fmt(resK.swCondition, 1)}`} /> for the local clusters data of part (a).
                </p>
              </div>
            ),
          },
          {
            title: 'Regularised LDA and shrinkage',
            body: (
              <div className="prose small">
                <p>
                  Replace <M tex="S_W" /> by <M tex="S_W + \gamma\,\tfrac{\operatorname{tr} S_W}{p} I" /> (equivalently shrink the pooled covariance towards a scaled identity). The ridge
                  bounds the condition number, damps the small-eigenvalue directions and biases <M tex="w" /> towards the mean difference. Friedman's regularised discriminant analysis
                  also shrinks the class covariances towards their pool; the Ledoit–Wolf estimator chooses the shrinkage intensity analytically. In all cases the tuning constant must be
                  selected by cross-validation, since training accuracy is monotone-optimistic in flexibility.
                </p>
              </div>
            ),
          },
          {
            title: 'Gaussian class assumption and equal covariance (LDA versus QDA)',
            body: (
              <div className="prose small">
                <p>
                  Fisher's criterion needs no distribution: it is a ratio of sums of squares. The <em>classifier</em> in lesson 5 adds the model <M tex="x \mid y = k \sim N(\mu_k, \Sigma)" />{' '}
                  with a <b>common</b> <M tex="\Sigma" />. If the classes have different covariances the Bayes boundary is quadratic (QDA), and pooling them into one <M tex="S_W" /> gives a
                  compromise direction that can be badly wrong when one class is tight and another wide. The laboratory's unequal-covariance dataset demonstrates the failure.
                </p>
                <Button small onClick={() => navigate('wrong')}>
                  Open “What can go wrong?” (unequal covariances)
                </Button>
              </div>
            ),
          },
          {
            title: 'Two things called LDA: dimensionality reduction versus classification',
            body: (
              <div className="prose small">
                <p>
                  <b>Fisher's linear discriminant</b> is a projection method: find up to <M tex="K-1" /> directions maximising between-to-within scatter, then look at the scores. No
                  distribution is assumed and no decision is made. <b>Gaussian LDA classification</b> is a generative model with a Bayes decision rule; it yields linear discriminant functions{' '}
                  <M tex="\delta_k(x)" /> and posterior probabilities. The two coincide in an important special case — with two classes the Bayes direction{' '}
                  <M tex="\hat\Sigma^{-1}(m_1 - m_2)" /> is Fisher's <M tex="w_1" /> — and share <M tex="S_W" /> and the class means in general, which is why one name covers both. Lesson 5
                  keeps the terminology precise.
                </p>
              </div>
            ),
          },
        ]}
      />
      <Callout kind="info" title="Reading the flags returned by lda()">
        <M tex="\texttt{swSingular}" /> is true when <M tex="\operatorname{rank} S_W < p" /> at a relative tolerance of <M tex="10^{-10}" />; <M tex="\texttt{swCondition}" /> is{' '}
        <M tex="\lambda_{\max}(S_W)/\lambda_{\min}(S_W)" /> (infinite when singular); <M tex="\texttt{maxDims} = \min(K-1, \operatorname{rank} S_W, \operatorname{rank} S_B)" />. For the wide
        data above these read: singular = {String(res0.swSingular)}, rank = {res0.swRank}, condition = {fmt(res0.swCondition, 1)}, maxDims = {res0.maxDims}.
      </Callout>
    </Section>
  );
}
