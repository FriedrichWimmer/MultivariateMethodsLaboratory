import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { useAnalysis, useStore } from '../../../state/store';
import { M, MBlock } from '../../common/Math';
import { Plot } from '../../common/Plot';
import { Section, Card, Callout, Interpretation, StatTile, Badge, Accordion } from '../../common/Panels';
import { Slider, Button } from '../../common/Controls';
import { fmt } from '../../../lib/linalg';
import { methodColor, neutralMark, ink, withAlpha, categorical } from '../../../lib/theme';
import { pct, sci, texNum, pcLabels, brokenStick, elbowIndex } from './util';

/** Lesson 3 — the scree plot, proportions of variance and the choice of k. */
export function Scree() {
  const a = useAnalysis();
  const { prep, setPrep, navigate } = useStore();
  const { pca: P, n, p, dataset } = a;

  const c = useMemo(() => {
    const lam = P.eigenvalues;
    const r = lam.length;
    const sigma = P.singularValues;
    const expl = P.explained;
    const cum = P.cumulative;
    const total = P.totalVariance;
    const meanLam = total / p; // trace(S)/p over all p eigenvalues (zeros included)
    const kaiser = lam.filter((l) => l > meanLam).length;
    const bs = brokenStick(p);
    let bsK = 0;
    for (let j = 0; j < r; j++) {
      if (expl[j] > bs[j]) bsK++;
      else break;
    }
    const elbow = elbowIndex(lam);
    return { lam, r, sigma, expl, cum, total, meanLam, kaiser, bs, bsK, elbow };
  }, [P, p]);

  const kmax = Math.max(1, Math.min(p, n - 1, c.r));
  const k = Math.min(Math.max(1, prep.k), kmax);
  const retained = c.cum[k - 1];
  const labels = pcLabels(c.r);
  const std = prep.scaling === 'standardize';

  const barData: Data[] = [
    { type: 'bar', x: labels.slice(0, k), y: c.lam.slice(0, k), name: `retained (j ≤ k = ${k})`, marker: { color: methodColor.PCA }, hovertemplate: '%{x}: λ = %{y:.4f}<extra></extra>' },
    { type: 'bar', x: labels.slice(k), y: c.lam.slice(k), name: 'discarded (j > k)', marker: { color: withAlpha(methodColor.PCA, 0.35) }, hovertemplate: '%{x}: λ = %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'lines', x: labels, y: labels.map(() => c.meanLam), name: `Kaiser threshold: mean λ = ${fmt(c.meanLam, 3)}`, line: { color: neutralMark, dash: 'dash', width: 1.5 }, hoverinfo: 'skip' },
  ];
  const barLayout: Record<string, unknown> = {
    showlegend: true,
    barmode: 'overlay',
    xaxis: { title: { text: 'component j' } },
    yaxis: { title: { text: 'eigenvalue λ_j (variance of PC_j)' }, rangemode: 'tozero' },
  };

  const propData: Data[] = [
    { type: 'scatter', mode: 'lines+markers', x: labels, y: c.expl, name: 'proportion λ_j / Σλ', line: { color: categorical[0] }, marker: { color: categorical[0], size: 7 }, hovertemplate: '%{x}: %{y:.1%}<extra></extra>' },
    { type: 'scatter', mode: 'lines+markers', x: labels, y: c.cum, name: 'cumulative', line: { color: categorical[1] }, marker: { color: categorical[1], size: 7 }, hovertemplate: '%{x}: %{y:.1%} cumulative<extra></extra>' },
    { type: 'scatter', mode: 'lines+markers', x: labels, y: c.bs.slice(0, c.r), name: 'broken-stick expectation b_j', line: { color: neutralMark, dash: 'dot', width: 1.5 }, marker: { color: neutralMark, size: 6, symbol: 'diamond-open' }, hovertemplate: '%{x}: b = %{y:.1%}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: [labels[k - 1]], y: [retained], name: `retained at k = ${k}`, marker: { color: ink.primary, size: 13, symbol: 'circle-open', line: { width: 2, color: ink.primary } }, hovertemplate: 'k = %{x}: %{y:.1%}<extra></extra>' },
  ];
  const propLayout: Record<string, unknown> = {
    showlegend: true,
    xaxis: { title: { text: 'component j' } },
    yaxis: { title: { text: 'proportion of total variance' }, tickformat: '.0%', range: [0, 1.06] },
  };

  const heuristics = [
    {
      title: `Kaiser rule — retain λ_j > mean λ (here λ̄ = ${fmt(c.meanLam, 3)}${std ? ', i.e. the classical "λ > 1"' : ''}): keeps ${c.kaiser} component${c.kaiser === 1 ? '' : 's'}`,
      body: (
        <div className="prose">
          <p>
            A component that explains less than an average variable's worth of variance, <M tex="\lambda_j < \operatorname{tr} S / p" />, is judged not worth keeping. For correlation PCA{' '}
            <M tex="\operatorname{tr} R = p" />, so the threshold is <M tex="\lambda_j > 1" />. Limitation: it ignores sampling variability entirely — with many variables and modest <M tex="n" /> the leading sample
            eigenvalues are biased upwards and the trailing ones downwards, so the rule tends to retain too many components in large <M tex="p" /> and is sensitive to the scaling choice of lesson 5.
          </p>
        </div>
      ),
    },
    {
      title: `Broken-stick rule — retain while proportion_j > b_j: keeps ${c.bsK} component${c.bsK === 1 ? '' : 's'}`,
      body: (
        <div className="prose">
          <p>
            If a stick of unit length is broken at <M tex="p-1" /> uniformly random points, the expected length of the <M tex="j" />-th longest piece is
          </p>
          <MBlock tex={String.raw`b_j = \frac{1}{p}\sum_{i=j}^{p}\frac{1}{i}, \qquad b_1 = ${texNum(c.bs[0], 4)},\ b_2 = ${texNum(c.bs[1] ?? 0, 4)},\ \dots`} />
          <p>
            Components whose proportion of variance exceeds the corresponding <M tex="b_j" /> are declared "more structured than chance". The null model (random partition of the total variance) is arbitrary;
            the rule is conservative and knows nothing about the scientific question.
          </p>
        </div>
      ),
    },
    {
      title: `Elbow (scree) — components up to the elbow: here j = ${c.elbow}`,
      body: (
        <div className="prose">
          <p>
            Cattell's scree test looks for the point where the eigenvalues stop falling steeply and start forming a flat "rubble" of noise-like values. Implemented here as the scree point farthest from the chord
            joining <M tex="(1,\lambda_1)" /> and <M tex="(r,\lambda_r)" /> after rescaling both axes to <M tex="[0,1]" />. It is a visual heuristic: with gradually decaying eigenvalues there is no elbow, and
            different analysts (and algorithms) locate it differently. Cross-validated reconstruction error or Horn's parallel analysis (comparing to eigenvalues of permuted data) are principled alternatives.
          </p>
        </div>
      ),
    },
  ];

  return (
    <Section
      id="pca-scree"
      title="3 · How many components? Eigenvalues, proportions and the scree plot"
      subtitle="Each λ_j is a variance; their sum is the total variance, so cumulative proportions tell how much dispersion a k-dimensional projection keeps."
      right={<Badge method="PCA" />}
    >
      <div className="prose">
        <p>
          Because the scores are uncorrelated with variances <M tex="\lambda_1 \ge \dots \ge \lambda_r" /> and <M tex="\sum_j \lambda_j = \operatorname{tr} S" />, the fraction of total variance kept by the
          first <M tex="k" /> components is
        </p>
        <MBlock tex={String.raw`\frac{\lambda_1 + \dots + \lambda_k}{\lambda_1 + \dots + \lambda_r} = \frac{\sigma_1^{2} + \dots + \sigma_k^{2}}{\|X_c\|_F^{2}} .`} />
        <p>
          The scree plot shows the eigenvalues in decreasing order; the second chart shows the individual and cumulative proportions on a common percentage axis. The slider sets the retained dimension{' '}
          <M tex="k" /> shared by the whole laboratory (it also drives the MDS embedding and the reconstruction of lesson 4).
        </p>
      </div>

      <div className="grid side">
        <div className="stack">
          <Card title="Retained dimension" plane>
            <Slider label="Retained components k" value={k} min={1} max={kmax} step={1} onChange={(v) => setPrep({ k: v })} />
            <div className="stats" style={{ marginTop: 6 }}>
              <StatTile label="cumulative variance at k" value={pct(retained)} note={`λ₁ + … + λ_${k} = ${fmt(c.cum[k - 1] * c.total, 3)}`} />
              <StatTile label="Kaiser rule keeps" value={String(c.kaiser)} note={`λ_j > ${fmt(c.meanLam, 3)}`} />
              <StatTile label="broken stick keeps" value={String(c.bsK)} note="proportion_j > b_j" />
              <StatTile label="elbow at" value={`j = ${c.elbow}`} note="farthest from chord" />
            </div>
          </Card>
          <div className="pipeline">
            <div className="stage">
              <div className="stage-title">Original dimension</div>
              <div>
                <M tex={`p = ${p}`} /> variables
              </div>
              <div className="small muted">
                total variance <M tex={`\\operatorname{tr} S = ${texNum(c.total, 3)}`} />
              </div>
            </div>
            <div className="arrow">→</div>
            <div className="stage">
              <div className="stage-title">Retained</div>
              <div>
                <M tex={`k = ${k}`} /> component{k === 1 ? '' : 's'}
              </div>
              <div className="small muted">
                <M tex={`\\sum_{j \\le k}\\lambda_j = ${texNum(retained * c.total, 3)}`} />
              </div>
            </div>
            <div className="arrow">→</div>
            <div className="stage">
              <div className="stage-title">Information retained</div>
              <div>
                <b>{pct(retained)}</b> of the variance
              </div>
              <div className="small muted">
                discarded: {p - k} direction{p - k === 1 ? '' : 's'}, {pct(1 - retained)}
              </div>
            </div>
          </div>
        </div>
        <div className="stack">
          <Plot data={barData} layout={barLayout} height={300} title="Scree plot: eigenvalues of S" />
          <Plot data={propData} layout={propLayout} height={280} title="Proportion and cumulative proportion of variance explained" />
        </div>
      </div>

      <Card title="Eigenvalues, singular values and proportions">
        <div className="table-wrap">
          <table className="summary-table">
            <thead>
              <tr>
                <th>
                  <M tex="j" />
                </th>
                <th>
                  <M tex="\lambda_j" />
                </th>
                <th>
                  <M tex="\sigma_j = \sqrt{(n-1)\lambda_j}" />
                </th>
                <th>proportion</th>
                <th>cumulative</th>
                <th>
                  <M tex="\lambda_j > \bar\lambda" />
                </th>
                <th>
                  broken stick <M tex="b_j" />
                </th>
                <th>status</th>
              </tr>
            </thead>
            <tbody>
              {c.lam.map((l, j) => (
                <tr key={j} style={j < k ? { fontWeight: 600 } : undefined}>
                  <td>{j + 1}</td>
                  <td>{fmt(l, 4)}</td>
                  <td>{fmt(c.sigma[j], 4)}</td>
                  <td>{pct(c.expl[j])}</td>
                  <td>{pct(c.cum[j])}</td>
                  <td>{l > c.meanLam ? 'yes' : 'no'}</td>
                  <td>
                    {pct(c.bs[j])} {c.expl[j] > c.bs[j] ? '(exceeded)' : ''}
                  </td>
                  <td>{j < k ? 'retained' : 'discarded'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Callout kind="warning" title='"Variance explained" is not "scientific importance"'>
        PCA is unsupervised: it never sees a response or a class label. A direction of large variance may be measurement scale (lesson 5), a nuisance factor such as batch or size, or genuinely uninformative
        spread; conversely a low-variance direction can carry all the predictive signal. On the current dataset PC1 explains {pct(c.expl[0])} of the variance — that number says nothing about whether PC1
        separates groups or predicts anything.{' '}
        <Button small onClick={() => navigate('lda', 'lda-vs-pca')}>
          See variance versus separation in the LDA laboratory
        </Button>
      </Callout>

      <Accordion items={heuristics} />

      <Interpretation
        items={{
          seeing: (
            <>
              For <b>{dataset.name}</b> the eigenvalues fall from λ₁ = {fmt(c.lam[0], 4)} ({pct(c.expl[0])}) to λ_{c.r} = {fmt(c.lam[c.r - 1], 4)} ({pct(c.expl[c.r - 1])}). With k = {k} the retained components
              hold {pct(retained)} of the total variance {fmt(c.total, 3)}; the three heuristics suggest {c.kaiser} (Kaiser), {c.bsK} (broken stick) and {c.elbow} (elbow) components — {c.kaiser === c.bsK && c.bsK === c.elbow ? 'they agree here, which is not guaranteed' : 'they disagree, which is common'}.
            </>
          ),
          why: (
            <>
              A steep drop between consecutive eigenvalues means the data are concentrated near a low-dimensional subspace; the largest ratio here is λ₁/λ₂ = {c.r > 1 && c.lam[1] > 0 ? fmt(c.lam[0] / c.lam[1], 2) : '∞'}.
              The cumulative curve is concave because the eigenvalues are sorted: each additional component adds less than the previous one, so the marginal value of dimension k + 1 is λ_{k + 1}
              {k < c.r ? ` = ${fmt(c.lam[k], 4)} (${pct(c.expl[k])})` : ' = 0 (everything is already retained)'}.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\sum_{j=1}^{r}\lambda_j = \operatorname{tr} S, \qquad \text{proportion}_k = \frac{\lambda_k}{\operatorname{tr} S}, \qquad \text{cumulative}_k = \frac{\sum_{j\le k}\lambda_j}{\operatorname{tr} S} = 1 - \frac{\|X_c - \hat X_k\|_F^{2}}{\|X_c\|_F^{2}} .`} />
              The last equality (Eckart–Young, lesson 4) identifies "variance discarded" with the squared relative reconstruction error: at k = {k} that error is {pct(1 - retained)} of{' '}
              <M tex="\|X_c\|_F^{2}" />.
            </>
          ),
          stats: (
            <>
              {std ? (
                <>
                  Under standardisation tr R = p = {p}, so λ̄ = 1 and an eigenvalue below 1 means the component explains less than one original variable. Here {c.kaiser} eigenvalue{c.kaiser === 1 ? ' exceeds' : 's exceed'} 1.
                </>
              ) : (
                <>
                  The eigenvalues are in squared measurement units; the Kaiser threshold λ̄ = tr S / p = {fmt(c.meanLam, 3)} is the average variance per variable. If the variables have different units this comparison
                  is not meaningful — standardise first (lesson 5).
                </>
              )}{' '}
              Sample eigenvalues are biased: the largest overestimates and the smallest underestimates the population values, increasingly so as p/n = {fmt(p / n, 2)} grows.
            </>
          ),
          careful: (
            <>
              None of the rules is a test of anything; they are conventions. Retaining k = {k} components for downstream regression or classification should be judged by predictive performance, not by{' '}
              {pct(retained)}. And the rank cap matters: with n = {n} observations at most min(n − 1, p) = {Math.min(n - 1, p)} eigenvalues can be non-zero, so a high cumulative proportion is partly an artefact when
              p is close to n.
            </>
          ),
        }}
      />
    </Section>
  );
}
