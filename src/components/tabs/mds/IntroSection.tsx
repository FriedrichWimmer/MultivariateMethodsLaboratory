import { useMemo, type ReactNode } from 'react';
import type { Data, Layout, Shape } from 'plotly.js';
import { useAnalysis } from '../../../state/store';
import { Section, Card, Callout, Interpretation, StatTile, Badge } from '../../common/Panels';
import { Plot } from '../../common/Plot';
import { MatrixView } from '../../common/MatrixView';
import { M, MBlock } from '../../common/Math';
import { metricLabels, type Metric } from '../../../lib/mds';
import { fmt, colStds } from '../../../lib/linalg';
import { plotlySequential, ink } from '../../../lib/theme';
import { orderByClass, permuteMatrix, classBoundaries, distanceSummary, withinBetween, classNamesOf, texNum } from './helpers';

const metricTex: Record<Metric, string> = {
  euclidean: 'd_{il} = \\Bigl(\\sum_{j=1}^{p} (x_{ij} - x_{lj})^2\\Bigr)^{1/2}',
  manhattan: 'd_{il} = \\sum_{j=1}^{p} \\lvert x_{ij} - x_{lj} \\rvert',
  chebyshev: 'd_{il} = \\max_{1 \\le j \\le p} \\lvert x_{ij} - x_{lj} \\rvert',
  minkowski3: 'd_{il} = \\Bigl(\\sum_{j=1}^{p} \\lvert x_{ij} - x_{lj} \\rvert^{3}\\Bigr)^{1/3}',
  sqeuclidean: 'd_{il} = \\sum_{j=1}^{p} (x_{ij} - x_{lj})^2',
};

const metricRows: { metric: Metric; when: ReactNode; exact: ReactNode }[] = [
  {
    metric: 'euclidean',
    when: (
      <>
        The metric of ordinary geometry. Appropriate when the variables are commensurable (same units, or standardised) and straight-line separation in the variable space is meaningful. Invariant under rotations of the variable space, which is why PCA and MDS can agree.
      </>
    ),
    exact: (
      <>
        <b>Exact.</b> <M tex="B" /> is positive semi-definite with <M tex="\operatorname{rank}(B) \le \min(n-1,p)" />; <M tex="X_k" /> reproduces <M tex="D" /> exactly once <M tex="k = \operatorname{rank}(B)" />.
      </>
    ),
  },
  {
    metric: 'manhattan',
    when: (
      <>
        Sums absolute coordinate differences: each variable contributes linearly, so one large discrepancy is penalised less than under <M tex="L_2" />. A robust choice for count-like or ordinal-like variables, or when outlying coordinates should not dominate. Not rotation invariant.
      </>
    ),
    exact: (
      <>
        <b>Approximate.</b> <M tex="L_1" /> distances satisfy the triangle inequality but are generally not realisable by points in Euclidean space, so <M tex="B" /> acquires negative eigenvalues.
      </>
    ),
  },
  {
    metric: 'chebyshev',
    when: (
      <>
        Only the largest coordinate discrepancy counts: two observations are as different as their most different variable (tolerance-type comparisons). Highly sensitive to the scale of individual variables.
      </>
    ),
    exact: (
      <>
        <b>Approximate.</b> Generally non-Euclidean; negative eigenvalues of <M tex="B" /> are typical.
      </>
    ),
  },
  {
    metric: 'minkowski3',
    when: (
      <>
        Interpolates between <M tex="L_2" /> and <M tex="L_\infty" />: larger coordinate differences are weighted more heavily than under <M tex="L_2" />. Rarely a default; useful to see how the exponent shifts the emphasis.
      </>
    ),
    exact: (
      <>
        <b>Approximate.</b> Generally non-Euclidean.
      </>
    ),
  },
  {
    metric: 'sqeuclidean',
    when: (
      <>
        Not a metric — the triangle inequality fails — but the natural quantity of variance decompositions (Ward clustering, <M tex="k" />-means). Fed to classical MDS as if it were a distance, the algorithm squares it a second time.
      </>
    ),
    exact: (
      <>
        <b>Inexact.</b> <M tex="D^{(2)}" /> then holds fourth powers and <M tex="B" /> is indefinite. Classical MDS already squares its input: supplying squared distances is a common mistake, reproduced here on purpose.
      </>
    ),
  },
];

export default function IntroSection() {
  const a = useAnalysis();
  const { D, n, p, y, K, prep, dataset } = a;
  const metric = prep.metric;
  const classNames = useMemo(() => classNamesOf(dataset), [dataset]);
  const order = useMemo(() => orderByClass(y, n), [y, n]);
  const Dord = useMemo(() => permuteMatrix(D, order), [D, order]);
  const boundaries = useMemo(() => classBoundaries(y, order), [y, order]);
  const summary = useMemo(() => distanceSummary(D), [D]);
  const wb = useMemo(() => (y && K >= 2 ? withinBetween(D, y) : null), [D, y, K]);
  const sds = useMemo(() => colStds(dataset.X), [dataset]);
  const sdMax = Math.max(...sds);
  const sdMin = Math.min(...sds);
  const labels = useMemo(() => order.map((i) => `#${i + 1}${y && classNames ? ` · ${classNames[y[i]]}` : ''}`), [order, y, classNames]);

  const heat = useMemo<Data[]>(
    () => [
      {
        type: 'heatmap',
        z: Dord,
        x: labels,
        y: labels,
        colorscale: plotlySequential,
        zmin: 0,
        zmax: summary.max,
        hovertemplate: '%{y}<br>%{x}<br>d = %{z:.3f}<extra></extra>',
        colorbar: { thickness: 10, len: 0.92, title: { text: 'd', side: 'right' }, tickfont: { size: 10 } },
      },
    ],
    [Dord, labels, summary.max],
  );
  const shapes = useMemo<Partial<Shape>[]>(
    () =>
      boundaries.flatMap((b) => [
        { type: 'line', x0: b - 0.5, x1: b - 0.5, y0: -0.5, y1: n - 0.5, line: { color: ink.primary, width: 1 }, opacity: 0.55 },
        { type: 'line', y0: b - 0.5, y1: b - 0.5, x0: -0.5, x1: n - 0.5, line: { color: ink.primary, width: 1 }, opacity: 0.55 },
      ]),
    [boundaries, n],
  );
  const heatLayout: Partial<Layout> = {
    xaxis: { showticklabels: false, ticks: '', title: { text: y ? 'observations, ordered by class' : 'observations (dataset order)' } },
    yaxis: { showticklabels: false, ticks: '', autorange: 'reversed', title: { text: 'observations' } },
    shapes,
    margin: { l: 40, r: 10, t: 10, b: 40 },
  };

  const m6 = Math.min(6, n);
  const D6 = useMemo(() => D.slice(0, m6).map((r) => r.slice(0, m6)), [D, m6]);
  const lab6 = Array.from({ length: m6 }, (_, i) => `#${i + 1}`);
  const ratio = wb && wb.within > 0 ? wb.between / wb.within : NaN;
  const scalingWord = prep.scaling === 'none' ? 'raw' : prep.scaling === 'center' ? 'centred' : 'standardised';

  return (
    <Section id="mds-intro" title="1 · Start from distances, not variables" subtitle="The distance matrix D is the only input of multidimensional scaling. Look at it before decomposing it.">
      <div className="prose">
        <p>
          Every other laboratory begins with the data matrix <M tex="X" /> and its variables. Suppose instead that the <M tex="p" /> variables are unavailable — only the dissimilarities between the <M tex="n" /> observations survive: travel times between cities, judged similarities between stimuli, alignment scores between sequences. The question of multidimensional scaling (MDS) is whether a configuration of <M tex="n" /> points in <M tex="\mathbb R^k" /> exists — or approximately exists — whose Euclidean inter-point distances <M tex="\hat d_{ij}" /> reproduce the given <M tex="d_{ij}" />.
        </p>
        <p>
          In this laboratory the distances are computed from the {scalingWord} matrix <M tex="X_c" /> under the metric chosen above, precisely so that every step can be checked against the variables we pretend not to have. Below, <M tex="D" /> for the active dataset: <M tex={`n = ${n}`} /> observations, hence <M tex={`n(n-1)/2 = ${summary.pairs}`} /> distinct distances.
        </p>
      </div>

      <div className="grid side-r" style={{ marginTop: 12 }}>
        <div>
          <Plot data={heat} layout={heatLayout} height={440} title={`Distance matrix D under the ${metricLabels[metric]} metric${y ? ' — rows and columns ordered by class' : ''}`} />
          <div className="plot-caption">
            {y ? (
              <>
                Observations are sorted by class so that within-class distances form blocks on the diagonal; thin lines mark the {K} class boundaries. Light is small, dark is large; the diagonal is <M tex="d_{ii} = 0" />.
              </>
            ) : (
              <>Unlabelled data: rows are in dataset order. Light is small, dark is large; the diagonal is <M tex="d_{ii} = 0" />.</>
            )}
          </div>
        </div>
        <div className="stack">
          <Card title={`Top-left ${m6} × ${m6} block of D (dataset order)`} plane>
            <MatrixView M={D6} rowLabels={lab6} colLabels={lab6} digits={2} heat="sequential" heatMax={summary.max} caption={`Distances between the first ${m6} observations under the ${metricLabels[metric]} metric; heat scale shared with the heatmap.`} />
          </Card>
          <div className="stats">
            <StatTile label="pairs" value={summary.pairs} note={<M tex="n(n-1)/2" />} />
            <StatTile label="mean distance" value={fmt(summary.mean, 3)} />
            <StatTile label="max distance" value={fmt(summary.max, 3)} note={`#${summary.argmax[0] + 1} and #${summary.argmax[1] + 1}`} />
            <StatTile label="smallest non-zero" value={fmt(summary.minNonzero, 3)} />
            {wb && <StatTile label="mean within-class" value={fmt(wb.within, 3)} note={`${wb.nWithin} pairs`} />}
            {wb && <StatTile label="mean between-class" value={fmt(wb.between, 3)} note={`${wb.nBetween} pairs`} />}
          </div>
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              Every entry of the <M tex={`${n}\\times ${n}`} /> matrix <M tex="D" /> under the {metricLabels[metric]} metric. <M tex="D" /> is symmetric with a zero diagonal, so it carries <M tex={`${summary.pairs}`} /> distinct numbers.{' '}
              {y ? (
                <>
                  Rows and columns are sorted by class, so the <b>light blocks along the diagonal</b> are within-class distances and the darker off-diagonal blocks are between-class distances.
                </>
              ) : (
                <>The dataset has no labels, so rows keep their original order and no block structure is imposed.</>
              )}{' '}
              The mean distance is {fmt(summary.mean, 3)}, the largest {fmt(summary.max, 3)} (observations #{summary.argmax[0] + 1} and #{summary.argmax[1] + 1}), the smallest non-zero {fmt(summary.minNonzero, 3)}. The {m6} × {m6} block lists the first {m6} observations numerically; for instance <M tex={`d_{12} = ${texNum(n > 1 ? D[0][1] : 0, 3)}`} />.
            </>
          ),
          why: (
            <>
              {wb ? (
                <>
                  Within-class distances average {fmt(wb.within, 3)} over {wb.nWithin} pairs; between-class distances average {fmt(wb.between, 3)} over {wb.nBetween} pairs — a ratio of {fmt(ratio, 2)}.{' '}
                  {ratio > 1.5
                    ? 'That contrast is what makes the diagonal blocks visibly lighter: members of one class sit close to each other relative to everyone else.'
                    : 'The ratio is close to one, so the classes are not separated in this metric and the block structure is faint — MDS will not create a separation that is absent from D.'}
                </>
              ) : (
                <>Without labels the only structure that can appear is the one generated by the data themselves; clusters would show as light square blocks after a suitable reordering, which is not attempted here.</>
              )}{' '}
              Light means small: the sequential colour scale starts at <M tex="d = 0" /> on the diagonal and ends at the maximum {fmt(summary.max, 3)}.
            </>
          ),
          math: (
            <>
              <MBlock tex={metricTex[metric]} />
              Distances are invariant under translations, rotations and reflections of the observation space, so <M tex="D" /> retains no information about the origin or orientation of <M tex="X" />: any configuration that reproduces <M tex="D" /> is determined only up to such a rigid motion. Counting entries, <M tex="X" /> holds <M tex={`np = ${n * p}`} /> numbers and <M tex="D" /> holds <M tex={`${summary.pairs}`} />;{' '}
              {summary.pairs > n * p ? (
                <>
                  the distances are therefore highly redundant, which is exactly what the rank bound <M tex="\operatorname{rank}(B) \le p" /> of Section 2 expresses.
                </>
              ) : (
                <>fewer than the data themselves, so the distances are not redundant for this small sample.</>
              )}
            </>
          ),
          stats: (
            <>
              The distances are computed on the {scalingWord} matrix.{' '}
              {prep.scaling === 'standardize' ? (
                <>After standardisation every variable has unit variance, so differences are measured in standard-deviation units and each variable contributes equally to <M tex="D" />.</>
              ) : (
                <>
                  The raw standard deviations range from {fmt(sdMin, 3)} to {fmt(sdMax, 3)} (ratio {fmt(sdMax / sdMin, 1)});{' '}
                  {sdMax / sdMin > 3 ? 'the widest variable dominates the distances — switch to Standardise to remove this dependence on units.' : 'the variables are on comparable scales, so no single variable dominates the distances.'}
                </>
              )}{' '}
              {metric === 'euclidean' && prep.scaling !== 'none' && (
                <>
                  For centred data the mean squared Euclidean distance over pairs equals <M tex="2\,\operatorname{tr}(S)" /> exactly: here <M tex={`${texNum(summary.meanSq, 3)} = 2\\times ${texNum(a.pca.totalVariance, 3)}`} />, so the distance matrix and the covariance matrix encode the same total dispersion.
                </>
              )}
            </>
          ),
          careful: (
            <>
              Classical MDS (Section 2) is exact only for Euclidean distances:{' '}
              {metric === 'euclidean' ? (
                <>the current metric qualifies, so <M tex="B" /> will be positive semi-definite.</>
              ) : (
                <>
                  the {metricLabels[metric]} metric does not qualify in general, so expect negative eigenvalues of <M tex="B" /> in Section 3 ({a.mds.negative} at present).
                </>
              )}{' '}
              Changing the metric is not a rescaling of <M tex="D" />: the ordering of pairs changes, not only the units. And <M tex="D" /> has forgotten the variables — an MDS map has axes without loadings, so on its own it cannot say which variables drive a separation.
            </>
          ),
        }}
      />

      <div className="divider" />
      <h3>Which distance?</h3>
      <p className="secondary small" style={{ marginTop: 4 }}>
        The current metric is marked. Only the first row leads to an exact classical solution; the others are shown so that the failure mode is visible rather than hidden.
      </p>
      <table className="summary-table" style={{ marginTop: 8 }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Definition</th>
            <th>Appropriate when</th>
            <th>Classical MDS</th>
          </tr>
        </thead>
        <tbody>
          {metricRows.map((r) => (
            <tr key={r.metric} style={r.metric === metric ? { background: 'var(--accent-soft)' } : undefined}>
              <td>
                {metricLabels[r.metric]}
                {r.metric === metric && (
                  <>
                    {' '}
                    <Badge>current</Badge>
                  </>
                )}
              </td>
              <td>
                <M tex={metricTex[r.metric]} />
              </td>
              <td>{r.when}</td>
              <td>{r.exact}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Callout kind="theorem" title="Why Euclidean is special">
        Classical MDS recovers coordinates through inner products (Section 2). Inner products are defined by Euclidean geometry: <M tex="d_{ij}^2 = \|x_i\|^2 + \|x_j\|^2 - 2x_i^T x_j" />. For any other metric the same algebra is still executed, but the matrix it produces is no longer a Gram matrix of any point set, and its negative eigenvalues measure exactly how far the input is from being Euclidean.
      </Callout>
    </Section>
  );
}
