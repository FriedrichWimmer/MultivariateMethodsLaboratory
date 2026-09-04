import { useMemo } from 'react';
import type { Layout } from 'plotly.js';
import { add, column, covariance, fmt, maxAbs, scale, sub, trace } from '../../../lib/linalg';
import type { LDAResult } from '../../../lib/lda';
import { M, MBlock, texVector } from '../../common/Math';
import { MatrixEquation, MatrixView } from '../../common/MatrixView';
import { Badge, Callout, Card, ClassLegend, Interpretation, MarkerShape, Section, StatTile } from '../../common/Panels';
import { Plot } from '../../common/Plot';
import { ScatterSVG } from '../../common/ScatterSVG';
import { classHistograms, classProjections, meanLineShapes, pct, type LabelledData } from './helpers';

interface Props {
  source: LabelledData;
  res: LDAResult;
  /** 'raw' | 'centred' | 'standardised' — how the analysed matrix was prepared */
  scalingLabel: string;
}

/** Lesson 1 — labelled observations, scatter matrices, the Fisher criterion and the generalised eigenproblem. */
export default function IntroSection({ source, res, scalingLabel }: Props) {
  const { X, y, classNames, variableNames } = source;
  const { n, p, K } = res;
  const m = res.maxDims;

  const S = useMemo(() => covariance(X), [X]);
  const check = useMemo(() => maxAbs(sub(add(res.SW, res.SB), scale(S, n - 1))), [res, S, n]);
  const heatMax = useMemo(() => maxAbs(res.ST), [res]);
  const trW = useMemo(() => trace(res.SW), [res]);
  const trB = useMemo(() => trace(res.SB), [res]);
  const lambda1 = res.eigenvalues[0] ?? 0;
  const F1 = K > 1 ? (lambda1 * (n - K)) / (K - 1) : NaN;
  const wLabels = useMemo(() => Array.from({ length: m }, (_, j) => `w${j + 1}`), [m]);

  const ld1 = useMemo(() => (m >= 1 ? column(res.scores, 0) : []), [res, m]);
  const ld1Proj = useMemo(() => classProjections(ld1, y, res.classes, classNames), [ld1, y, res, classNames]);
  const ld1Hist = useMemo(() => classHistograms(ld1, y, res.classes, classNames), [ld1, y, res, classNames]);
  const ld1Layout = useMemo<Partial<Layout>>(
    () => ({ barmode: 'overlay', showlegend: true, xaxis: { title: { text: 'LD1 score  z₁ = (x − m)ᵀ w₁' } }, yaxis: { title: { text: 'count' } }, shapes: meanLineShapes(ld1Proj) }),
    [ld1Proj],
  );
  const scores2 = useMemo(() => (m >= 2 ? res.scores.map((r) => [r[0], r[1]]) : []), [res, m]);

  const sepIndex = trW + trB > 0 ? trB / (trW + trB) : NaN;
  const w1 = m >= 1 ? column(res.W, 0) : [];
  const largestLoading = w1.length ? w1.reduce((b, v, j, arr) => (Math.abs(v) > Math.abs(arr[b]) ? j : b), 0) : -1;
  const ldSep = ld1Proj.length >= 2 ? ld1Proj.map((c) => c.mean) : [];
  const ldRange = ldSep.length ? Math.max(...ldSep) - Math.min(...ldSep) : NaN;
  const pooledSd = m >= 1 ? Math.sqrt(res.eigenvalues.length ? 1 / Math.max(n - K, 1) * quadWithin(res, w1) : NaN) : NaN;

  return (
    <Section
      id="lda-intro"
      title="1 · Labelled observations: what LDA uses that PCA ignores"
      subtitle="Within-, between- and total scatter; the Fisher criterion; the generalised eigenproblem"
      right={<Badge method="LDA" />}
    >
      <div className="prose">
        <p>
          SVD, PCA and MDS are <em>unsupervised</em>: they see the data matrix <M tex="X \in \mathbb{R}^{n\times p}" /> and nothing else. Linear discriminant analysis is{' '}
          <em>supervised</em>: every observation <M tex="x_i" /> carries a class label <M tex="y_i \in \{1,\dots,K\}" />, and the method is allowed to use <M tex="y" /> when it chooses
          directions. PCA asks how <M tex="X" /> varies; LDA asks how the <em>classes</em> differ. Both produce linear projections <M tex="z = X_c w" />, but they optimise different criteria
          and can therefore pick very different directions from the same <M tex="X" />.
        </p>
      </div>

      <div className="table-wrap">
        <table className="summary-table">
          <thead>
            <tr>
              <th />
              <th>
                <Badge method="PCA" /> principal component analysis
              </th>
              <th>
                <Badge method="LDA" /> linear discriminant analysis
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Input</td>
              <td>
                <M tex="X" /> only (the labels are ignored)
              </td>
              <td>
                <M tex="X" /> <b>and</b> <M tex="y" />
              </td>
            </tr>
            <tr>
              <td>Question</td>
              <td>Along which directions does the data vary most?</td>
              <td>Along which directions are the classes most separated relative to their internal spread?</td>
            </tr>
            <tr>
              <td>Criterion</td>
              <td>
                <M tex="\max_{\|w\|=1} w^{\mathsf T} S w" />
              </td>
              <td>
                <M tex="\max_{w} J(w) = \dfrac{w^{\mathsf T} S_B w}{w^{\mathsf T} S_W w}" />
              </td>
            </tr>
            <tr>
              <td>Solution</td>
              <td>
                eigenvectors of <M tex="S" /> (equivalently the SVD of <M tex="X_c" />)
              </td>
              <td>
                generalised eigenvectors: <M tex="S_B w = \lambda S_W w" />
              </td>
            </tr>
            <tr>
              <td>Number of directions</td>
              <td>
                <M tex="\operatorname{rank}(S) \le \min(n-1,\,p)" />
              </td>
              <td>
                at most <M tex="K-1" />
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <Callout kind="definition" title="Scatter matrices">
        <p>
          Let <M tex="n_k" /> be the size of class <M tex="k" />, <M tex="m_k = \frac{1}{n_k}\sum_{i\in k} x_i" /> its mean and <M tex="m = \frac1n\sum_i x_i" /> the grand mean. The{' '}
          <em>within-class</em>, <em>between-class</em> and <em>total</em> scatter matrices are
        </p>
        <MBlock tex="S_W = \sum_{k=1}^{K}\sum_{i\in k}(x_i - m_k)(x_i - m_k)^{\mathsf T},\qquad S_B = \sum_{k=1}^{K} n_k\,(m_k - m)(m_k - m)^{\mathsf T},\qquad S_T = \sum_{i=1}^{n}(x_i - m)(x_i - m)^{\mathsf T}." />
        <p>
          Writing <M tex="x_i - m = (x_i - m_k) + (m_k - m)" /> and using <M tex="\sum_{i\in k}(x_i - m_k) = 0" /> kills the cross terms, so{' '}
          <M tex="S_T = S_W + S_B = X_c^{\mathsf T} X_c = (n-1)\,S" />. <M tex="S_W" /> measures variability <em>inside</em> the classes; <M tex="S_B" /> measures how far the class means sit
          from the grand mean, each weighted by its class size <M tex="n_k" />. They are <M tex="p\times p" /> and symmetric; <M tex="S_W" /> is positive definite whenever the pooled
          within-class deviations span <M tex="\mathbb{R}^p" />, while <M tex="S_B" /> has rank at most <M tex="K-1" /> (lesson 4).
        </p>
      </Callout>

      {source.local && (
        <Callout kind="info" title="Local stand-in data">
          The active dataset carries no labels, so this lesson uses a local labelled dataset ({source.name}: <M tex={`n = ${n},\\ p = ${p},\\ K = ${K}`} />). Select a labelled dataset in
          the data laboratory to see these matrices for your own data.
        </Callout>
      )}

      <Card title={`Scatter matrices of the ${source.local ? 'local' : scalingLabel} data (${source.name}): n = ${n}, p = ${p}, K = ${K}`} plane>
        <MatrixEquation
          items={[
            <MatrixView key="sw" M={res.SW} title="S_W" rowLabels={variableNames} colLabels={variableNames} digits={2} heat="diverging" heatMax={heatMax} maxRows={8} maxCols={8} compact />,
            '+',
            <MatrixView key="sb" M={res.SB} title="S_B" rowLabels={variableNames} colLabels={variableNames} digits={2} heat="diverging" heatMax={heatMax} maxRows={8} maxCols={8} compact />,
            '=',
            <MatrixView key="st" M={res.ST} title="S_T = (n − 1) S" rowLabels={variableNames} colLabels={variableNames} digits={2} heat="diverging" heatMax={heatMax} maxRows={8} maxCols={8} compact />,
          ]}
        />
        <div className="stats" style={{ marginTop: 12 }}>
          <StatTile label={<M tex="\max\,|S_W + S_B - (n-1)S|" />} value={fmt(check, 6)} note="numerical check of the decomposition" />
          <StatTile label={<M tex="\operatorname{tr} S_W" />} value={fmt(trW, 2)} note="total within-class scatter" />
          <StatTile label={<M tex="\operatorname{tr} S_B" />} value={fmt(trB, 2)} note="total between-class scatter" />
          <StatTile label={<M tex="\operatorname{tr} S_B / \operatorname{tr} S_T" />} value={pct(sepIndex)} note="share of the total scatter that lies between class means" />
        </div>
      </Card>

      <Card title="Class means and class sizes">
        <div className="table-wrap">
          <table className="summary-table">
            <thead>
              <tr>
                <th>class</th>
                <th>
                  <M tex="n_k" />
                </th>
                <th>share</th>
                {variableNames.map((v) => (
                  <th key={v}>{v}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {res.classes.map((c, k) => (
                <tr key={c}>
                  <td>
                    <span className="row" style={{ gap: 6, display: 'inline-flex' }}>
                      <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden>
                        <MarkerShape k={c} r={5} />
                      </svg>
                      {classNames[c] ?? `class ${c}`}
                    </span>
                  </td>
                  <td>{res.classSizes[k]}</td>
                  <td>{pct(res.classSizes[k] / n)}</td>
                  {res.classMeans[k].map((v, j) => (
                    <td key={j}>{fmt(v, 2)}</td>
                  ))}
                </tr>
              ))}
              <tr>
                <td>
                  <M tex="m" /> (grand mean)
                </td>
                <td>{n}</td>
                <td>100%</td>
                {res.grandMean.map((v, j) => (
                  <td key={j}>{fmt(v, 2)}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <div className="small muted" style={{ marginTop: 6 }}>
          Each row of <M tex="S_B" /> is built from the differences <M tex="m_k - m" /> in this table, weighted by <M tex="n_k" />; the <M tex="K" /> vectors <M tex="m_k - m" /> satisfy{' '}
          <M tex="\sum_k n_k (m_k - m) = 0" />, which is why <M tex="S_B" /> cannot have full rank when <M tex="K \le p" />.
        </div>
      </Card>

      <Callout kind="theorem" title="Fisher's criterion and the generalised eigenproblem">
        <p>
          A direction <M tex="w" /> is good for discrimination when the projected class means are far apart compared with the projected within-class spread. Along <M tex="w" /> the between-class
          scatter is <M tex="w^{\mathsf T} S_B w" /> and the within-class scatter is <M tex="w^{\mathsf T} S_W w" />, so Fisher's criterion is the generalised Rayleigh quotient
        </p>
        <MBlock tex="J(w) = \frac{w^{\mathsf T} S_B w}{w^{\mathsf T} S_W w}, \qquad J(cw) = J(w)\ \text{for all } c \neq 0 ." />
        <p>
          Setting the gradient to zero gives <M tex="S_B w = J(w)\, S_W w" />: every stationary point is a solution of the <em>generalised eigenproblem</em>{' '}
          <M tex="S_B w = \lambda S_W w" /> with <M tex="\lambda = J(w)" />. When <M tex="S_W" /> is invertible this is the ordinary eigenproblem of <M tex="S_W^{-1} S_B" />; its
          eigenvalues <M tex="\lambda_1 \ge \lambda_2 \ge \dots" /> are the attainable Fisher ratios and the maximiser of <M tex="J" /> is the leading eigenvector <M tex="w_1" />. Because{' '}
          <M tex="\operatorname{rank}(S_B) \le K-1" />, at most <M tex="K-1" /> eigenvalues are non-zero. Numerically the laboratory whitens with <M tex="S_W = Q D Q^{\mathsf T}" />, solves the
          symmetric problem for <M tex="D^{-1/2} Q^{\mathsf T} S_B Q D^{-1/2}" /> and maps back, which is more stable than forming <M tex="S_W^{-1} S_B" />. Each <M tex="w_j" /> is defined only
          up to a non-zero scalar; the laboratory reports unit-norm vectors with the largest entry positive.
        </p>
      </Callout>

      <div className="grid c2">
        <Card title="Discriminant directions W (columns w₁, …, w_m)">
          {m >= 1 ? (
            <>
              <MatrixView M={res.W} title="W" rowLabels={variableNames} colLabels={wLabels} digits={3} heat="diverging" caption="unit-norm columns; sign fixed so the largest |entry| is positive" />
              <div className="stats" style={{ marginTop: 12 }}>
                {res.eigenvalues.map((l, j) => (
                  <StatTile key={j} label={<M tex={`\\lambda_{${j + 1}} = J(w_{${j + 1}})`} />} value={fmt(l, 3)} note={`${pct(res.explained[j])} of Σⱼ λⱼ`} />
                ))}
                <StatTile label={<M tex="m = \min(K-1,\ \operatorname{rank} S_W)" />} value={m} note={`K − 1 = ${K - 1}, rank S_W = ${res.swRank}`} />
              </div>
            </>
          ) : (
            <Callout kind="danger">No discriminant direction could be computed (S_B or S_W has no usable range).</Callout>
          )}
        </Card>
        <Card title={m >= 2 ? 'Discriminant scores Z = X_c W (first two directions)' : 'Scores on the single discriminant direction'}>
          {m >= 2 ? (
            <>
              <ScatterSVG points={scores2} labels={y} classNames={classNames} width={460} height={380} xLabel="LD1 = (x − m)ᵀ w₁" yLabel="LD2 = (x − m)ᵀ w₂" />
              <ClassLegend classNames={res.classes.map((c) => classNames[c] ?? `class ${c}`)} />
            </>
          ) : m === 1 ? (
            <Plot data={ld1Hist} layout={ld1Layout} height={320} />
          ) : (
            <div className="muted small">Nothing to plot.</div>
          )}
        </Card>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              Three <M tex={`${p}\\times ${p}`} /> matrices for the {source.local ? 'local' : scalingLabel} data: <M tex="S_W" /> collects deviations around the <M tex={`K = ${K}`} /> class means,{' '}
              <M tex="S_B" /> the size-weighted deviations of the class means from the grand mean, and their sum equals the total scatter. Here <M tex={`\\operatorname{tr} S_W = ${fmt(trW, 2)}`} />{' '}
              and <M tex={`\\operatorname{tr} S_B = ${fmt(trB, 2)}`} />, so {pct(sepIndex)} of the total scatter is between-class. The right-hand panel shows the observations projected on the
              discriminant direction{m >= 2 ? 's' : ''} <M tex="w_1" />
              {m >= 2 ? <M tex=", w_2" /> : null}, the directions along which the <M tex="J" /> ratio is largest.
            </>
          ),
          why: (
            <>
              The decomposition <M tex="S_T = S_W + S_B" /> is an identity, not an approximation: the largest entry of <M tex="S_W + S_B - (n-1)S" /> is <M tex={`${fmt(check, 6)}`} />, rounding
              error only. Cells of <M tex="S_B" /> are large exactly where class means differ along a variable (or along a pair of variables jointly). The class means table shows which
              variables drive that:{' '}
              {largestLoading >= 0 ? (
                <>
                  the leading direction <M tex="w_1" /> puts its largest weight ({fmt(w1[largestLoading], 3)}) on <b>{variableNames[largestLoading]}</b>.
                </>
              ) : (
                'no direction was computed.'
              )}
            </>
          ),
          math: (
            <>
              The generalised eigenvalues are <M tex={`(${res.eigenvalues.map((l) => fmt(l, 3)).join(',\\ ')})`} />; <M tex={`\\lambda_1 = ${fmt(lambda1, 3)}`} /> is the maximum of{' '}
              <M tex="J(w)" /> over all <M tex="w \in \mathbb{R}^p" />, attained at{' '}
              {w1.length ? <M tex={`w_1 = ${texVector(w1, 3, false)}^{\\mathsf T}`} /> : <M tex="w_1" />}. Exactly <M tex={`m = ${m}`} /> directions are meaningful because{' '}
              <M tex={`\\operatorname{rank} S_B \\le K - 1 = ${K - 1}`} /> and <M tex={`\\operatorname{rank} S_W = ${res.swRank}`} />.
            </>
          ),
          stats: (
            <>
              Along <M tex="w_1" /> the between-class scatter is <M tex={`${fmt(lambda1, 2)}`} /> times the within-class scatter. Equivalently, a one-way ANOVA of the LD1 scores would report{' '}
              <M tex={`F = \\lambda_1 (n-K)/(K-1) = ${fmt(F1, 1)}`} /> on <M tex={`(${K - 1},\\ ${n - K})`} /> degrees of freedom — but the direction was <em>chosen</em> to maximise this
              ratio, so the usual F reference distribution does not apply. The projected class means on LD1 span {Number.isFinite(ldRange) ? fmt(ldRange, 2) : '—'} units against a pooled
              within-class standard deviation of {Number.isFinite(pooledSd) ? fmt(pooledSd, 2) : '—'}.
            </>
          ),
          careful: (
            <>
              Scatter matrices are in squared data units, so <M tex="J" /> is invariant to rescaling a variable but the entries of <M tex="S_W, S_B" /> and the coefficients of{' '}
              <M tex="w" /> are not (current preprocessing: {source.local ? 'raw local data' : scalingLabel}). <M tex="S_W" /> is{' '}
              {res.swSingular ? (
                <>
                  <b>singular</b> (rank {res.swRank} &lt; {p}); the solution lives on its range and lesson 4 explains why
                </>
              ) : (
                <>
                  invertible with condition number <M tex={`\\kappa(S_W) = ${fmt(res.swCondition, 1)}`} />
                </>
              )}
              . The eigenvectors are defined only up to sign, and the <M tex="\lambda_j" /> describe the <em>training</em> sample.
            </>
          ),
        }}
      />
    </Section>
  );
}

/** w ᵀ S_W w for the leading direction (pooled within-class scatter along w). */
function quadWithin(res: LDAResult, w: number[]): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) {
    let t = 0;
    for (let j = 0; j < w.length; j++) t += res.SW[i][j] * w[j];
    s += w[i] * t;
  }
  return s;
}
