import { useMemo } from 'react';
import type { Data } from 'plotly.js';
import { useAnalysis } from '../../../state/store';
import { sub, maxAbs, svd, fmt, type Matrix } from '../../../lib/linalg';
import { lowRankSummaries } from '../../../lib/svdlab';
import { plotlyDiverging } from '../../../lib/theme';
import { Plot } from '../../common/Plot';
import { M, MBlock } from '../../common/Math';
import { MatrixView, MatrixEquation } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile, Derivation } from '../../common/Panels';
import { Slider } from '../../common/Controls';
import { DatasetControls, PrepControls } from '../../common/DatasetControls';
import { rankOneBlock, texNum, fmtPct, subscript } from './util';

/** Lesson 3 — truncated SVD of the analysed data matrix and the Eckart–Young theorem. */
export function LowRankSection({ k, setK }: { k: number; setK: (k: number) => void }) {
  const a = useAnalysis();
  const X = a.pca.Xc;
  const res = a.svd;
  const n = a.n;
  const p = a.p;
  const r = res.s.length;
  const kk = Math.max(1, Math.min(k, r));
  const names = a.dataset.variableNames;

  const summaries = useMemo(() => lowRankSummaries(X, res), [X, res]);
  const cur = summaries[kk];
  const resid = useMemo(() => sub(X, cur.Xk), [X, cur]);
  const specNorm = useMemo(() => (resid.length ? svd(resid).s[0] : 0), [resid]);
  const zmax = useMemo(() => maxAbs(X), [X]);
  const total = useMemo(() => res.s.reduce((acc, x) => acc + x * x, 0), [res]);
  const eyGap = Math.abs(cur.error - cur.eckartYoung);
  const sigmaNext = kk < r ? res.s[kk] : 0;
  const rows = useMemo(() => Array.from({ length: n }, (_, i) => i + 1), [n]);
  const blockRows = Math.min(6, n);
  const Xblock = useMemo(() => X.slice(0, blockRows), [X, blockRows]);
  const rowLabels = Array.from({ length: blockRows }, (_, i) => `${i + 1}`);

  const steps = useMemo(
    () =>
      Array.from({ length: kk }, (_, t) => {
        const j = t + 1;
        const term = rankOneBlock(res, t, blockRows);
        const prev = summaries[t].Xk.slice(0, blockRows);
        const now = summaries[j].Xk.slice(0, blockRows);
        return {
          title: (
            <>
              Add the rank-one term <M tex={`\\sigma_{${j}} u_{${j}} v_{${j}}^T`} /> with <M tex={`\\sigma_{${j}} = ${texNum(res.s[t], 3)}`} />
            </>
          ),
          body: (
            <MatrixEquation
              items={[
                <MatrixView M={prev} title={`X${subscript(j - 1)}`} digits={2} heat="diverging" heatMax={zmax} maxCols={6} compact />,
                '+',
                <MatrixView M={term} title={`σ${subscript(j)}u${subscript(j)}v${subscript(j)}ᵀ`} digits={2} heat="diverging" heatMax={zmax} maxCols={6} compact />,
                '=',
                <MatrixView M={now} title={`X${subscript(j)}`} digits={2} heat="diverging" heatMax={zmax} maxCols={6} compact />,
              ]}
            />
          ),
          note: (
            <>
              Over the whole matrix: <M tex={`\\|X - X_{${j}}\\|_F = ${texNum(summaries[j].error, 4)}`} /> and{' '}
              <M tex={`\\sqrt{\\textstyle\\sum_{t > ${j}} \\sigma_t^2} = ${texNum(summaries[j].eckartYoung, 4)}`} />; energy kept{' '}
              <M tex={`\\sum_{t \\le ${j}} \\sigma_t^2 / \\sum_t \\sigma_t^2 = ${texNum(summaries[j].energy, 4)}`} />.
            </>
          ),
        };
      }),
    [kk, res, summaries, blockRows, zmax],
  );

  const heat = (z: Matrix, label: string): Data => ({
    type: 'heatmap',
    z,
    x: names,
    y: rows,
    colorscale: plotlyDiverging,
    zmin: -zmax,
    zmax,
    showscale: true,
    colorbar: { thickness: 8, len: 0.9, outlinewidth: 0, tickfont: { size: 10 } },
    hovertemplate: `${label}<br>row %{y} · %{x}: %{z:.3f}<extra></extra>`,
  });
  const heatLayout: Record<string, unknown> = {
    xaxis: { tickangle: -35 },
    yaxis: { autorange: 'reversed', title: { text: 'observation' } },
    margin: { b: 72, l: 56, r: 10 },
  };

  const proof = [
    {
      title: 'Orthogonal invariance of the Frobenius norm',
      body: (
        <>
          For orthogonal <M tex="Q, P" />, <M tex="\|QAP^T\|_F^2 = \operatorname{tr}(P A^T Q^T Q A P^T) = \operatorname{tr}(A^TA) = \|A\|_F^2" />. Writing{' '}
          <M tex="X - X_k = U(\Sigma - \Sigma_k)V^T" /> with <M tex="\Sigma_k" /> the diagonal truncated after <M tex="k" /> entries gives{' '}
          <M tex="\|X - X_k\|_F^2 = \|\Sigma - \Sigma_k\|_F^2 = \sum_{j > k}\sigma_j^2" />, and likewise <M tex="\|X - X_k\|_2 = \sigma_{k+1}" />.
        </>
      ),
      note: (
        <>
          Live: <M tex={`\\|X - X_{${kk}}\\|_F = ${texNum(cur.error, 4)}`} />, <M tex={`\\sqrt{\\sum_{j>${kk}}\\sigma_j^2} = ${texNum(cur.eckartYoung, 4)}`} />;{' '}
          <M tex={`\\|X - X_{${kk}}\\|_2 = ${texNum(specNorm, 4)}`} /> against <M tex={`\\sigma_{${kk + 1}} = ${texNum(sigmaNext, 4)}`} />.
        </>
      ),
    },
    {
      title: 'Weyl’s inequality for singular values',
      body: (
        <>
          For any <M tex="A, C" /> of the same shape, <M tex="\sigma_{i+j-1}(A + C) \le \sigma_i(A) + \sigma_j(C)" />. Proof idea: by the variational characterisation{' '}
          <M tex="\sigma_m(A) = \min_{\dim \mathcal S = p - m + 1}\ \max_{x \in \mathcal S,\ \|x\| = 1} \|Ax\|" />; intersecting the minimising subspaces of{' '}
          <M tex="\sigma_i(A)" /> and <M tex="\sigma_j(C)" /> yields a subspace of dimension at least <M tex="p - (i + j - 1) + 1" /> on which{' '}
          <M tex="\|(A + C)x\| \le \sigma_i(A) + \sigma_j(C)" />.
        </>
      ),
    },
    {
      title: 'Apply it with A = X − B and C = B, rank B ≤ k',
      body: (
        <>
          Since <M tex="\operatorname{rank} B \le k" />, <M tex="\sigma_{k+1}(B) = 0" />. Take <M tex="j = k + 1" />:{' '}
          <MBlock tex="\sigma_{i+k}(X) \;\le\; \sigma_i(X - B) + \sigma_{k+1}(B) \;=\; \sigma_i(X - B), \qquad i = 1, \dots, r - k." />
          Every singular value of the residual <M tex="X - B" /> dominates a tail singular value of <M tex="X" />.
        </>
      ),
    },
    {
      title: 'Sum the squares (Frobenius) or take i = 1 (spectral)',
      body: (
        <>
          <MBlock tex="\|X - B\|_F^2 = \sum_{i} \sigma_i(X - B)^2 \;\ge\; \sum_{i=1}^{r-k} \sigma_{i+k}(X)^2 = \sum_{j > k}\sigma_j^2 = \|X - X_k\|_F^2 ," />
          and with <M tex="i = 1" />: <M tex="\|X - B\|_2 = \sigma_1(X - B) \ge \sigma_{k+1}(X) = \|X - X_k\|_2" />. Both bounds are attained by <M tex="B = X_k" /> (step 1),
          whose rank is <M tex="\min(k, \operatorname{rank}X)" />.
        </>
      ),
      note: (
        <>
          Uniqueness: in the Frobenius norm the minimiser is unique if and only if <M tex="\sigma_k > \sigma_{k+1}" /> (here{' '}
          <M tex={`\\sigma_{${kk}} = ${texNum(res.s[kk - 1], 3)}`} />, <M tex={`\\sigma_{${kk + 1}} = ${texNum(sigmaNext, 3)}`} />); in the spectral norm it is not unique in general.
          Mirsky (1960) extended the statement to every unitarily invariant norm.
        </>
      ),
    },
  ];

  return (
    <Section
      id="svd-lowrank"
      title="3 · Low-rank approximation and the Eckart–Young theorem"
      subtitle="Keep the k largest singular triplets of the analysed data matrix and drop the rest: the truncated SVD is the best rank-k approximation in every unitarily invariant norm."
    >
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<>rank <M tex="k" /></>} value={kk} min={1} max={r} step={1} onChange={setK} format={(v) => `${v} of ${r}`} />
          <div className="stats">
            <StatTile label={<M tex="\|X - X_k\|_F" />} value={fmt(cur.error, 3)} note={`relative ${fmtPct(cur.relativeError, 1)}`} />
            <StatTile label={<M tex="\sqrt{\sum_{j>k}\sigma_j^2}" />} value={fmt(cur.eckartYoung, 3)} note={`gap ${fmt(eyGap, 1)}`} />
            <StatTile label="Energy kept" value={fmtPct(cur.energy, 1)} note={<M tex="\sum_{j\le k}\sigma_j^2 / \sum_j \sigma_j^2" />} />
            <StatTile label={<M tex="\|X - X_k\|_2" />} value={fmt(specNorm, 3)} note={`σ_{k+1} = ${fmt(sigmaNext, 3)}`} />
          </div>
          <div className="divider" />
          <div className="small secondary">Global data (shared with every laboratory):</div>
          <DatasetControls compact allowUpload={false} />
          <PrepControls showMetric={false} showK={false} />
        </div>
        <div className="stack">
          <div className="grid c3">
            <Plot data={[heat(X, 'X')]} layout={heatLayout} height={360} title={`X  (${n} × ${p}, ${a.prep.scaling === 'none' ? 'raw' : a.prep.scaling === 'center' ? 'centred' : 'standardised'})`} />
            <Plot data={[heat(cur.Xk, `X_${kk}`)]} layout={heatLayout} height={360} title={`X_${kk} = U_${kk} Σ_${kk} V_${kk}ᵀ  (rank ${kk})`} />
            <Plot data={[heat(resid, 'residual')]} layout={heatLayout} height={360} title={`X − X_${kk}  (‖·‖_F = ${fmt(cur.error, 3)})`} />
          </div>
          <div className="plot-caption">
            Rows are observations, columns variables; one diverging colour scale (symmetric about 0, range ±{fmt(zmax, 2)}) is shared by the three panels, so the residual can be
            compared with the signal at a glance.
          </div>
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        <Interpretation
          items={{
            seeing: (
              <>
                The analysed matrix <M tex="X" /> (<M tex={`${n}\\times${p}`} />), its rank-<M tex={`${kk}`} /> truncation <M tex={`X_{${kk}}`} /> and what is left over.{' '}
                <M tex={`k = ${kk}`} /> keeps <M tex={fmtPct(cur.energy, 1).replace('%', '\\%')} /> of <M tex="\sum_j\sigma_j^2" />; the residual has Frobenius norm{' '}
                <M tex={texNum(cur.error, 3)} />, which equals <M tex={`\\sqrt{\\sum_{j>${kk}}\\sigma_j^2} = ${texNum(cur.eckartYoung, 3)}`} /> to <M tex={texNum(eyGap, 1)} />.
              </>
            ),
            why: (
              <>
                Each singular triplet contributes a rank-one layer <M tex="\sigma_j u_j v_j^T" /> whose squared Frobenius norm is exactly <M tex="\sigma_j^2" />, and the layers are
                mutually orthogonal in the trace inner product. Removing the layers <M tex={`j > ${kk}`} /> therefore removes exactly <M tex={`\\sum_{j>${kk}}\\sigma_j^2 = ${texNum(total - total * cur.energy, 3)}`} />{' '}
                of the total <M tex={`\\sum_j \\sigma_j^2 = ${texNum(total, 3)}`} />. Vertical stripes in <M tex={`X_{${kk}}`} /> are variables with large loadings on the retained{' '}
                <M tex="v_j" />; the residual looks like unstructured noise when <M tex="k" /> exceeds the effective rank.
              </>
            ),
            math: (
              <>
                Eckart–Young: for every <M tex="B" /> with <M tex={`\\operatorname{rank}B \\le ${kk}`} />, <M tex={`\\|X - B\\|_F \\ge \\|X - X_{${kk}}\\|_F`} /> and{' '}
                <M tex={`\\|X - B\\|_2 \\ge \\sigma_{${kk + 1}} = ${texNum(sigmaNext, 3)}`} /> (verified numerically: <M tex={`\\|X - X_{${kk}}\\|_2 = ${texNum(specNorm, 3)}`} />). The
                progressive strip below shows the sum being built one layer at a time.
              </>
            ),
            stats: (
              <>
                For a centred matrix <M tex={`X_{${kk}}`} /> is the PCA reconstruction <M tex={`\\hat X_{${kk}} = Z_{${kk}} V_{${kk}}^T`} /> from the first <M tex={`${kk}`} /> components;
                "energy kept" is the cumulative proportion of variance explained, <M tex={`${fmtPct(cur.energy, 1).replace('%', '\\%')}`} />. The per-observation residual norms
                identify points poorly described by the retained components.
              </>
            ),
            careful: (
              <>
                Optimality is in the Frobenius (or spectral) norm of the analysed matrix, so it depends on the preprocessing: standardising changes <M tex="X" /> and hence which{' '}
                <M tex="k" /> is "enough". Energy is not variance unless <M tex="X" /> is centred. Nothing here says the discarded layers are noise; a weak but systematic pattern
                (a small class difference, say) can live in <M tex={`\\sigma_{${kk + 1}}, \\dots`} />.
              </>
            ),
          }}
        />
      </div>

      <div className="grid side-r" style={{ marginTop: 14 }}>
        <Card title="Reconstruction error for every k">
          <div className="table-wrap">
            <table className="summary-table">
              <thead>
                <tr>
                  <th>
                    <M tex="k" />
                  </th>
                  <th>
                    <M tex="\|X - X_k\|_F" />
                  </th>
                  <th>
                    <M tex="\sqrt{\sum_{j>k}\sigma_j^2}" />
                  </th>
                  <th>relative error</th>
                  <th>energy kept</th>
                  <th>
                    <M tex="\sigma_k" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {summaries.map((sm) => (
                  <tr
                    key={sm.k}
                    onClick={() => sm.k > 0 && setK(sm.k)}
                    style={sm.k === kk ? { background: 'var(--accent-soft)' } : sm.k > 0 ? { cursor: 'pointer' } : undefined}
                    title={sm.k > 0 ? 'Click to set k' : undefined}
                  >
                    <td className="mono">{sm.k}</td>
                    <td className="mono">{fmt(sm.error, 4)}</td>
                    <td className="mono">{fmt(sm.eckartYoung, 4)}</td>
                    <td className="mono">{fmtPct(sm.relativeError, 2)}</td>
                    <td className="mono">{fmtPct(sm.energy, 2)}</td>
                    <td className="mono">{sm.k > 0 ? fmt(res.s[sm.k - 1], 4) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            The second and third columns agree to rounding for every <M tex="k" />: the reconstruction error never has to be computed, it is read off the singular values.
          </div>
        </Card>
        <Callout kind="theorem" title="Eckart–Young–Mirsky theorem">
          Let <M tex="X = \sum_{j=1}^{r}\sigma_j u_j v_j^T" /> with <M tex="\sigma_1 \ge \dots \ge \sigma_r \ge 0" /> and <M tex="X_k = \sum_{j\le k}\sigma_j u_j v_j^T" />. For every
          matrix <M tex="B \in \mathbb{R}^{n\times p}" /> with <M tex="\operatorname{rank} B \le k" />,
          <MBlock tex="\|X - B\|_F \;\ge\; \|X - X_k\|_F = \Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}, \qquad \|X - B\|_2 \;\ge\; \|X - X_k\|_2 = \sigma_{k+1}," />
          with equality in both attained by <M tex="B = X_k" />. Mirsky: the same holds for every unitarily invariant norm. The Frobenius minimiser is unique if and only if{' '}
          <M tex="\sigma_k > \sigma_{k+1}" />.
        </Callout>
      </div>

      <div className="grid c2" style={{ marginTop: 14 }}>
        <Card title="Progressive reconstruction (first rows of X)">
          <div className="small secondary" style={{ marginBottom: 8 }}>
            Target block: the first <M tex={`${blockRows}`} /> rows of <M tex="X" /> (all <M tex={`${p}`} /> variables, up to 6 shown). Reveal the layers one at a time.
          </div>
          <MatrixView M={Xblock} title={`X (rows 1–${blockRows})`} digits={2} heat="diverging" heatMax={zmax} rowLabels={rowLabels} colLabels={names} maxCols={6} compact />
          <div style={{ marginTop: 10 }}>
            <Derivation steps={steps} initiallyRevealed={1} title={<>Build <M tex={`X_{${kk}} = \\sum_{j \\le ${kk}} \\sigma_j u_j v_j^T`} /> layer by layer</>} />
          </div>
        </Card>
        <Card title="Proof sketch (Weyl’s inequality)">
          <Derivation steps={proof} initiallyRevealed={1} />
        </Card>
      </div>
    </Section>
  );
}
