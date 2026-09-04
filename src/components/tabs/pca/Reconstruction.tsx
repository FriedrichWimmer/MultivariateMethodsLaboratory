import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { useAnalysis, useStore } from '../../../state/store';
import { M, MBlock } from '../../common/Math';
import { Plot } from '../../common/Plot';
import { DataTable } from '../../common/DataTable';
import { Section, Card, Callout, Interpretation, StatTile, Badge } from '../../common/Panels';
import { Slider, Button } from '../../common/Controls';
import { fmt } from '../../../lib/linalg';
import { pcaReconstruct, reconstructionErrors } from '../../../lib/pca';
import { methodColor, neutralMark, ink, sequential } from '../../../lib/theme';
import { pct, sci, texNum, residualNorm } from './util';

/** Lesson 4 — rank-k reconstruction X̂_k = Z_k V_kᵀ and the Eckart–Young theorem. */
export function Reconstruction() {
  const a = useAnalysis();
  const { prep, navigate } = useStore();
  const { pca: P, n, p, dataset } = a;
  const names = dataset.variableNames;
  const r = P.V[0].length;
  const kmax = Math.max(1, Math.min(p, n - 1, r));
  const k = Math.min(Math.max(1, prep.k), kmax);

  const [obs, setObs] = useState(0);
  const i = Math.min(obs, n - 1);

  const c = useMemo(() => {
    const r1 = pcaReconstruct(P, 1);
    const r2 = pcaReconstruct(P, Math.min(2, r));
    const rk = pcaReconstruct(P, k);
    const errs = reconstructionErrors(P); // k = 0..r
    const s2 = P.singularValues.map((s) => s * s);
    const ey = errs.map((_, m) => Math.sqrt(s2.slice(m).reduce((x, y) => x + y, 0)));
    const eyGap = errs.reduce((mx, e, m) => Math.max(mx, Math.abs(e - ey[m])), 0);
    const total = errs[0];
    const rel = total > 0 ? errs[k] / total : 0;
    const cumFromError = 1 - rel * rel;
    const cumGap = Math.abs(cumFromError - P.cumulative[k - 1]);
    return { r1, r2, rk, errs, ey, eyGap, total, rel, cumFromError, cumGap };
  }, [P, k, r]);

  const row = useMemo(() => {
    const x = P.Xc[i];
    const z = P.scores[i];
    const h1 = c.r1.analysed[i];
    const h2 = c.r2.analysed[i];
    const hk = c.rk.analysed[i];
    const tail = (m: number) => Math.sqrt(z.slice(m).reduce((s, v) => s + v * v, 0));
    return {
      x,
      z,
      h1,
      h2,
      hk,
      res1: residualNorm(x, h1),
      res2: residualNorm(x, h2),
      resk: residualNorm(x, hk),
      tailk: tail(k),
      norm: Math.sqrt(x.reduce((s, v) => s + v * v, 0)),
      xo: dataset.X[i],
      hko: c.rk.original[i],
    };
  }, [P, c, i, k, dataset]);

  const barData: Data[] = [
    { type: 'bar', x: names, y: row.x, name: `x_${i + 1} (analysed scale)`, marker: { color: neutralMark }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
    { type: 'bar', x: names, y: row.h1, name: 'reconstruction with PC1', marker: { color: sequential[3] }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' },
  ];
  if (r >= 2) barData.push({ type: 'bar', x: names, y: row.h2, name: 'with PC1–PC2', marker: { color: sequential[7] }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' });
  if (k > 2) barData.push({ type: 'bar', x: names, y: row.hk, name: `with k = ${k} PCs`, marker: { color: sequential[11] }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' });
  const barLayout: Record<string, unknown> = { showlegend: true, barmode: 'group', xaxis: { title: { text: 'variable' } }, yaxis: { title: { text: 'value (analysed scale)' } } };

  const ks = c.errs.map((_, m) => m);
  const errData: Data[] = [
    { type: 'scatter', mode: 'lines+markers', x: ks, y: c.errs, name: '‖X_c − X̂_k‖_F (computed)', line: { color: methodColor.PCA }, marker: { color: methodColor.PCA, size: 7 }, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'lines+markers', x: ks, y: c.ey, name: '√(Σ_{j>k} σ_j²)  (Eckart–Young)', line: { color: neutralMark, dash: 'dash' }, marker: { color: neutralMark, size: 9, symbol: 'diamond-open' }, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: [k], y: [c.errs[k]], name: `current k = ${k}`, marker: { color: ink.primary, size: 13, symbol: 'circle-open', line: { width: 2, color: ink.primary } }, hovertemplate: 'k = %{x}: %{y:.4f}<extra></extra>' },
  ];
  const errLayout: Record<string, unknown> = { showlegend: true, xaxis: { title: { text: 'number of components k' }, dtick: 1 }, yaxis: { title: { text: 'Frobenius reconstruction error' }, rangemode: 'tozero' } };

  const stdz = prep.scaling === 'standardize';
  const raw = prep.scaling === 'none';

  return (
    <Section
      id="pca-reconstruction"
      title="4 · Reconstruction and the Eckart–Young theorem"
      subtitle="Keeping k scores and k loadings gives the best rank-k approximation of the centred data; the error is the tail of the singular values."
      right={<Badge method="PCA" />}
    >
      <div className="prose">
        <p>
          With <M tex="Z_k = X_c V_k" /> (the first <M tex="k" /> score columns) and <M tex="V_k" /> (the first <M tex="k" /> loadings), the rank-<M tex="k" /> reconstruction is
        </p>
        <MBlock tex={String.raw`\hat X_k = Z_k V_k^{T} = X_c V_k V_k^{T} = U_k \Sigma_k V_k^{T} = X_k ,`} />
        <p>
          i.e. the orthogonal projection of every row of <M tex="X_c" /> onto the <M tex="k" />-dimensional principal subspace, and at the same time the truncated SVD <M tex="X_k" /> of the SVD laboratory. Row{' '}
          <M tex="i" /> of <M tex="\hat X_k" /> is <M tex="\hat x_i = \sum_{j \le k} z_{ij}\, v_j" />: the observation rebuilt from its first <M tex="k" /> coordinates in the rotated basis. Choose an observation
          below (slider, or click a row of the table) and vary the global <M tex="k" /> in lesson 3.
        </p>
      </div>

      <div className="grid side">
        <div className="stack">
          <Card title="Observation" plane>
            <Slider label="Observation i" value={i + 1} min={1} max={n} step={1} onChange={(v) => setObs(v - 1)} />
            <div className="small muted">
              k = {k} (set in lesson 3). {dataset.y && dataset.classNames ? `Class: ${dataset.classNames[dataset.y[i]]}.` : ''}
            </div>
            <div className="stats" style={{ marginTop: 6 }}>
              <StatTile label="‖x_i − x̂_i‖ with PC1" value={fmt(row.res1, 4)} note={`of ‖x_i‖ = ${fmt(row.norm, 4)}`} />
              {r >= 2 && <StatTile label="with PC1–PC2" value={fmt(row.res2, 4)} note={row.norm > 0 ? `${pct(row.res2 / row.norm)} of ‖x_i‖` : ''} />}
              <StatTile label={`with k = ${k}`} value={fmt(row.resk, 4)} note={row.norm > 0 ? `${pct(row.resk / row.norm)} of ‖x_i‖` : ''} />
              <StatTile label={<M tex={`\\sqrt{\\sum_{j>k} z_{ij}^{2}}`} />} value={fmt(row.tailk, 4)} note={`gap ${sci(Math.abs(row.tailk - row.resk))}`} />
            </div>
          </Card>
          <Card title="Rows of X_c (click to select)" plane>
            <DataTable X={P.Xc} variableNames={names} y={dataset.y} classNames={dataset.classNames} maxRows={10} digits={2} highlight={[i]} onRowClick={setObs} />
            <div className="small muted">Showing the first 10 rows; use the slider for the rest.</div>
          </Card>
        </div>
        <div className="stack">
          <Plot data={barData} layout={barLayout} height={300} title={`Observation ${i + 1}: original values and reconstructions by variable`} />
          <div className="table-wrap">
            <table className="summary-table">
              <thead>
                <tr>
                  <th>variable</th>
                  <th>
                    <M tex="x_{ij}" /> (analysed)
                  </th>
                  <th>
                    <M tex="\hat x_{ij}" /> PC1
                  </th>
                  {r >= 2 && (
                    <th>
                      <M tex="\hat x_{ij}" /> PC1–PC2
                    </th>
                  )}
                  <th>
                    <M tex="\hat x_{ij}" /> k = {k}
                  </th>
                  <th>residual (k)</th>
                </tr>
              </thead>
              <tbody>
                {names.map((nm, j) => (
                  <tr key={nm}>
                    <td>{nm}</td>
                    <td>{fmt(row.x[j], 3)}</td>
                    <td>{fmt(row.h1[j], 3)}</td>
                    {r >= 2 && <td>{fmt(row.h2[j], 3)}</td>}
                    <td>{fmt(row.hk[j], 3)}</td>
                    <td>{fmt(row.x[j] - row.hk[j], 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small muted">
            Scores of this observation: z = ({row.z.map((v) => fmt(v, 3)).join(', ')}). The residual after k components is exactly the norm of the discarded scores, because the loadings are orthonormal.
          </div>
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              Observation {i + 1} of <b>{dataset.name}</b> in the analysed scale, next to its reconstructions. With PC1 alone the residual norm is {fmt(row.res1, 4)}; with two components{' '}
              {r >= 2 ? fmt(row.res2, 4) : '–'}; with the retained k = {k} it is {fmt(row.resk, 4)}, i.e. {row.norm > 0 ? pct(row.resk / row.norm) : '–'} of the observation's own norm {fmt(row.norm, 4)}.
              {k >= r && ' With k = r every observation is reconstructed exactly.'}
            </>
          ),
          why: (
            <>
              Each added component adds the term <M tex="z_{ij} v_j" /> to the reconstruction, so the bars move towards the original by an amount proportional to the score z_{`{${i + 1},j}`}. For this observation
              the scores are ({row.z.map((v) => fmt(v, 2)).join(', ')}): {Math.abs(row.z[0]) >= Math.max(...row.z.slice(1).map(Math.abs), 0) ? 'the first score dominates, so PC1 already captures most of it' : 'a later score is larger than the first — PC1 is the best direction on average over all observations, not for this particular one'}.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\hat x_i^{(k)} = \sum_{j\le k} z_{ij} v_j, \qquad x_i - \hat x_i^{(k)} = \sum_{j>k} z_{ij} v_j, \qquad \|x_i - \hat x_i^{(k)}\|^{2} = \sum_{j>k} z_{ij}^{2}`} />
              because <M tex="v_1,\dots,v_r" /> are orthonormal and, when <M tex="r = \operatorname{rank} X_c" />, span the row space. Live check: {fmt(row.resk, 5)} versus {fmt(row.tailk, 5)}.
            </>
          ),
          stats: (
            <>
              Reconstruction error per observation is a diagnostic: an observation whose residual is large relative to the others is poorly described by the common low-dimensional structure — a candidate outlier in
              the orthogonal complement (the "orthogonal distance" of PCA diagnostics), invisible in the score plot itself.
            </>
          ),
          careful: (
            <>
              The values are in the analysed scale{stdz ? ' (standard deviations from the mean)' : raw ? ' (raw, uncentred — the reconstruction then passes through the origin)' : ' (deviations from the mean)'}; the
              table further down maps them back to the original units. A small residual for one observation does not validate k — the Frobenius curve below aggregates over all n = {n} rows.
            </>
          ),
        }}
      />

      <Callout kind="theorem" title="Theorem (Eckart–Young–Mirsky)">
        Let <M tex="X_c = U\Sigma V^{T}" /> and let <M tex="X_k = U_k\Sigma_kV_k^{T}" /> be its rank-<M tex="k" /> truncation. Then for every matrix <M tex="B" /> of rank at most <M tex="k" />,
        <MBlock tex={String.raw`\|X_c - B\|_F \;\ge\; \|X_c - X_k\|_F = \sqrt{\sum_{j>k}\sigma_j^{2}}, \qquad \|X_c - B\|_2 \;\ge\; \|X_c - X_k\|_2 = \sigma_{k+1}.`} />
        Since <M tex="\hat X_k = X_k" />, PCA's rank-<M tex="k" /> reconstruction is optimal in both norms, and "variance explained" is the same statement as "squared Frobenius error minimised":{' '}
        <M tex="\|X_c - \hat X_k\|_F^{2}/\|X_c\|_F^{2} = 1 - \text{cumulative}_k" />.{' '}
        <Button small onClick={() => navigate('svd', 'svd-lowrank')}>
          Low-rank approximation in the SVD laboratory
        </Button>
      </Callout>

      <div className="grid side-r">
        <div>
          <Plot data={errData} layout={errLayout} height={320} title="Frobenius reconstruction error against k" />
          <div className="plot-caption">Solid: ‖X_c − Z_kV_kᵀ‖_F computed by explicit reconstruction for k = 0, …, r. Dashed: the Eckart–Young value from the singular values alone.</div>
        </div>
        <div className="stats">
          <StatTile label={<M tex={`\\|X_c - \\hat X_{${k}}\\|_F`} />} value={fmt(c.errs[k], 4)} note={`‖X_c‖_F = ${fmt(c.total, 4)}`} />
          <StatTile label="relative error" value={pct(c.rel)} note="‖X_c − X̂_k‖_F / ‖X_c‖_F" />
          <StatTile label="1 − (relative error)²" value={pct(c.cumFromError, 2)} note={`cumulative_k = ${pct(P.cumulative[k - 1], 2)}; gap ${sci(c.cumGap)}`} />
          <StatTile label="max |computed − Eckart–Young|" value={sci(c.eyGap)} note="over k = 0..r" />
        </div>
      </div>

      <Card title="Back to original units">
        <div className="prose">
          <p>
            The analysed matrix is <M tex={stdz ? 'X_s = (X - \\mathbf 1\\bar x^{T})D_s^{-1}' : raw ? 'X' : 'X_c = X - \\mathbf 1\\bar x^{T}'} />, so the reconstruction in original units is{' '}
            <M tex={stdz ? '\\hat x_{ij}^{\\mathrm{orig}} = \\bar x_j + s_j\\,\\hat x_{ij}' : raw ? '\\hat x_{ij}^{\\mathrm{orig}} = \\hat x_{ij}' : '\\hat x_{ij}^{\\mathrm{orig}} = \\bar x_j + \\hat x_{ij}'} />
            {stdz ? ' — the mean is added back and the standard deviation multiplies the residual, so variables with large spread receive large absolute errors even when their standardised error is small.' : raw ? ' — nothing was removed, so nothing is added back; note that the fitted subspace passes through the origin, not through the mean.' : ' — the mean is added back; residuals are unchanged in size.'}
          </p>
        </div>
        <div className="table-wrap">
          <table className="summary-table">
            <thead>
              <tr>
                <th>variable</th>
                <th>
                  <M tex="\bar x_j" />
                </th>
                {stdz && (
                  <th>
                    <M tex="s_j" />
                  </th>
                )}
                <th>
                  <M tex="x_{ij}" /> (original units)
                </th>
                <th>
                  <M tex="\hat x_{ij}" /> with k = {k}
                </th>
                <th>residual</th>
              </tr>
            </thead>
            <tbody>
              {names.map((nm, j) => (
                <tr key={nm}>
                  <td>{nm}</td>
                  <td>{fmt(P.means[j], 3)}</td>
                  {stdz && <td>{fmt(P.stds[j], 3)}</td>}
                  <td>{fmt(row.xo[j], 3)}</td>
                  <td>{fmt(row.hko[j], 3)}</td>
                  <td>{fmt(row.xo[j] - row.hko[j], 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Interpretation
        title="Interpretation — the error curve"
        items={{
          seeing: (
            <>
              The Frobenius error falls from ‖X_c‖_F = {fmt(c.total, 4)} at k = 0 to 0 at k = r = {r}; at the retained k = {k} it is {fmt(c.errs[k], 4)}, a relative error of {pct(c.rel)}. The dashed Eckart–Young curve
              √(Σ_{`{j>k}`} σ_j²) coincides with it to {sci(c.eyGap)}.
            </>
          ),
          why: (
            <>
              Removing component j removes exactly σ_j² = (n − 1)λ_j from the squared Frobenius norm, nothing more and nothing less, because the components are orthogonal. The curve is therefore steep where
              the eigenvalues are large and flat in the "rubble" of small eigenvalues — it is the scree plot read cumulatively from the right.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\|X_c - \hat X_k\|_F^{2} = \sum_{j>k}\sigma_j^{2} = (n-1)\sum_{j>k}\lambda_j, \qquad 1 - \frac{\|X_c - \hat X_k\|_F^{2}}{\|X_c\|_F^{2}} = \frac{\sum_{j\le k}\lambda_j}{\sum_j \lambda_j} .`} />
              Numerically 1 − ({pct(c.rel, 2)})² = {pct(c.cumFromError, 2)} against the cumulative proportion {pct(P.cumulative[k - 1], 2)} of lesson 3 (gap {sci(c.cumGap)}).
            </>
          ),
          stats: (
            <>
              This is the least-squares view of PCA: <M tex="\hat X_k" /> minimises the total squared distance between the observations and their projections onto a k-dimensional affine subspace through the mean
              (Pearson, 1901). The reconstruction error is therefore a natural loss for choosing k by cross-validation — leaving out entries and predicting them from a rank-k fit — which, unlike the heuristics of
              lesson 3, accounts for sampling noise.
            </>
          ),
          careful: (
            <>
              Optimality is in the Frobenius (and spectral) norm on the <em>analysed</em> matrix. {stdz ? 'Standardisation changes the norm: it is optimal for the standardised data, not for the original units.' : raw ? 'Without centring the "best rank-k matrix" includes the mean as (mostly) its first component — the approximation describes location, not dispersion.' : 'Changing the scaling of the variables changes the norm and hence the optimal subspace (lesson 5).'}{' '}
              The theorem says nothing about statistical optimality for prediction of anything outside X.
            </>
          ),
        }}
      />
    </Section>
  );
}
