import { useMemo, useState } from 'react';
import { useAnalysis, useStore } from '../../state/store';
import { DatasetControls, PrepControls } from '../common/DatasetControls';
import { Section, Card, Callout, Interpretation, ClassLegend, StatTile, Badge } from '../common/Panels';
import { ScatterSVG } from '../common/ScatterSVG';
import { Plot } from '../common/Plot';
import { MatrixView, MatrixEquation } from '../common/MatrixView';
import { DataTable } from '../common/DataTable';
import { Select } from '../common/Controls';
import { M, MBlock, texMatrix } from '../common/Math';
import { colMeans, colStds, correlation, covariance, fmt, centerColumns, standardizeColumns, matmul, transpose, symmetricEigen, svd } from '../../lib/linalg';
import { pca } from '../../lib/pca';
import { distanceMatrix, classicalMDS } from '../../lib/mds';
import { lda } from '../../lib/lda';
import { plotlyDiverging, accent } from '../../lib/theme';

export default function DataLab() {
  const a = useAnalysis();
  const { dataset, prep } = useStore();
  const p = a.p;
  const names = dataset.variableNames;
  const [vi, setVi] = useState(0);
  const [vj, setVj] = useState(1);
  const i = Math.min(vi, p - 1);
  const j = Math.min(vj, p - 1);

  const stats = useMemo(() => {
    const means = colMeans(a.X);
    const sds = colStds(a.X);
    const mins = names.map((_, c) => Math.min(...a.X.map((r) => r[c])));
    const maxs = names.map((_, c) => Math.max(...a.X.map((r) => r[c])));
    const R = correlation(a.X);
    const S = covariance(a.X);
    let rmax = 0;
    let pair: [number, number] = [0, 1];
    for (let x = 0; x < p; x++)
      for (let y = x + 1; y < p; y++)
        if (Math.abs(R[x][y]) > rmax) {
          rmax = Math.abs(R[x][y]);
          pair = [x, y];
        }
    const sdRatio = Math.max(...sds) / Math.max(Math.min(...sds), 1e-12);
    return { means, sds, mins, maxs, R, S, rmax, pair, sdRatio };
  }, [a.X, names, p]);

  const points = useMemo(() => a.X.map((r) => [r[i], r[j]]), [a.X, i, j]);

  const varOptions = names.map((nm, k) => ({ value: String(k), label: nm }));

  return (
    <>
      <div className="topbar">
        <div className="title">
          <h2>Data laboratory</h2>
          <div className="lede">
            Choose or generate the dataset that every other laboratory analyses, inspect the raw matrix <M tex="X" /> next to its graphical representation, and decide how it should be centred and scaled.
          </div>
        </div>
      </div>

      <Section id="data-lab" title="Choose and generate data" subtitle="All generators are seeded: the same parameters always reproduce the same matrix. Upload a CSV to analyse your own data.">
        <div className="grid side">
          <div className="stack">
            <Card title="Dataset">
              <DatasetControls />
            </Card>
            <Card title="Preprocessing (applies everywhere)">
              <PrepControls />
            </Card>
          </div>
          <div className="stack">
            <div className="row" style={{ alignItems: 'flex-end' }}>
              <Select label="Horizontal variable" value={String(i)} options={varOptions} onChange={(v) => setVi(Number(v))} />
              <Select label="Vertical variable" value={String(j)} options={varOptions} onChange={(v) => setVj(Number(v))} />
              {dataset.classNames && dataset.y && <ClassLegend classNames={dataset.classNames} />}
            </div>
            <ScatterSVG points={points} labels={dataset.y} classNames={dataset.classNames} xLabel={names[i]} yLabel={names[j]} height={380} width={560} equalAspect={false} title={`${dataset.name}: ${names[i]} against ${names[j]}`} />
            <div className="stats">
              <StatTile label="Observations n" value={a.n} />
              <StatTile label="Variables p" value={a.p} />
              <StatTile label="Classes K" value={a.y ? a.K : '—'} note={a.y ? 'labels available' : 'unlabelled'} />
              <StatTile label="Largest |correlation|" value={fmt(stats.rmax, 2)} note={`${names[stats.pair[0]]} & ${names[stats.pair[1]]}`} />
              <StatTile label="Ratio of largest to smallest sd" value={fmt(stats.sdRatio, 1)} note={stats.sdRatio > 10 ? 'scales differ strongly' : 'comparable scales'} />
            </div>
            <Interpretation
              items={{
                seeing: (
                  <>
                    Each marker is one row of <M tex="X" /> plotted in two of its {p} coordinates ({names[i]}, {names[j]}); {a.y ? 'colour and marker shape encode the class label, which the unsupervised methods will ignore.' : 'there are no class labels.'}
                  </>
                ),
                why: <>{dataset.description}</>,
                math: (
                  <>
                    The cloud's shape is summarised by the sample covariance <M tex="S" />: here <M tex={`s_{${i + 1}${i + 1}} = ${fmt(stats.S[i][i], 3)}`} />, <M tex={`s_{${j + 1}${j + 1}} = ${fmt(stats.S[j][j], 3)}`} /> and <M tex={`r_{${i + 1}${j + 1}} = ${fmt(stats.R[i][j], 3)}`} />. PCA will diagonalise exactly this matrix.
                  </>
                ),
                stats: (
                  <>
                    {stats.sdRatio > 10
                      ? `The standard deviations differ by a factor of ${fmt(stats.sdRatio, 0)}: any covariance-based analysis will be dominated by the variable with the largest scale unless the data are standardised.`
                      : `Standard deviations are within a factor of ${fmt(stats.sdRatio, 1)} of each other, so covariance- and correlation-based analyses will give similar (but not identical) results.`}
                  </>
                ),
                careful: (
                  <>
                    A two-variable view can hide structure in the other {Math.max(p - 2, 0)} coordinates; the full covariance heatmap below and the PCA laboratory show the whole picture. Simulated data satisfy the Gaussian assumptions by construction — uploaded data may not.
                  </>
                ),
              }}
            />
          </div>
        </div>
      </Section>

      <Section id="data-matrix" title="The raw data matrix next to its graphical summary" subtitle="Rows are observations i = 1…n, columns are variables. Every method in this laboratory starts from this matrix or from something computed from it.">
        <div className="grid c2">
          <div className="stack">
            <h3>
              <M tex="X" /> ({a.n} × {a.p})
            </h3>
            <DataTable X={a.X} variableNames={names} y={dataset.y} classNames={dataset.classNames} maxRows={150} />
            <table className="data-table" style={{ fontSize: 12.5 }}>
              <thead>
                <tr>
                  <th>variable</th>
                  <th>mean x̄</th>
                  <th>sd s</th>
                  <th>min</th>
                  <th>max</th>
                </tr>
              </thead>
              <tbody>
                {names.map((nm, c) => (
                  <tr key={nm}>
                    <td style={{ textAlign: 'left', color: 'var(--ink)' }}>{nm}</td>
                    <td>{fmt(stats.means[c], 3)}</td>
                    <td>{fmt(stats.sds[c], 3)}</td>
                    <td>{fmt(stats.mins[c], 2)}</td>
                    <td>{fmt(stats.maxs[c], 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="stack">
            <Plot
              title="Distribution of each variable (box plots, original units)"
              height={300}
              data={names.map((nm, c) => ({ type: 'box' as const, y: a.X.map((r) => r[c]), name: nm, marker: { color: accent, size: 3 }, line: { color: accent, width: 1.5 }, fillcolor: 'rgba(42,120,214,0.10)', boxpoints: 'outliers' as const, hoverinfo: 'y+name' as const }))}
              layout={{ yaxis: { title: { text: 'value' } }, margin: { b: 60 } }}
            />
            <Plot
              title="Correlation matrix R (diverging scale, −1 … 1)"
              height={Math.max(260, 40 + 32 * p)}
              data={[
                {
                  type: 'heatmap',
                  z: stats.R,
                  x: names,
                  y: names,
                  colorscale: plotlyDiverging,
                  zmin: -1,
                  zmax: 1,
                  zmid: 0,
                  colorbar: { thickness: 10, len: 0.9, tickfont: { size: 10 } },
                  hovertemplate: 'r(%{y}, %{x}) = %{z:.3f}<extra></extra>',
                },
              ]}
              layout={{ yaxis: { autorange: 'reversed' as const }, margin: { l: 90, b: 70 } }}
            />
          </div>
        </div>
      </Section>

      <Section id="data-preprocess" title="Centring, standardisation, covariance or correlation" subtitle="The choice made here changes which matrix the methods decompose. Nothing else in the laboratory is as consequential.">
        <PreprocessDemo />
        <div className="grid c3" style={{ marginTop: 14 }}>
          <Callout kind="definition" title="Raw (no centring)">
            The SVD of <M tex="X" /> itself describes directions through the <i>origin</i>. The leading singular vector then mostly captures the mean vector, not the dispersion. Appropriate only when the origin is meaningful (e.g. counts, spectra) and the analysis is deliberately about the uncentred second-moment matrix <M tex="X^TX/(n-1)" />.
          </Callout>
          <Callout kind="definition" title="Centring (covariance PCA)">
            <M tex="X_c = X - \mathbf 1\bar x^T" /> and <M tex="S = X_c^TX_c/(n-1)" />. Use when the variables share units and their variances are scientifically comparable — a variable with larger variance <i>should</i> then carry more weight.
          </Callout>
          <Callout kind="definition" title="Standardisation (correlation PCA)">
            <M tex="X_s = X_c\,\mathrm{diag}(s)^{-1}" /> so that <M tex="X_s^TX_s/(n-1) = R" />. Use when units differ or scales are arbitrary; every variable gets total variance 1 and the analysis becomes invariant to rescaling — at the cost of discarding the information carried by the relative variances.
          </Callout>
        </div>
        <Callout kind="info" title="Distances and the retained dimension">
          The <b>distance metric</b> chosen above is used by the MDS laboratory (classical MDS is exact only for Euclidean distances), and <b>k</b> is the number of dimensions retained in every truncated representation: <M tex="X_k" />, <M tex="Z_k" />, the MDS configuration, and the number of discriminant directions displayed. Currently: {prep.metric} metric, k = {prep.k}.
        </Callout>
      </Section>

      <WorkedExample />
    </>
  );
}

function PreprocessDemo() {
  const a = useAnalysis();
  const { dataset } = useStore();
  const names = dataset.variableNames;
  const demo = useMemo(() => {
    const { Xc, means } = centerColumns(a.X);
    const { Xs, stds } = standardizeColumns(a.X);
    const S = covariance(a.X);
    const R = correlation(a.X);
    return { Xc, Xs, means, stds, S, R };
  }, [a.X]);
  const rows = Math.min(6, a.n);
  const rl = Array.from({ length: rows }, (_, i) => `${i + 1}`);
  return (
    <div className="stack">
      <div className="matrix-eq" style={{ alignItems: 'flex-start', gap: 18 }}>
        <MatrixView M={a.X.slice(0, rows)} title="X (first rows)" rowLabels={rl} colLabels={names} digits={2} compact />
        <span className="op">→</span>
        <MatrixView M={demo.Xc.slice(0, rows)} title="X_c = X − 1x̄ᵀ" rowLabels={rl} colLabels={names} digits={2} compact heat="diverging" />
        <span className="op">→</span>
        <MatrixView M={demo.Xs.slice(0, rows)} title="X_s = X_c diag(s)⁻¹" rowLabels={rl} colLabels={names} digits={2} compact heat="diverging" />
      </div>
      <div className="matrix-eq" style={{ alignItems: 'flex-start', gap: 18 }}>
        <MatrixView M={[demo.means]} title="x̄ᵀ (column means)" colLabels={names} digits={3} compact />
        <MatrixView M={[demo.stds]} title="sᵀ (column standard deviations)" colLabels={names} digits={3} compact />
        <MatrixView M={demo.S} title="S = X_cᵀX_c/(n−1)" rowLabels={names} colLabels={names} digits={3} heat="diverging" compact />
        <MatrixView M={demo.R} title="R = X_sᵀX_s/(n−1)" rowLabels={names} colLabels={names} digits={3} heat="diverging" heatMax={1} compact />
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              The same six observations in three coordinate systems: original units, deviations from the mean, and deviations measured in standard-deviation units. The active preprocessing is <b>{a.prep.scaling === 'none' ? 'raw' : a.prep.scaling === 'center' ? 'centring' : 'standardisation'}</b>, so the other laboratories decompose{' '}
              {a.prep.scaling === 'none' ? <M tex="X" /> : a.prep.scaling === 'center' ? <M tex="X_c" /> : <M tex="X_s" />}.
            </>
          ),
          why: (
            <>
              Centring removes the location; standardisation additionally removes the scale. On this dataset the diagonal of <M tex="S" /> ranges from {fmt(Math.min(...demo.S.map((r, i) => r[i])), 3)} to {fmt(Math.max(...demo.S.map((r, i) => r[i])), 3)}, whereas the diagonal of <M tex="R" /> is identically 1.
            </>
          ),
          math: (
            <>
              <M tex="R = D_s^{-1} S D_s^{-1}" /> with <M tex="D_s = \mathrm{diag}(s_1,\dots,s_p)" />. The eigenvectors of <M tex="S" /> and <M tex="R" /> are not related by any simple transformation, so covariance PCA and correlation PCA are genuinely different analyses, not re-parametrisations of one another.
            </>
          ),
          stats: (
            <>
              Total variance is <M tex={`\\operatorname{tr}(S) = ${fmt(demo.S.reduce((s, r, i) => s + r[i], 0), 3)}`} /> in original units and <M tex={`\\operatorname{tr}(R) = ${a.p}`} /> after standardisation; "variance explained" percentages refer to whichever total is in force.
            </>
          ),
          careful: <>Standardising a variable with near-zero variance amplifies noise; standardising binary or count variables changes their meaning; and neither choice makes a nonlinear structure linear.</>,
        }}
      />
    </div>
  );
}

/** A tiny integer example whose every quantity can be verified by hand. */
function WorkedExample() {
  const ex = useMemo(() => {
    const X = [
      [7, 3],
      [6, 6],
      [4, 5],
      [3, 2],
    ];
    const y = [0, 0, 1, 1];
    const { Xc, means } = centerColumns(X);
    const G = matmul(transpose(Xc), Xc);
    const S = covariance(X);
    const eig = symmetricEigen(S);
    const dec = svd(Xc);
    const pc = pca(X, 'center');
    const D = distanceMatrix(Xc, 'euclidean');
    const mds = classicalMDS(D, 2);
    const ld = lda(X, y);
    return { X, y, Xc, means, G, S, eig, dec, pc, D, mds, ld };
  }, []);
  const rl = ['1', '2', '3', '4'];
  const cl = ['x₁', 'x₂'];
  return (
    <Section id="data-worked" title="A worked example you can verify by hand" subtitle="Four observations, two variables. Every quantity below is computed by the same code that analyses the large datasets — and every value can be checked with pencil and paper.">
      <div className="grid c2">
        <div className="stack">
          <div className="matrix-eq" style={{ gap: 16, alignItems: 'flex-start' }}>
            <MatrixView M={ex.X} title="X" rowLabels={rl} colLabels={cl} digits={0} />
            <MatrixView M={[ex.means]} title="x̄ᵀ = (5, 4)" colLabels={cl} digits={0} />
            <MatrixView M={ex.Xc} title="X_c" rowLabels={rl} colLabels={cl} digits={0} heat="diverging" />
          </div>
          <MBlock tex={String.raw`X_c^TX_c = \begin{bmatrix}10&3\\3&10\end{bmatrix},\qquad S = \tfrac13 X_c^TX_c = \begin{bmatrix}10/3&1\\1&10/3\end{bmatrix}`} />
          <MatrixEquation items={[<MatrixView key="g" M={ex.G} title="X_cᵀX_c (computed)" digits={3} />, <MatrixView key="s" M={ex.S} title="S (computed)" digits={3} />]} />
          <div className="prose small">
            <p>
              Eigenvalues of <M tex="S" /> solve <M tex="(10/3-\lambda)^2 = 1" />, so <M tex="\lambda_1 = 13/3 \approx 4.333" /> and <M tex="\lambda_2 = 7/3 \approx 2.333" />, with eigenvectors <M tex="(1,1)/\sqrt2" /> and <M tex="(1,-1)/\sqrt2" />. Computed: λ = ({ex.eig.values.map((v) => fmt(v, 3)).join(', ')}), <M tex={`V = ${texMatrix(ex.eig.vectors, 3)}`} />.
            </p>
            <p>
              Singular values of <M tex="X_c" /> are <M tex="\sqrt{13} \approx 3.606" /> and <M tex="\sqrt7 \approx 2.646" />; computed σ = ({ex.dec.s.map((v) => fmt(v, 3)).join(', ')}). Check: <M tex="\sigma_1^2/(n-1) = 13/3" /> — computed {fmt((ex.dec.s[0] ** 2) / 3, 3)}. Scores <M tex="z_1 = X_cv_1 = (1,3,0,-4)/\sqrt2" />: computed ({ex.pc.scores.map((r) => fmt(r[0], 3)).join(', ')}), whose sample variance {fmt(ex.pc.scores.reduce((s, r) => s + r[0] ** 2, 0) / 3, 3)} equals <M tex="\lambda_1" />.
            </p>
          </div>
        </div>
        <div className="stack">
          <div className="matrix-eq" style={{ gap: 16, alignItems: 'flex-start' }}>
            <MatrixView M={ex.D} title="D (Euclidean, from X_c)" rowLabels={rl} colLabels={rl} digits={3} heat="sequential" />
            <MatrixView M={ex.mds.B} title="B = −½JD⁽²⁾J = X_cX_cᵀ" rowLabels={rl} colLabels={rl} digits={2} heat="diverging" />
          </div>
          <div className="prose small">
            <p>
              <Badge method="MDS" /> Eigenvalues of <M tex="B" />: ({ex.mds.eigenvalues.map((v) => fmt(v, 3)).join(', ')}) — exactly <M tex="\sigma_1^2 = 13,\ \sigma_2^2 = 7" /> and two zeros, because <M tex="B = X_cX_c^T" /> has rank 2. The MDS coordinates <M tex="V_2\Lambda_2^{1/2}" /> are ({ex.mds.coords.map((r) => `(${fmt(r[0], 2)}, ${fmt(r[1], 2)})`).join(', ')}): the PCA scores up to sign. Stress-1 = {fmt(ex.mds.stress1, 6)}.
            </p>
            <p>
              <Badge method="LDA" /> With labels y = (A, A, B, B): class means <M tex="m_A = (6.5, 4.5)" />, <M tex="m_B = (3.5, 3.5)" />, so <M tex="S_B = \begin{bmatrix}9&3\\3&1\end{bmatrix}" /> and <M tex="S_W = \begin{bmatrix}1&0\\0&9\end{bmatrix}" /> (check: <M tex="S_W + S_B = X_c^TX_c" />). Computed <M tex={`S_W = ${texMatrix(ex.ld.SW, 2)}`} />, <M tex={`S_B = ${texMatrix(ex.ld.SB, 2)}`} />. The Fisher direction <M tex="w \propto S_W^{-1}(m_A - m_B) = (3, 1/9)" /> gives <M tex="\lambda = (m_A-m_B)^TS_W^{-1}(m_A-m_B) = 9 + 1/9 \approx 9.111" />; computed w = ({ex.ld.W.map((r) => fmt(r[0], 4)).join(', ')}), λ = {fmt(ex.ld.eigenvalues[0], 3)}, and only K − 1 = 1 direction exists.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
