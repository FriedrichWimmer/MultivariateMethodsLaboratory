import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { useAnalysis } from '../../state/store';
import { generateDataset, defaultParams } from '../../lib/datasets';
import { pca } from '../../lib/pca';
import { lda } from '../../lib/lda';
import { distanceMatrix, classicalMDS, metricLabels, type Metric } from '../../lib/mds';
import { svd, symmetricEigen, cholesky, conditionNumber, covariance, centerColumns, gram, column, dot, fmt, trace, colMeans, colStds, EPS } from '../../lib/linalg';
import { makeRNG } from '../../lib/random';
import { methodColor, neutralMark, accent, accent2, diverging, ink, classColor } from '../../lib/theme';
import { M, MBlock, texVector } from '../common/Math';
import { Plot } from '../common/Plot';
import { ScatterSVG } from '../common/ScatterSVG';
import { Slider, Select } from '../common/Controls';
import { Section, Card, Callout, Interpretation, StatTile, Badge, ClassLegend } from '../common/Panels';
import {
  angleBetweenDeg,
  axialAngleDeg,
  axialDiffDeg,
  leverages,
  kappaFromSingularValues,
  varianceInflation,
  naiveVariance,
  twoPassVariance,
  welfordVariance,
  laeuchli,
  texSci,
  fmtSci,
  fmtDeg,
  fmtPct,
  relErr,
  digitsLost,
} from './diag/helpers';
import { LineLegend } from './diag/LineLegend';

const COLL_N = 100;
const COLL_P = 5;
const ZERO_COLOR = ink.axis;
const POS_COLOR = diverging[1];
const NEG_COLOR = diverging[7];
const LOG_FLOOR = 1e-20;

const clampLog = (v: number) => Math.max(v, LOG_FLOOR);

/** Bar traces for a spectrum split into "above tolerance" and "numerically zero" groups (log scale). */
function spectrumTraces(values: number[], tol: number, aboveLabel: string, color: string, tolLabel = 'tolerance'): Data[] {
  const idx = values.map((_, j) => j + 1);
  const above = idx.filter((_, j) => values[j] > tol);
  const below = idx.filter((_, j) => !(values[j] > tol));
  const traces: Data[] = [
    { type: 'bar', x: above, y: above.map((j) => clampLog(values[j - 1])), name: aboveLabel, marker: { color }, hovertemplate: 'j=%{x}<br>%{y:.3e}<extra></extra>' },
  ];
  if (below.length) traces.push({ type: 'bar', x: below, y: below.map((j) => clampLog(values[j - 1])), name: 'numerically zero (below tolerance)', marker: { color: ZERO_COLOR }, hovertemplate: 'j=%{x}<br>%{y:.3e}<extra></extra>' });
  traces.push({ type: 'scatter', mode: 'lines', x: [0.5, values.length + 0.5], y: [clampLog(tol), clampLog(tol)], name: tolLabel, line: { color: accent2, dash: 'dot', width: 1.5 }, hoverinfo: 'skip' });
  return traces;
}

const logBarLayout = (xTitle: string, yTitle: string, n: number): Record<string, unknown> => ({
  showlegend: true,
  barmode: 'overlay',
  xaxis: { title: { text: xTitle }, tickmode: n <= 24 ? 'array' : 'auto', tickvals: n <= 24 ? Array.from({ length: n }, (_, j) => j + 1) : undefined, range: [0.4, n + 0.6] },
  yaxis: { type: 'log', title: { text: yTitle }, exponentformat: 'power' },
});

// ---------------------------------------------------------------------------
// 1. Rank deficiency and singular covariance
// ---------------------------------------------------------------------------

function RankSection() {
  const [noise, setNoise] = useState(0);
  const [seed, setSeed] = useState(7);

  const r = useMemo(() => {
    const ds = generateDataset('collinear', { ...defaultParams, n: COLL_N, p: COLL_P, noise, seed });
    const n = ds.X.length;
    const p = ds.X[0].length;
    const { Xc } = centerColumns(ds.X);
    const dec = svd(Xc);
    const S = covariance(ds.X);
    const eig = symmetricEigen(S);
    const L = cholesky(S);
    const lamSvd = dec.s.map((s) => (s * s) / (n - 1));
    const eigTol = Math.max(eig.values[0], 1e-300) * p * EPS * 10;
    const nonzeroEig = eig.values.filter((v) => v > eigTol).length;
    const minDiagL = L ? Math.min(...L.map((row, i) => row[i])) : NaN;
    const negEig = eig.values.filter((v) => v < 0).length;
    return { ds, n, p, dec, S, eig, L, lamSvd, noiseSd: noise * 0.2, eigTol, nonzeroEig, minDiagL, negEig };
  }, [noise, seed]);

  const wide = useMemo(() => {
    const ds = generateDataset('pGreaterN', { ...defaultParams, n: 10, p: 20, K: 3, seed });
    const n = ds.X.length;
    const p = ds.X[0].length;
    const { Xc } = centerColumns(ds.X);
    const dec = svd(Xc);
    const S = covariance(ds.X);
    const eig = symmetricEigen(S);
    const L = cholesky(S);
    const eigTol = Math.max(eig.values[0], 1e-300) * p * EPS * 10;
    const nonzero = eig.values.filter((v) => v > eigTol).length;
    return { ds, n, p, dec, S, eig, L, eigTol, nonzero };
  }, [seed]);

  const sigmaMin = r.dec.s[r.dec.s.length - 1];
  const deficient = r.dec.rank < r.p;

  return (
    <Section
      id="diag-rank"
      title="1 · Rank deficiency and a singular covariance matrix"
      subtitle={
        <>
          Near-collinear data: <M tex="x_3,\dots,x_p" /> are linear combinations of <M tex="x_1, x_2" /> plus noise. Turn the noise to zero and watch <M tex="\operatorname{rank}(X_c)" /> collapse to 2.
        </>
      }
    >
      <div className="grid side">
        <div className="controls-panel">
          <Slider label="Collinearity noise (sd of the residual, ×0.2)" value={noise} min={0} max={1} step={0.01} onChange={setNoise} format={(v) => `${v.toFixed(2)} → sd ${(v * 0.2).toFixed(3)}`} />
          <Slider label="Seed" value={seed} min={1} max={99} step={1} onChange={setSeed} />
          <div className="small secondary">
            Local dataset <b>collinear</b>: <M tex={`n = ${r.n},\\ p = ${r.p}`} />. The SVD rank tolerance is <M tex="\max(n,p)\,\varepsilon\,\sigma_1 \cdot 10" /> with <M tex="\varepsilon \approx 2.2\times 10^{-16}" />.
          </div>
          <div className="stats">
            <StatTile label={<>numerical rank of <M tex="X_c" /></>} value={`${r.dec.rank} / ${r.p}`} note={deficient ? 'rank deficient' : 'full column rank'} />
            <StatTile label="rank tolerance" value={<M tex={texSci(r.dec.tol)} />} note="from svd(X_c).tol" />
            <StatTile label={<M tex="\sigma_1" />} value={fmt(r.dec.s[0], 3)} />
            <StatTile label={<M tex={`\\sigma_{${r.p}}`} />} value={<M tex={texSci(sigmaMin)} />} note={sigmaMin > r.dec.tol ? 'above tolerance' : 'below tolerance'} />
          </div>
          {r.L ? (
            <Callout kind="good" title="cholesky(S) succeeded">
              <M tex="S = LL^T" /> exists numerically; the smallest diagonal entry of <M tex="L" /> is <M tex={texSci(r.minDiagL)} />. Small pivots signal that positive definiteness is fragile.
            </Callout>
          ) : (
            <Callout kind="danger" title="cholesky(S) returned null">
              A pivot <M tex="L_{jj}^2 = S_{jj} - \sum_{k<j} L_{jk}^2" /> became non-positive: <M tex="S" /> is not numerically positive definite. Anything that inverts <M tex="S" /> (Mahalanobis distances, regression, classical LDA) is undefined here.
            </Callout>
          )}
        </div>
        <div className="stack">
          <Plot
            title="Singular values of X_c (log scale)"
            data={spectrumTraces(r.dec.s, r.dec.tol, 'σⱼ above tolerance', methodColor.SVD, 'rank tolerance')}
            layout={logBarLayout('index j', 'σⱼ (log scale)', r.p)}
            height={300}
          />
          <div className="plot-caption">
            The y axis is logarithmic; values are floored at <M tex="10^{-20}" /> for display. Bars in grey fall below the rank tolerance and are treated as zero.
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>j</th>
                <th>
                  <M tex="\sigma_j" /> (SVD of <M tex="X_c" />)
                </th>
                <th>
                  <M tex="\sigma_j^2/(n-1)" />
                </th>
                <th>
                  <M tex="\lambda_j(S)" /> (Jacobi on <M tex="S" />)
                </th>
                <th>difference</th>
              </tr>
            </thead>
            <tbody>
              {r.dec.s.map((s, j) => (
                <tr key={j}>
                  <td>{j + 1}</td>
                  <td className="mono">{fmtSci(s, 4)}</td>
                  <td className="mono">{fmtSci(r.lamSvd[j], 4)}</td>
                  <td className="mono">{fmtSci(r.eig.values[j], 4)}</td>
                  <td className="mono">{fmtSci(r.eig.values[j] - r.lamSvd[j], 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="divider" />
      <h3>
        The <M tex="p > n" /> case: at most <M tex="n-1" /> non-zero eigenvalues
      </h3>
      <div className="grid side-r">
        <div className="stack">
          <Plot
            title={`Eigenvalues of S for n = ${wide.n}, p = ${wide.p} (log scale)`}
            data={spectrumTraces(wide.eig.values.map((v) => Math.max(v, 0)), wide.eigTol, 'λⱼ above tolerance', methodColor.PCA, 'eigenvalue tolerance p·ε·λ₁·10')}
            layout={logBarLayout('index j', 'λⱼ(S) (log scale)', wide.p)}
            height={300}
          />
          <div className="plot-caption">
            Local dataset <b>pGreaterN</b>. Negative computed eigenvalues (rounding noise) are shown at the display floor.
          </div>
        </div>
        <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <StatTile label={<M tex="n - 1" />} value={wide.n - 1} note="upper bound on the rank of X_c" />
          <StatTile label={<>rank of <M tex="X_c" /> (SVD)</>} value={wide.dec.rank} note={`of min(n, p) = ${Math.min(wide.n, wide.p)} singular values`} />
          <StatTile label={<>non-zero <M tex="\lambda_j(S)" /></>} value={`${wide.nonzero} / ${wide.p}`} note={`${wide.p - wide.nonzero} eigenvalues are numerically zero`} />
          <StatTile label="cholesky(S)" value={wide.L ? 'factor exists' : 'null'} note={wide.L ? 'unexpected' : 'S is singular by construction'} />
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              With residual sd {r.noiseSd.toFixed(3)} the centred matrix has numerical rank {r.dec.rank} of {r.p}: {deficient ? `${r.p - r.dec.rank} singular value(s) fall below the tolerance ${fmtSci(r.dec.tol, 2)}` : `all singular values exceed the tolerance ${fmtSci(r.dec.tol, 2)}, although σ${r.p} = ${fmtSci(sigmaMin, 2)} is tiny compared with σ₁ = ${fmt(r.dec.s[0], 2)}`}. In the wide example only {wide.nonzero} of {wide.p} eigenvalues of <M tex="S" /> are non-zero, exactly <M tex="n-1" /> = {wide.n - 1}.
            </>
          ),
          why: (
            <>
              Each extra column is <M tex="x_j = w_{1j}x_1 + w_{2j}x_2 + e_j" />; when <M tex="e_j" /> vanishes, every column lies in the two-dimensional span of <M tex="x_1,x_2" />, so <M tex="\operatorname{rank}(X_c)\le 2" />. Centring removes one further direction in the <M tex="p>n" /> case: the <M tex="n" /> rows of <M tex="X_c" /> sum to zero, so they span at most <M tex="n-1" /> dimensions.
            </>
          ),
          math: (
            <>
              <M tex="\operatorname{rank}(S) = \operatorname{rank}(X_c^T X_c) = \operatorname{rank}(X_c) \le \min(n-1, p)" />. A symmetric matrix has a Cholesky factor iff it is positive definite; with a zero eigenvalue the pivot <M tex="L_{jj}^2" /> collapses to rounding noise ({r.L ? `here the smallest pivot is ${fmtSci(r.minDiagL, 2)}` : 'here it became non-positive and the factorisation was abandoned'}). Rank is a discontinuous function of the data, so any numerical rank is a threshold decision: the SVD uses <M tex={`\\sigma_j > ${texSci(r.dec.tol, 1)}`} />, and the eigen-route needs the coarser <M tex="\lambda_j > p\,\varepsilon\,\lambda_1\cdot 10" /> because its small eigenvalues are only accurate to about <M tex="\varepsilon\lambda_1" /> in absolute terms.
            </>
          ),
          stats: (
            <>
              A singular or near-singular <M tex="S" /> means some linear combination of the variables is (almost) constant in the sample: the data occupy an affine subspace of dimension {r.dec.rank}. PCA is still well defined (the trailing components simply have variance ≈ 0), but every method that needs <M tex="S^{-1}" /> — Mahalanobis distance, Hotelling's <M tex="T^2" />, Fisher's LDA with a singular <M tex="S_W" /> — is ill-posed until the problem is restricted to the range or regularised.
            </>
          ),
          careful: (
            <>
              The table shows that <M tex="\sigma_j^2/(n-1)" /> and the Jacobi eigenvalues agree for the leading components but differ at the level of <M tex="\varepsilon\lambda_1" /> for the trailing ones{r.negEig > 0 ? `; ${r.negEig} eigenvalue(s) of S even came out negative` : ''}. Never read a numerical rank off the eigenvalues of <M tex="S" /> without a tolerance, and never compare singular values across datasets without dividing by <M tex="\sigma_1" />.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 2. Conditioning and multicollinearity
// ---------------------------------------------------------------------------

function ConditioningSection() {
  const [noise, setNoise] = useState(0.05);
  const [seed, setSeed] = useState(7);

  const r = useMemo(() => {
    const ds = generateDataset('collinear', { ...defaultParams, n: COLL_N, p: COLL_P, noise, seed });
    const n = ds.X.length;
    const p = ds.X[0].length;
    const { Xc } = centerColumns(ds.X);
    const dec = svd(Xc);
    const S = covariance(ds.X);
    const kX = conditionNumber(Xc);
    const kS = conditionNumber(S);
    const vr = column(dec.V, dec.V[0].length - 1);
    const sigmaR = dec.s[dec.s.length - 1];
    const sdComb = sigmaR / Math.sqrt(n - 1);
    const sds = colStds(ds.X);
    const vifs = varianceInflation(ds.X);
    const grid = Array.from({ length: 25 }, (_, i) => 0.002 * Math.pow(500, i / 24));
    const curve = grid.map((g) => {
      const d = generateDataset('collinear', { ...defaultParams, n: COLL_N, p: COLL_P, noise: g, seed });
      const kx = conditionNumber(centerColumns(d.X).Xc);
      const ks = conditionNumber(covariance(d.X));
      return { noise: g, kx, ks };
    });
    const ratio = Number.isFinite(kS) && Number.isFinite(kX) ? kS / (kX * kX) : NaN;
    const saturated = !Number.isFinite(kX) || kX * kX > 1 / EPS;
    return { ds, n, p, dec, S, kX, kS, vr, sigmaR, sdComb, sds, vifs, curve, ratio, saturated };
  }, [noise, seed]);

  const finiteOrNull = (v: number) => (Number.isFinite(v) ? v : null);
  const curveTraces: Data[] = [
    { type: 'scatter', mode: 'lines+markers', x: r.curve.map((c) => c.noise * 0.2), y: r.curve.map((c) => finiteOrNull(c.kx)), name: 'κ₂(X_c)', line: { color: methodColor.SVD }, marker: { size: 5 } },
    { type: 'scatter', mode: 'lines+markers', x: r.curve.map((c) => c.noise * 0.2), y: r.curve.map((c) => finiteOrNull(c.ks)), name: 'κ₂(S)', line: { color: methodColor.PCA }, marker: { size: 5 } },
    { type: 'scatter', mode: 'lines', x: r.curve.map((c) => c.noise * 0.2), y: r.curve.map((c) => finiteOrNull(c.kx * c.kx)), name: 'κ₂(X_c)²', line: { color: neutralMark, dash: 'dash' } },
    { type: 'scatter', mode: 'lines', x: [r.curve[0].noise * 0.2, r.curve[r.curve.length - 1].noise * 0.2], y: [1 / EPS, 1 / EPS], name: '1/ε (floor of double precision)', line: { color: accent2, dash: 'dot', width: 1.5 } },
    { type: 'scatter', mode: 'lines', x: [Math.max(noise * 0.2, 0.0004), Math.max(noise * 0.2, 0.0004)], y: [1, 1e33], name: 'current noise', line: { color: ink.muted, width: 1 } },
  ];
  const loadTraces: Data[] = [{ type: 'bar', x: r.ds.variableNames, y: r.vr, marker: { color: methodColor.SVD }, name: 'v_r', hovertemplate: '%{x}: %{y:.4f}<extra></extra>' }];

  return (
    <Section
      id="diag-conditioning"
      title="2 · Numerical conditioning and multicollinearity"
      subtitle={
        <>
          <M tex="\kappa_2(X_c) = \sigma_1/\sigma_r" /> measures how close <M tex="X_c" /> is to rank deficiency; forming <M tex="S" /> squares it.
        </>
      }
    >
      <div className="grid side">
        <div className="controls-panel">
          <Slider label="Collinearity noise (sd of the residual, ×0.2)" value={noise} min={0} max={1} step={0.01} onChange={setNoise} format={(v) => `${v.toFixed(2)} → sd ${(v * 0.2).toFixed(3)}`} />
          <Slider label="Seed" value={seed} min={1} max={99} step={1} onChange={setSeed} />
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label={<M tex="\kappa_2(X_c)" />} value={<M tex={texSci(r.kX)} />} note={`≈ ${digitsLost(r.kX)} digits lost`} />
            <StatTile label={<M tex="\kappa_2(S)" />} value={<M tex={texSci(r.kS)} />} note={`≈ ${digitsLost(r.kS)} digits lost`} />
            <StatTile label={<M tex="\kappa_2(X_c)^2" />} value={<M tex={texSci(r.kX * r.kX)} />} note="predicted κ₂(S)" />
            <StatTile label={<M tex="\kappa_2(S)/\kappa_2(X_c)^2" />} value={Number.isFinite(r.ratio) ? r.ratio.toFixed(3) : '—'} note={r.saturated ? 'saturated at 1/ε' : 'should be ≈ 1'} />
          </div>
          {r.saturated && (
            <Callout kind="warning" title="Beyond the precision floor">
              <M tex="\kappa_2(X_c)^2" /> exceeds <M tex="1/\varepsilon \approx 4.5\times 10^{15}" />. The smallest eigenvalue of the computed <M tex="S" /> is pure rounding noise, so <M tex="\kappa_2(S)" /> stalls near <M tex="1/\varepsilon" /> instead of following the square law.
            </Callout>
          )}
        </div>
        <div className="stack">
          <Plot title="Condition numbers against the collinearity noise (log–log)" data={curveTraces} layout={{ showlegend: true, xaxis: { type: 'log', title: { text: 'residual sd' }, exponentformat: 'power' }, yaxis: { type: 'log', title: { text: 'condition number' }, exponentformat: 'power' } }} height={320} />
          <div className="plot-caption">
            The dashed curve <M tex="\kappa_2(X_c)^2" /> coincides with <M tex="\kappa_2(S)" /> until it hits <M tex="1/\varepsilon" />: the square law is exact in real arithmetic, and double precision cannot represent a larger condition number.
          </div>
        </div>
      </div>

      <div className="divider" />
      <div className="grid c2">
        <Card title={<>The near-null right singular vector v_r</>}>
          <Plot data={loadTraces} layout={{ xaxis: { title: { text: 'variable' } }, yaxis: { title: { text: 'coefficient' }, zeroline: true } }} height={220} />
          <MBlock tex={`v_r = ${texVector(r.vr, 3)}^T,\\qquad \\operatorname{sd}\\big(X_c v_r\\big) = \\frac{\\sigma_r}{\\sqrt{n-1}} = ${texSci(r.sdComb)}`} />
          <p className="small secondary">
            The combination <M tex="\sum_j v_{rj}x_j" /> has standard deviation {fmtSci(r.sdComb, 2)} while the individual variables have sds between {fmt(Math.min(...r.sds), 2)} and {fmt(Math.max(...r.sds), 2)}: it is the linear relation that is (almost) exactly satisfied by the data.
          </p>
        </Card>
        <Card title="Variance-inflation view of the same fact">
          <table className="summary-table">
            <thead>
              <tr>
                <th>variable</th>
                <th>
                  <M tex="R_j^2" /> on the others
                </th>
                <th>
                  <M tex="\mathrm{VIF}_j = 1/(1-R_j^2)" />
                </th>
              </tr>
            </thead>
            <tbody>
              {r.ds.variableNames.map((v, j) => (
                <tr key={v}>
                  <td>{v}</td>
                  <td className="mono">{r.vifs.rSquared[j] > 0.999999 ? '> 0.999999' : r.vifs.rSquared[j].toFixed(6)}</td>
                  <td className="mono">{fmtSci(r.vifs.vif[j], 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small secondary" style={{ marginTop: 8 }}>
            <M tex="\mathrm{VIF}_j = (R^{-1})_{jj}" /> for the correlation matrix <M tex="R" />
            {r.vifs.pseudo ? '; R was numerically singular, so a pseudo-inverse on its range was used' : ''}. A VIF above 10 is the conventional alarm; here the largest is {fmtSci(Math.max(...r.vifs.vif), 2)}.
          </p>
        </Card>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              At residual sd {(noise * 0.2).toFixed(3)}, <M tex="\kappa_2(X_c)" /> = {fmtSci(r.kX, 2)} and <M tex="\kappa_2(S)" /> = {fmtSci(r.kS, 2)}; their ratio to the predicted square is {Number.isFinite(r.ratio) ? r.ratio.toFixed(3) : 'undefined'}. The trailing singular vector <M tex="v_r" /> identifies a combination with sd {fmtSci(r.sdComb, 2)}, and the largest VIF is {fmtSci(Math.max(...r.vifs.vif), 2)}.
            </>
          ),
          why: (
            <>
              Reducing the residual noise pushes the columns <M tex="x_3,\dots,x_p" /> into the plane of <M tex="x_1,x_2" />; <M tex="\sigma_r" /> shrinks like the residual sd (times <M tex="\sqrt{n-1}" />) while <M tex="\sigma_1" /> stays put, so <M tex="\kappa_2" /> grows as 1/noise — a straight line of slope −1 on the log–log plot, slope −2 for <M tex="S" />.
            </>
          ),
          math: (
            <>
              If <M tex="X_c = U\Sigma V^T" /> then <M tex="S = V\frac{\Sigma^2}{n-1}V^T" />, so <M tex="\lambda_j = \sigma_j^2/(n-1)" /> and <M tex="\kappa_2(S) = \lambda_1/\lambda_p = \kappa_2(X_c)^2" />. A backward-stable eigen-solver perturbs each eigenvalue by about <M tex="\varepsilon\lambda_1" />, hence the relative error of <M tex="\lambda_p" /> is <M tex="\varepsilon\kappa_2(S)" /> while the SVD gives <M tex="\sigma_p" /> with relative error <M tex="\varepsilon\kappa_2(X_c)" />: about {digitsLost(r.kS)} versus {digitsLost(r.kX)} decimal digits lost here.
            </>
          ),
          stats: (
            <>
              Multicollinearity does not bias PCA, but it makes the trailing loadings — and anything based on <M tex="S^{-1}" /> such as regression coefficients or discriminant weights — extremely sensitive to the sample: their sampling variance is inflated by the VIF, i.e. by a factor {fmtSci(Math.max(...r.vifs.vif), 1)} for the worst variable. The near-constant combination <M tex="X_c v_r" /> is the statistical content of the warning: one variable is redundant given the others.
            </>
          ),
          careful: (
            <>
              A large <M tex="\kappa_2" /> can also be produced by mere differences in units (see section 6); compare after standardising before calling variables collinear. Near collinearity, sign and even order of the trailing components are arbitrary. Remedies: drop or merge redundant variables, ridge/shrinkage <M tex="S + \gamma I" />, or principal-component regression on the leading <M tex="k" /> directions.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 3. Eigenvalue multiplicity
// ---------------------------------------------------------------------------

const EQ_P = 5;
const EQ_N = 200;
const G2_SEEDS = Array.from({ length: 16 }, (_, i) => i + 1);

function MultiplicitySection() {
  const [rho, setRho] = useState(0.6);
  const [seedA, setSeedA] = useState(3);
  const [seedB, setSeedB] = useState(4);
  const [rho2, setRho2] = useState(0.1);

  const eq = useMemo(() => {
    const dsA = generateDataset('correlated', { ...defaultParams, n: EQ_N, p: EQ_P, correlation: rho, variance: 1, noise: 0, seed: seedA });
    const dsB = generateDataset('correlated', { ...defaultParams, n: EQ_N, p: EQ_P, correlation: rho, variance: 1, noise: 0, seed: seedB });
    const pA = pca(dsA.X, 'center');
    const pB = pca(dsB.X, 'center');
    const angles = Array.from({ length: EQ_P }, (_, j) => angleBetweenDeg(column(pA.V, j), column(pB.V, j)));
    const theory = [1 + (EQ_P - 1) * rho, ...new Array(EQ_P - 1).fill(1 - rho)];
    const eigDiff = Array.from({ length: EQ_P }, (_, j) => Math.abs(pA.eigenvalues[j] - pB.eigenvalues[j]));
    const ones = new Array(EQ_P).fill(1 / Math.sqrt(EQ_P));
    const angleToOnes = [angleBetweenDeg(column(pA.V, 0), ones), angleBetweenDeg(column(pB.V, 0), ones)];
    return { pA, pB, angles, theory, eigDiff, angleToOnes, gap: theory[0] - theory[1] };
  }, [rho, seedA, seedB]);

  const g2 = useMemo(() => {
    const rows = G2_SEEDS.map((seed) => {
      const ds = generateDataset('gaussian2d', { ...defaultParams, n: 150, correlation: rho2, variance: 1, noise: 0, seed });
      const res = pca(ds.X, 'center');
      return { seed, angle: axialAngleDeg(column(res.V, 0)), l1: res.eigenvalues[0], l2: res.eigenvalues[1] };
    });
    const popAngle = rho2 > 0 ? 45 : rho2 < 0 ? -45 : NaN;
    const popGap = 2 * Math.abs(rho2);
    const meanGap = rows.reduce((a, b) => a + (b.l1 - b.l2), 0) / rows.length;
    const dev = Number.isFinite(popAngle) ? rows.map((r) => axialDiffDeg(r.angle, popAngle)) : rows.map((r) => r.angle);
    const rms = Math.sqrt(dev.reduce((a, d) => a + d * d, 0) / dev.length);
    const min = Math.min(...rows.map((r) => r.angle));
    const max = Math.max(...rows.map((r) => r.angle));
    return { rows, popAngle, popGap, meanGap, rms, min, max };
  }, [rho2]);

  const eigTraces: Data[] = [
    { type: 'bar', x: eq.theory.map((_, j) => j + 1), y: eq.pA.eigenvalues, name: `sample λⱼ, seed ${seedA}`, marker: { color: accent } },
    { type: 'bar', x: eq.theory.map((_, j) => j + 1), y: eq.pB.eigenvalues, name: `sample λⱼ, seed ${seedB}`, marker: { color: accent2 } },
    { type: 'bar', x: eq.theory.map((_, j) => j + 1), y: eq.theory, name: 'population λⱼ', marker: { color: neutralMark } },
  ];
  const angleTraces: Data[] = [{ type: 'bar', x: eq.angles.map((_, j) => j + 1), y: eq.angles, name: 'angle between v_j(A) and v_j(B)', marker: { color: methodColor.PCA }, hovertemplate: 'j=%{x}: %{y:.1f}°<extra></extra>' }];
  const g2Traces: Data[] = [
    { type: 'scatter', mode: 'markers', x: g2.rows.map((r) => r.seed), y: g2.rows.map((r) => r.angle), name: 'sample PC1 angle', marker: { color: methodColor.PCA, size: 8 } },
    ...(Number.isFinite(g2.popAngle) ? [{ type: 'scatter', mode: 'lines', x: [0.5, G2_SEEDS.length + 0.5], y: [g2.popAngle, g2.popAngle], name: 'population PC1 angle', line: { color: neutralMark, dash: 'dash' } } as Data] : []),
  ];

  return (
    <Section
      id="diag-multiplicity"
      title="3 · Eigenvalue multiplicity: stable eigenvalues, unstable eigenvectors"
      subtitle={
        <>
          When eigenvalues coincide, the eigenvectors are only defined up to a rotation inside the eigenspace. Two independent samples make this visible.
        </>
      }
    >
      <h3>
        Equicorrelation: <M tex="\lambda_1 = 1 + (p-1)\rho" />, <M tex="\lambda_2 = \dots = \lambda_p = 1-\rho" />
      </h3>
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<>common correlation <M tex="\rho" /></>} value={rho} min={0.05} max={0.9} step={0.05} onChange={setRho} />
          <Slider label="Seed A" value={seedA} min={1} max={99} step={1} onChange={setSeedA} />
          <Slider label="Seed B" value={seedB} min={1} max={99} step={1} onChange={setSeedB} />
          <div className="small secondary">
            Local dataset <b>correlated</b> with <M tex={`n = ${EQ_N},\\ p = ${EQ_P}`} />, unit variances, no extra noise. Population eigenvalues: {eq.theory[0].toFixed(2)} and {eq.theory[1].toFixed(2)} (multiplicity {EQ_P - 1}).
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label={<>angle <M tex="v_1(A), v_1(B)" /></>} value={fmtDeg(eq.angles[0])} note="leading eigenvector: stable" />
            <StatTile label={<>angles <M tex="v_j(A), v_j(B)" />, <M tex="j\ge 2" /></>} value={`${fmtDeg(Math.min(...eq.angles.slice(1)), 0)} – ${fmtDeg(Math.max(...eq.angles.slice(1)), 0)}`} note="trailing eigenvectors: arbitrary" />
            <StatTile label={<>max <M tex="|\lambda_j(A) - \lambda_j(B)|" /></>} value={fmt(Math.max(...eq.eigDiff), 3)} note="eigenvalues: stable" />
            <StatTile label={<>angle of <M tex="v_1" /> to <M tex="(1,\dots,1)/\sqrt p" /></>} value={`${fmtDeg(eq.angleToOnes[0])} / ${fmtDeg(eq.angleToOnes[1])}`} note="seed A / seed B" />
          </div>
        </div>
        <div className="grid c2">
          <Plot title="Eigenvalues of S: two seeds and the population" data={eigTraces} layout={{ showlegend: true, barmode: 'group', xaxis: { title: { text: 'j' }, tickvals: eq.theory.map((_, j) => j + 1) }, yaxis: { title: { text: 'λⱼ' } } }} height={300} />
          <Plot title="Angle between corresponding eigenvectors of the two samples" data={angleTraces} layout={{ showlegend: true, xaxis: { title: { text: 'j' }, tickvals: eq.theory.map((_, j) => j + 1) }, yaxis: { title: { text: 'degrees' }, range: [0, 95] } }} height={300} />
        </div>
      </div>
      <Callout kind="theorem" title="What is and is not identifiable">
        The eigenspace of a repeated eigenvalue is unique; a basis inside it is not. Here the trailing eigenspace is the orthogonal complement of <M tex="v_1" />, so the two samples' trailing <em>subspaces</em> differ only by the angle between their leading vectors, {fmtDeg(eq.angles[0])} — yet the individual trailing vectors differ by up to {fmtDeg(Math.max(...eq.angles.slice(1)), 0)}. Reporting or interpreting <M tex="v_2,\dots,v_p" /> separately is meaningless under (near-)multiplicity; report the subspace, or the projector <M tex="I - v_1v_1^T" />.
      </Callout>

      <div className="divider" />
      <h3>
        Two dimensions with <M tex="\rho \to 0" />: <M tex="\lambda_1 \approx \lambda_2" /> and PC1 points anywhere
      </h3>
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<>correlation <M tex="\rho" /></>} value={rho2} min={0} max={0.9} step={0.02} onChange={setRho2} />
          <div className="small secondary">
            Local dataset <b>gaussian2d</b>, <M tex="n = 150" />, unit variances, one PCA per seed for seeds 1–{G2_SEEDS.length}. Population: <M tex="\lambda_1 - \lambda_2 = 2\rho" /> = {g2.popGap.toFixed(2)}, PC1 at {Number.isFinite(g2.popAngle) ? `${g2.popAngle}°` : 'any angle (isotropic)'}.
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label="range of sample PC1 angles" value={`${fmtDeg(g2.min, 0)} … ${fmtDeg(g2.max, 0)}`} note={`${G2_SEEDS.length} seeds`} />
            <StatTile label={Number.isFinite(g2.popAngle) ? 'RMS deviation from 45°' : 'RMS angle'} value={fmtDeg(g2.rms)} />
            <StatTile label={<>mean sample gap <M tex="\lambda_1-\lambda_2" /></>} value={g2.meanGap.toFixed(3)} note={`population ${g2.popGap.toFixed(2)}`} />
            <StatTile label={<>sampling sd of <M tex="s_{12}" /> (≈)</>} value={Math.sqrt((1 + rho2 * rho2) / 150).toFixed(3)} note="√((1+ρ²)/n) for unit variances" />
          </div>
        </div>
        <Plot title="PC1 direction against the random seed" data={g2Traces} layout={{ showlegend: true, xaxis: { title: { text: 'seed' }, tickvals: G2_SEEDS }, yaxis: { title: { text: 'PC1 angle (degrees, axial)' }, range: [-95, 95] } }} height={320} />
      </div>
      <Callout kind="theorem" title="Davis–Kahan bound">
        <MBlock tex="\sin\angle(\hat v_1, v_1) \;\le\; \frac{2\,\|S - \Sigma\|_2}{\lambda_1 - \lambda_2}" />
        (Yu, Wang & Samworth's form of the Davis–Kahan <M tex="\sin\Theta" /> theorem). The sampling error <M tex="\|S-\Sigma\|_2" /> is of order <M tex="1/\sqrt n" /> whatever <M tex="\rho" /> is; only the eigengap in the denominator changes. With the current gap {g2.popGap.toFixed(2)} the bound {g2.popGap > 0 ? `allows sin θ up to about ${Math.min(1, (2 * Math.sqrt((1 + rho2 * rho2) / 150) * 1.4) / g2.popGap).toFixed(2)} for a typical ‖S − Σ‖₂` : 'is vacuous: the direction is unidentifiable'}.
      </Callout>

      <Interpretation
        items={{
          seeing: (
            <>
              Under equicorrelation <M tex="\rho" /> = {rho.toFixed(2)} the two samples agree on the eigenvalues to within {fmt(Math.max(...eq.eigDiff), 3)} and on <M tex="v_1" /> to within {fmtDeg(eq.angles[0])}, but their trailing eigenvectors differ by {fmtDeg(Math.min(...eq.angles.slice(1)), 0)} to {fmtDeg(Math.max(...eq.angles.slice(1)), 0)}. In two dimensions with <M tex="\rho" /> = {rho2.toFixed(2)} the PC1 angle wanders over {fmtDeg(g2.min, 0)} … {fmtDeg(g2.max, 0)} across seeds.
            </>
          ),
          why: (
            <>
              The trailing population eigenvalue <M tex="1-\rho" /> has multiplicity {EQ_P - 1}: any orthonormal basis of the {EQ_P - 1}-dimensional complement of <M tex="(1,\dots,1)" /> is a valid set of eigenvectors, and sampling noise picks one at random. With <M tex="\rho \to 0" /> the 2-D covariance approaches <M tex="I" />, whose every direction is an eigenvector.
            </>
          ),
          math: (
            <>
              Eigenvalues are Lipschitz in the matrix (Weyl: <M tex="|\hat\lambda_j - \lambda_j| \le \|S-\Sigma\|_2" />), eigenvectors are not: their perturbation is governed by <M tex="\|S-\Sigma\|_2/\text{gap}" />. The present gaps are {eq.gap.toFixed(2)} (leading, equicorrelation), exactly 0 (trailing, equicorrelation) and {g2.popGap.toFixed(2)} (2-D).
            </>
          ),
          stats: (
            <>
              A scree plot with a plateau — eigenvalues within sampling error of each other — says the corresponding components are not separately interpretable. Retain or discard them as a block, and quote the variance explained by the block rather than by each member. Sample eigenvalues are also biased: the largest is inflated and the smallest deflated (here the sample <M tex="\lambda_1" /> is {eq.pA.eigenvalues[0].toFixed(3)} against a population {eq.theory[0].toFixed(3)}).
            </>
          ),
          careful: (
            <>
              Sign indeterminacy is separate from this: the library flips signs so the largest loading is positive, which is why the angles above use <M tex="|v_j^Tv_j'|" />. Bootstrapping the loadings without aligning eigenvectors (Procrustes) will report spurious instability even when the subspace is perfectly stable.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 4. Negative eigenvalues in classical MDS
// ---------------------------------------------------------------------------

function MDSNegativeSection() {
  const [metric, setMetric] = useState<Metric>('manhattan');
  const [seed, setSeed] = useState(11);

  const r = useMemo(() => {
    const ds = generateDataset('clusters', { ...defaultParams, n: 120, p: 4, K: 3, separation: 3, seed });
    const { Xc } = centerColumns(ds.X);
    const D = distanceMatrix(Xc, metric);
    const mds = classicalMDS(D, 2);
    const lamMin = mds.eigenvalues[mds.eigenvalues.length - 1];
    const cLingoes = lamMin < 0 ? -lamMin : 0;
    const n = ds.X.length;
    return { ds, n, D, mds, lamMin, cLingoes, sumNeg: mds.eigenvalues.filter((v) => v < 0).reduce((a, b) => a + b, 0), sumPos: mds.eigenvalues.filter((v) => v > 0).reduce((a, b) => a + b, 0) };
  }, [metric, seed]);

  const ev = r.mds.eigenvalues;
  const idx = ev.map((_, j) => j + 1);
  const specTraces: Data[] = [
    { type: 'bar', x: idx.filter((_, j) => ev[j] >= 0), y: ev.filter((v) => v >= 0), name: 'positive eigenvalues of B', marker: { color: POS_COLOR } },
    { type: 'bar', x: idx.filter((_, j) => ev[j] < 0), y: ev.filter((v) => v < 0), name: 'negative eigenvalues of B', marker: { color: NEG_COLOR } },
  ];
  const pts = r.mds.coords.map((c) => [c[0] ?? 0, c[1] ?? 0]);

  return (
    <Section
      id="diag-mds-negative"
      title="4 · Negative eigenvalues in classical MDS"
      subtitle={
        <>
          <M tex="B = -\tfrac12 J D^{(2)} J" /> is positive semi-definite if and only if <M tex="D" /> is Euclidean. Non-Euclidean metrics produce negative eigenvalues.
        </>
      }
    >
      <div className="grid side">
        <div className="controls-panel">
          <Select<Metric> label="Distance metric (local to this section)" value={metric} onChange={setMetric} options={(Object.keys(metricLabels) as Metric[]).map((m) => ({ value: m, label: metricLabels[m] }))} />
          <Slider label="Seed" value={seed} min={1} max={99} step={1} onChange={setSeed} />
          <div className="small secondary">
            Local dataset <b>clusters</b>: <M tex={`n = ${r.n},\\ p = 4,\\ K = 3`} />, centred. The full spectrum of <M tex="B" /> has <M tex="n" /> eigenvalues, one of which is exactly 0 (the constant vector <M tex="\mathbf 1" /> is in the null space of <M tex="J" />).
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label="positive / negative" value={`${r.mds.positive} / ${r.mds.negative}`} note="eigenvalues above / below tolerance" />
            <StatTile label="negative mass" value={fmtPct(r.mds.negativeMass, 2)} note="Σ|λ₋| / Σ|λ|" />
            <StatTile label="stress-1 (2-D)" value={fmtPct(r.mds.stress1, 2)} />
            <StatTile label={<M tex="\lambda_{\min}(B)" />} value={fmt(r.lamMin, 3)} note={r.lamMin < 0 ? `Lingoes constant c = ${fmt(r.cLingoes, 3)}` : 'no correction needed'} />
          </div>
        </div>
        <div className="stack">
          <Plot title={`Eigenvalue spectrum of B under ${metricLabels[metric]}`} data={specTraces} layout={{ showlegend: true, barmode: 'overlay', xaxis: { title: { text: 'index j (decreasing λⱼ)' } }, yaxis: { title: { text: 'λⱼ(B)' }, zeroline: true } }} height={300} />
        </div>
      </div>
      <div className="grid side-r" style={{ marginTop: 12 }}>
        <div>
          <ScatterSVG points={pts} labels={r.ds.y} classNames={r.ds.classNames} width={520} height={360} xLabel="MDS axis 1" yLabel="MDS axis 2" title="Two-dimensional configuration (leading positive eigenvalues)" />
          <ClassLegend classNames={r.ds.classNames ?? []} />
        </div>
        <div className="stack">
          <Callout kind="theorem" title="Schoenberg / Young–Householder">
            A dissimilarity matrix <M tex="D" /> with zero diagonal is embeddable in some Euclidean space iff <M tex="B = -\tfrac12 J D^{(2)} J \succeq 0" />; the smallest embedding dimension is <M tex="\operatorname{rank}(B)" />. Euclidean input on centred data gives <M tex="B = X_cX_c^T" />, hence <M tex="\lambda_j(B) = \sigma_j^2" /> and no negative values.
          </Callout>
          <Callout kind="info" title="Remedies">
            <ul>
              <li>
                Drop the negative eigenvalues (what classical MDS silently does): valid when the negative mass is small; here it is {fmtPct(r.mds.negativeMass, 2)}.
              </li>
              <li>
                Additive constant. Lingoes: replace <M tex="d_{ij}^2" /> by <M tex="d_{ij}^2 + 2c" /> (<M tex="i \ne j" />), which turns <M tex="B" /> into <M tex="B + cJ" /> and shifts every non-trivial eigenvalue by <M tex="c" />; <M tex="c = -\lambda_{\min}" /> = {fmt(r.cLingoes, 3)} makes all of them non-negative. Cailliez's constant is added to <M tex="d_{ij}" /> itself and requires a <M tex="2n\times 2n" /> eigenproblem.
              </li>
              <li>Non-metric MDS optimises stress over a monotone transformation of the dissimilarities and never forms <M tex="B" />.</li>
            </ul>
          </Callout>
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              With {metricLabels[metric]} distances the doubly centred matrix has {r.mds.positive} positive and {r.mds.negative} negative eigenvalues; the negative mass is {fmtPct(r.mds.negativeMass, 2)} and the 2-D stress-1 is {fmtPct(r.mds.stress1, 2)}. {metric === 'euclidean' ? 'Euclidean input gives no negative eigenvalues beyond rounding, as the theorem predicts.' : 'The spectrum crosses zero: this dissimilarity matrix cannot be realised exactly by points in any Euclidean space.'}
            </>
          ),
          why: (
            <>
              The <M tex="L_1" />, <M tex="L_\infty" /> and squared-Euclidean metrics satisfy the triangle inequality (or not at all, for squared distances) but not the Euclidean four-point conditions; doubly centring their squares produces a matrix that is not a Gram matrix <M tex="YY^T" /> of any configuration <M tex="Y" />.
            </>
          ),
          math: (
            <>
              Write <M tex="B = V\Lambda V^T" />. The Torgerson coordinates <M tex="Y_k = V_k\Lambda_k^{1/2}" /> exist only for <M tex="\lambda_j > 0" />; the strain <M tex="\|B - YY^T\|_F" /> is minimised by keeping the <M tex="k" /> largest positive eigenvalues, and the discarded negative part contributes <M tex="\sum_{\lambda_j<0}\lambda_j^2" /> = {fmt(ev.filter((v) => v < 0).reduce((a, b) => a + b * b, 0), 3)} to the squared strain that no Euclidean configuration can remove.
            </>
          ),
          stats: (
            <>
              A substantial negative mass means the chosen dissimilarity is not compatible with the Euclidean picture the map suggests; distances read off the plot are systematically distorted. For clustering-type questions this is often harmless (the cluster structure is still visible), but for quantitative claims about distances use stress or the Shepard diagram, or switch to non-metric MDS.
            </>
          ),
          careful: (
            <>
              "Variance explained" in classical MDS is computed relative to the positive eigenvalues only (here <M tex="\sum\lambda_+" /> = {fmt(r.sumPos, 1)}, <M tex="\sum\lambda_-" /> = {fmt(r.sumNeg, 1)}), so it overstates fit when negatives are large. An additive constant makes the input Euclidean but also changes the dissimilarities being represented.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 5. Floating-point precision
// ---------------------------------------------------------------------------

function PrecisionSection() {
  const [k, setK] = useState(8);
  const [e, setE] = useState(-9);

  const varDemo = useMemo(() => {
    const rng = makeRNG(2024);
    const n = 200;
    const z = Array.from({ length: n }, () => rng.normal());
    const offset = Math.pow(10, k);
    const x = z.map((v) => offset + v);
    const ref = twoPassVariance(z);
    const naive = naiveVariance(x);
    const twoPass = twoPassVariance(x);
    const welford = welfordVariance(x);
    const meanSq = x.reduce((a, b) => a + b * b, 0) / n;
    return { n, offset, naive, twoPass, welford, ref, meanSq, ulp: Math.pow(2, Math.floor(Math.log2(meanSq)) - 52) };
  }, [k]);

  const lae = useMemo(() => {
    const delta = Math.pow(10, e);
    const X = laeuchli(delta);
    const dec = svd(X);
    const G = gram(X);
    const eig = symmetricEigen(G);
    const sTrue = [Math.sqrt(2 + delta * delta), delta];
    const sGram = eig.values.map((l) => (l > 0 ? Math.sqrt(l) : l === 0 ? 0 : NaN));
    const kappa = sTrue[0] / sTrue[1];
    const storedDiag = 1 + delta * delta;
    const es = Array.from({ length: 33 }, (_, i) => -2 - i * 0.25);
    const curve = es.map((ee) => {
      const d = Math.pow(10, ee);
      const Xd = laeuchli(d);
      const s2svd = svd(Xd).s[1];
      const l2 = symmetricEigen(gram(Xd)).values[1];
      const s2g = l2 > 0 ? Math.sqrt(l2) : 0;
      return { delta: d, errSvd: Math.abs(s2svd - d) / d, errGram: Math.abs(s2g - d) / d };
    });
    return { delta, X, dec, G, eig, sTrue, sGram, kappa, storedDiag, curve };
  }, [e]);

  const errTraces: Data[] = [
    { type: 'scatter', mode: 'lines+markers', x: lae.curve.map((c) => c.delta), y: lae.curve.map((c) => Math.max(c.errGram, 1e-17)), name: 'relative error of σ₂ via eigen(XᵀX)', line: { color: methodColor.PCA }, marker: { size: 5 } },
    { type: 'scatter', mode: 'lines+markers', x: lae.curve.map((c) => c.delta), y: lae.curve.map((c) => Math.max(c.errSvd, 1e-17)), name: 'relative error of σ₂ via SVD of X', line: { color: methodColor.SVD }, marker: { size: 5 } },
    { type: 'scatter', mode: 'lines', x: [1e-10, 1e-2], y: [EPS, EPS], name: 'ε machine', line: { color: accent2, dash: 'dot', width: 1.5 } },
    { type: 'scatter', mode: 'lines', x: [lae.delta, lae.delta], y: [1e-17, 10], name: 'current δ', line: { color: ink.muted, width: 1 } },
  ];

  const naiveErr = relErr(varDemo.naive, varDemo.ref);
  const twoErr = relErr(varDemo.twoPass, varDemo.ref);

  return (
    <Section
      id="diag-precision"
      title="5 · Floating-point precision: cancellation and squaring"
      subtitle={
        <>
          Doubles carry about 16 significant digits (<M tex="\varepsilon \approx 2.2\times 10^{-16}" />). Two live experiments show how a formula, not the data, throws them away.
        </>
      }
    >
      <h3>Experiment A — variance of shifted data</h3>
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<>offset <M tex="10^k" /> added to N(0,1) data</>} value={k} min={0} max={9} step={1} onChange={setK} format={(v) => `k = ${v}`} />
          <div className="small secondary">
            <M tex={`x_i = 10^{${k}} + z_i,\\ z_i \\sim N(0,1),\\ n = ${varDemo.n}`} /> (fixed seed). In exact arithmetic <M tex="\operatorname{var}(x) = \operatorname{var}(z)" />, so the sample variance of <M tex="z" /> serves as the reference.
          </div>
          <MBlock tex="\text{naive: } \frac{n}{n-1}\Big(\frac1n\sum x_i^2 - \bar x^2\Big),\qquad \text{two-pass: } \frac{1}{n-1}\sum (x_i-\bar x)^2" />
        </div>
        <div className="stack">
          <table className="summary-table">
            <thead>
              <tr>
                <th>formula</th>
                <th>result</th>
                <th>relative error</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>reference (two-pass on unshifted z)</td>
                <td className="mono">{varDemo.ref.toPrecision(12)}</td>
                <td className="mono">—</td>
              </tr>
              <tr>
                <td>naive one-pass</td>
                <td className="mono" style={{ color: naiveErr > 1e-6 ? 'var(--danger)' : undefined }}>
                  {Number.isFinite(varDemo.naive) ? varDemo.naive.toPrecision(12) : String(varDemo.naive)}
                </td>
                <td className="mono">{fmtSci(naiveErr, 2)}</td>
              </tr>
              <tr>
                <td>two-pass</td>
                <td className="mono">{varDemo.twoPass.toPrecision(12)}</td>
                <td className="mono">{fmtSci(twoErr, 2)}</td>
              </tr>
              <tr>
                <td>Welford (one pass, stable)</td>
                <td className="mono">{varDemo.welford.toPrecision(12)}</td>
                <td className="mono">{fmtSci(relErr(varDemo.welford, varDemo.ref), 2)}</td>
              </tr>
            </tbody>
          </table>
          <div className="small secondary">
            The naive formula subtracts two numbers of size <M tex={`\\bar x^2 \\approx ${texSci(varDemo.meanSq, 2)}`} /> whose spacing in double precision is <M tex={texSci(varDemo.ulp, 1)} />; the difference it is trying to resolve is ≈ 1. {naiveErr > 1 ? 'The result is garbage' : naiveErr > 1e-6 ? 'Digits are visibly lost' : 'The cancellation is still benign'}{varDemo.naive < 0 ? ' — and negative, which no variance can be.' : '.'}
          </div>
        </div>
      </div>

      <div className="divider" />
      <h3>
        Experiment B — the Läuchli matrix: <M tex="\sigma_2" /> survives the SVD but not <M tex="X^TX" />
      </h3>
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<M tex="\log_{10}\delta" />} value={e} min={-10} max={-2} step={0.5} onChange={setE} format={(v) => `δ = 10^${v}`} />
          <MBlock tex={`X = \\begin{bmatrix} 1 & 1 \\\\ \\delta & 0 \\\\ 0 & \\delta \\end{bmatrix},\\quad X^TX = \\begin{bmatrix} 1+\\delta^2 & 1 \\\\ 1 & 1+\\delta^2 \\end{bmatrix},\\quad \\sigma_1 = \\sqrt{2+\\delta^2},\\ \\sigma_2 = \\delta`} />
          <div className="small secondary">
            Both columns have norm <M tex="\sqrt{1+\delta^2}\approx 1" />; the matrix is well scaled. Stored diagonal entry: <M tex={`\\mathrm{fl}(1+\\delta^2) - 1 = ${texSci(lae.storedDiag - 1, 3)}`} /> {lae.storedDiag === 1 ? '— the information about δ has been erased before any eigen-solver runs.' : '(still representable).'}
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label={<M tex="\kappa_2(X)" />} value={<M tex={texSci(lae.kappa)} />} note={`ε·κ ≈ ${fmtSci(EPS * lae.kappa, 1)}`} />
            <StatTile label={<M tex="\kappa_2(X^TX) = \kappa_2(X)^2" />} value={<M tex={texSci(lae.kappa * lae.kappa)} />} note={`ε·κ² ≈ ${fmtSci(EPS * lae.kappa * lae.kappa, 1)}`} />
          </div>
        </div>
        <div className="stack">
          <table className="summary-table">
            <thead>
              <tr>
                <th>quantity</th>
                <th>exact</th>
                <th>one-sided Jacobi SVD of X</th>
                <th>
                  <M tex="\sqrt{\lambda_j(X^TX)}" />
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <M tex="\sigma_1" />
                </td>
                <td className="mono">{lae.sTrue[0].toPrecision(10)}</td>
                <td className="mono">{lae.dec.s[0].toPrecision(10)}</td>
                <td className="mono">{Number.isFinite(lae.sGram[0]) ? lae.sGram[0].toPrecision(10) : 'undefined'}</td>
              </tr>
              <tr>
                <td>
                  <M tex="\sigma_2" />
                </td>
                <td className="mono">{fmtSci(lae.sTrue[1], 6)}</td>
                <td className="mono">{fmtSci(lae.dec.s[1], 6)}</td>
                <td className="mono" style={{ color: relErr(lae.sGram[1], lae.sTrue[1]) > 1e-6 ? 'var(--danger)' : undefined }}>
                  {Number.isFinite(lae.sGram[1]) ? fmtSci(lae.sGram[1], 6) : `undefined (λ₂ = ${fmtSci(lae.eig.values[1], 2)} < 0)`}
                </td>
              </tr>
              <tr>
                <td>
                  relative error of <M tex="\sigma_2" />
                </td>
                <td className="mono">—</td>
                <td className="mono">{fmtSci(relErr(lae.dec.s[1], lae.sTrue[1]), 2)}</td>
                <td className="mono">{fmtSci(relErr(Number.isFinite(lae.sGram[1]) ? lae.sGram[1] : 0, lae.sTrue[1]), 2)}</td>
              </tr>
            </tbody>
          </table>
          <Plot title="Relative error of σ₂ against δ (log–log)" data={errTraces} layout={{ showlegend: true, xaxis: { type: 'log', title: { text: 'δ' }, exponentformat: 'power' }, yaxis: { type: 'log', title: { text: 'relative error (floored at 1e−17)' }, exponentformat: 'power' } }} height={300} />
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              With <M tex={`k = ${k}`} /> the naive variance is {Number.isFinite(varDemo.naive) ? varDemo.naive.toPrecision(6) : 'not finite'} against a reference of {varDemo.ref.toPrecision(6)} (relative error {fmtSci(naiveErr, 1)}), while the two-pass and Welford formulas agree to {fmtSci(Math.max(twoErr, 1e-17), 1)}. With <M tex={`\\delta = 10^{${e}}`} /> the SVD returns <M tex="\sigma_2" /> with relative error {fmtSci(relErr(lae.dec.s[1], lae.sTrue[1]), 1)}, whereas the eigenvalue route gives {Number.isFinite(lae.sGram[1]) ? fmtSci(lae.sGram[1], 3) : 'an undefined value'} instead of {fmtSci(lae.sTrue[1], 3)}.
            </>
          ),
          why: (
            <>
              Both failures are catastrophic cancellation: a small quantity is computed as the difference of two large, individually rounded numbers. In A the large numbers are <M tex="\sum x_i^2/n" /> and <M tex="\bar x^2" />; in B the entry <M tex="1+\delta^2" /> is rounded to <M tex="1" /> as soon as <M tex="\delta^2 < \varepsilon/2" />, i.e. <M tex="\delta \lesssim 1.05\times 10^{-8}" />, which makes <M tex="X^TX" /> exactly singular in floating point.
            </>
          ),
          math: (
            <>
              Squaring doubles the exponent: <M tex="\kappa_2(X^TX) = \kappa_2(X)^2" />, so an eigen-solver that is backward stable on <M tex="X^TX" /> returns <M tex="\lambda_2" /> with absolute error <M tex="\approx\varepsilon\lambda_1" />, i.e. relative error <M tex="\varepsilon\kappa^2" /> = {fmtSci(EPS * lae.kappa * lae.kappa, 1)}, while the one-sided Jacobi SVD works on the columns of <M tex="X" /> and delivers relative error <M tex="\approx\varepsilon\kappa" /> = {fmtSci(EPS * lae.kappa, 1)} or better. When <M tex="\varepsilon\kappa^2 \ge 1" /> no digit of <M tex="\sigma_2" /> survives the Gram route.
            </>
          ),
          stats: (
            <>
              The statistical objects at risk are exactly the small ones: residual variances, trailing eigenvalues, the within-class scatter along nearly constant directions, and Mahalanobis distances (which divide by them). A dataset measured on a large offset scale (years, instrument readings around a calibration constant) or with nearly collinear columns is not "ill-posed"; it merely requires the right formula.
            </>
          ),
          careful: (
            <>
              Centre before you square (the two-pass formula is the matrix analogue of forming <M tex="X_c" /> before <M tex="X_c^TX_c" />), prefer a streaming update such as Welford's for one-pass computation, and prefer the SVD of <M tex="X_c" /> to the eigendecomposition of <M tex="S" /> whenever <M tex="\kappa_2(X_c) \gtrsim 10^{7}" />, since then <M tex="\varepsilon\kappa^2 \gtrsim 10^{-2}" />.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 6. Metric and standardisation
// ---------------------------------------------------------------------------

function MetricScalingSection() {
  const [scaleExp, setScaleExp] = useState(2);
  const [metric, setMetric] = useState<Metric>('euclidean');
  const [seed, setSeed] = useState(5);

  const r = useMemo(() => {
    const factor = Math.round(Math.pow(10, scaleExp));
    const ds = generateDataset('scales', { ...defaultParams, n: 120, p: 3, scaleFactor: factor, seed });
    const y = ds.y ?? [];
    const pC = pca(ds.X, 'center');
    const pS = pca(ds.X, 'standardize');
    const Dc = distanceMatrix(pC.Xc, metric);
    const Ds = distanceMatrix(pS.Xc, metric);
    const mC = classicalMDS(Dc, 2);
    const mS = classicalMDS(Ds, 2);
    const trS = trace(pC.S);
    const share1 = pC.S[0][0] / trS;
    const load1C = Math.abs(pC.V[0][0]);
    const load1S = Math.abs(pS.V[0][0]);
    let Jraw = NaN;
    let Jstd = NaN;
    try {
      Jraw = lda(pC.Xc, y).eigenvalues[0] ?? NaN;
      Jstd = lda(pS.Xc, y).eigenvalues[0] ?? NaN;
    } catch {
      /* singular within-class scatter: leave undefined */
    }
    // between-class separation visible along MDS axis 1: standardised mean difference of coordinate 1
    const sepAlong = (coords: number[][]) => {
      const a = coords.filter((_, i) => y[i] === 0).map((c) => c[0] ?? 0);
      const b = coords.filter((_, i) => y[i] === 1).map((c) => c[0] ?? 0);
      const ma = a.reduce((s, v) => s + v, 0) / a.length;
      const mb = b.reduce((s, v) => s + v, 0) / b.length;
      const sd = Math.sqrt((a.reduce((s, v) => s + (v - ma) ** 2, 0) + b.reduce((s, v) => s + (v - mb) ** 2, 0)) / (a.length + b.length - 2));
      return sd > 0 ? Math.abs(ma - mb) / sd : 0;
    };
    return { factor, ds, pC, pS, mC, mS, share1, load1C, load1S, Jraw, Jstd, sepC: sepAlong(mC.coords), sepS: sepAlong(mS.coords) };
  }, [scaleExp, metric, seed]);

  const ptsC = r.mC.coords.map((c) => [c[0] ?? 0, c[1] ?? 0]);
  const ptsS = r.mS.coords.map((c) => [c[0] ?? 0, c[1] ?? 0]);

  return (
    <Section
      id="diag-metric-scaling"
      title="6 · Choice of distance metric and standardisation"
      subtitle={
        <>
          Squared Euclidean distances decompose over variables exactly like the total variance, so a variable with large units dominates MDS for the same reason it dominates covariance PCA.
        </>
      }
    >
      <div className="grid side">
        <div className="controls-panel">
          <Slider label={<>scale factor applied to <M tex="x_1" /></>} value={scaleExp} min={0} max={3} step={0.25} onChange={setScaleExp} format={(v) => `× ${Math.round(Math.pow(10, v))}`} />
          <Select<Metric> label="Distance metric (local)" value={metric} onChange={setMetric} options={(Object.keys(metricLabels) as Metric[]).map((m) => ({ value: m, label: metricLabels[m] }))} />
          <Slider label="Seed" value={seed} min={1} max={99} step={1} onChange={setSeed} />
          <div className="small secondary">
            Local dataset <b>scales</b>: <M tex="n = 120,\ p = 3" />, two classes that differ along <M tex="x_2" /> only; <M tex="x_1" /> is multiplied by {r.factor}.
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label={<M tex="s_{11}/\operatorname{tr}S" />} value={fmtPct(r.share1)} note="share of total variance and of Σ d²ᵢⱼ due to x₁" />
            <StatTile label="PC1 % (covariance / correlation)" value={`${fmtPct(r.pC.explained[0], 0)} / ${fmtPct(r.pS.explained[0], 0)}`} />
            <StatTile label={<>|loading of <M tex="x_1" /> on PC1|</>} value={`${r.load1C.toFixed(3)} / ${r.load1S.toFixed(3)}`} note="covariance / correlation" />
            <StatTile label="Fisher J (raw / standardised)" value={`${Number.isFinite(r.Jraw) ? r.Jraw.toFixed(3) : '—'} / ${Number.isFinite(r.Jstd) ? r.Jstd.toFixed(3) : '—'}`} note="LDA is invariant to rescaling" />
          </div>
        </div>
        <div className="grid c2">
          <div>
            <ScatterSVG points={ptsC} labels={r.ds.y} classNames={r.ds.classNames} width={420} height={330} xLabel="MDS axis 1" yLabel="MDS axis 2" title={`Centred only (${metricLabels[metric]})`} caption={`stress-1 ${fmtPct(r.mC.stress1, 1)} · class separation along axis 1: ${r.sepC.toFixed(2)} sd`} />
          </div>
          <div>
            <ScatterSVG points={ptsS} labels={r.ds.y} classNames={r.ds.classNames} width={420} height={330} xLabel="MDS axis 1" yLabel="MDS axis 2" title={`Standardised (${metricLabels[metric]})`} caption={`stress-1 ${fmtPct(r.mS.stress1, 1)} · class separation along axis 1: ${r.sepS.toFixed(2)} sd`} />
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <ClassLegend classNames={r.ds.classNames ?? []} />
          </div>
        </div>
      </div>
      <MBlock tex="\sum_{i<j} d_{ij}^2 = \sum_{i<j}\sum_{k=1}^p (x_{ik}-x_{jk})^2 = n\sum_{k=1}^p \sum_i (x_{ik}-\bar x_k)^2 = n(n-1)\operatorname{tr}S" />
      <Interpretation
        items={{
          seeing: (
            <>
              With <M tex="x_1" /> scaled by {r.factor}, variable 1 carries {fmtPct(r.share1)} of the total variance — and by the identity above, the same share of all squared Euclidean distances. Covariance PCA puts {fmtPct(r.pC.explained[0], 0)} of the variance on PC1 with |loading| {r.load1C.toFixed(3)} on <M tex="x_1" />; after standardising PC1 explains {fmtPct(r.pS.explained[0], 0)}. The class separation along the first MDS axis moves from {r.sepC.toFixed(2)} to {r.sepS.toFixed(2)} within-class sds.
            </>
          ),
          why: (
            <>
              Multiplying a column by <M tex="c" /> multiplies its variance and its contribution to every squared distance by <M tex="c^2" />; the other variables keep theirs. Both covariance PCA and Euclidean MDS optimise sums of squares, so both follow the loud variable — the informative direction (<M tex="x_2" />) is drowned out.
            </>
          ),
          math: (
            <>
              For centred Euclidean input <M tex="B = X_cX_c^T" /> and the MDS axes are the PCA scores up to sign: the two methods are the same eigenproblem seen from the <M tex="n\times n" /> and the <M tex="p\times p" /> side. Standardising replaces <M tex="S" /> by the correlation matrix <M tex="R" /> and the distances by <M tex="\sum_k (x_{ik}-x_{jk})^2/s_k^2" /> — a Mahalanobis distance with a diagonal metric. For a non-Euclidean metric the identity fails, but the dominance of the loud variable persists. Fisher's criterion is invariant under any invertible linear transform, which is why <M tex="J" /> is {Number.isFinite(r.Jraw) && Number.isFinite(r.Jstd) ? `${r.Jraw.toFixed(3)} both before and after` : 'unchanged by'} standardising.
            </>
          ),
          stats: (
            <>
              The choice of scale is a modelling decision, not a numerical detail: covariance PCA answers "where is the variance in the given units", correlation PCA answers "which variables move together". If the units are arbitrary (grams versus kilograms) or the variables are incommensurable, standardise; if the units are shared and meaningful, the covariance answer may be the right one.
            </>
          ),
          careful: (
            <>
              Standardising is not a free lunch: it gives a nearly constant, noisy variable the same weight as a highly informative one, and it changes the geometry the distances represent. Stress-1 comparisons across the two panels ({fmtPct(r.mC.stress1, 1)} versus {fmtPct(r.mS.stress1, 1)}) are between different distance matrices, not a measure of which one is "better".
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 7. Outliers and leverage
// ---------------------------------------------------------------------------

function OutliersSection() {
  const [mag, setMag] = useState(8);
  const [count, setCount] = useState(5);
  const [seed, setSeed] = useState(3);

  const r = useMemo(() => {
    const ds = generateDataset('outliers', { ...defaultParams, n: 100, p: 3, correlation: 0.7, outlierMagnitude: mag, outlierCount: count, seed });
    const n = ds.X.length;
    const p = ds.X[0].length;
    const m = Math.min(Math.max(0, Math.round(count)), Math.floor(n / 2));
    const full = pca(ds.X, 'center');
    const cleanX = ds.X.slice(m);
    const clean = pca(cleanX, 'center');
    const v1 = column(full.V, 0);
    const v1c = column(clean.V, 0);
    const angle = angleBetweenDeg(v1, v1c);
    const h = leverages(full.svd.U);
    const labels = ds.X.map((_, i) => (i < m ? 1 : 0));
    const cleanMeanFull = colMeans(full.Xc.slice(m));
    const rowNorms2 = full.Xc.map((row) => dot(row, row));
    const totalScatter = rowNorms2.reduce((a, b) => a + b, 0);
    const outlierScatter = rowNorms2.slice(0, m).reduce((a, b) => a + b, 0);
    const maxLev = Math.max(...h);
    const maxLevIdx = h.indexOf(maxLev);
    return { ds, n, p, m, full, clean, v1, v1c, angle, h, labels, cleanMeanFull, scatterShare: totalScatter > 0 ? outlierScatter / totalScatter : 0, maxLev, maxLevIdx, rank: full.rank };
  }, [mag, count, seed]);

  const classNames = ['regular observation', 'outlier'];
  const pts = r.full.Xc.map((row) => [row[0], row[1]]);
  const idx = r.h.map((_, i) => i + 1);
  const levTraces: Data[] = [
    { type: 'bar', x: idx.filter((_, i) => r.labels[i] === 0), y: r.h.filter((_, i) => r.labels[i] === 0), name: 'regular observation', marker: { color: classColor(0) }, hovertemplate: 'i=%{x}: h=%{y:.3f}<extra></extra>' },
    { type: 'bar', x: idx.filter((_, i) => r.labels[i] === 1), y: r.h.filter((_, i) => r.labels[i] === 1), name: 'outlier', marker: { color: classColor(1) }, hovertemplate: 'i=%{x}: h=%{y:.3f}<extra></extra>' },
    { type: 'scatter', mode: 'lines', x: [0.5, r.n + 0.5], y: [r.rank / r.n, r.rank / r.n], name: 'mean leverage r/n', line: { color: neutralMark, dash: 'dash', width: 1.2 } },
    { type: 'scatter', mode: 'lines', x: [0.5, r.n + 0.5], y: [(2 * r.rank) / r.n, (2 * r.rank) / r.n], name: 'conventional cutoff 2r/n', line: { color: accent2, dash: 'dot', width: 1.2 } },
  ];

  return (
    <Section
      id="diag-outliers"
      title="7 · Outliers, leverage and the least-squares nature of PCA"
      subtitle={
        <>
          A few extreme points can rotate <M tex="v_1" />: each observation enters <M tex="S" /> with weight proportional to its squared distance from the centre.
        </>
      }
    >
      <div className="grid side">
        <div className="controls-panel">
          <Slider label="Outlier magnitude" value={mag} min={1} max={20} step={0.5} onChange={setMag} />
          <Slider label="Number of outliers" value={count} min={0} max={20} step={1} onChange={setCount} />
          <Slider label="Seed" value={seed} min={1} max={99} step={1} onChange={setSeed} />
          <div className="small secondary">
            Local dataset <b>outliers</b>: <M tex={`n = ${r.n},\\ p = ${r.p}`} />, AR(1) correlation 0.7; the first {r.m} rows are replaced by points at ≈ ±{mag} along <M tex="(1,-1,0)" />, i.e. across the main axis of the cloud.
          </div>
          <div className="stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatTile label={<>angle between <M tex="v_1" /> with / without outliers</>} value={fmtDeg(r.angle)} note={`in ℝ^${r.p}`} />
            <StatTile label={<M tex="\lambda_1" />} value={`${r.full.eigenvalues[0].toFixed(2)} / ${r.clean.eigenvalues[0].toFixed(2)}`} note="with / without outliers" />
            <StatTile label="share of total scatter from outliers" value={fmtPct(r.scatterShare)} note={`${r.m} of ${r.n} rows`} />
            <StatTile label="largest leverage" value={r.maxLev.toFixed(3)} note={`row ${r.maxLevIdx + 1}; mean r/n = ${(r.rank / r.n).toFixed(3)}`} />
          </div>
        </div>
        <div className="stack">
          <ScatterSVG
            points={pts}
            labels={r.labels}
            classNames={classNames}
            width={640}
            height={380}
            xLabel="x₁ (centred)"
            yLabel="x₂ (centred)"
            title="First two coordinates with the PC1 axes"
            lines={[
              { angle: Math.atan2(r.v1[1], r.v1[0]), color: methodColor.PCA, width: 2 },
              { angle: Math.atan2(r.v1c[1], r.v1c[0]), through: [r.cleanMeanFull[0], r.cleanMeanFull[1]], color: neutralMark, dashed: true, width: 2 },
            ]}
            hoverInfo={(i) => `#${i + 1} ${r.labels[i] ? 'outlier' : 'regular'}\nh = ${r.h[i].toFixed(3)}`}
          />
          <div className="row">
            <ClassLegend classNames={classNames} />
            <LineLegend
              items={[
                { color: methodColor.PCA, label: <>PC1 with outliers</> },
                { color: neutralMark, dashed: true, label: <>PC1 without outliers (through the clean mean)</> },
              ]}
            />
          </div>
        </div>
      </div>
      <Plot title="Leverage hᵢ = ‖uᵢ‖² of every observation (rows of U)" data={levTraces} layout={{ showlegend: true, barmode: 'overlay', xaxis: { title: { text: 'observation i' } }, yaxis: { title: { text: 'hᵢ' }, rangemode: 'tozero' } }} height={280} />
      <div className="plot-caption">
        <M tex={`\\sum_i h_i = \\operatorname{rank}(X_c) = ${r.rank}`} />; the leverage measures how much the fitted subspace is pulled by observation <M tex="i" />. The PC1 axis in the plot is the projection of <M tex="v_1" /> onto the <M tex="(x_1,x_2)" /> plane.
      </div>
      <Callout kind="info" title="Robust remedies">
        Minimum-covariance-determinant (MCD) or M-estimates of scatter replace <M tex="S" /> before the eigendecomposition; spherical PCA projects each centred observation to the unit sphere (<M tex="x_i/\|x_i\|" />) so that no point can dominate; robust Mahalanobis distances (with a robust location and scatter) flag outliers before analysis; ROBPCA combines projection pursuit with MCD in high dimensions. For MDS, the analogous step is to down-weight large dissimilarities (e.g. weighted stress) or to use a rank-based (non-metric) criterion.
      </Callout>
      <Interpretation
        items={{
          seeing: (
            <>
              {r.m} outliers of magnitude {mag} rotate the first principal axis by {fmtDeg(r.angle)} and change <M tex="\lambda_1" /> from {r.clean.eigenvalues[0].toFixed(2)} to {r.full.eigenvalues[0].toFixed(2)}. They are {fmtPct(r.m / r.n, 0)} of the observations but carry {fmtPct(r.scatterShare)} of the total scatter <M tex="\sum_i\|x_i-\bar x\|^2" />; the largest leverage is {r.maxLev.toFixed(3)} against a mean of {(r.rank / r.n).toFixed(3)}.
            </>
          ),
          why: (
            <>
              <M tex="S = \frac1{n-1}\sum_i (x_i-\bar x)(x_i-\bar x)^T" /> is a sum of rank-one terms weighted by squared distances: a point {mag} sds away counts like {Math.round(mag * mag)} ordinary points. The outliers were placed across the correlation axis, so they add variance precisely where the clean data have least of it.
            </>
          ),
          math: (
            <>
              The influence function of the leading eigenvector at a point <M tex="x" /> is proportional to <M tex="\sum_{j\ge 2}\frac{(v_j^Tx)(v_1^Tx)}{\lambda_1-\lambda_j}v_j" />: unbounded in <M tex="\|x\|" /> and amplified by small eigengaps. The leverage <M tex="h_i = \|u_i\|^2 = x_i^T(X_c^TX_c)^{+}x_i" /> is the diagonal of the projector onto the column space of <M tex="X_c" />, so a single point can approach <M tex="h_i = 1" /> and define a component by itself.
            </>
          ),
          stats: (
            <>
              Classical PCA and classical MDS have breakdown point 0: one sufficiently extreme point moves the estimate arbitrarily far. Before interpreting components, inspect leverages and robust distances; if genuine, outliers may deserve their own component or a separate model, and if erroneous they should be corrected rather than silently absorbed into <M tex="v_1" />.
            </>
          ),
          careful: (
            <>
              Leverage is computed from the contaminated fit, so masking can occur: several outliers in the same direction share the leverage and each looks less extreme (here the {r.m} outliers average {r.m > 0 ? (r.h.slice(0, r.m).reduce((a, b) => a + b, 0) / r.m).toFixed(3) : '—'}). Deleting points by eye and refitting is not a robust procedure either — use an estimator with a positive breakdown point.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 8. Why the SVD
// ---------------------------------------------------------------------------

function WhySVDSection() {
  const a = useAnalysis();
  const r = useMemo(() => {
    const s = a.svd.s;
    const kX = kappaFromSingularValues(s, a.svd.tol);
    const kS = conditionNumber(a.pca.S);
    const minNP = Math.min(a.n, a.p);
    const lamSvd = a.pca.eigenvalues;
    const lamEig = a.pca.eigen.values;
    const maxDisc = lamSvd.length ? Math.max(...lamSvd.map((l, j) => Math.abs(l - (lamEig[j] ?? 0)))) / (lamSvd[0] || 1) : 0;
    const negEig = lamEig.filter((v) => v < 0).length;
    const flopsA = a.n * a.p * a.p + a.p ** 3;
    const flopsB = a.n * a.p * minNP;
    return { kX, kS, minNP, maxDisc, negEig, flopsA, flopsB, rank: a.svd.rank };
  }, [a]);

  const scaleLabel = a.prep.scaling === 'standardize' ? 'correlation matrix R' : a.prep.scaling === 'center' ? 'covariance matrix S' : 'second-moment matrix XᵀX/(n−1)';

  return (
    <Section
      id="diag-why-svd"
      title="8 · Why practical implementations use the SVD"
      subtitle={
        <>
          Two routes to the same principal components: eigen-decompose <M tex="S" />, or take the thin SVD of <M tex="X_c" />. Live numbers refer to the global dataset (<b>{a.dataset.name}</b>, <M tex={`n = ${a.n},\\ p = ${a.p}`} />, {scaleLabel}).
        </>
      }
    >
      <div className="stats">
        <StatTile label={<M tex="\kappa_2(X_c)" />} value={<M tex={texSci(r.kX)} />} note={`σ₁/σ_r over r = ${a.svd.s.length}`} />
        <StatTile label={<M tex="\kappa_2(S)" />} value={<M tex={texSci(r.kS)} />} note="≈ κ₂(X_c)²" />
        <StatTile label={<M tex="\min(n,p)" />} value={r.minNP} note={`rank ${r.rank}`} />
        <StatTile label={<>max <M tex="|\lambda_j^{\text{eig}} - \sigma_j^2/(n-1)|/\lambda_1" /></>} value={<M tex={texSci(r.maxDisc, 1)} />} note={r.negEig ? `${r.negEig} negative eigenvalue(s) from the eigen route` : 'both routes agree to this level'} />
      </div>
      <table className="summary-table" style={{ marginTop: 12 }}>
        <thead>
          <tr>
            <th>aspect</th>
            <th>
              Route A — form <M tex="S = X_c^TX_c/(n-1)" />, eigen-decompose
            </th>
            <th>
              Route B — thin SVD <M tex="X_c = U\Sigma V^T" />
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>operations</td>
            <td>
              <M tex="X_c^TX_c" />: <M tex="O(np^2)" />, then a <M tex="p\times p" /> eigenproblem <M tex="O(p^3)" /> — here ≈ {fmtSci(r.flopsA, 2)} basic operations
            </td>
            <td>
              <M tex="O(np\min(n,p))" /> (Golub–Kahan bidiagonalisation, or one-sided Jacobi) — here ≈ {fmtSci(r.flopsB, 2)}
            </td>
          </tr>
          <tr>
            <td>conditioning</td>
            <td>
              <M tex="\kappa_2(S) = \kappa_2(X_c)^2" /> = {fmtSci(r.kS, 2)}; relative error of the smallest eigenvalue ≈ <M tex="\varepsilon\kappa^2" /> = {fmtSci(EPS * r.kS, 1)}
            </td>
            <td>
              <M tex="\kappa_2(X_c)" /> = {fmtSci(r.kX, 2)}; relative error of the smallest singular value ≈ <M tex="\varepsilon\kappa" /> = {fmtSci(EPS * r.kX, 1)}
            </td>
          </tr>
          <tr>
            <td>
              <M tex="p > n" />
            </td>
            <td>
              <M tex="S" /> is <M tex="p\times p" /> but has rank <M tex="\le n-1" />: <M tex="p-n+1" /> zero eigenvalues are computed only to be discarded
            </td>
            <td>
              only <M tex="\min(n,p)" /> = {r.minNP} singular values exist; <M tex="U" /> is <M tex="n\times r" />, <M tex="V" /> is <M tex="p\times r" />
            </td>
          </tr>
          <tr>
            <td>rank revelation</td>
            <td>
              trailing eigenvalues are <M tex="\pm\varepsilon\lambda_1" /> noise and can be negative{r.negEig ? ` (${r.negEig} here)` : ''}; the rank decision needs a tolerance of order <M tex="\varepsilon\lambda_1" />
            </td>
            <td>
              singular values are non-negative and accurate to <M tex="\varepsilon\sigma_1" />; tolerance <M tex="\max(n,p)\varepsilon\sigma_1" /> resolves variances down to <M tex="\varepsilon^2\lambda_1" />
            </td>
          </tr>
          <tr>
            <td>memory</td>
            <td>
              <M tex="X" /> plus a dense <M tex="p\times p" /> matrix: {a.p * a.p} extra entries
            </td>
            <td>
              <M tex="U" />, <M tex="\Sigma" />, <M tex="V" />: about <M tex="(n+p)r" /> = {(a.n + a.p) * r.minNP} entries, no <M tex="p\times p" /> product
            </td>
          </tr>
          <tr>
            <td>scores and reconstruction</td>
            <td>
              <M tex="Z = X_cV" /> requires another multiplication; <M tex="\hat X_k = ZV_k^T" />
            </td>
            <td>
              <M tex="Z = U\Sigma" /> is available at once; <M tex="\hat X_k = U_k\Sigma_kV_k^T" /> is the Eckart–Young optimum
            </td>
          </tr>
          <tr>
            <td>squaring</td>
            <td>
              information below <M tex="\sqrt\varepsilon\,\sigma_1 \approx 1.5\times 10^{-8}\sigma_1" /> is destroyed when the product is formed (Läuchli, section 5)
            </td>
            <td>works on the columns of <M tex="X_c" /> directly; no quantity is squared before it is resolved</td>
          </tr>
        </tbody>
      </table>
      <Interpretation
        items={{
          seeing: (
            <>
              For the current dataset <M tex="\kappa_2(X_c)" /> = {fmtSci(r.kX, 2)} and <M tex="\kappa_2(S)" /> = {fmtSci(r.kS, 2)}; the two routes agree on the eigenvalues to a relative {fmtSci(r.maxDisc, 1)} of <M tex="\lambda_1" />
              {r.negEig ? `, but the eigen route produced ${r.negEig} negative eigenvalue(s)` : ''}. The operation counts are of the same order here ({fmtSci(r.flopsA, 1)} versus {fmtSci(r.flopsB, 1)}) because <M tex="p" /> is small; the accuracy argument does not depend on size.
            </>
          ),
          why: (
            <>
              Both routes are exact in real arithmetic (<M tex="S = V\Sigma^2V^T/(n-1)" />), so any difference is rounding. Forming <M tex="X_c^TX_c" /> squares the singular values and therefore squares the condition number before the decomposition starts, whereas the SVD algorithms apply orthogonal transformations to <M tex="X_c" /> itself, which preserve the singular values exactly.
            </>
          ),
          math: (
            <>
              Backward stability: the computed SVD is the exact SVD of <M tex="X_c + E" /> with <M tex="\|E\|_2 \le c\,\varepsilon\|X_c\|_2" />, so by Weyl's inequality <M tex="|\hat\sigma_j - \sigma_j| \le c\,\varepsilon\sigma_1" /> for every <M tex="j" />. The eigen route gives <M tex="|\hat\lambda_j - \lambda_j| \le c'\varepsilon\lambda_1" />, and since <M tex="\lambda_j = \sigma_j^2/(n-1)" /> this is an error of <M tex="c'\varepsilon\sigma_1^2/(2\sigma_j)" /> in <M tex="\sigma_j" /> — larger by the factor <M tex="\sigma_1/\sigma_j" />, which for the trailing component is <M tex="\kappa_2(X_c)" /> = {fmtSci(r.kX, 1)} here.
            </>
          ),
          stats: (
            <>
              Statistically the two routes are the same PCA; the SVD route is simply the one whose small eigenvalues, residual variances and reconstruction errors can be trusted. That matters exactly when it is interesting — near-collinear predictors, <M tex="p > n" /> genomics or spectroscopy data, and any downstream step that divides by a trailing eigenvalue (Mahalanobis distances, whitening, LDA).
            </>
          ),
          careful: (
            <>
              The SVD does not repair a badly scaled problem: if <M tex="\kappa_2(X_c)" /> is large because of units, standardise first (section 6). Randomised and truncated SVDs trade accuracy in the trailing components for speed, so they should not be used to decide numerical rank. Finally, when <M tex="n \gg p" /> and only <M tex="S" /> is available (streaming, privacy), route A with a stable eigen-solver is acceptable provided <M tex="\varepsilon\kappa_2(S) \ll 1" />, i.e. <M tex="\kappa_2(X_c) \ll 10^8" />.
            </>
          ),
        }}
      />
    </Section>
  );
}

// ---------------------------------------------------------------------------

export default function Diagnostics() {
  return (
    <div>
      <div className="topbar">
        <div className="title">
          <h2>Diagnostics &amp; numerical stability</h2>
          <div className="lede">
            Eight live experiments on what happens to SVD, PCA, MDS and LDA when the covariance is singular, badly conditioned, degenerate, non-Euclidean, numerically fragile, badly scaled or contaminated — and why the SVD is the workhorse. Sections 1–7 use their own local datasets; section 8 uses the global dataset.
          </div>
        </div>
      </div>
      <RankSection />
      <ConditioningSection />
      <MultiplicitySection />
      <MDSNegativeSection />
      <PrecisionSection />
      <MetricScalingSection />
      <OutliersSection />
      <WhySVDSection />
    </div>
  );
}
