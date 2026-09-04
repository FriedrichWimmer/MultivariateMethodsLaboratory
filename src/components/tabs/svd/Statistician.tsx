import { useMemo, useState } from 'react';
import type { Data } from 'plotly.js';
import { useAnalysis, useStore } from '../../../state/store';
import { conditionNumber, svd, gram, symmetricEigen, column, matmul, frobenius, sub, fmt, EPS, type Matrix } from '../../../lib/linalg';
import { methodColor, accent2 } from '../../../lib/theme';
import { Plot } from '../../common/Plot';
import { M, MBlock, texMatrix } from '../../common/Math';
import { Section, Card, Callout, Interpretation, StatTile, Accordion, Badge } from '../../common/Panels';
import { Slider, Button } from '../../common/Controls';
import { coefficientVarianceFactors, ridgeFilterFactors, sumInverseSquares, texNum, fmtPct } from './util';

/** Lesson 4 — why a statistician should care: nine concrete reasons, each computed on the current data. */
export function StatisticianSection({ k }: { k: number }) {
  const a = useAnalysis();
  const { navigate } = useStore();
  const [expo, setExpo] = useState(6);
  const [logLam, setLogLam] = useState(0);

  const Xc = a.pca.Xc;
  const res = a.svd;
  const s = res.s;
  const r = s.length;
  const n = a.n;
  const p = a.p;
  const kk = Math.max(1, Math.min(k, r));
  const names = a.dataset.variableNames;
  const centred = a.prep.scaling !== 'none';

  const core = useMemo(() => {
    const kappaX = conditionNumber(Xc);
    const kappaS = conditionNumber(a.pca.S);
    const ratio = Number.isFinite(kappaX) && Number.isFinite(kappaS) ? kappaS / (kappaX * kappaX) : NaN;
    const vr = column(res.V, r - 1);
    const sdLast = s[r - 1] / Math.sqrt(Math.max(n - 1, 1));
    const varSds = a.pca.S.map((row, j) => Math.sqrt(Math.max(row[j], 0)));
    const cvf = coefficientVarianceFactors(res);
    const fullRank = cvf.retained === p;
    const gramDiag = Xc[0].map((_, j) => Xc.reduce((acc, row) => acc + row[j] * row[j], 0));
    const vif = gramDiag.map((gj, j) => (fullRank ? cvf.diag[j] * gj : Infinity));
    const lambdaSVD = s.map((x) => (x * x) / Math.max(n - 1, 1));
    const eigenVals = a.pca.eigen.values;
    const m = Math.min(eigenVals.length, lambdaSVD.length);
    let eigDiff = 0;
    for (let j = 0; j < m; j++) eigDiff = Math.max(eigDiff, Math.abs(eigenVals[j] - lambdaSVD[j]));
    const Z = matmul(Xc, res.V);
    const US = res.U.map((row) => row.map((u, j) => u * s[j]));
    const scoreGap = frobenius(sub(Z, US));
    const total = s.reduce((acc, x) => acc + x * x, 0);
    const v1 = column(res.V, 0);
    const order = v1.map((_, j) => j).sort((i, j) => Math.abs(v1[j]) - Math.abs(v1[i]));
    const olsTrace = sumInverseSquares(s, r, res.tol);
    const rankBound = Math.min(centred ? n - 1 : n, p);
    return { kappaX, kappaS, ratio, vr, sdLast, varSds, cvf, fullRank, vif, lambdaSVD, eigenVals, m, eigDiff, scoreGap, total, v1, order, olsTrace, rankBound };
  }, [a, Xc, res, s, r, n, p, centred]);

  const pcrTrace = useMemo(() => sumInverseSquares(s, kk, res.tol), [s, kk, res.tol]);

  const lauchli = useMemo(() => {
    const eps = Math.pow(10, -expo);
    const L: Matrix = [
      [1, 1],
      [eps, 0],
      [0, eps],
    ];
    const direct = svd(L).s;
    const G = gram(L);
    const eig = symmetricEigen(G).values;
    const viaGram = eig.map((l) => Math.sqrt(Math.max(l, 0)));
    const exact = [Math.sqrt(2 + eps * eps), eps];
    const lost = G[0][0] === 1;
    return { eps, L, G, direct, eig, viaGram, exact, lost, kappa: exact[0] / exact[1], relDirect: Math.abs(direct[1] - eps) / eps, relGram: Math.abs(viaGram[1] - eps) / eps };
  }, [expo]);

  const lamMax = Math.max(1, Math.ceil(Math.log10(Math.max(s[0] * s[0], 10))));
  const lv = Math.min(Math.max(logLam, -3), lamMax);
  const ridge = useMemo(() => {
    const lambda = Math.pow(10, lv);
    const f = ridgeFilterFactors(s, lambda);
    const df = f.reduce((acc, x) => acc + x, 0);
    const varRidge = s.reduce((acc, x, j) => acc + (x > 0 ? (f[j] * f[j]) / (x * x) : 0), 0);
    const pcr = s.map((_, j) => (j < kk ? 1 : 0));
    const half = s.filter((x) => x * x < lambda).length;
    return { lambda, f, df, varRidge, pcr, half };
  }, [s, lv, kk]);

  const js = s.map((_, j) => j + 1);
  const ridgeTraces: Data[] = [
    {
      type: 'scatter',
      mode: 'lines+markers',
      x: js,
      y: ridge.f,
      name: `ridge f_j, λ = ${fmt(ridge.lambda, 3)}`,
      line: { color: methodColor.SVD, width: 1.6 },
      marker: { color: methodColor.SVD, size: 6 },
      hovertemplate: 'j = %{x}<br>f_j = %{y:.4f}<extra>ridge</extra>',
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      x: js,
      y: ridge.pcr,
      name: `PCR, keep j ≤ ${kk}`,
      line: { color: accent2, width: 1.4, dash: 'dot', shape: 'hv' },
      marker: { color: accent2, size: 5, symbol: 'square' },
      hovertemplate: 'j = %{x}<br>f_j = %{y}<extra>PCR</extra>',
    },
  ];
  const ridgeLayout: Record<string, unknown> = {
    showlegend: true,
    xaxis: { title: { text: 'singular direction j (decreasing σ_j)' }, dtick: 1 },
    yaxis: { title: { text: 'filter factor f_j' }, range: [-0.05, 1.08] },
  };

  const storageK = kk * (n + p + 1);
  const storageFull = n * p;
  const energyK = s.slice(0, kk).reduce((acc, x) => acc + x * x, 0) / (core.total || 1);

  const items = [
    {
      id: 'svd-why-stability',
      title: (
        <>
          Numerical stability: <M tex="\kappa_2(X^TX) = \kappa_2(X)^2" />
        </>
      ),
      body: (
        <div className="stack">
          <div className="prose small">
            <p>
              The eigenvalues of <M tex="X_c^T X_c" /> are <M tex="\sigma_j^2" />, so forming the Gram (or covariance) matrix squares the condition number. On the current data{' '}
              <M tex={`\\kappa_2(X_c) = \\sigma_1/\\sigma_{${r}} = ${texNum(core.kappaX, 3)}`} /> while <M tex={`\\kappa_2(S) = ${texNum(core.kappaS, 3)}`} />; their ratio{' '}
              <M tex={`\\kappa_2(S)/\\kappa_2(X_c)^2 = ${texNum(core.ratio, 4)}`} />. Forming <M tex="X^TX" /> in floating point commits rounding errors of relative size{' '}
              <M tex="\varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}" /> in entries of size <M tex="\sigma_1^2" />, so any direction with{' '}
              <M tex="\sigma_j^2 \lesssim \varepsilon_{\text{mach}}\, \sigma_1^2" />, i.e. <M tex="\sigma_j/\sigma_1 \lesssim 1.5 \times 10^{-8}" />, becomes indistinguishable from an exact null
              direction. The SVD (here a one-sided Jacobi method) rotates the columns of <M tex="X" /> directly and never squares anything, so small singular values are
              recovered with high <em>relative</em> accuracy.
            </p>
          </div>
          <div className="stats">
            <StatTile label={<M tex="\kappa_2(X_c)" />} value={fmt(core.kappaX, 2)} />
            <StatTile label={<M tex="\kappa_2(X_c)^2" />} value={fmt(core.kappaX * core.kappaX, 2)} />
            <StatTile label={<M tex="\kappa_2(S)" />} value={fmt(core.kappaS, 2)} note="from the SVD of S" />
            <StatTile label="Rounding floor" value={fmt(EPS * s[0] * s[0], 2)} note={<M tex="\varepsilon_{\text{mach}}\sigma_1^2" />} />
          </div>
          <Card title="Läuchli’s example: a singular value that the Gram matrix cannot see" plane>
            <div className="grid side">
              <div className="controls-panel">
                <Slider label={<>exponent <M tex="e" />, <M tex="\varepsilon = 10^{-e}" /></>} value={expo} min={3} max={9} step={1} onChange={setExpo} format={(v) => `ε = 1e−${v}`} />
                <MBlock tex={`L = \\begin{bmatrix} 1 & 1 \\\\ \\varepsilon & 0 \\\\ 0 & \\varepsilon \\end{bmatrix}, \\qquad L^TL = \\begin{bmatrix} 1+\\varepsilon^2 & 1 \\\\ 1 & 1+\\varepsilon^2 \\end{bmatrix}`} />
                <div className="small secondary">
                  Exact singular values <M tex="\sigma_1 = \sqrt{2 + \varepsilon^2}" />, <M tex="\sigma_2 = \varepsilon" />; <M tex={`\\kappa_2(L) = ${texNum(lauchli.kappa, 2)}`} />,{' '}
                  <M tex={`\\kappa_2(L^TL) = ${texNum(lauchli.kappa ** 2, 2)}`} />.
                </div>
              </div>
              <div className="stack">
                <table className="summary-table">
                  <thead>
                    <tr>
                      <th>quantity</th>
                      <th>exact</th>
                      <th>SVD of L</th>
                      <th>
                        eigenvalues of <M tex="L^TL" />
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <M tex="\sigma_1" />
                      </td>
                      <td className="mono">{lauchli.exact[0].toPrecision(10)}</td>
                      <td className="mono">{lauchli.direct[0].toPrecision(10)}</td>
                      <td className="mono">{lauchli.viaGram[0].toPrecision(10)}</td>
                    </tr>
                    <tr>
                      <td>
                        <M tex="\sigma_2" />
                      </td>
                      <td className="mono">{lauchli.exact[1].toExponential(4)}</td>
                      <td className="mono">{lauchli.direct[1].toExponential(4)}</td>
                      <td className="mono">{lauchli.viaGram[1].toExponential(4)}</td>
                    </tr>
                    <tr>
                      <td>
                        relative error in <M tex="\sigma_2" />
                      </td>
                      <td className="mono">—</td>
                      <td className="mono">{lauchli.relDirect.toExponential(2)}</td>
                      <td className="mono">{lauchli.relGram.toExponential(2)}</td>
                    </tr>
                    <tr>
                      <td>
                        stored <M tex="(L^TL)_{11} - 1" />
                      </td>
                      <td className="mono">{(lauchli.eps * lauchli.eps).toExponential(2)}</td>
                      <td className="mono">—</td>
                      <td className="mono">{(lauchli.G[0][0] - 1).toExponential(2)}</td>
                    </tr>
                  </tbody>
                </table>
                {lauchli.lost ? (
                  <Callout kind="danger" title="Information destroyed before the eigensolver even starts">
                    With <M tex={`\\varepsilon = 10^{-${expo}}`} />, <M tex="1 + \varepsilon^2" /> rounds to exactly 1 in double precision (<M tex="\varepsilon^2 < 2^{-53}" />). The stored{' '}
                    <M tex="L^TL" /> is the singular matrix of ones, its second eigenvalue is <M tex={texNum(lauchli.eig[1], 2)} /> instead of <M tex={`\\varepsilon^2 = ${texNum(lauchli.eps ** 2, 2)}`} />, and no
                    eigensolver, however accurate, can recover <M tex="\sigma_2" />. The SVD applied to <M tex="L" /> itself returns it with relative error{' '}
                    <M tex={texNum(lauchli.relDirect, 2)} />.
                  </Callout>
                ) : (
                  <Callout kind="warning" title="Accuracy already degrading">
                    <M tex="1 + \varepsilon^2" /> is still representable, but <M tex="\sigma_2" /> computed through <M tex="L^TL" /> carries relative error{' '}
                    <M tex={texNum(lauchli.relGram, 2)} /> against <M tex={texNum(lauchli.relDirect, 2)} /> for the SVD. Increase <M tex="e" /> to 8 or 9 to see the eigenvalue
                    route fail completely.
                  </Callout>
                )}
              </div>
            </div>
          </Card>
        </div>
      ),
    },
    {
      id: 'svd-why-rank',
      title: 'Rank deficiency and the numerical rank',
      body: (
        <div className="stack">
          <div className="prose small">
            <p>
              The exact rank is a discontinuous function of the entries, so software counts singular values above a tolerance: here{' '}
              <M tex={`\\text{tol} = 10\\,\\max(n,p)\\,\\varepsilon_{\\text{mach}}\\,\\sigma_1 = ${texNum(res.tol, 2)}`} /> and{' '}
              <M tex={`\\operatorname{rank}_{\\text{num}} X_c = ${res.rank}`} /> of <M tex={`p = ${p}`} />.{' '}
              {centred ? (
                <>
                  Centring imposes <M tex="\mathbf 1^T X_c = 0" />, one linear constraint on the rows, so <M tex={`\\operatorname{rank} X_c \\le \\min(n-1, p) = ${core.rankBound}`} />
                  {n - 1 < p ? (
                    <>
                      : with <M tex={`n - 1 = ${n - 1} < p = ${p}`} /> the covariance matrix <M tex="S" /> is necessarily singular. PCA still works through the SVD (only{' '}
                      <M tex={`${res.rank}`} /> components carry variance), but <M tex="S^{-1}" /> — needed by Mahalanobis distances, LDA and GLS — does not exist without regularisation.
                    </>
                  ) : (
                    '.'
                  )}
                </>
              ) : (
                <>
                  Without centring the bound is <M tex={`\\min(n, p) = ${core.rankBound}`} />.
                </>
              )}
            </p>
          </div>
          <div className="stats">
            {s.map((x, j) => (
              <StatTile key={j} label={<M tex={`\\sigma_{${j + 1}}`} />} value={fmt(x, 3)} note={x > res.tol ? `σ_j / σ_1 = ${fmt(x / s[0], 4)}` : 'below tolerance: null direction'} />
            ))}
          </div>
        </div>
      ),
    },
    {
      id: 'svd-why-collinearity',
      title: 'Multicollinearity is a small singular value',
      body: (
        <div className="grid side-r">
          <div className="prose small">
            <p>
              A small <M tex={`\\sigma_{${r}}`} /> means there is a unit vector <M tex={`v_{${r}}`} /> with <M tex={`\\|X_c v_{${r}}\\| = \\sigma_{${r}} = ${texNum(s[r - 1], 4)}`} />: the linear
              combination <M tex={`\\sum_j v_{${r}j}\\, x_j`} /> of the (centred) variables is nearly constant, with sample standard deviation{' '}
              <M tex={`\\sigma_{${r}}/\\sqrt{n-1} = ${texNum(core.sdLast, 4)}`} /> against a smallest variable standard deviation of <M tex={texNum(Math.min(...core.varSds), 4)} />. The
              entries of <M tex={`v_{${r}}`} /> (table) name the variables involved. The classical variance inflation factor is the same object viewed through the Gram inverse:{' '}
              <M tex="\mathrm{VIF}_j = [(X_c^TX_c)^{-1}]_{jj}\,\|x_j\|^2 = 1/(1 - R_j^2)" />, where <M tex="R_j^2" /> is from regressing <M tex="x_j" /> on the other columns.
            </p>
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>variable</th>
                <th>
                  <M tex={`v_{${r}}`} /> entry
                </th>
                <th>sd</th>
                <th>VIF</th>
              </tr>
            </thead>
            <tbody>
              {names.map((nm, j) => (
                <tr key={j}>
                  <td>{nm}</td>
                  <td className="mono">{fmt(core.vr[j], 3)}</td>
                  <td className="mono">{fmt(core.varSds[j], 3)}</td>
                  <td className="mono">{fmt(core.vif[j], 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ),
    },
    {
      id: 'svd-why-leastsquares',
      title: (
        <>
          Least squares through the pseudo-inverse <M tex="X^+ = V\Sigma^+U^T" />
        </>
      ),
      body: (
        <div className="grid side-r">
          <div className="prose small">
            <MBlock tex="\hat\beta = X^+ y = \sum_{\sigma_j > 0} \frac{u_j^T y}{\sigma_j}\, v_j, \qquad \operatorname{Cov}(\hat\beta) = \sigma^2 (X^TX)^{-1} = \sigma^2 V\Sigma^{-2}V^T = \sigma^2 \sum_j \frac{v_j v_j^T}{\sigma_j^2}." />
            <p>
              The pseudo-inverse solves <M tex="\min_\beta \|y - X\beta\|" /> and, when <M tex="X" /> is rank deficient, returns the minimum-norm minimiser (directions with{' '}
              <M tex="\sigma_j = 0" /> are simply omitted). The covariance formula shows where instability comes from: coefficient variance is a sum over singular directions weighted by{' '}
              <M tex="1/\sigma_j^2" />, so the smallest singular value inflates variance quadratically. On the current design{' '}
              <M tex={`\\operatorname{tr}\\operatorname{Cov}(\\hat\\beta)/\\sigma^2 = \\sum_j \\sigma_j^{-2} = ${texNum(core.olsTrace, 4)}`} />, of which the direction{' '}
              <M tex={`v_{${core.cvf.retained}}`} /> alone contributes <M tex={`\\sigma_{${core.cvf.retained}}^{-2} = ${texNum(1 / (s[core.cvf.retained - 1] ** 2), 4)}`} />. Perturbing{' '}
              <M tex="y" /> by <M tex="\delta y" /> moves the estimate by <M tex="\|\delta\hat\beta\| = \|X^+\delta y\| \le \|\delta y\|/\sigma_r" />, while <M tex="\|\hat y\| \le \sigma_1\|\hat\beta\|" />, hence{' '}
              <M tex="\|\delta\hat\beta\|/\|\hat\beta\| \le \kappa_2(X)\,\|\delta y\|/\|\hat y\|" /> with <M tex={`\\kappa_2 = ${texNum(core.kappaX, 3)}`} /> here.
            </p>
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>coefficient</th>
                <th>
                  <M tex="\operatorname{Var}(\hat\beta_j)/\sigma^2" />
                </th>
                <th>
                  share from <M tex={`v_{${core.cvf.retained}}`} />
                </th>
              </tr>
            </thead>
            <tbody>
              {names.map((nm, j) => (
                <tr key={j}>
                  <td>{nm}</td>
                  <td className="mono">{fmt(core.cvf.diag[j], 4)}</td>
                  <td className="mono">{fmtPct(core.cvf.lastShare[j], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ),
    },
    {
      id: 'svd-why-pca',
      title: (
        <>
          PCA is the SVD of the centred matrix: <M tex="\lambda_j = \sigma_j^2/(n-1)" />, <M tex="Z = U\Sigma" />
        </>
      ),
      body: (
        <div className="stack">
          <div className="prose small">
            <p>
              <M tex="S = \frac{1}{n-1}X_c^TX_c = V\,\frac{\Sigma^T\Sigma}{n-1}\,V^T" /> is an eigendecomposition of <M tex="S" />: the loadings are the right singular vectors and{' '}
              <M tex="\lambda_j = \sigma_j^2/(n-1)" />. The scores are <M tex="Z = X_cV = U\Sigma" />, so <M tex="u_j" /> is the <M tex="j" />-th score vector scaled to unit length. The table
              compares the two routes on the current data (Jacobi eigensolver on <M tex="S" /> versus one-sided Jacobi SVD of <M tex="X_c" />): they agree to{' '}
              <M tex={texNum(core.eigDiff, 2)} /> in the eigenvalues, and <M tex={`\\|X_cV - U\\Sigma\\|_F = ${texNum(core.scoreGap, 2)}`} />.
            </p>
          </div>
          <div className="grid side-r">
            <table className="summary-table">
              <thead>
                <tr>
                  <th>j</th>
                  <th>
                    <M tex="\sigma_j" />
                  </th>
                  <th>
                    <M tex="\sigma_j^2/(n-1)" />
                  </th>
                  <th>
                    <M tex="\lambda_j(S)" />
                  </th>
                  <th>difference</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: core.m }, (_, j) => (
                  <tr key={j}>
                    <td className="mono">{j + 1}</td>
                    <td className="mono">{fmt(s[j], 4)}</td>
                    <td className="mono">{fmt(core.lambdaSVD[j], 5)}</td>
                    <td className="mono">{fmt(core.eigenVals[j], 5)}</td>
                    <td className="mono">{fmt(core.lambdaSVD[j] - core.eigenVals[j], 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="stack">
              <Callout kind="info" title="Continue in the PCA laboratory">
                The PCA tab derives the principal components from this factorisation and shows scores, loadings and the biplot for the same data.
              </Callout>
              <Button primary onClick={() => navigate('pca', 'pca-from-svd')}>
                Open PCA → from the SVD
              </Button>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 'svd-why-compression',
      title: 'Matrix compression: k(n + p + 1) numbers instead of np',
      body: (
        <div className="grid side-r">
          <div className="prose small">
            <p>
              Storing <M tex="X_k" /> as <M tex="U_k" /> (<M tex="n\times k" />), <M tex="\sigma_1,\dots,\sigma_k" /> and <M tex="V_k" /> (<M tex="p\times k" />) costs{' '}
              <M tex={`k(n+p+1) = ${kk}\\,(${n}+${p}+1) = ${storageK}`} /> numbers against <M tex={`np = ${storageFull}`} /> for the full matrix — a ratio of{' '}
              <M tex={texNum(storageK / storageFull, 3)} /> while keeping <M tex={fmtPct(energyK, 1).replace('%', '\\%')} /> of <M tex="\sum_j\sigma_j^2" /> (<M tex="k" /> is the slider of
              lesson 3). The trade-off is only favourable when <M tex="k \ll \min(n,p)" />: with <M tex={`p = ${p}`} /> variables there is little to gain, whereas an image or a
              term–document matrix with thousands of columns compresses by orders of magnitude. Statistically the same idea is dimension reduction: the <M tex="k" /> score
              columns <M tex="U_k\Sigma_k" /> replace <M tex="p" /> correlated variables.
            </p>
          </div>
          <div className="stats">
            <StatTile label="Full storage np" value={storageFull.toLocaleString()} />
            <StatTile label={<M tex="k(n+p+1)" />} value={storageK.toLocaleString()} note={`k = ${kk}`} />
            <StatTile label="Compression ratio" value={fmt(storageK / storageFull, 3)} note={storageK < storageFull ? 'smaller' : 'no saving at this k'} />
            <StatTile label="Energy kept" value={fmtPct(energyK, 1)} />
          </div>
        </div>
      ),
    },
    {
      id: 'svd-why-latent',
      title: 'Latent structure: rank-one terms as factors',
      body: (
        <div className="grid side-r">
          <div className="prose small">
            <p>
              <M tex="X = \sum_j \sigma_j u_j v_j^T" /> writes the data as a sum of "factors": <M tex="v_j" /> is a pattern over variables, <M tex="u_j" /> the pattern over observations that
              carries it, <M tex="\sigma_j" /> its strength. The first factor accounts for <M tex={`\\sigma_1^2/\\sum_j\\sigma_j^2 = ${fmtPct(s[0] ** 2 / (core.total || 1), 1).replace('%', '\\%')}`} />{' '}
              of the total; its loading vector is dominated by {names[core.order[0]]} (<M tex={texNum(core.v1[core.order[0]], 3)} />)
              {p > 1 && (
                <>
                  {' '}
                  and {names[core.order[1]]} (<M tex={texNum(core.v1[core.order[1]], 3)} />)
                </>
              )}
              . Caution on identifiability: the SVD itself is unique (for distinct <M tex="\sigma_j" />), but the factor model <M tex="X \approx F L^T" /> is not — <M tex="F \mapsto FQ" />,{' '}
              <M tex="L \mapsto LQ" /> for any orthogonal <M tex="Q" /> gives the same fit. The SVD picks the particular rotation in which the factors are uncorrelated and ordered by
              strength; a rotated solution (varimax, say) may be easier to name but is not "more true".
            </p>
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>variable</th>
                <th>
                  <M tex="v_1" /> loading
                </th>
                <th>
                  <M tex="v_2" /> loading
                </th>
              </tr>
            </thead>
            <tbody>
              {core.order.map((j) => (
                <tr key={j}>
                  <td>{names[j]}</td>
                  <td className="mono">{fmt(core.v1[j], 3)}</td>
                  <td className="mono">{r > 1 ? fmt(res.V[j][1], 3) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ),
    },
    {
      id: 'svd-why-pcr',
      title: 'Regression on components: principal component regression drops the small-σ directions',
      body: (
        <div className="stack">
          <div className="prose small">
            <MBlock tex="\hat\beta_{\text{PCR}(k)} = \sum_{j \le k} \frac{u_j^T y}{\sigma_j}\, v_j, \qquad \frac{\operatorname{tr}\operatorname{Cov}(\hat\beta_{\text{PCR}(k)})}{\sigma^2} = \sum_{j\le k}\sigma_j^{-2} ." />
            <p>
              Regressing <M tex="y" /> on the first <M tex="k" /> score columns <M tex="U_k\Sigma_k" /> and mapping back to the original coefficients is ordinary least squares with the
              directions <M tex="v_{k+1},\dots,v_r" /> removed: filter factors <M tex="f_j = 1" /> for <M tex="j \le k" /> and <M tex="0" /> beyond. With the lesson-3 value{' '}
              <M tex={`k = ${kk}`} /> the variance trace falls from <M tex={`\\sum_j \\sigma_j^{-2} = ${texNum(core.olsTrace, 4)}`} /> to{' '}
              <M tex={`\\sum_{j\\le ${kk}}\\sigma_j^{-2} = ${texNum(pcrTrace, 4)}`} />, a reduction of <M tex={fmtPct(1 - pcrTrace / (core.olsTrace || 1), 1).replace('%', '\\%')} />. The price is
              bias equal to the component of <M tex="\beta" /> in <M tex={`\\operatorname{span}\\{v_{${kk + 1}},\\dots,v_{${r}}\\}`} /> — and nothing guarantees it is small: the variance
              ordering of <M tex="X" /> knows nothing about <M tex="y" />. A low-variance direction can be the one that predicts.
            </p>
          </div>
          <div className="stats">
            <StatTile label={<>OLS <M tex="\sum_j\sigma_j^{-2}" /></>} value={fmt(core.olsTrace, 4)} />
            <StatTile label={<>PCR <M tex={`\\sum_{j\\le ${kk}}\\sigma_j^{-2}`} /></>} value={fmt(pcrTrace, 4)} />
            <StatTile label="Variance removed" value={fmtPct(1 - pcrTrace / (core.olsTrace || 1), 1)} note={`directions ${kk + 1}…${r} dropped`} />
          </div>
        </div>
      ),
    },
    {
      id: 'svd-why-ridge',
      title: (
        <>
          Regularisation: ridge filter factors <M tex="f_j = \sigma_j^2/(\sigma_j^2 + \lambda)" />
        </>
      ),
      body: (
        <div className="stack">
          <div className="prose small">
            <MBlock tex="\hat\beta_\lambda = (X^TX + \lambda I)^{-1}X^Ty = \sum_j \underbrace{\frac{\sigma_j^2}{\sigma_j^2 + \lambda}}_{f_j}\, \frac{u_j^T y}{\sigma_j}\, v_j, \qquad \operatorname{df}(\lambda) = \sum_j f_j, \qquad \frac{\operatorname{tr}\operatorname{Cov}(\hat\beta_\lambda)}{\sigma^2} = \sum_j \frac{f_j^2}{\sigma_j^2}." />
            <p>
              Ridge regression is the same computation as PCR with a smooth filter instead of a hard cut: directions with <M tex="\sigma_j^2 \gg \lambda" /> pass almost unchanged, directions
              with <M tex="\sigma_j^2 \ll \lambda" /> are shrunk towards zero, and <M tex="\sigma_j^2 = \lambda" /> is halved. Because the filter multiplies <M tex="1/\sigma_j" /> by{' '}
              <M tex="f_j" />, the explosive terms <M tex="1/\sigma_j^2" /> in the variance become <M tex="f_j^2/\sigma_j^2 \le \sigma_j^2/\lambda^2" />: bounded however small{' '}
              <M tex="\sigma_j" /> gets. Everything about the estimator — bias, variance, effective degrees of freedom — is a function of <M tex="\lambda" /> and the singular values.
            </p>
          </div>
          <div className="grid side">
            <div className="controls-panel">
              <Slider label={<M tex="\log_{10}\lambda" />} value={lv} min={-3} max={lamMax} step={0.1} onChange={setLogLam} format={(v) => `λ = ${fmt(Math.pow(10, v), 3)}`} />
              <div className="stats">
                <StatTile label={<M tex="\operatorname{df}(\lambda) = \sum_j f_j" />} value={fmt(ridge.df, 3)} note={`of ${r} directions`} />
                <StatTile label={<M tex="\sum_j f_j^2/\sigma_j^2" />} value={fmt(ridge.varRidge, 4)} note={`OLS: ${fmt(core.olsTrace, 4)}`} />
                <StatTile label={<M tex="\#\{\sigma_j^2 < \lambda\}" />} value={ridge.half} note="directions shrunk by more than half" />
              </div>
            </div>
            <div className="stack">
              <Plot data={ridgeTraces} layout={ridgeLayout} height={300} title="Filter factors by singular direction" />
              <div className="plot-caption">
                Singular values <M tex={`(\\sigma_1,\\dots,\\sigma_{${r}}) = (${s.map((x) => texNum(x, 3)).join(', ')})`} />; squared, they set where the ridge curve bends.
              </div>
              <Interpretation
                defaultOpen={false}
                items={{
                  seeing: (
                    <>
                      Ridge factors <M tex="f_j" /> for <M tex={`\\lambda = ${texNum(ridge.lambda, 3)}`} /> (solid) fall from <M tex={texNum(ridge.f[0], 3)} /> at{' '}
                      <M tex="j = 1" /> to <M tex={texNum(ridge.f[r - 1], 3)} /> at <M tex={`j = ${r}`} />; the PCR step function (dotted) keeps <M tex={`${kk}`} /> directions
                      exactly and discards the rest. Effective degrees of freedom <M tex={`\\operatorname{df}(\\lambda) = ${texNum(ridge.df, 2)}`} /> against <M tex={`${kk}`} /> for
                      PCR.
                    </>
                  ),
                  why: (
                    <>
                      <M tex="f_j = \sigma_j^2/(\sigma_j^2+\lambda)" /> is a monotone function of <M tex="\sigma_j" />, so the curve decreases along <M tex="j" /> and crosses{' '}
                      <M tex="1/2" /> where <M tex="\sigma_j^2 = \lambda" />: currently <M tex={`${ridge.half}`} /> of the <M tex={`${r}`} /> squared singular values lie below{' '}
                      <M tex="\lambda" />. Widely spread singular values (here <M tex={`\\kappa_2 = ${texNum(core.kappaX, 2)}`} />) make the transition gradual on the index axis.
                    </>
                  ),
                  math: (
                    <>
                      <M tex="(X^TX + \lambda I)^{-1}X^T = V(\Sigma^T\Sigma + \lambda I)^{-1}\Sigma^T U^T" />, whose diagonal entries are <M tex="\sigma_j/(\sigma_j^2+\lambda) = f_j/\sigma_j" />. The
                      variance trace <M tex={`\\sum_j f_j^2/\\sigma_j^2 = ${texNum(ridge.varRidge, 4)}`} /> compares with the OLS value <M tex={texNum(core.olsTrace, 4)} />.
                    </>
                  ),
                  stats: (
                    <>
                      Ridge trades bias for variance continuously and never sets a coefficient to zero; PCR is its hard-threshold cousin. Choosing <M tex="\lambda" /> (or{' '}
                      <M tex="k" />) is a statistical decision — cross-validation or GCV, with <M tex="\operatorname{df}(\lambda)" /> as the complexity measure — not a property of{' '}
                      <M tex="X" /> alone.
                    </>
                  ),
                  careful: (
                    <>
                      The filter factors depend on the scale of the columns: standardising <M tex="X" /> changes every <M tex="\sigma_j" /> and hence the meaning of a given{' '}
                      <M tex="\lambda" /> (compare the "Centre" and "Standardise" preprocessing in lesson 3). The intercept is normally left unpenalised, which is why{' '}
                      <M tex="X" /> is centred first.
                    </>
                  ),
                }}
              />
            </div>
          </div>
        </div>
      ),
    },
  ];

  return (
    <Section
      id="svd-statistician"
      title="4 · Why should a statistician care about the SVD?"
      subtitle="Nine reasons, each checked on the current data. The common thread: every quantity that misbehaves in multivariate statistics misbehaves through a small singular value."
      right={<Badge method="SVD" />}
    >
      <Accordion items={items} defaultOpen={[0]} />
    </Section>
  );
}
