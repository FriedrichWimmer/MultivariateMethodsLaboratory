import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { useAnalysis, useStore } from '../../../state/store';
import { M, MBlock, texVector } from '../../common/Math';
import { Plot } from '../../common/Plot';
import { ScatterSVG } from '../../common/ScatterSVG';
import { MatrixView } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile, Derivation, ClassLegend, Badge } from '../../common/Panels';
import { Select, Slider, Button } from '../../common/Controls';
import { centerColumns, covariance, colVariances, gram, fmt } from '../../../lib/linalg';
import { projectionVariance, projectOnDirection, sampleVariance } from '../../../lib/pca';
import { methodColor, neutralMark, ink } from '../../../lib/theme';
import { axial, deg, rad, pct, sci, texNum, eigen2, pcLabels } from './util';

/** Lesson 2 — a principal direction as the maximiser of projected variance, in a two-variable plane. */
export function Geometry() {
  const a = useAnalysis();
  const { prep } = useStore();
  const { pca: P, n, p, dataset } = a;
  const names = dataset.variableNames;
  const Xc = P.Xc;

  const [sel, setSel] = useState<[number, number] | null>(null);
  const [theta, setThetaRaw] = useState(0.35);
  const setTheta = (t: number) => setThetaRaw(axial(t));

  // default: the two variables with the largest variance in the analysed matrix
  const defaults = useMemo<[number, number]>(() => {
    const v = colVariances(Xc);
    const order = v.map((_, j) => j).sort((x, y) => v[y] - v[x]);
    return [order[0], order[1] ?? order[0]];
  }, [Xc]);
  const valid = sel !== null && sel[0] < p && sel[1] < p && sel[0] !== sel[1];
  const [ii, jj] = valid ? (sel as [number, number]) : defaults;

  const geo = useMemo(() => {
    const Y = Xc.map((r) => [r[ii], r[jj]]);
    const { Xc: Y2, means } = centerColumns(Y);
    const S2 = covariance(Y2);
    const e = eigen2(S2);
    const grid: number[] = [];
    const prof: number[] = [];
    for (let d = -90; d <= 90; d += 1) {
      const t = rad(d);
      grid.push(d);
      prof.push(projectionVariance(S2, [Math.cos(t), Math.sin(t)]));
    }
    const covZ12 = e.v1[0] * (S2[0][0] * e.v2[0] + S2[0][1] * e.v2[1]) + e.v1[1] * (S2[1][0] * e.v2[0] + S2[1][1] * e.v2[1]);
    return { Y2, means, S2, ...e, grid, prof, covZ12, total: S2[0][0] + S2[1][1] };
  }, [Xc, ii, jj]);

  const cur = useMemo(() => {
    const w: [number, number] = [Math.cos(theta), Math.sin(theta)];
    const z = projectOnDirection(geo.Y2, w);
    const vQuad = projectionVariance(geo.S2, w);
    const vSample = sampleVariance(z);
    const feet = z.map((zi) => [zi * w[0], zi * w[1]] as [number, number]);
    const frac = geo.l1 > 0 ? vQuad / geo.l1 : 0;
    const phi = axial(theta - geo.theta1);
    const closed = geo.l1 * Math.cos(phi) ** 2 + geo.l2 * Math.sin(phi) ** 2;
    return { w, z, vQuad, vSample, feet, frac, phi, closed };
  }, [geo, theta]);

  // uncorrelatedness of the global scores: ZᵀZ/(n−1) = Λ
  const scoresCheck = useMemo(() => {
    const Z = P.scores;
    const m = Math.min(Z[0].length, 4);
    const G = gram(Z.map((row) => row.slice(0, m))).map((row) => row.map((x) => x / Math.max(n - 1, 1)));
    let off = 0;
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) if (i !== j) off = Math.max(off, Math.abs(G[i][j]));
    return { G, off, m };
  }, [P, n]);

  const varOptions = names.map((nm, j) => ({ value: String(j), label: nm }));
  const rx = 2 * Math.sqrt(geo.l1);
  const ry = 2 * Math.sqrt(geo.l2);
  const thetaDeg = deg(theta);
  const labelled = !!dataset.y;

  const profileData: Data[] = [
    { type: 'scatter', mode: 'lines', x: geo.grid, y: geo.prof, name: 'wᵀSw', line: { color: methodColor.PCA, width: 2 }, hovertemplate: 'θ = %{x}°<br>Var = %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: [thetaDeg], y: [cur.vQuad], name: 'current w', marker: { color: ink.primary, size: 11 }, hovertemplate: 'θ = %{x:.1f}°<br>Var = %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: [deg(geo.theta1)], y: [geo.l1], name: 'PC1: λ₁ (maximum)', marker: { color: methodColor.PCA, size: 10, symbol: 'diamond' }, hovertemplate: 'θ₁ = %{x:.1f}°<br>λ₁ = %{y:.4f}<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: [deg(geo.theta2)], y: [geo.l2], name: 'PC2: λ₂ (minimum)', marker: { color: methodColor.PCA, size: 10, symbol: 'diamond-open' }, hovertemplate: 'θ₂ = %{x:.1f}°<br>λ₂ = %{y:.4f}<extra></extra>' },
  ];
  const profileLayout: Partial<Layout> = {
    showlegend: true,
    xaxis: { title: { text: 'θ (degrees)' }, range: [-90, 90], dtick: 30 },
    yaxis: { title: { text: 'Var(X_c w) = wᵀ S w' }, rangemode: 'tozero' },
  };

  const steps = [
    {
      title: 'The optimisation problem',
      body: (
        <>
          A principal direction is a unit vector <M tex="w \in \mathbb R^{p}" /> along which the projected data <M tex="z = X_c w" /> has maximal sample variance:
          <MBlock tex={String.raw`\max_{w}\ \operatorname{Var}(X_c w) = \frac{1}{n-1}\,\|X_c w\|^{2} = \frac{1}{n-1}\,w^{T}X_c^{T}X_c\,w = w^{T}Sw \qquad \text{subject to } w^{T}w = 1.`} />
        </>
      ),
      note: 'Without the constraint the objective is unbounded (replacing w by tw multiplies it by t²); the unit-norm constraint picks one representative of each direction. Because w and −w give the same line, a direction is an axis — the handle above is axial.',
    },
    {
      title: 'Rayleigh quotient',
      body: (
        <>
          Equivalently maximise the Rayleigh quotient <M tex="R(w) = \dfrac{w^{T}Sw}{w^{T}w}" /> over <M tex="w \ne 0" />; it is homogeneous of degree zero, so the supremum over all <M tex="w" /> equals the maximum over
          the unit sphere. The Rayleigh–Ritz theorem states that for symmetric <M tex="S" /> with eigenvalues <M tex="\lambda_1 \ge \dots \ge \lambda_p" />,
          <MBlock tex={String.raw`\lambda_p \le R(w) \le \lambda_1 \quad \text{for all } w \ne 0, \qquad R(v_1) = \lambda_1, \quad R(v_p) = \lambda_p .`} />
        </>
      ),
      note: `In the plane shown above R(w) ranges over [λ₂, λ₁] = [${fmt(geo.l2, 4)}, ${fmt(geo.l1, 4)}]; the current handle gives R(w) = ${fmt(cur.vQuad, 4)}.`,
    },
    {
      title: 'Lagrangian and stationarity',
      body: (
        <>
          <MBlock tex={String.raw`\mathcal L(w,\lambda) = w^{T}Sw - \lambda\,(w^{T}w - 1), \qquad \nabla_{w}\mathcal L = 2Sw - 2\lambda w = 0 \;\Longleftrightarrow\; Sw = \lambda w .`} />
          Every stationary point is a unit eigenvector of <M tex="S" />, and there the objective equals <M tex="w^{T}Sw = \lambda\,w^{T}w = \lambda" />. The maximum is therefore the largest eigenvalue{' '}
          <M tex="\lambda_1" />, attained at <M tex="w = v_1" /> — the multiplier <em>is</em> the variance.
        </>
      ),
      note: `Writing w = cos φ · v₁ + sin φ · v₂ (φ = angle from PC1) gives wᵀSw = λ₁cos²φ + λ₂sin²φ. At the current φ = ${fmt(deg(cur.phi), 1)}° this is ${fmt(cur.closed, 4)}; the direct computation gives ${fmt(cur.vQuad, 4)} (gap ${sci(Math.abs(cur.closed - cur.vQuad))}).`,
    },
    {
      title: 'The second component: orthogonality constraint',
      body: (
        <>
          For <M tex="v_2" /> maximise <M tex="w^{T}Sw" /> subject to <M tex="w^{T}w = 1" /> <em>and</em> <M tex="w^{T}v_1 = 0" />:
          <MBlock tex={String.raw`\mathcal L = w^{T}Sw - \lambda\,(w^{T}w-1) - \mu\, w^{T}v_1, \qquad 2Sw - 2\lambda w - \mu v_1 = 0 .`} />
          Premultiplying by <M tex="v_1^{T}" /> and using <M tex="v_1^{T}Sw = (Sv_1)^{T}w = \lambda_1 v_1^{T}w = 0" /> gives <M tex="\mu = 0" />, so again <M tex="Sw = \lambda w" />: <M tex="w" /> is an eigenvector orthogonal to{' '}
          <M tex="v_1" />, and the best such choice is <M tex="v_2" /> with value <M tex="\lambda_2" />. Induction (Courant–Fischer) yields the full orthonormal basis <M tex="v_1, \dots, v_p" />.
        </>
      ),
      note: `Here θ₁ = ${fmt(deg(geo.theta1), 1)}° and θ₂ = ${fmt(deg(geo.theta2), 1)}°: the two axes are 90° apart, as they must be.`,
    },
    {
      title: 'The components are uncorrelated',
      body: (
        <>
          <MBlock tex={String.raw`\operatorname{Cov}(z_j, z_k) = \frac{1}{n-1}\,v_j^{T}X_c^{T}X_c\,v_k = v_j^{T}Sv_k = \lambda_k\, v_j^{T}v_k = \lambda_j\,\delta_{jk}, \qquad \text{so} \qquad \frac{1}{n-1}Z^{T}Z = \Lambda .`} />
          PCA is thus a rotation of the centred data into new coordinates in which the sample covariance is diagonal; the total variance <M tex="\operatorname{tr} S" /> is preserved and merely reallocated.
        </>
      ),
      note: `Live check on the active dataset: the largest off-diagonal entry of ZᵀZ/(n−1) for the first ${scoresCheck.m} components is ${sci(scoresCheck.off)}; in the plane above v₁ᵀSv₂ = ${sci(geo.covZ12)}.`,
    },
  ];

  return (
    <Section
      id="pca-geometry"
      title="2 · The geometry of a principal direction"
      subtitle="Drag the direction w through the mean and watch the variance of the projected points; the maximiser is the first eigenvector of S."
      right={<Badge method="PCA" />}
    >
      <div className="prose">
        <p>
          Two variables of the analysed matrix are shown, each re-centred so that the origin is the sample mean. Every observation <M tex="x_i" /> is projected orthogonally onto the line through the mean in
          the direction <M tex="w = (\cos\theta, \sin\theta)" />: the projection has coordinate <M tex="z_i = x_i^{T}w" /> along the line and foot <M tex="z_i\,w" />. The dashed ellipse is the covariance ellipse
          with semi-axes <M tex="2\sqrt{\lambda_1}, 2\sqrt{\lambda_2}" /> along the eigenvectors of the <M tex="2\times 2" /> covariance matrix <M tex="S" /> of the two variables.
          {prep.scaling === 'none' && (
            <>
              {' '}
              (The global preprocessing is "Raw"; this panel nevertheless centres the two variables, so its <M tex="S" /> is a genuine covariance and differs from the global second-moment matrix.)
            </>
          )}
        </p>
      </div>

      <div className="grid side">
        <div className="stack">
          <Card title="Direction and variables" plane>
            <Select<string>
              label="Horizontal variable"
              value={String(ii)}
              options={varOptions}
              onChange={(v) => {
                const k = Number(v);
                setSel(k === jj ? [k, ii] : [k, jj]);
              }}
            />
            <Select<string>
              label="Vertical variable"
              value={String(jj)}
              options={varOptions}
              onChange={(v) => {
                const k = Number(v);
                setSel(k === ii ? [jj, k] : [ii, k]);
              }}
            />
            <Slider label="θ (degrees)" value={Math.round(thetaDeg)} min={-90} max={90} step={1} onChange={(d) => setTheta(rad(d))} format={(v) => `${v}°`} />
            <div className="row">
              <Button small primary onClick={() => setTheta(geo.theta1)}>
                Snap to PC1
              </Button>
              <Button small onClick={() => setTheta(geo.theta2)}>
                Snap to PC2
              </Button>
              <Button small onClick={() => setTheta(0)}>
                θ = 0°
              </Button>
            </div>
            <div className="kbd-hint">Drag the handle labelled w, or click anywhere in the plot to re-orient it.</div>
          </Card>
          <Card title="Live quantities" plane>
            <div className="stack" style={{ gap: 6 }}>
              <div>
                <M tex={`w = ${texVector(cur.w, 3)}`} />, <M tex={`\\theta = ${texNum(thetaDeg, 1)}^{\\circ}`} />
              </div>
              <div>
                <M tex={`w^{T}Sw = ${texNum(cur.vQuad, 4)}`} />
              </div>
              <div>
                <M tex={`\\operatorname{Var}(z) = ${texNum(cur.vSample, 4)}`} /> <span className="muted small">(sample variance of z = X_c w; gap {sci(Math.abs(cur.vSample - cur.vQuad))})</span>
              </div>
              <div>
                <M tex={`\\lambda_1 = ${texNum(geo.l1, 4)},\\; \\lambda_2 = ${texNum(geo.l2, 4)}`} />
              </div>
              <div>
                fraction of <M tex="\lambda_1" />: <b>{pct(cur.frac)}</b>; angle to PC1: <M tex={`\\varphi = ${texNum(Math.abs(deg(cur.phi)), 1)}^{\\circ}`} />
              </div>
              <div className="small muted mono">
                z₁…z₆ = {cur.z.slice(0, 6).map((v) => fmt(v, 2)).join(', ')}
                {n > 6 ? ', …' : ''}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <ScatterSVG
            points={geo.Y2}
            labels={dataset.y}
            classNames={dataset.classNames}
            pointColor={labelled ? undefined : neutralMark}
            pointOpacity={0.7}
            width={560}
            height={440}
            xLabel={`${names[ii]} (centred)`}
            yLabel={`${names[jj]} (centred)`}
            title="Projection onto the line through the mean in direction w"
            segments={geo.Y2.map((pt, i) => ({ from: [pt[0], pt[1]] as [number, number], to: cur.feet[i], color: neutralMark, opacity: 0.3 }))}
            extraPoints={cur.feet.map((f) => ({ x: f[0], y: f[1], r: 2.4, color: ink.primary, opacity: 0.6 }))}
            ellipses={[{ cx: 0, cy: 0, rx, ry, angle: Math.atan2(geo.v1[1], geo.v1[0]), color: methodColor.PCA, dashed: true }]}
            vectors={[
              { x: rx * geo.v1[0], y: rx * geo.v1[1], color: methodColor.PCA, label: 'PC1' },
              { x: ry * geo.v2[0], y: ry * geo.v2[1], color: methodColor.PCA, label: 'PC2', dashed: true },
            ]}
            direction={{ angle: theta, onChange: setTheta, axial: true, label: 'w', color: ink.primary }}
            hoverInfo={(i) => `#${i + 1}  (${geo.Y2[i][0].toFixed(2)}, ${geo.Y2[i][1].toFixed(2)})\nz = ${cur.z[i].toFixed(3)}${dataset.y && dataset.classNames ? `\n${dataset.classNames[dataset.y[i]]}` : ''}`}
            caption={
              <>
                Arrows PC1/PC2 have length <M tex="2\sqrt{\lambda_j}" />; grey segments join each observation to its foot on the w-line; small dark dots are the projected points. Dashed curve: covariance ellipse.
              </>
            }
          />
          {labelled && dataset.classNames && <ClassLegend classNames={dataset.classNames} />}
        </div>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              The variables <b>{names[ii]}</b> and <b>{names[jj]}</b>, mean-centred, with the direction handle at θ = {fmt(thetaDeg, 1)}°. Projecting the {n} observations onto this line gives a variable z with
              variance {fmt(cur.vQuad, 4)}, which is {pct(cur.frac)} of λ₁ = {fmt(geo.l1, 4)}. The handle is {fmt(Math.abs(deg(cur.phi)), 1)}° away from PC1 (θ₁ = {fmt(deg(geo.theta1), 1)}°).
            </>
          ),
          why: (
            <>
              The projected spread is largest along the long axis of the ellipse because the ellipse's semi-axes are exactly <M tex="2\sqrt{\lambda_j}" />: along <M tex="v_1" /> the projection keeps all of{' '}
              λ₁ = {fmt(geo.l1, 4)}, along <M tex="v_2" /> only λ₂ = {fmt(geo.l2, 4)} (ratio λ₁/λ₂ = {geo.l2 > 0 ? fmt(geo.l1 / geo.l2, 2) : '∞'}). The grey feet bunch together when w is far from PC1 and spread out as it
              approaches it.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\operatorname{Var}(X_c w) = w^{T}Sw = \lambda_1\cos^{2}\varphi + \lambda_2\sin^{2}\varphi, \qquad \varphi = \angle(w, v_1),`} />
              here {fmt(geo.l1, 4)}·cos²({fmt(Math.abs(deg(cur.phi)), 1)}°) + {fmt(geo.l2, 4)}·sin²({fmt(Math.abs(deg(cur.phi)), 1)}°) = {fmt(cur.closed, 4)}. The total{' '}
              <M tex={`\\operatorname{tr} S = \\lambda_1 + \\lambda_2 = ${texNum(geo.total, 4)}`} /> is shared between any two orthogonal directions.
            </>
          ),
          stats: (
            <>
              z = X_c w is a new derived variable, the weighted sum {fmt(cur.w[0], 3)}·({names[ii]} − mean) + {fmt(cur.w[1], 3)}·({names[jj]} − mean). Its sample variance {fmt(cur.vSample, 4)} equals the quadratic form{' '}
              wᵀSw = {fmt(cur.vQuad, 4)} (difference {sci(Math.abs(cur.vSample - cur.vQuad))}), so the variance of any linear combination is read directly off <M tex="S" /> without recomputing from the data.
            </>
          ),
          careful: (
            <>
              The ellipse summarises second moments only: under bivariate normality the <M tex="2\sqrt{\lambda}" /> contour encloses about 86.5% of the mass (Mahalanobis radius 2); for skewed or clustered clouds
              {labelled ? ' — note the class structure visible here — ' : ' '}it is not a density contour. When λ₁ ≈ λ₂ the direction of PC1 is unstable (here λ₁/λ₂ = {geo.l2 > 0 ? fmt(geo.l1 / geo.l2, 2) : '∞'}
              ). And w, −w describe the same line: the sign of z is a convention.
            </>
          ),
        }}
      />

      <div className="grid c2">
        <div>
          <Plot data={profileData} layout={profileLayout} height={320} title="Projected variance as a function of the direction angle" />
          <div className="plot-caption">
            wᵀSw over θ ∈ [−90°, 90°] (axial: θ and θ ± 180° coincide). Diamonds mark the eigen-directions; the black dot is the current handle.
          </div>
        </div>
        <Card title="Uncorrelated scores on the active dataset">
          <MatrixView
            M={scoresCheck.G}
            title="ZᵀZ/(n−1) for the first components"
            rowLabels={pcLabels(scoresCheck.m)}
            colLabels={pcLabels(scoresCheck.m)}
            digits={4}
            heat="diverging"
            compact
            caption={`Diagonal = λ_j; largest off-diagonal magnitude ${sci(scoresCheck.off)}. ${prep.scaling === 'none' ? 'Without centring this is a second-moment matrix, not a covariance.' : 'This is the sample covariance matrix of the scores.'}`}
          />
          <div className="stats" style={{ marginTop: 8 }}>
            <StatTile label="global λ₁" value={fmt(P.eigenvalues[0], 4)} note={`${pct(P.explained[0])} of tr S`} />
            <StatTile label="global λ₂" value={P.eigenvalues.length > 1 ? fmt(P.eigenvalues[1], 4) : '–'} note={P.explained.length > 1 ? `${pct(P.explained[1])} of tr S` : ''} />
          </div>
        </Card>
      </div>

      <Interpretation
        title="Interpretation — the variance profile"
        items={{
          seeing: (
            <>
              The curve is the quadratic form wᵀSw as w rotates through every direction in the plane. It is a sinusoid in 2θ with maximum λ₁ = {fmt(geo.l1, 4)} at θ₁ = {fmt(deg(geo.theta1), 1)}° and minimum λ₂ ={' '}
              {fmt(geo.l2, 4)} at θ₂ = {fmt(deg(geo.theta2), 1)}°, exactly 90° apart. The black dot ({fmt(thetaDeg, 1)}°, {fmt(cur.vQuad, 4)}) is the current handle.
            </>
          ),
          why: (
            <>
              Substituting w = (cos θ, sin θ) gives wᵀSw = S₁₁cos²θ + 2S₁₂ sin θ cos θ + S₂₂ sin²θ = ½(S₁₁ + S₂₂) + ½(S₁₁ − S₂₂)cos 2θ + S₁₂ sin 2θ, a single harmonic in 2θ whose mean level is ½ tr S ={' '}
              {fmt(geo.total / 2, 4)} and whose amplitude is ½(λ₁ − λ₂) = {fmt((geo.l1 - geo.l2) / 2, 4)}.
            </>
          ),
          math: (
            <>
              The stationary points of wᵀSw on the unit circle solve <M tex="Sw = \lambda w" /> (step 3 of the derivation): in two dimensions there are exactly two axes, the eigenvectors, and the curve has no other
              extrema. The matrix ZᵀZ/(n−1) on the right shows the same fact for the full analysis: <M tex="v_j^{T}Sv_k = \lambda_j\delta_{jk}" />, off-diagonals {sci(scoresCheck.off)}.
            </>
          ),
          stats: (
            <>
              The gap between maximum and minimum measures how anisotropic the cloud is: λ₁ − λ₂ = {fmt(geo.l1 - geo.l2, 4)} out of tr S = {fmt(geo.total, 4)}. The uncorrelatedness of the scores means each
              component carries non-redundant variance — PC2 adds information that PC1 cannot, which is why cumulative proportions of variance add up (lesson 3).
            </>
          ),
          careful: (
            <>
              "Uncorrelated" is not "independent": PCA removes linear association only. A flat profile (λ₁ ≈ λ₂) does not mean the data are structureless — it means no <em>linear</em> direction is preferred, which is
              exactly the situation in which PC1 is arbitrary and unstable under resampling.
            </>
          ),
        }}
      />

      <Callout kind="theorem" title="Theorem (variational characterisation of the principal directions)">
        For a symmetric positive semi-definite <M tex="S" /> with eigenpairs <M tex="(\lambda_j, v_j)" />, <M tex="\lambda_1 \ge \dots \ge \lambda_p" />:{' '}
        <M tex="\max_{\|w\|=1} w^{T}Sw = \lambda_1" /> attained at <M tex="v_1" />; and for <M tex="k \ge 2" />, <M tex="\max\{\,w^{T}Sw : \|w\|=1,\ w \perp v_1,\dots,v_{k-1}\} = \lambda_k" /> attained at{' '}
        <M tex="v_k" />. The resulting scores satisfy <M tex="\tfrac{1}{n-1}Z^{T}Z = \Lambda" />.
      </Callout>
      <Derivation title="Derivation: from the variance criterion to the eigenproblem" steps={steps} initiallyRevealed={1} />
    </Section>
  );
}
