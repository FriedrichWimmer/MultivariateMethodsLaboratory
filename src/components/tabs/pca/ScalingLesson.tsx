import { useMemo, useState } from 'react';
import { M, MBlock, texVector } from '../../common/Math';
import { ScatterSVG } from '../../common/ScatterSVG';
import { MatrixView } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile, Badge } from '../../common/Panels';
import { Slider, Segmented, Button } from '../../common/Controls';
import { column, colMeans, colVariances, fmt } from '../../../lib/linalg';
import { pca, type Scaling } from '../../../lib/pca';
import { generateDataset, defaultParams } from '../../../lib/datasets';
import { methodColor, accent2, neutralMark, ink } from '../../../lib/theme';
import { pct, sci, texNum, pcLabels, scalingName, decomposedMatrixTex } from './util';

const VAR = ['x1', 'x2', 'x3'];
const VAR_TEX = ['x_1', 'x_2', 'x_3'];

/** Lesson 5 — covariance versus correlation PCA on a dataset with one badly scaled variable. */
export function ScalingLesson() {
  const [logF, setLogF] = useState(2);
  const [offset, setOffset] = useState(3);
  const [seed, setSeed] = useState(7);
  const [mode, setMode] = useState<Scaling>('center');
  const factor = Math.round(Math.pow(10, logF));

  // local dataset: three AR(1)-correlated N(0,1) variables, x1 multiplied by `factor`, x2 translated by `offset`
  const ds = useMemo(() => generateDataset('scales', { ...defaultParams, n: 150, p: 3, correlation: 0.6, scaleFactor: factor, seed }), [factor, seed]);
  const X = useMemo(() => ds.X.map((row) => row.map((x, j) => (j === 1 ? x + offset : x))), [ds, offset]);
  const n = X.length;

  const fits = useMemo(() => ({ none: pca(X, 'none'), center: pca(X, 'center'), standardize: pca(X, 'standardize') }), [X]);

  const st = useMemo(() => {
    const vars = colVariances(X);
    const stds = vars.map(Math.sqrt);
    const means = colMeans(X);
    const meanNorm = Math.hypot(...means);
    const trRaw = fits.none.totalVariance; // tr(XᵀX)/(n−1) = tr S + n‖x̄‖²/(n−1)
    const trS = fits.center.totalVariance;
    const meanShare = trRaw > 0 ? ((n / (n - 1)) * meanNorm * meanNorm) / trRaw : 0;
    const vCov = column(fits.center.V, 0);
    const vCor = column(fits.standardize.V, 0);
    const vRaw = column(fits.none.V, 0);
    const cosCovCor = Math.abs(vCov.reduce((s, v, j) => s + v * vCor[j], 0));
    return { vars, stds, means, meanNorm, trRaw, trS, meanShare, vCov, vCor, vRaw, cosCovCor, varRatio: vars[0] / vars[1] };
  }, [X, fits, n]);

  const sel = fits[mode];
  const selV1 = column(sel.V, 0);
  const selL1 = sel.eigenvalues[0];
  const selPts = sel.Xc.map((row) => [row[0], row[1]]);
  const selArrowLen = 2 * Math.sqrt(selL1);
  const selMean: [number, number] = mode === 'none' ? [st.means[0], st.means[1]] : [0, 0];
  const selAxis: [string, string] = mode === 'none' ? ['x1 (original units, uncentred)', 'x2 (original units, uncentred)'] : mode === 'center' ? ['x1 − mean (original units)', 'x2 − mean (original units)'] : ['(x1 − mean)/s1', '(x2 − mean)/s2'];
  const selTitle = mode === 'none' ? 'Raw SVD of X: first singular direction through the origin' : mode === 'center' ? 'Covariance PCA: PC1 through the mean, original units' : 'Correlation PCA: PC1 through the mean, standardised units';

  const cov = fits.center;
  const cor = fits.standardize;
  const lC = cov.eigenvalues[0];
  const lR = cor.eigenvalues[0];
  const s = st.stds;
  const ptsA = cov.Xc.map((row) => [row[0], row[1]]);
  const ptsB = cor.Xc.map((row) => [row[0], row[1]]);
  const arrowsA = [
    { x: 2 * Math.sqrt(lC) * st.vCov[0], y: 2 * Math.sqrt(lC) * st.vCov[1], color: methodColor.PCA, label: 'PC1 (covariance)' },
    { x: 2 * Math.sqrt(lR) * st.vCor[0] * s[0], y: 2 * Math.sqrt(lR) * st.vCor[1] * s[1], color: accent2, dashed: true, label: 'PC1 (correlation), mapped' },
  ];
  const arrowsB = [
    { x: 2 * Math.sqrt(lR) * st.vCor[0], y: 2 * Math.sqrt(lR) * st.vCor[1], color: accent2, label: 'PC1 (correlation)' },
    { x: (2 * Math.sqrt(lC) * st.vCov[0]) / s[0], y: (2 * Math.sqrt(lC) * st.vCov[1]) / s[1], color: methodColor.PCA, dashed: true, label: 'PC1 (covariance), mapped' },
  ];
  const pcs = pcLabels(3);

  const fitSummary = (f: typeof cov, label: string) => (
    <div className="stats">
      <StatTile label={`${label}: λ₁ share`} value={pct(f.explained[0])} note={`λ₁ = ${fmt(f.eigenvalues[0], 3)}`} />
      <StatTile label="λ₂ share" value={pct(f.explained[1])} note={`λ₂ = ${fmt(f.eigenvalues[1], 3)}`} />
      <StatTile label="|v₁₁| (weight of x1 in PC1)" value={fmt(Math.abs(f.V[0][0]), 3)} note={`|v₂₁| = ${fmt(Math.abs(f.V[1][0]), 3)}, |v₃₁| = ${fmt(Math.abs(f.V[2][0]), 3)}`} />
    </div>
  );

  return (
    <Section
      id="pca-scaling"
      title="5 · Scaling and standardisation: which matrix should PCA diagonalise?"
      subtitle="Covariance PCA is not invariant to the units of the variables; correlation PCA is — at the price of discarding the information carried by the relative variances."
      right={<Badge method="PCA" />}
    >
      <div className="prose">
        <p>
          This lesson uses its own dataset: <M tex="n = 150" /> observations of three variables with an AR(1) correlation structure (<M tex="\rho = 0.6" /> between neighbours), each with unit variance in the
          population — except that <M tex="x_1" /> has been multiplied by a factor <M tex={`${factor}`} />, as if it were recorded in grams instead of kilograms, and <M tex="x_2" /> has been translated by{' '}
          <M tex={`\\mu = ${offset}`} />. Nothing about the <em>relationships</em> between the variables has changed; only their units and origin.
        </p>
      </div>

      <div className="grid side">
        <div className="stack">
          <Card title="Local dataset" plane>
            <Slider label="Scale factor for x1" value={logF} min={0} max={3} step={0.1} onChange={setLogF} format={() => `×${factor}`} hint="log scale: ×1 … ×1000" />
            <Slider label="Translation μ of x2" value={offset} min={0} max={10} step={0.5} onChange={setOffset} format={(v) => `+${v.toFixed(1)}`} hint="moves the cloud away from the origin — matters only without centring" />
            <div className="row">
              <Button small onClick={() => setSeed((v) => (v % 999) + 1)}>
                Resample (seed {seed})
              </Button>
            </div>
            <Segmented<Scaling>
              label="Matrix decomposed"
              value={mode}
              onChange={setMode}
              options={[
                { value: 'none', label: 'Raw', hint: 'SVD of X itself — no centring' },
                { value: 'center', label: 'Centre', hint: 'covariance PCA' },
                { value: 'standardize', label: 'Standardise', hint: 'correlation PCA' },
              ]}
            />
          </Card>
          <Card title="Sample moments" plane>
            <div className="stats">
              <StatTile label="s(x1)" value={fmt(s[0], 2)} note={`s(x2) = ${fmt(s[1], 2)}, s(x3) = ${fmt(s[2], 2)}`} />
              <StatTile label="Var(x1)/Var(x2)" value={fmt(st.varRatio, 1)} note={`≈ ${factor}² up to sampling`} />
              <StatTile label="‖x̄‖" value={fmt(st.meanNorm, 2)} note={`x̄ = (${st.means.map((m) => fmt(m, 1)).join(', ')})`} />
              <StatTile label="mean's share of tr(XᵀX)/(n−1)" value={pct(st.meanShare)} note="what an uncentred SVD 'explains' with location" />
            </div>
          </Card>
        </div>
        <div className="stack">
          <div className="grid side-r">
            <ScatterSVG
              points={selPts}
              pointColor={neutralMark}
              pointOpacity={0.65}
              width={520}
              height={400}
              xLabel={selAxis[0]}
              yLabel={selAxis[1]}
              title={selTitle}
              vectors={[{ x: selArrowLen * selV1[0], y: selArrowLen * selV1[1], from: [0, 0], color: methodColor.PCA, label: 'PC1' }]}
              lines={[{ angle: Math.atan2(selV1[1], selV1[0]), through: [0, 0], color: methodColor.PCA, dashed: true, opacity: 0.6 }]}
              extraPoints={[{ x: selMean[0], y: selMean[1], shape: 'cross', color: ink.primary, r: 6, label: 'x̄', labelPosition: 'right' }]}
              include={[[0, 0]]}
              caption={
                <>
                  Projection of the three-dimensional first direction <M tex="v_1" /> onto the <M tex="(x_1, x_2)" /> plane, drawn with length <M tex="2\sqrt{\lambda_1}" /> from the point the decomposition treats as
                  its origin. Equal aspect ratio: one unit of <M tex="x_1" /> is one unit of <M tex="x_2" />.
                </>
              }
            />
            <div className="stack">
              <MatrixView M={sel.S} title={mode === 'none' ? 'XᵀX/(n−1)' : mode === 'center' ? 'S (covariance)' : 'R (correlation)'} rowLabels={VAR} colLabels={VAR} digits={mode === 'center' && factor > 30 ? 0 : 3} heat="diverging" compact />
              <MatrixView M={sel.V} title={`V — ${scalingName(mode)}`} rowLabels={VAR} colLabels={pcs} digits={3} heat="diverging" highlightCols={[0]} compact />
              <div className="small">
                <M tex={`v_1 = ${texVector(selV1, 3)}`} />, <M tex={`\\lambda_1 = ${texNum(selL1, 3)}`} /> ({pct(sel.explained[0])} of <M tex={`\\operatorname{tr} = ${texNum(sel.totalVariance, 3)}`} />)
              </div>
            </div>
          </div>
        </div>
      </div>

      <Interpretation
        title={`Interpretation — ${scalingName(mode)} decomposition`}
        items={{
          seeing: (
            <>
              The {scalingName(mode)} decomposition of the local dataset (<M tex={decomposedMatrixTex(mode)} />). Its first direction is v₁ = ({selV1.map((v) => fmt(v, 3)).join(', ')}) with λ₁ = {fmt(selL1, 3)},{' '}
              {pct(sel.explained[0])} of the trace {fmt(sel.totalVariance, 3)}. {mode === 'none' ? `The cross marks the sample mean; the direction is anchored at the origin, ${fmt(st.meanNorm, 2)} units away from it.` : mode === 'center' ? `In original units the cloud is ${fmt(Math.sqrt(st.varRatio), 0)} times wider along x1 than along x2, so PC1 is essentially the x1 axis: |v₁₁| = ${fmt(Math.abs(selV1[0]), 3)}.` : `In standardised units all three variables have unit variance and PC1 is a balanced combination: |v₁₁| = ${fmt(Math.abs(selV1[0]), 3)}, |v₂₁| = ${fmt(Math.abs(selV1[1]), 3)}, |v₃₁| = ${fmt(Math.abs(selV1[2]), 3)}.`}
            </>
          ),
          why: (
            <>
              {mode === 'none' ? (
                <>
                  Without centring the diagonalised matrix is <M tex="\tfrac{1}{n-1}X^{T}X = S + \tfrac{n}{n-1}\bar x\bar x^{T}" />. The rank-one term <M tex="\bar x\bar x^{T}" /> has eigenvalue{' '}
                  <M tex={`\\tfrac{n}{n-1}\\|\\bar x\\|^{2} = ${texNum((n / (n - 1)) * st.meanNorm ** 2, 2)}`} /> along <M tex="\bar x" />, which is {pct(st.meanShare)} of the trace: the leading singular
                  direction is pulled towards the mean vector and the associated "variance explained" is largely location. Set the factor to ×1 and increase μ to see v₁ swing towards x₂.
                </>
              ) : mode === 'center' ? (
                <>
                  Covariance PCA maximises variance in the units the data come in. Multiplying x1 by {factor} multiplies its variance by {factor}² ≈ {fmt(st.varRatio, 0)}× that of x2, so the direction of maximal
                  variance must lie almost along x1 whatever the correlations are: λ₁ = {fmt(lC, 2)} is almost entirely Var(x1) = {fmt(st.vars[0], 2)}.
                </>
              ) : (
                <>
                  Dividing each centred column by its standard deviation removes the units: <M tex="R" /> has unit diagonal and its eigenvectors depend only on the correlations{' '}
                  <M tex={`r_{12} = ${texNum(cor.S[0][1], 3)},\\ r_{23} = ${texNum(cor.S[1][2], 3)},\\ r_{13} = ${texNum(cor.S[0][2], 3)}`} />. With positive neighbour correlations the leading eigenvector
                  weights all variables with the same sign, more heavily the middle one.
                </>
              )}
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\text{covariance: } S = \tfrac{1}{n-1}X_c^{T}X_c, \qquad \text{correlation: } R = D_s^{-1} S D_s^{-1}, \qquad \text{raw: } \tfrac{1}{n-1}X^{T}X = S + \tfrac{n}{n-1}\bar x\bar x^{T},`} />
              with <M tex="D_s = \operatorname{diag}(s_1, s_2, s_3)" />. Rescaling <M tex="x_1 \to c\,x_1" /> changes <M tex="S" /> to <M tex="D_cSD_c" /> — <em>not</em> a similarity transform — so eigenvectors
              and eigenvalues change; <M tex="R" /> is unchanged, hence scale-invariant.
            </>
          ),
          stats: (
            <>
              tr S = Σ Var(x_j) = {fmt(st.trS, 2)}, of which x1 alone contributes {pct(st.vars[0] / st.trS)}; tr R = 3 by construction, so under standardisation each variable is given exactly one unit of variance
              to "compete" with. The angle between the covariance and correlation first directions is {fmt((Math.acos(Math.min(1, st.cosCovCor)) * 180) / Math.PI, 1)}° (|cos| = {fmt(st.cosCovCor, 3)}).
            </>
          ),
          careful: (
            <>
              {mode === 'none'
                ? 'An uncentred SVD is a legitimate matrix factorisation but it is not PCA: PCA is defined on deviations from the mean. Quote "variance explained" only after centring.'
                : mode === 'center'
                  ? 'Reading "PC1 = x1" as a substantive finding would be an artefact of the unit chosen for x1. Whenever variables have different units, covariance PCA answers a question about units, not about structure.'
                  : 'Standardisation is a choice of weights, not a neutral act: it forces equal variance on variables whose natural spread may be scientifically meaningful, and it inflates the role of near-constant noisy variables.'}
            </>
          ),
        }}
      />

      <div className="grid c2">
        <Card title="(a) Covariance PCA — original units">
          <MatrixView M={cov.V} title="V (loadings)" rowLabels={VAR} colLabels={pcs} digits={3} heat="diverging" highlightCols={[0]} compact caption={`Column 1: |v₁₁| = ${fmt(Math.abs(st.vCov[0]), 3)} — PC1 is ${Math.abs(st.vCov[0]) > 0.95 ? 'almost entirely x1' : 'a mixture of the variables'}.`} />
          {fitSummary(cov, 'covariance')}
          <ScatterSVG
            points={ptsA}
            pointColor={neutralMark}
            pointOpacity={0.65}
            width={480}
            height={340}
            xLabel="x1 − mean (original units)"
            yLabel="x2 − mean (original units)"
            vectors={arrowsA}
            include={[[0, 0]]}
            caption={
              <>
                Equal-aspect view in original units: the cloud is stretched {fmt(Math.sqrt(st.varRatio), 0)}× along <M tex="x_1" />. Solid arrow: covariance PC1, length <M tex="2\sqrt{\lambda_1}" />. Dashed: the
                correlation PC1 mapped back to original units, <M tex="(u_1 s_1, u_2 s_2)" />.
              </>
            }
          />
        </Card>
        <Card title="(b) Correlation PCA — standardised units">
          <MatrixView M={cor.V} title="V (loadings)" rowLabels={VAR} colLabels={pcs} digits={3} heat="diverging" highlightCols={[0]} compact caption={`Column 1: |v₁₁| = ${fmt(Math.abs(st.vCor[0]), 3)}, |v₂₁| = ${fmt(Math.abs(st.vCor[1]), 3)}, |v₃₁| = ${fmt(Math.abs(st.vCor[2]), 3)} — a balanced combination.`} />
          {fitSummary(cor, 'correlation')}
          <ScatterSVG
            points={ptsB}
            pointColor={neutralMark}
            pointOpacity={0.65}
            width={480}
            height={340}
            xLabel="(x1 − mean)/s1"
            yLabel="(x2 − mean)/s2"
            vectors={arrowsB}
            include={[[0, 0]]}
            caption={
              <>
                The same observations after standardisation: both axes now have unit variance and the correlation <M tex={`r_{12} = ${texNum(cor.S[0][1], 2)}`} /> becomes visible as the tilt of the cloud. Solid: correlation
                PC1. Dashed: covariance PC1 mapped into standardised units, <M tex="(v_1/s_1, v_2/s_2)" />.
              </>
            }
          />
        </Card>
      </div>

      <Interpretation
        title="Interpretation — covariance versus correlation PCA"
        items={{
          seeing: (
            <>
              The same 150 observations analysed twice. On the covariance matrix PC1 has |v₁₁| = {fmt(Math.abs(st.vCov[0]), 3)} and explains {pct(cov.explained[0])}; on the correlation matrix |v₁₁| ={' '}
              {fmt(Math.abs(st.vCor[0]), 3)} and PC1 explains {pct(cor.explained[0])}. The leading direction flips from "{Math.abs(st.vCov[0]) > 0.95 ? 'almost entirely x1' : 'x1-dominated'}" to a combination in
              which x2 carries weight {fmt(Math.abs(st.vCor[1]), 3)} — the variable along which the two hidden groups of this dataset actually differ.
            </>
          ),
          why: (
            <>
              Covariance PCA weights each variable by its variance, and Var(x1)/Var(x2) = {fmt(st.varRatio, 0)}; the dashed correlation direction, mapped into original units, is visually indistinguishable from the
              x1 axis in panel (a) because one standardised unit of x2 is only 1/{fmt(s[0] / s[1], 0)} of a unit of x1. In panel (b) the covariance direction, mapped into standardised units, collapses onto the x1
              axis for the mirror-image reason: it has essentially no x2 content to begin with.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`R = D_s^{-1} S D_s^{-1}, \qquad R u = \lambda^{R} u \;\not\Rightarrow\; S(D_s^{-1}u) \propto D_s^{-1}u :`} />
              the eigenvectors of <M tex="R" /> and of <M tex="S" /> are not related by a simple mapping (a congruence <M tex="D_s^{-1}SD_s^{-1}" /> preserves inertia, not eigenvectors). Here{' '}
              <M tex={`|\\cos\\angle(v_1^{S}, v_1^{R})| = ${texNum(st.cosCovCor, 3)}`} /> as directions in <M tex="\mathbb R^{3}" />, and the eigenvalue spectra ({cov.eigenvalues.map((l) => fmt(l, 2)).join(', ')})
              versus ({cor.eigenvalues.map((l) => fmt(l, 3)).join(', ')}) are not rescalings of one another.
            </>
          ),
          stats: (
            <>
              Use the <b>covariance</b> matrix when the variables share a unit and their variances are meaningful — e.g. all lengths in cm, spectral intensities, log-returns — because then "large variance" is a
              substantive statement and standardising would throw it away. Use the <b>correlation</b> matrix when units differ or scales are arbitrary (questionnaire items, mixed physical quantities), because then
              the covariance PCA merely reports the unit choice, as here with ×{factor}. Centring is required in both cases: an uncentred decomposition describes the mean (currently {pct(st.meanShare)} of
              the raw trace), not the dispersion.
            </>
          ),
          careful: (
            <>
              Standardisation makes PCA invariant to the units of each variable but not to the choice of variables, and it destroys the information in the relative variances: after standardising, a noisy
              near-constant variable counts as much as a highly informative one. Neither choice is "correct" in general — the decision must be argued from the measurement scales, and reported. Also
              remember that the labels of this dataset (two groups shifted along x2) played no role: PCA did not "find" the groups under standardisation, it merely stopped hiding x2.
            </>
          ),
        }}
      />

      <div className="grid c3">
        <Callout kind="definition" title="Covariance PCA">
          Diagonalise <M tex="S" />. Appropriate when all variables are measured in the same unit and their variances are comparable and meaningful. The eigenvalues are variances in those units; rescaling a
          variable changes the answer.
        </Callout>
        <Callout kind="definition" title="Correlation PCA">
          Diagonalise <M tex="R" />, equivalently run PCA on <M tex="X_s = X_c D_s^{-1}" />. Appropriate when units differ or are arbitrary. Scale-invariant; <M tex="\operatorname{tr} R = p" /> so eigenvalues
          are in "number of variables" units. Discards the relative-variance information.
        </Callout>
        <Callout kind="warning" title="Centring is not optional">
          PCA proper is defined on <M tex="X_c" />. The SVD of uncentred <M tex="X" /> diagonalises <M tex="S + \tfrac{n}{n-1}\bar x\bar x^{T}" />: its first direction points towards the mean, and its
          "variance explained" is inflated by location — here by {pct(st.meanShare)} of the raw trace. Centre first; standardise only when the units call for it.
        </Callout>
      </div>
    </Section>
  );
}
