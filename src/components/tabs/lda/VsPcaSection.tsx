import { useMemo, useState } from 'react';
import type { Layout } from 'plotly.js';
import { column, fmt } from '../../../lib/linalg';
import { fisherCriterion, lda } from '../../../lib/lda';
import { pca, projectOnDirection, sampleVariance } from '../../../lib/pca';
import { defaultParams, generateDataset } from '../../../lib/datasets';
import { ink, methodColor } from '../../../lib/theme';
import { M, MBlock } from '../../common/Math';
import { Slider } from '../../common/Controls';
import { Badge, Card, ClassLegend, Interpretation, Section, StatTile } from '../../common/Panels';
import { Plot } from '../../common/Plot';
import { ScatterSVG } from '../../common/ScatterSVG';
import { axialAngle, axialAngleBetween, classHistograms, classProjections, defaultClassNames, fmtDeg, meanLineShapes, midpointThreshold, pct, unitFromAngle } from './helpers';

/** Lesson 3 — variance is not separation: PCA and LDA side by side on a dataset built to tell them apart. */
export default function VsPcaSection() {
  const [sep, setSep] = useState(3);
  const [seed, setSeed] = useState(7);

  const ds = useMemo(() => generateDataset('pcaVsLda', { ...defaultParams, n: 200, separation: sep, variance: 1, seed }), [sep, seed]);
  const y = useMemo(() => ds.y ?? [], [ds]);
  const classNames = ds.classNames ?? defaultClassNames(2);
  const pcaRes = useMemo(() => pca(ds.X, 'center'), [ds]);
  const ldaRes = useMemo(() => lda(pcaRes.Xc, y), [pcaRes, y]);
  const v1 = useMemo(() => column(pcaRes.V, 0), [pcaRes]);
  const w1 = useMemo(() => (ldaRes.maxDims > 0 ? column(ldaRes.W, 0) : [1, 0]), [ldaRes]);

  const zP = useMemo(() => projectOnDirection(pcaRes.Xc, v1), [pcaRes, v1]);
  const zL = useMemo(() => projectOnDirection(pcaRes.Xc, w1), [pcaRes, w1]);
  const varP = useMemo(() => sampleVariance(zP), [zP]);
  const varL = useMemo(() => sampleVariance(zL), [zL]);
  const fcP = useMemo(() => fisherCriterion(ldaRes.SB, ldaRes.SW, v1), [ldaRes, v1]);
  const fcL = useMemo(() => fisherCriterion(ldaRes.SB, ldaRes.SW, w1), [ldaRes, w1]);
  const ruleP = useMemo(() => midpointThreshold(zP, y, ldaRes.classes), [zP, y, ldaRes]);
  const ruleL = useMemo(() => midpointThreshold(zL, y, ldaRes.classes), [zL, y, ldaRes]);
  const projP = useMemo(() => classProjections(zP, y, ldaRes.classes, classNames), [zP, y, ldaRes, classNames]);
  const projL = useMemo(() => classProjections(zL, y, ldaRes.classes, classNames), [zL, y, ldaRes, classNames]);
  const histP = useMemo(() => classHistograms(zP, y, ldaRes.classes, classNames), [zP, y, ldaRes, classNames]);
  const histL = useMemo(() => classHistograms(zL, y, ldaRes.classes, classNames), [zL, y, ldaRes, classNames]);
  const angle = axialAngleBetween(v1, w1);
  const thetaP = axialAngle(Math.atan2(v1[1], v1[0]));
  const thetaL = axialAngle(Math.atan2(w1[1], w1[0]));
  const dirP = unitFromAngle(thetaP);
  const dirL = unitFromAngle(thetaL);
  const halfRange = useMemo(() => {
    let h = 0;
    for (const r of pcaRes.Xc) h = Math.max(h, Math.abs(r[0]), Math.abs(r[1]));
    return h || 1;
  }, [pcaRes]);
  const L = halfRange * 0.5;
  const totalVar = pcaRes.totalVariance;

  const layoutP = useMemo<Record<string, unknown>>(
    () => ({ barmode: 'overlay', showlegend: true, xaxis: { title: { text: 'z = X_c v₁ (PC1 score)' } }, yaxis: { title: { text: 'count' } }, shapes: [...meanLineShapes(projP), thresholdShape(ruleP.threshold)] }),
    [projP, ruleP.threshold],
  );
  const layoutL = useMemo<Record<string, unknown>>(
    () => ({ barmode: 'overlay', showlegend: true, xaxis: { title: { text: 'z = X_c w₁ (LDA score)' } }, yaxis: { title: { text: 'count' } }, shapes: [...meanLineShapes(projL), thresholdShape(ruleL.threshold)] }),
    [projL, ruleL.threshold],
  );

  const gapP = projP.length >= 2 ? Math.abs(projP[0].mean - projP[1].mean) : NaN;
  const gapL = projL.length >= 2 ? Math.abs(projL[0].mean - projL[1].mean) : NaN;
  const pooledP = Math.sqrt(fcP.within / Math.max(ldaRes.n - ldaRes.K, 1));
  const pooledL = Math.sqrt(fcL.within / Math.max(ldaRes.n - ldaRes.K, 1));

  return (
    <Section
      id="lda-vs-pca"
      title="3 · PCA versus LDA: the direction of maximum variance is not the direction of maximum separation"
      subtitle="A local two-class dataset whose classes are elongated along one axis and separated along another"
      right={
        <span className="row">
          <Badge method="PCA" />
          <Badge method="LDA" />
        </span>
      }
    >
      <div className="grid c2">
        <Card plane>
          <div className="prose">
            <p>
              <Badge method="PCA" /> <b>PCA asks:</b> “In which directions does the data vary the most?” It maximises <M tex="w^{\mathsf T} S w" /> and never looks at the labels. Variance
              caused by within-class spread counts exactly as much as variance caused by class differences.
            </p>
          </div>
        </Card>
        <Card plane>
          <div className="prose">
            <p>
              <Badge method="LDA" /> <b>LDA asks:</b> “In which directions are the classes separated the most relative to within-class variation?” It maximises{' '}
              <M tex="J(w) = w^{\mathsf T} S_B w / w^{\mathsf T} S_W w" />: a direction with little total variance can win if almost all of that variance is between the class means.
            </p>
          </div>
        </Card>
      </div>

      <div className="grid c2">
        <Slider label="Class separation (in units of the thin within-class sd)" value={sep} min={0.5} max={8} step={0.25} onChange={setSep} />
        <Slider label="Random seed" value={seed} min={1} max={999} step={1} onChange={setSeed} />
      </div>

      <div className="grid c3">
        <div>
          <ScatterSVG
            points={pcaRes.Xc}
            labels={y}
            classNames={classNames}
            width={440}
            height={400}
            xLabel="x₁ (centred)"
            yLabel="x₂ (centred)"
            title="Both directions on the centred data"
            lines={[
              { angle: thetaP, color: methodColor.PCA, dashed: true, opacity: 0.55 },
              { angle: thetaL, color: methodColor.LDA, dashed: true, opacity: 0.55 },
            ]}
            vectors={[
              { x: L * dirP[0], y: L * dirP[1], color: methodColor.PCA, label: 'PC1' },
              { x: L * dirL[0], y: L * dirL[1], color: methodColor.LDA, label: 'LDA' },
            ]}
            caption={
              <>
                <M tex={`n = ${ldaRes.n}`} />, two classes with a common covariance. PC1 (blue) follows the long axis of the cloud; the Fisher direction (orange) is {angle.toFixed(1)}° away
                from it.
              </>
            }
          />
          <ClassLegend
            classNames={ldaRes.classes.map((c) => classNames[c] ?? `class ${c}`)}
            extra={
              <>
                <span className="item">
                  <span style={{ display: 'inline-block', width: 16, height: 3, background: methodColor.PCA, borderRadius: 2 }} aria-hidden /> PC1
                </span>
                <span className="item">
                  <span style={{ display: 'inline-block', width: 16, height: 3, background: methodColor.LDA, borderRadius: 2 }} aria-hidden /> LDA
                </span>
              </>
            }
          />
        </div>
        <div className="stack">
          <Plot data={histP} layout={layoutP} height={300} title="Projection onto PC1 (unsupervised choice)" />
          <div className="stats">
            <StatTile label={<M tex="\operatorname{Var}(X_c v_1) = \lambda_1(S)" />} value={fmt(varP, 3)} note={`${pct(varP / totalVar)} of the total variance`} />
            <StatTile label={<M tex="J(v_1)" />} value={fmt(fcP.J, 3)} note={`between ${fmt(fcP.between, 1)} / within ${fmt(fcP.within, 1)}`} />
            <StatTile label="midpoint-rule accuracy" value={pct(ruleP.accuracy)} note={`threshold at z = ${fmt(ruleP.threshold, 2)}; ${ruleP.correct}/${ldaRes.n} correct`} />
          </div>
        </div>
        <div className="stack">
          <Plot data={histL} layout={layoutL} height={300} title="Projection onto w₁ (supervised choice)" />
          <div className="stats">
            <StatTile label={<M tex="\operatorname{Var}(X_c w_1) = w_1^{\mathsf T} S w_1" />} value={fmt(varL, 3)} note={`${pct(varL / totalVar)} of the total variance`} />
            <StatTile label={<M tex="J(w_1) = \lambda_1" />} value={fmt(fcL.J, 3)} note={`between ${fmt(fcL.between, 1)} / within ${fmt(fcL.within, 1)}`} />
            <StatTile label="midpoint-rule accuracy" value={pct(ruleL.accuracy)} note={`threshold at z = ${fmt(ruleL.threshold, 2)}; ${ruleL.correct}/${ldaRes.n} correct`} />
          </div>
        </div>
      </div>

      <div className="table-wrap">
        <table className="summary-table">
          <thead>
            <tr>
              <th />
              <th>
                <Badge method="PCA" /> PC1 direction <M tex="v_1" />
              </th>
              <th>
                <Badge method="LDA" /> Fisher direction <M tex="w_1" />
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Angle of the direction</td>
              <td>{fmtDeg(thetaP)}</td>
              <td>{fmtDeg(thetaL)}</td>
            </tr>
            <tr>
              <td>
                Variance along the direction <M tex="w^{\mathsf T} S w" />
              </td>
              <td>
                {fmt(varP, 3)} ({pct(varP / totalVar)} of total)
              </td>
              <td>
                {fmt(varL, 3)} ({pct(varL / totalVar)} of total)
              </td>
            </tr>
            <tr>
              <td>
                Between-class scatter <M tex="w^{\mathsf T} S_B w" />
              </td>
              <td>{fmt(fcP.between, 2)}</td>
              <td>{fmt(fcL.between, 2)}</td>
            </tr>
            <tr>
              <td>
                Within-class scatter <M tex="w^{\mathsf T} S_W w" />
              </td>
              <td>{fmt(fcP.within, 2)}</td>
              <td>{fmt(fcL.within, 2)}</td>
            </tr>
            <tr>
              <td>
                Fisher criterion <M tex="J(w)" />
              </td>
              <td>{fmt(fcP.J, 3)}</td>
              <td>{fmt(fcL.J, 3)}</td>
            </tr>
            <tr>
              <td>Gap between projected class means / pooled sd along the line</td>
              <td>
                {fmt(gapP, 2)} / {fmt(pooledP, 2)} = {fmt(gapP / pooledP, 2)}
              </td>
              <td>
                {fmt(gapL, 2)} / {fmt(pooledL, 2)} = {fmt(gapL / pooledL, 2)}
              </td>
            </tr>
            <tr>
              <td>Accuracy of the midpoint threshold (training data)</td>
              <td>{pct(ruleP.accuracy)}</td>
              <td>{pct(ruleL.accuracy)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <MBlock tex="(n-1)\,S = S_W + S_B \quad\Longrightarrow\quad w^{\mathsf T} S w \text{ is large whenever } w^{\mathsf T} S_W w \text{ is large, even if } w^{\mathsf T} S_B w = 0 ." />

      <Interpretation
        items={{
          seeing: (
            <>
              Left: the same <M tex={`n = ${ldaRes.n}`} /> points with the two competing directions, {angle.toFixed(1)}° apart. Middle: projected onto PC1 the two class histograms sit almost on
              top of each other (projected means {fmt(gapP, 2)} apart against a pooled within-class sd of {fmt(pooledP, 2)}); a midpoint threshold classifies {pct(ruleP.accuracy)} of the
              training points correctly. Right: projected onto <M tex="w_1" /> the histograms separate (gap {fmt(gapL, 2)} against sd {fmt(pooledL, 2)}), and the same simple rule reaches{' '}
              {pct(ruleL.accuracy)}.
            </>
          ),
          why: (
            <>
              The generator elongates both classes along one axis and separates the means along the thin orthogonal axis. PC1 captures {pct(varP / totalVar)} of the total variance (
              <M tex={`\\lambda_1(S) = ${fmt(varP, 3)}`} />) but almost all of it is within-class: <M tex={`v_1^{\\mathsf T} S_W v_1 = ${fmt(fcP.within, 1)}`} /> against{' '}
              <M tex={`v_1^{\\mathsf T} S_B v_1 = ${fmt(fcP.between, 1)}`} />, so <M tex={`J(v_1) = ${fmt(fcP.J, 3)}`} />. The Fisher direction keeps only {pct(varL / totalVar)} of the variance
              (<M tex={`${fmt(varL, 3)}`} />) but <M tex={`J(w_1) = ${fmt(fcL.J, 3)}`} />, {fcP.J > 0 ? `${fmt(fcL.J / fcP.J, 0)}×` : 'infinitely'} larger.
            </>
          ),
          math: (
            <>
              PCA maximises <M tex="w^{\mathsf T} S w" /> subject to <M tex="\|w\| = 1" />; LDA maximises the ratio <M tex="w^{\mathsf T} S_B w / w^{\mathsf T} S_W w" />, which is invariant to
              the scale of <M tex="w" /> and to any linear rescaling of the variables. Because <M tex="(n-1)S = S_W + S_B" />, the variance along a direction can be dominated by{' '}
              <M tex="S_W" />; PCA cannot tell the two contributions apart, LDA divides by the within part. With two classes <M tex="S_B" /> has rank one and{' '}
              <M tex="w_1 \propto S_W^{-1}(m_1 - m_2)" /> (lesson 5): the within-class covariance rotates the naive mean-difference direction towards the thin axis.
            </>
          ),
          stats: (
            <>
              The midpoint threshold is the two-class Gaussian LDA rule with equal priors applied to one coordinate. Its training accuracy on PC1, {pct(ruleP.accuracy)}, is close to what
              guessing the larger class would give; on <M tex="w_1" /> it is {pct(ruleL.accuracy)}. Increase the separation slider and watch <M tex="J(w_1)" /> grow roughly with the square of the
              separation while <M tex="\lambda_1(S)" /> barely moves: PCA is almost blind to the change that matters for classification.
            </>
          ),
          careful: (
            <>
              This dataset is constructed to make the disagreement extreme; when class differences happen to lie along the high-variance axis, PCA and LDA nearly coincide (compare lesson 2
              on the active dataset). Accuracies here are computed on the {ldaRes.n} training points and are optimistic. The PC1 sign and the sign of <M tex="w_1" /> are arbitrary; only the
              lines matter. Reducing the seed slider re-draws the sample: the qualitative conclusion is stable, the numbers are not.
            </>
          ),
        }}
      />
    </Section>
  );
}

function thresholdShape(t: number) {
  return { type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: t, x1: t, y0: 0, y1: 1, line: { color: ink.primary, width: 1.5, dash: 'dot' as const } };
}
