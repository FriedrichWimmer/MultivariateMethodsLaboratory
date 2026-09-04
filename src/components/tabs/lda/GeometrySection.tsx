import { useEffect, useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { fmt, quadForm } from '../../../lib/linalg';
import { fisherCriterion, type LDAResult } from '../../../lib/lda';
import { projectOnDirection } from '../../../lib/pca';
import { classColor, ink, methodColor } from '../../../lib/theme';
import { M, MBlock } from '../../common/Math';
import { Button, Select } from '../../common/Controls';
import { Badge, Callout, ClassLegend, Interpretation, Section, StatTile } from '../../common/Panels';
import { Plot } from '../../common/Plot';
import { ScatterSVG } from '../../common/ScatterSVG';
import { axialAngleBetween, classHistograms, classLetter, classProjections, deg, fisherCurve, fmtDeg, meanLineShapes, pct, unitFromAngle, type LabelledData, type TwoD } from './helpers';

interface Props {
  source: LabelledData;
  /** LDA on all p variables (for comparison with the 2-D optimum) */
  res: LDAResult;
  twoD: TwoD;
  pair: [number, number];
  setPair: (p: [number, number]) => void;
}

/** Lesson 2 — the geometry of the Fisher criterion in two dimensions, with a draggable direction. */
export default function GeometrySection({ source, res, twoD, pair, setPair }: Props) {
  const [thetaState, setThetaState] = useState<number | null>(null);
  // a new dataset or variable pair resets the handle to the first principal direction
  useEffect(() => setThetaState(null), [twoD]);
  const theta = thetaState ?? twoD.thetaPCA;

  const { X2, y, classNames, res: res2, cov } = twoD;
  const { n, K } = res2;
  const w = useMemo(() => unitFromAngle(theta), [theta]);
  const z = useMemo(() => projectOnDirection(X2, w), [X2, w]);
  const fc = useMemo(() => fisherCriterion(res2.SB, res2.SW, w), [res2, w]);
  const varW = useMemo(() => quadForm(cov, w), [cov, w]);
  const projs = useMemo(() => classProjections(z, y, res2.classes, classNames), [z, y, res2, classNames]);
  const curve = useMemo(() => fisherCurve(res2.SB, res2.SW, cov), [res2, cov]);
  const fcLDA = useMemo(() => fisherCriterion(res2.SB, res2.SW, twoD.w1), [res2, twoD.w1]);
  const fcPCA = useMemo(() => fisherCriterion(res2.SB, res2.SW, twoD.v1), [res2, twoD.v1]);
  const varLDA = useMemo(() => quadForm(cov, twoD.w1), [cov, twoD.w1]);
  const varPCA = twoD.covEigen.values[0] ?? NaN;
  const angleBetween = axialAngleBetween(twoD.w1, twoD.v1);
  const minJ = useMemo(() => Math.min(...curve.J), [curve]);
  const F = K > 1 ? (fc.J * (n - K)) / (K - 1) : NaN;

  const L = twoD.halfRange * 0.55;
  const ldaDir = useMemo(() => unitFromAngle(twoD.thetaLDA), [twoD.thetaLDA]);
  const pcaDir = useMemo(() => unitFromAngle(twoD.thetaPCA), [twoD.thetaPCA]);

  const varOptions = source.variableNames.map((v, j) => ({ value: String(j), label: v }));
  const histData = useMemo(() => classHistograms(z, y, res2.classes, classNames), [z, y, res2, classNames]);
  const histLayout = useMemo<Record<string, unknown>>(
    () => ({ barmode: 'overlay', showlegend: true, xaxis: { title: { text: 'projected value  z = X_c w' } }, yaxis: { title: { text: 'count' } }, shapes: meanLineShapes(projs) }),
    [projs],
  );

  const curveData = useMemo<Data[]>(
    () => [
      { x: curve.thetaDeg, y: curve.J, type: 'scatter', mode: 'lines', name: 'J(θ) — Fisher criterion', line: { color: ink.primary, width: 2 }, hovertemplate: 'θ = %{x}°<br>J = %{y:.3f}<extra></extra>' },
      { x: [deg(twoD.thetaLDA)], y: [fcLDA.J], type: 'scatter', mode: 'markers', name: 'Fisher optimum (LDA)', marker: { color: methodColor.LDA, size: 12, symbol: 'diamond' }, hovertemplate: 'LDA: θ = %{x:.1f}°, J = %{y:.3f}<extra></extra>' },
      { x: [deg(twoD.thetaPCA)], y: [fcPCA.J], type: 'scatter', mode: 'markers', name: 'PC1 (maximum variance)', marker: { color: methodColor.PCA, size: 11, symbol: 'square' }, hovertemplate: 'PC1: θ = %{x:.1f}°, J = %{y:.3f}<extra></extra>' },
      { x: [deg(theta)], y: [fc.J], type: 'scatter', mode: 'markers', name: 'current w', marker: { color: ink.primary, size: 13, symbol: 'x' }, hovertemplate: 'current: θ = %{x:.1f}°, J = %{y:.3f}<extra></extra>' },
    ],
    [curve, twoD.thetaLDA, twoD.thetaPCA, fcLDA.J, fcPCA.J, theta, fc.J],
  );
  const curveLayout = useMemo<Record<string, unknown>>(
    () => ({
      showlegend: true,
      xaxis: { title: { text: 'direction angle θ (degrees)' }, range: [-90, 90], dtick: 30 },
      yaxis: { title: { text: 'J(θ)' }, rangemode: 'tozero' },
    }),
    [],
  );

  const varianceData = useMemo<Data[]>(
    () => [
      { x: curve.thetaDeg, y: curve.variance, type: 'scatter', mode: 'lines', name: 'wᵀSw — variance along w', line: { color: methodColor.PCA, width: 2 }, hovertemplate: 'θ = %{x}°<br>variance = %{y:.3f}<extra></extra>' },
      { x: [deg(twoD.thetaPCA)], y: [varPCA], type: 'scatter', mode: 'markers', name: 'PC1 (λ₁, maximum variance)', marker: { color: methodColor.PCA, size: 11, symbol: 'square' }, hovertemplate: 'PC1: θ = %{x:.1f}°, variance = %{y:.3f}<extra></extra>' },
      { x: [deg(twoD.thetaLDA)], y: [varLDA], type: 'scatter', mode: 'markers', name: 'Fisher optimum (LDA)', marker: { color: methodColor.LDA, size: 12, symbol: 'diamond' }, hovertemplate: 'LDA: θ = %{x:.1f}°, variance = %{y:.3f}<extra></extra>' },
      { x: [deg(theta)], y: [varW], type: 'scatter', mode: 'markers', name: 'current w', marker: { color: ink.primary, size: 13, symbol: 'x' }, hovertemplate: 'current: θ = %{x:.1f}°, variance = %{y:.3f}<extra></extra>' },
    ],
    [curve, twoD.thetaPCA, twoD.thetaLDA, varPCA, varLDA, theta, varW],
  );
  const varianceLayout = useMemo<Record<string, unknown>>(
    () => ({ showlegend: true, xaxis: { title: { text: 'direction angle θ (degrees)' }, range: [-90, 90], dtick: 30 }, yaxis: { title: { text: 'wᵀSw' }, rangemode: 'tozero' } }),
    [],
  );

  const letterMap = res2.classes.map((c, k) => `${classLetter(k)} = ${classNames[c] ?? `class ${c}`}`).join(', ');
  const meansText = projs.map((pr, k) => `${classLetter(k)}: ${fmt(pr.mean, 2)}`).join(', ');
  const sdText = projs.map((pr, k) => `${classLetter(k)}: ${fmt(pr.sd, 2)}`).join(', ');

  return (
    <Section
      id="lda-geometry"
      title="2 · The geometry of the Fisher criterion"
      subtitle="Project two variables onto a direction w, watch the class means and within-class spread along the line, and find the direction that maximises J(w)"
      right={<Badge method="LDA" />}
    >
      <div className="prose">
        <p>
          Projecting an observation onto a unit direction <M tex="w" /> gives the scalar <M tex="z_i = w^{\mathsf T}(x_i - m)" />. The projected class means are <M tex="w^{\mathsf T}(m_k - m)" />{' '}
          and the two scatter quantities become sums of squares along the line:
        </p>
        <MBlock tex="w^{\mathsf T} S_B w = \sum_k n_k \bigl(w^{\mathsf T}(m_k - m)\bigr)^2, \qquad w^{\mathsf T} S_W w = \sum_k \sum_{i\in k}\bigl(w^{\mathsf T}(x_i - m_k)\bigr)^2 ." />
        <p>
          Drag the handle (or click anywhere in the plot) to rotate <M tex="w" />. The handle starts at the first principal direction; the buttons snap it to the Fisher optimum or back to PC1. Only
          the two selected variables are used here, centred at their grand mean, so the 2-D optimum need not equal the full-<M tex="p" /> optimum from lesson 1.
        </p>
      </div>

      {twoD.local && (
        <Callout kind="info" title="Local stand-in data">
          The active dataset is unlabelled; this lesson uses {twoD.sourceName} (three spherical clusters, <M tex="n = 150" />, <M tex="p = 2" />).
        </Callout>
      )}

      <div className="row" style={{ alignItems: 'flex-end' }}>
        <div style={{ minWidth: 200 }}>
          <Select<string>
            label="Horizontal variable"
            value={String(pair[0])}
            options={varOptions.map((o) => ({ ...o, disabled: Number(o.value) === pair[1] }))}
            onChange={(v) => setPair([Number(v), pair[1]])}
          />
        </div>
        <div style={{ minWidth: 200 }}>
          <Select<string>
            label="Vertical variable"
            value={String(pair[1])}
            options={varOptions.map((o) => ({ ...o, disabled: Number(o.value) === pair[0] }))}
            onChange={(v) => setPair([pair[0], Number(v)])}
          />
        </div>
        <Button primary small onClick={() => setThetaState(twoD.thetaLDA)}>
          Find the optimum
        </Button>
        <Button small onClick={() => setThetaState(twoD.thetaPCA)}>
          Snap to PC1
        </Button>
        <Button small onClick={() => setThetaState(0)}>
          Horizontal axis
        </Button>
      </div>

      <div className="grid c2">
        <div>
          <ScatterSVG
            points={X2}
            labels={y}
            classNames={classNames}
            width={520}
            height={440}
            xLabel={`${twoD.names[0]} (centred)`}
            yLabel={`${twoD.names[1]} (centred)`}
            title={`Direction w at θ = ${fmtDeg(theta)} — drag the handle`}
            direction={{ angle: theta, onChange: setThetaState, axial: true, label: 'w', color: ink.primary }}
            vectors={[
              { x: L * ldaDir[0], y: L * ldaDir[1], color: methodColor.LDA, label: 'LDA' },
              { x: L * pcaDir[0], y: L * pcaDir[1], color: methodColor.PCA, label: 'PC1' },
            ]}
            extraPoints={projs.map((pr, k) => ({
              x: pr.mean * w[0],
              y: pr.mean * w[1],
              shape: 'class' as const,
              classIndex: pr.label,
              color: classColor(pr.label),
              r: 6,
              label: `m_${classLetter(k)}`,
              labelPosition: 'above' as const,
            }))}
            render={(sx, sy) => (
              <g>
                {projs.map((pr, k) => {
                  const off = (k - (K - 1) / 2) * 7;
                  const ox = off * Math.sin(theta);
                  const oy = off * Math.cos(theta);
                  const lo = pr.mean - pr.sd;
                  const hi = pr.mean + pr.sd;
                  return (
                    <line
                      key={pr.label}
                      x1={sx(lo * w[0]) + ox}
                      y1={sy(lo * w[1]) + oy}
                      x2={sx(hi * w[0]) + ox}
                      y2={sy(hi * w[1]) + oy}
                      stroke={classColor(pr.label)}
                      strokeWidth={4}
                      strokeLinecap="round"
                      opacity={0.85}
                    />
                  );
                })}
              </g>
            )}
            caption={
              <>
                Class markers show the projected class means <M tex="m_A, m_B, \dots" /> on the line ({letterMap}); the thick bars extend one within-class standard deviation of <M tex="z" /> on
                either side of each projected mean. Arrows: Fisher direction (LDA) and first principal direction (PC1).
              </>
            }
          />
          <ClassLegend
            classNames={res2.classes.map((c) => classNames[c] ?? `class ${c}`)}
            extra={
              <>
                <span className="item">
                  <span style={{ display: 'inline-block', width: 16, height: 3, background: methodColor.LDA, borderRadius: 2 }} aria-hidden /> LDA direction
                </span>
                <span className="item">
                  <span style={{ display: 'inline-block', width: 16, height: 3, background: methodColor.PCA, borderRadius: 2 }} aria-hidden /> PC1
                </span>
              </>
            }
          />
        </div>
        <div className="stack">
          <div className="stats">
            <StatTile label={<M tex="\theta" />} value={fmtDeg(theta)} note="axial: w and −w are the same direction" />
            <StatTile label={<M tex="w^{\mathsf T} S_B w" />} value={fmt(fc.between, 2)} note="between-class scatter along w" />
            <StatTile label={<M tex="w^{\mathsf T} S_W w" />} value={fmt(fc.within, 2)} note="within-class scatter along w" />
            <StatTile label={<M tex="J(w)" />} value={fmt(fc.J, 3)} note={`optimum λ₁ = ${fmt(fcLDA.J, 3)} at θ = ${fmtDeg(twoD.thetaLDA)}`} />
            <StatTile label={<M tex="w^{\mathsf T} S w" />} value={fmt(varW, 3)} note={`maximum ${fmt(varPCA, 3)} at PC1, θ = ${fmtDeg(twoD.thetaPCA)}`} />
            <StatTile label="LDA vs PC1" value={`${angleBetween.toFixed(1)}°`} note="angle between the two directions" />
          </div>
          <Plot data={histData} layout={histLayout} height={300} title="Projected values z = X_c w, one histogram per class" />
        </div>
      </div>

      <div className="grid c2">
        <Plot data={curveData} layout={curveLayout} height={320} title="Fisher criterion J(θ) along w(θ) = (cos θ, sin θ)" />
        <Plot data={varianceData} layout={varianceLayout} height={320} title="Variance wᵀSw along the same directions (separate axis, same θ)" />
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              The dashed line is the candidate direction at <M tex={`\\theta = ${deg(theta).toFixed(1)}^\\circ`} />. Along it the projected class means are at {meansText} and the within-class
              standard deviations of <M tex="z" /> are {sdText}; the histograms show the same projected values per class. The lower chart traces <M tex="J(\theta)" /> for every direction in
              the plane: its maximum <M tex={`${fmt(fcLDA.J, 3)}`} /> is at the Fisher direction (<M tex={`\\theta = ${deg(twoD.thetaLDA).toFixed(1)}^\\circ`} />), its minimum is{' '}
              <M tex={`${fmt(minJ, 3)}`} />, and the dotted curve is the variance <M tex="w^{\mathsf T} S w" />, which peaks at PC1 (<M tex={`\\theta = ${deg(twoD.thetaPCA).toFixed(1)}^\\circ`} />).
            </>
          ),
          why: (
            <>
              <M tex={`J(w) = ${fmt(fc.between, 2)} / ${fmt(fc.within, 2)} = ${fmt(fc.J, 3)}`} /> compares the spread of the projected means with the spread inside the projected classes.
              Rotating <M tex="w" /> changes both: a direction with large total variance (<M tex="w^{\mathsf T} S w" /> large) is only useful if that variance comes from <M tex="S_B" />{' '}
              rather than <M tex="S_W" />. PC1 has variance <M tex={`${fmt(varPCA, 3)}`} /> but <M tex={`J = ${fmt(fcPCA.J, 3)}`} />; the Fisher direction has variance{' '}
              <M tex={`${fmt(varLDA, 3)}`} /> and <M tex={`J = ${fmt(fcLDA.J, 3)}`} />. The two directions differ by {angleBetween.toFixed(1)}°
              {angleBetween < 10 ? ' — on this dataset the high-variance direction happens to separate the classes as well.' : ' — maximal variance and maximal separation are different questions.'}
            </>
          ),
          math: (
            <>
              In two dimensions <M tex="J(\theta)" /> is <M tex="\pi" />-periodic, so <M tex="\theta \in (-90^\circ, 90^\circ]" /> covers every direction. The stationary points are the
              generalised eigenvectors of <M tex="(S_B, S_W)" />: the maximum <M tex={`\\lambda_1 = ${fmt(res2.eigenvalues[0] ?? 0, 3)}`} /> and the minimum{' '}
              <M tex={`\\lambda_2 = ${fmt(res2.eigenvalues[1] ?? 0, 3)}`} />
              {K === 2 ? (
                <>
                  ; with <M tex="K = 2" /> the matrix <M tex="S_B" /> has rank one, so the minimum is zero: some direction carries no mean difference at all
                </>
              ) : null}
              . The variance curve has its own stationary points at the eigenvectors of <M tex="S" /> (<M tex={`\\lambda_1(S) = ${fmt(varPCA, 3)}`} />,{' '}
              <M tex={`\\lambda_2(S) = ${fmt(twoD.covEigen.values[1] ?? 0, 3)}`} />). Since <M tex="(n-1)S = S_W + S_B" />, the two problems agree only when <M tex="S_W" /> is a multiple of the
              identity.
            </>
          ),
          stats: (
            <>
              For the current direction <M tex={`F = J(w)\\,(n-K)/(K-1) = ${fmt(F, 1)}`} /> is the one-way ANOVA F statistic of the projected scores{K === 2 ? <>, i.e. the squared two-sample t statistic</> : null}: LDA is the direction that maximises this statistic. It uses the class means and a <em>pooled</em> within-class spread, so it implicitly treats
              the classes as having a common covariance; the per-class bars ({sdText}) let you judge that assumption along the chosen direction.
            </>
          ),
          careful: (
            <>
              Only two of the <M tex={`p = ${source.variableNames.length}`} /> variables enter here; the full-<M tex="p" /> optimum from lesson 1 is <M tex={`\\lambda_1 = ${fmt(res.eigenvalues[0] ?? 0, 3)}`} />{' '}
              versus <M tex={`${fmt(fcLDA.J, 3)}`} /> in this plane. The projected means and standard deviations are training quantities, and <M tex="J" /> can be made arbitrarily large by
              adding variables when <M tex="n" /> is small (lesson 4). The direction is axial: <M tex="w" /> and <M tex="-w" /> give the same projection up to sign.
            </>
          ),
        }}
      />
      <div className="small muted">
        Summary: <M tex={`J(\\text{PC1}) = ${fmt(fcPCA.J, 3)}`} /> with variance {fmt(varPCA, 3)} ({pct(varPCA / (twoD.covEigen.values[0] + (twoD.covEigen.values[1] ?? 0)))} of the total),{' '}
        <M tex={`J(w_{\\text{LDA}}) = ${fmt(fcLDA.J, 3)}`} /> with variance {fmt(varLDA, 3)} ({pct(varLDA / (twoD.covEigen.values[0] + (twoD.covEigen.values[1] ?? 0)))} of the total).
      </div>
    </Section>
  );
}
