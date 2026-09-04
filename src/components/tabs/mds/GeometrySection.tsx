import { useEffect, useMemo, useState } from 'react';
import type { Data } from 'plotly.js';
import { useStore } from '../../../state/store';
import { Section, Card, Callout, Interpretation, StatTile, Badge, ClassLegend } from '../../common/Panels';
import { ScatterSVG } from '../../common/ScatterSVG';
import { Plot } from '../../common/Plot';
import { Slider, Button } from '../../common/Controls';
import { M, MBlock } from '../../common/Math';
import { metricLabels } from '../../../lib/mds';
import { fmt } from '../../../lib/linalg';
import { classColor, methodColor, ink, categorical, neutralMark } from '../../../lib/theme';
import { stressForK, strainForK, samplePairs, mostDistortedPair, shepardStats, classNamesOf, pct } from './helpers';

/** Lesson 3 — the configuration, a selected pair of observations, distortion, stress and the eigenvalue spectrum. */
export default function GeometrySection() {
  const { analysis: a, dataset, prep, setPrep } = useStore();
  const { mds, D } = a;
  const n = a.n;
  const classNames = classNamesOf(dataset);
  const coords2 = useMemo(() => mds.coords.map((r) => [r[0] ?? 0, r[1] ?? 0]), [mds.coords]);

  const [sel, setSel] = useState<number[]>([]);
  useEffect(() => setSel([]), [dataset, prep.metric]);
  const worst = useMemo(() => mostDistortedPair(D, mds.Dhat), [D, mds.Dhat]);
  const pair: [number, number] | null = sel.length === 2 ? [sel[0], sel[1]] : worst;
  const onPointClick = (i: number) => setSel((prev) => (prev.length >= 2 || prev.includes(i) ? [i] : [...prev, i]));

  const pairInfo = useMemo(() => {
    if (!pair) return null;
    const [i, j] = pair;
    const d = D[i][j];
    const dh = mds.Dhat[i][j];
    return { i, j, d, dh, abs: Math.abs(d - dh), rel: d > 0 ? (dh - d) / d : 0 };
  }, [pair, D, mds.Dhat]);

  const shepard = useMemo(() => {
    const pairs = samplePairs(n, 2500);
    return { x: pairs.map(([i, j]) => D[i][j]), y: pairs.map(([i, j]) => mds.Dhat[i][j]), count: pairs.length };
  }, [D, mds.Dhat, n]);
  const shStats = useMemo(() => shepardStats(D, mds.Dhat), [D, mds.Dhat]);

  const kMax = Math.max(1, Math.min(6, mds.positive));
  const stressCurve = useMemo(() => {
    const ks = Array.from({ length: kMax }, (_, t) => t + 1);
    return { ks, stress: ks.map((k) => stressForK(D, mds.eigen, k)), strain: ks.map((k) => strainForK(mds.eigenvalues, k)) };
  }, [D, mds.eigen, mds.eigenvalues, kMax]);

  const spectrum = useMemo(() => {
    const vals = mds.eigenvalues.slice(0, Math.min(n, 30));
    const idx = vals.map((_, i) => i + 1);
    return { idx, pos: vals.map((v) => (v > 0 ? v : 0)), neg: vals.map((v) => (v < 0 ? v : 0)), shown: vals.length };
  }, [mds.eigenvalues, n]);

  const maxD = Math.max(shStats.maxD, shStats.maxDhat, 1e-12);
  const shepardData: Data[] = [
    { x: [0, maxD], y: [0, maxD], type: 'scatter', mode: 'lines', name: 'd̂ = d (no distortion)', line: { color: ink.axis, width: 1.5 }, hoverinfo: 'skip' },
    { x: shepard.x, y: shepard.y, type: 'scatter', mode: 'markers', name: `pairs (${shepard.count} of ${(n * (n - 1)) / 2})`, marker: { color: neutralMark, size: 4, opacity: 0.45 }, hovertemplate: 'd = %{x:.3f}<br>d̂ = %{y:.3f}<extra></extra>' },
    ...(pairInfo ? [{ x: [pairInfo.d], y: [pairInfo.dh], type: 'scatter' as const, mode: 'markers' as const, name: `selected pair (${pairInfo.i + 1}, ${pairInfo.j + 1})`, marker: { color: methodColor.LDA, size: 13, symbol: 'diamond' as const, line: { color: ink.surface, width: 1.5 } }, hovertemplate: 'selected: d = %{x:.3f}, d̂ = %{y:.3f}<extra></extra>' }] : []),
  ];
  const spectrumData: Data[] = [
    { x: spectrum.idx, y: spectrum.pos, type: 'bar', name: 'positive eigenvalues (Euclidean part)', marker: { color: categorical[0] } , hovertemplate: 'λ_%{x} = %{y:.4g}<extra></extra>' },
    { x: spectrum.idx, y: spectrum.neg, type: 'bar', name: 'negative eigenvalues (non-Euclidean part)', marker: { color: '#e34948' }, hovertemplate: 'λ_%{x} = %{y:.4g}<extra></extra>' },
  ];
  const stressData: Data[] = [
    { x: stressCurve.ks, y: stressCurve.stress, type: 'scatter', mode: 'lines+markers', name: 'Kruskal stress-1', line: { color: methodColor.MDS, width: 2 }, marker: { size: 8 } },
    { x: stressCurve.ks, y: stressCurve.strain, type: 'scatter', mode: 'lines+markers', name: 'strain ‖B − X_kX_kᵀ‖_F / ‖B‖_F', line: { color: methodColor.SVD, width: 2 }, marker: { size: 8, symbol: 'square' } },
    { x: [mds.k], y: [mds.stress1], type: 'scatter', mode: 'markers', name: `current k = ${mds.k}`, marker: { color: ink.primary, size: 13, symbol: 'x' } },
  ];

  const metric = metricLabels[prep.metric];
  const segments = pairInfo && mds.k >= 1 ? [{ from: [coords2[pairInfo.i][0], coords2[pairInfo.i][1]] as [number, number], to: [coords2[pairInfo.j][0], coords2[pairInfo.j][1]] as [number, number], color: methodColor.LDA, width: 2, opacity: 0.9 }] : [];

  return (
    <Section id="mds-geometry" title="3 · The configuration and its distortions" subtitle="Click two observations to compare their original distance with the distance in the k-dimensional map; the Shepard diagram does this for every pair at once." right={<Badge method="MDS" />}>
      <div className="grid side-r">
        <div className="stack">
          <div className="row between">
            {classNames && <ClassLegend classNames={classNames} />}
            <span className="kbd-hint">click one point, then a second; a third click starts over</span>
          </div>
          <ScatterSVG
            points={coords2}
            labels={dataset.y}
            classNames={classNames}
            width={560}
            height={420}
            xLabel={`MDS 1 (λ₁ = ${fmt(mds.eigenvalues[0] ?? 0, 2)})`}
            yLabel={mds.k >= 2 ? `MDS 2 (λ₂ = ${fmt(mds.eigenvalues[1] ?? 0, 2)})` : 'k = 1: no second axis'}
            selected={pair ? [pair[0], pair[1]] : []}
            onPointClick={onPointClick}
            segments={segments}
            title={`Classical MDS configuration, k = ${mds.k}, ${metric} input distances`}
            hoverInfo={(i) => `#${i + 1}${dataset.y && classNames ? ' · ' + classNames[dataset.y[i]] : ''}\nclick to select`}
          />
        </div>
        <div className="stack">
          <Card title="Selected pair">
            {pairInfo ? (
              <div className="stack" style={{ gap: 6 }}>
                <div className="small secondary">
                  Observations {pairInfo.i + 1} and {pairInfo.j + 1}
                  {sel.length < 2 ? ' (the most distorted pair — click to choose your own)' : ''}
                </div>
                <div className="stats">
                  <StatTile label={`original d_ij (${prep.metric})`} value={fmt(pairInfo.d, 3)} />
                  <StatTile label="configuration d̂_ij" value={fmt(pairInfo.dh, 3)} />
                  <StatTile label="|d − d̂|" value={fmt(pairInfo.abs, 3)} note={`${pairInfo.rel >= 0 ? '+' : ''}${(pairInfo.rel * 100).toFixed(1)} % ${pairInfo.rel < 0 ? '(compressed)' : '(stretched)'}`} />
                </div>
                <Button small onClick={() => setSel([])}>
                  Reset to the most distorted pair
                </Button>
              </div>
            ) : (
              <div className="muted small">No pair selected.</div>
            )}
          </Card>
          <Card title="Global fit">
            <div className="stats">
              <StatTile label="Kruskal stress-1" value={fmt(mds.stress1, 4)} note="√(Σ(d−d̂)² / Σd²)" />
              <StatTile label="strain" value={fmt(mds.strain, 4)} note="‖B − X_kX_kᵀ‖_F / ‖B‖_F" />
              <StatTile label="positive / negative eigenvalues" value={`${mds.positive} / ${mds.negative}`} note={`|negative| mass ${pct(mds.negativeMass, 2)}`} />
              <StatTile label="pairs compressed" value={pct(shStats.compressed, 1)} note="d̂ < d — projection can only shorten Euclidean distances" />
            </div>
          </Card>
          <Slider label="Dimensions retained k (global)" value={Math.min(prep.k, Math.max(1, Math.min(a.p, n - 1)))} min={1} max={Math.max(1, Math.min(a.p, n - 1))} step={1} onChange={(v) => setPrep({ k: v })} />
        </div>
      </div>

      <div className="grid c3" style={{ marginTop: 14 }}>
        <Plot title="Shepard diagram: configuration distance d̂ against input distance d" data={shepardData} layout={{ showlegend: true, xaxis: { title: { text: `d_ij (${prep.metric})` } }, yaxis: { title: { text: 'd̂_ij (Euclidean, in the map)' }, scaleanchor: 'x' } }} height={330} />
        <Plot title={`Eigenvalues of B (first ${spectrum.shown} of ${n})`} data={spectrumData} layout={{ showlegend: true, barmode: 'relative', xaxis: { title: { text: 'index j' } }, yaxis: { title: { text: 'λ_j(B)' } } }} height={330} />
        <Plot title="Fit as a function of the retained dimension" data={stressData} layout={{ showlegend: true, xaxis: { title: { text: 'k' }, dtick: 1 }, yaxis: { title: { text: 'stress-1 / strain' }, rangemode: 'tozero' } }} height={330} />
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              The {n} observations placed in {mds.k} dimension{mds.k === 1 ? '' : 's'} so that their Euclidean distances in the map approximate the {metric} distances of the analysed data. {pairInfo && (
                <>
                  The highlighted pair ({pairInfo.i + 1}, {pairInfo.j + 1}) has d = {fmt(pairInfo.d, 3)} in the data and d̂ = {fmt(pairInfo.dh, 3)} in the map, a {Math.abs(pairInfo.rel * 100).toFixed(1)} % {pairInfo.rel < 0 ? 'compression' : 'stretch'}.
                </>
              )}
            </>
          ),
          why: (
            <>
              Stress-1 is {fmt(mds.stress1, 4)} and {pct(shStats.compressed, 1)} of all pairs are shortened: with Euclidean input, <M tex="B = X_cX_c^T" /> and the map is an orthogonal projection, which can only reduce distances. The spectrum has {mds.positive} positive and {mds.negative} negative eigenvalues{mds.negative > 0 ? `, carrying ${pct(mds.negativeMass, 2)} of the absolute eigenvalue mass — the ${prep.metric} distances are not Euclidean-embeddable` : ' — the distances are exactly Euclidean'}.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\text{stress-1} = \sqrt{\frac{\sum_{i<j}(d_{ij}-\hat d_{ij})^2}{\sum_{i<j}d_{ij}^2}},\qquad \text{strain} = \frac{\|B - X_kX_k^T\|_F}{\|B\|_F} = \sqrt{\frac{\sum_{j>k}\lambda_j^2 + \sum_{\lambda_j<0}\lambda_j^2}{\sum_j\lambda_j^2}}.`} />
              Classical MDS minimises the strain, not the stress: it is the Eckart–Young solution for <M tex="B" />. Stress-1 is reported because it is the quantity a reader of a map cares about; for Euclidean input it equals <M tex="0" /> exactly when <M tex="k = \operatorname{rank}(B)" />.
            </>
          ),
          stats: (
            <>
              The correlation between <M tex="d_{ij}" /> and <M tex="\hat d_{ij}" /> over all pairs is {fmt(shStats.correlation, 4)} and the mean absolute distortion is {fmt(shStats.meanAbsErr, 4)}. Increasing k from {mds.k} to {Math.min(mds.k + 1, kMax)} would change stress-1 from {fmt(stressCurve.stress[mds.k - 1] ?? mds.stress1, 4)} to {fmt(stressCurve.stress[Math.min(mds.k, kMax - 1)] ?? 0, 4)}.
            </>
          ),
          careful: (
            <>
              A map with low stress can still mislead about <i>neighbourhoods</i>: classical MDS weights all pairs equally, so it spends its accuracy on the many large distances and may misplace close neighbours. Negative eigenvalues are dropped silently; if their mass is large, the Euclidean map is a poor summary of the dissimilarities and non-metric MDS is the appropriate tool.
            </>
          ),
        }}
      />

      <div className="divider" />
      <h3>Four different things one might try to preserve</h3>
      <table className="summary-table" style={{ marginTop: 8 }}>
        <thead>
          <tr>
            <th>Preserve…</th>
            <th>Method</th>
            <th>Criterion</th>
            <th>What is sacrificed</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>variance (dispersion of the observations)</td>
            <td>
              <Badge method="PCA" />
            </td>
            <td>
              <M tex="\max_{\|w\|=1} w^TSw" />; equivalently minimum squared reconstruction error
            </td>
            <td>directions of small variance, whether or not they carry class or neighbourhood information</td>
          </tr>
          <tr>
            <td>pairwise distances</td>
            <td>
              <Badge method="MDS" />
            </td>
            <td>
              <M tex="\min\|B - XX^T\|_F" /> (classical) or stress (metric / non-metric)
            </td>
            <td>large distances dominate the fit; local neighbourhoods may be distorted</td>
          </tr>
          <tr>
            <td>neighbourhoods (who is close to whom)</td>
            <td>
              <span className="badge neutral">Isomap · t-SNE · UMAP</span>
            </td>
            <td>graph distances or neighbour probabilities (mentioned for contrast only — nonlinear, not covered here)</td>
            <td>global distances and axes lose their metric meaning</td>
          </tr>
          <tr>
            <td>class separation</td>
            <td>
              <Badge method="LDA" />
            </td>
            <td>
              <M tex="\max_w w^TS_Bw / w^TS_Ww" />
            </td>
            <td>everything about the data that does not distinguish the labelled groups</td>
          </tr>
        </tbody>
      </table>
      <Callout kind="warning" title="This distinction is the point">
        The same dataset gives different "best" two-dimensional pictures under these four criteria. None of them is wrong; each answers a different question. Deciding which structure to preserve must precede choosing the method.
      </Callout>
      <div className="small muted" style={{ marginTop: 6 }}>
        Colours: class identity uses the fixed categorical palette ({classColor(0)}, {classColor(1)}, …); the selected pair is drawn in {methodColor.LDA}.
      </div>
    </Section>
  );
}
