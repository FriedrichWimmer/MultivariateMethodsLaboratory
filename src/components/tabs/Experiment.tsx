import { useMemo, useState } from 'react';
import type { Data } from 'plotly.js';
import { useStore } from '../../state/store';
import { generateDataset, defaultParams, datasetCatalog, type DatasetKind, type DatasetParams } from '../../lib/datasets';
import { pca, type Scaling } from '../../lib/pca';
import { distanceMatrix, classicalMDS, metricLabels, type Metric } from '../../lib/mds';
import { lda, type LDAResult } from '../../lib/lda';
import { colMeans, colStds, conditionNumber, fmt, maxAbs, sub, scale as scaleM } from '../../lib/linalg';
import { Section, Card, Callout, Interpretation, StatTile, Badge, ClassLegend } from '../common/Panels';
import { ScatterSVG } from '../common/ScatterSVG';
import { Plot } from '../common/Plot';
import { MatrixView } from '../common/MatrixView';
import { DataTable } from '../common/DataTable';
import { Slider, Select, Segmented, Toggle, Button } from '../common/Controls';
import { M } from '../common/Math';
import { categorical, classColor } from '../../lib/theme';

type MethodKey = 'svd' | 'pca' | 'mds' | 'lda';

const sliderMeta: Partial<Record<keyof DatasetParams, { label: string; min: number; max: number; step: number }>> = {
  n: { label: 'Observations n', min: 20, max: 300, step: 10 },
  p: { label: 'Variables p', min: 2, max: 10, step: 1 },
  K: { label: 'Classes K', min: 2, max: 5, step: 1 },
  correlation: { label: 'Correlation ρ', min: -0.95, max: 0.95, step: 0.05 },
  variance: { label: 'Variance', min: 0.1, max: 5, step: 0.1 },
  noise: { label: 'Noise sd', min: 0, max: 2, step: 0.05 },
  separation: { label: 'Class separation', min: 0, max: 8, step: 0.25 },
  scaleFactor: { label: 'Scale factor for x₁', min: 1, max: 1000, step: 1 },
  outlierCount: { label: 'Outliers', min: 0, max: 20, step: 1 },
  outlierMagnitude: { label: 'Outlier magnitude', min: 1, max: 20, step: 0.5 },
};

export default function Experiment() {
  const { setKind, setParams, setPrep, navigate } = useStore();
  const [kind, setLocalKind] = useState<DatasetKind>('clusters');
  const [params, setLocalParams] = useState<DatasetParams>({ ...defaultParams, n: 120, p: 4, K: 3, seed: 7 });
  const [scaling, setScaling] = useState<Scaling>('center');
  const [metric, setMetric] = useState<Metric>('euclidean');
  const [k, setK] = useState(2);
  const [methods, setMethods] = useState<Record<MethodKey, boolean>>({ svd: true, pca: true, mds: true, lda: true });
  const desc = datasetCatalog.find((d) => d.kind === kind)!;

  const ds = useMemo(() => generateDataset(kind, params), [kind, params]);
  const p = ds.X[0].length;
  const kk = Math.max(1, Math.min(k, p, ds.X.length - 1));

  const run = useMemo(() => {
    const pcaRes = pca(ds.X, scaling);
    const D = distanceMatrix(pcaRes.Xc, metric);
    const mdsRes = classicalMDS(D, kk);
    let ldaRes: LDAResult | null = null;
    let ldaError: string | null = null;
    if (ds.y) {
      try {
        ldaRes = lda(scaling === 'none' ? ds.X : pcaRes.Xc, ds.y);
      } catch (e) {
        ldaError = (e as Error).message;
      }
    }
    const kappa = conditionNumber(pcaRes.Xc);
    const means = colMeans(ds.X);
    const sds = colStds(ds.X);
    const lamCheck = Math.max(...pcaRes.eigenvalues.map((l, j) => Math.abs(l - (pcaRes.eigen.values[j] ?? 0))));
    const stCheck = ldaRes ? maxAbs(sub(ldaRes.ST, scaleM(pcaRes.S, ds.X.length - 1))) : NaN;
    const explainedSum = pcaRes.explained.reduce((a, b) => a + b, 0);
    return { pcaRes, D, mdsRes, ldaRes, ldaError, kappa, means, sds, lamCheck, stCheck, explainedSum };
  }, [ds, scaling, metric, kk]);

  const { pcaRes, mdsRes, ldaRes } = run;
  const pcaPts = useMemo(() => pcaRes.scores.map((r) => [r[0] ?? 0, r[1] ?? 0]), [pcaRes.scores]);
  const mdsPts = useMemo(() => mdsRes.coords.map((r) => [r[0] ?? 0, r[1] ?? 0]), [mdsRes.coords]);
  const names = ds.variableNames;

  const csv = () => {
    const header = [...names, ...(ds.y ? ['class'] : [])].join(',');
    const rows = ds.X.map((r, i) => [...r.map((x) => x.toPrecision(8)), ...(ds.y ? [ds.classNames ? ds.classNames[ds.y[i]] : String(ds.y[i])] : [])].join(','));
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${kind}-seed${params.seed}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const scalingWord = scaling === 'none' ? 'raw' : scaling === 'center' ? 'centred' : 'standardised';
  const ldaScatter = ldaRes && ldaRes.maxDims >= 2;
  const ldaHist: Data[] = ldaRes && ds.y && !ldaScatter ? ldaRes.classes.map((cl, ki) => ({ x: ldaRes.scores.filter((_, i) => ds.y![i] === cl).map((r) => r[0]), type: 'histogram', name: ds.classNames?.[cl] ?? `class ${cl}`, opacity: 0.6, marker: { color: classColor(ki) }, nbinsx: 30 })) : [];

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>Experiment mode</h2>
          <div className="lede">Design your own experiment: choose the data-generating structure, the preprocessing, the distance, the target dimension and the methods. The visualisations and the matrices underneath them update together.</div>
        </div>
      </div>

      <Section id="exp-design" title="1 · Design" subtitle="This laboratory keeps its own dataset; press the button at the bottom to hand it to the rest of the dashboard.">
        <div className="grid c3">
          <Card title="Data-generating process">
            <div className="controls-panel">
              <Select label="Structure" value={kind} options={datasetCatalog.map((d) => ({ value: d.kind, label: d.label }))} onChange={(v) => setLocalKind(v)} />
              <div className="small secondary">{desc.short}</div>
              {desc.params
                .filter((key) => sliderMeta[key])
                .map((key) => {
                  const m = sliderMeta[key]!;
                  return <Slider key={key} label={m.label} value={params[key] as number} min={m.min} max={m.max} step={m.step} onChange={(v) => setLocalParams((pp) => ({ ...pp, [key]: v }))} />;
                })}
              {desc.params.includes('classProportions') && (
                <Slider
                  label="Share of first class"
                  value={params.classProportions[0]}
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  format={(v) => `${Math.round(v * 100)}%`}
                  onChange={(v) => {
                    const K = Math.max(2, Math.round(params.K));
                    setLocalParams((pp) => ({ ...pp, classProportions: [v, ...new Array(K - 1).fill((1 - v) / (K - 1))] }));
                  }}
                />
              )}
              <div className="row">
                <Slider label="Seed" value={params.seed} min={1} max={999} step={1} onChange={(v) => setLocalParams((pp) => ({ ...pp, seed: v }))} />
                <Button small onClick={() => setLocalParams((pp) => ({ ...pp, seed: Math.floor(Math.random() * 999) + 1 }))}>
                  New seed
                </Button>
              </div>
            </div>
          </Card>
          <Card title="Preprocessing and geometry">
            <div className="controls-panel">
              <Segmented<Scaling> label="Scaling" value={scaling} onChange={setScaling} options={[{ value: 'none', label: 'Raw' }, { value: 'center', label: 'Centre' }, { value: 'standardize', label: 'Standardise' }]} />
              <Select<Metric> label="Distance metric (MDS)" value={metric} onChange={setMetric} options={(Object.keys(metricLabels) as Metric[]).map((m) => ({ value: m, label: metricLabels[m] }))} />
              <Slider label="Target dimension k" value={kk} min={1} max={Math.max(1, Math.min(p, 3))} step={1} onChange={setK} />
            </div>
          </Card>
          <Card title="Methods">
            <div className="stack">
              <Toggle label="SVD — singular value spectrum" checked={methods.svd} onChange={(v) => setMethods((m) => ({ ...m, svd: v }))} />
              <Toggle label="PCA — scores and scree" checked={methods.pca} onChange={(v) => setMethods((m) => ({ ...m, pca: v }))} />
              <Toggle label="MDS — configuration and stress" checked={methods.mds} onChange={(v) => setMethods((m) => ({ ...m, mds: v }))} />
              <Toggle label={ds.y ? 'LDA — discriminant scores' : 'LDA — needs class labels (unavailable for this structure)'} checked={methods.lda && !!ds.y} onChange={(v) => setMethods((m) => ({ ...m, lda: v }))} hint={ds.y ? undefined : 'Choose a labelled structure'} />
              <div className="divider" />
              <div className="small secondary">
                <b>{ds.name}</b>: n = {ds.X.length}, p = {p}
                {ds.y ? `, K = ${new Set(ds.y).size}` : ', unlabelled'}; analysed matrix is {scalingWord}.
              </div>
              <div className="small muted">{ds.description}</div>
            </div>
          </Card>
        </div>
      </Section>

      <Section id="exp-results" title="2 · Results" subtitle="One panel per selected method, all computed from the same generated matrix.">
        {ds.classNames && ds.y && <ClassLegend classNames={ds.classNames} />}
        <div className="grid c2" style={{ marginTop: 8 }}>
          {methods.svd && (
            <Card title={<><Badge method="SVD" /> singular values of the {scalingWord} matrix</>}>
              <Plot data={[{ x: pcaRes.singularValues.map((_, i) => i + 1), y: pcaRes.singularValues, type: 'bar', marker: { color: categorical[0] }, hovertemplate: 'σ_%{x} = %{y:.4g}<extra></extra>' }]} layout={{ xaxis: { title: { text: 'j' }, dtick: 1 }, yaxis: { title: { text: 'σ_j' }, rangemode: 'tozero' } }} height={280} />
              <div className="stats">
                <StatTile label="numerical rank" value={`${pcaRes.rank} / ${pcaRes.singularValues.length}`} />
                <StatTile label="κ₂ = σ₁/σ_r" value={run.kappa === Infinity ? '∞' : fmt(run.kappa, 1)} />
                <StatTile label={`energy in first ${kk}`} value={`${(100 * pcaRes.singularValues.slice(0, kk).reduce((a, s) => a + s * s, 0) / Math.max(pcaRes.singularValues.reduce((a, s) => a + s * s, 0), 1e-300)).toFixed(1)} %`} note="Σ_{j≤k}σ_j² / Σσ_j²" />
              </div>
            </Card>
          )}
          {methods.pca && (
            <Card title={<><Badge method="PCA" /> scores and scree</>}>
              <ScatterSVG points={pcaPts} labels={ds.y} classNames={ds.classNames} width={480} height={300} xLabel={`PC1 (${(pcaRes.explained[0] * 100).toFixed(1)} %)`} yLabel={`PC2 (${((pcaRes.explained[1] ?? 0) * 100).toFixed(1)} %)`} pointRadius={3.5} />
              <Plot data={[{ x: pcaRes.eigenvalues.map((_, i) => `PC${i + 1}`), y: pcaRes.eigenvalues, type: 'bar', marker: { color: categorical[0] }, hovertemplate: '%{x}: λ = %{y:.4g}<extra></extra>' }]} layout={{ yaxis: { title: { text: 'λ_j' }, rangemode: 'tozero' } }} height={200} title={`Scree — cumulative ${(pcaRes.cumulative[kk - 1] * 100).toFixed(1)} % at k = ${kk}`} />
            </Card>
          )}
          {methods.mds && (
            <Card title={<><Badge method="MDS" /> classical configuration ({metricLabels[metric]})</>}>
              <ScatterSVG points={mdsPts} labels={ds.y} classNames={ds.classNames} width={480} height={300} xLabel="MDS 1" yLabel={kk >= 2 ? 'MDS 2' : 'k = 1'} pointRadius={3.5} />
              <div className="stats">
                <StatTile label="stress-1" value={fmt(mdsRes.stress1, 4)} />
                <StatTile label="strain" value={fmt(mdsRes.strain, 4)} />
                <StatTile label="negative eigenvalues of B" value={mdsRes.negative} note={`${(mdsRes.negativeMass * 100).toFixed(2)} % of |λ| mass`} />
              </div>
            </Card>
          )}
          {methods.lda && ds.y && (
            <Card title={<><Badge method="LDA" /> discriminant scores</>}>
              {ldaRes ? (
                <>
                  {ldaScatter ? (
                    <ScatterSVG points={ldaRes.scores.map((r) => [r[0], r[1]])} labels={ds.y} classNames={ds.classNames} width={480} height={300} xLabel={`LD1 (λ₁ = ${fmt(ldaRes.eigenvalues[0], 2)})`} yLabel={`LD2 (λ₂ = ${fmt(ldaRes.eigenvalues[1], 2)})`} pointRadius={3.5} />
                  ) : (
                    <Plot data={ldaHist} layout={{ barmode: 'overlay', showlegend: true, xaxis: { title: { text: 'LD1 score' } }, yaxis: { title: { text: 'count' } } }} height={300} />
                  )}
                  <div className="stats">
                    <StatTile label="Fisher eigenvalues" value={ldaRes.eigenvalues.map((l) => fmt(l, 2)).join(', ') || '—'} />
                    <StatTile label="directions (≤ K − 1)" value={`${ldaRes.maxDims} of ${ldaRes.K - 1}`} />
                    <StatTile label="S_W" value={ldaRes.swSingular ? 'singular' : 'full rank'} note={ldaRes.swSingular ? `rank ${ldaRes.swRank} < ${ldaRes.p}` : `κ = ${fmt(ldaRes.swCondition, 1)}`} />
                  </div>
                  {ldaRes.swSingular && <Callout kind="warning" title="Singular within-class scatter">The solution was computed on the range of <M tex="S_W" />; regularise or reduce p before trusting it.</Callout>}
                </>
              ) : (
                <Callout kind="danger" title="LDA failed">{run.ldaError}</Callout>
              )}
            </Card>
          )}
        </div>
        <Interpretation
          items={{
            seeing: <>{ds.name} (n = {ds.X.length}, p = {p}) analysed after {scalingWord === 'raw' ? 'no' : scalingWord} preprocessing, with {metricLabels[metric]} distances for MDS and k = {kk}.</>,
            why: (
              <>
                PC1 explains {(pcaRes.explained[0] * 100).toFixed(1)} % and the first {kk} component{kk === 1 ? '' : 's'} {(pcaRes.cumulative[kk - 1] * 100).toFixed(1)} %; stress-1 = {fmt(mdsRes.stress1, 4)}{metric === 'euclidean' ? ' (the MDS map is the PCA score plot)' : ` with ${mdsRes.negative} negative eigenvalues of B`};{' '}
                {ldaRes ? `LDA λ₁ = ${fmt(ldaRes.eigenvalues[0] ?? 0, 3)} with ${ldaRes.maxDims} direction${ldaRes.maxDims === 1 ? '' : 's'}` : 'no labels for LDA'}.
              </>
            ),
            math: <>Identities checked on this run: max |λ_j − σ_j²/(n−1)| = {fmt(run.lamCheck, 12)}; Σ explained = {fmt(run.explainedSum, 12)}{ldaRes ? <>; max |S_W + S_B − (n−1)S| = {fmt(run.stCheck, 10)}</> : null}.</>,
            stats: <>κ₂(X_c) = {run.kappa === Infinity ? '∞' : fmt(run.kappa, 1)}: {run.kappa > 1e3 ? 'ill-conditioned — trailing components and any inverse are unreliable' : 'well-conditioned'}. {ds.y ? `The smallest class has ${Math.min(...(ldaRes?.classSizes ?? [ds.X.length]))} observations.` : ''}</>,
            careful: <>Everything here is in-sample; the generating structure is known, which is never the case with real data. Change the seed to see the sampling variability of every quantity.</>,
          }}
        />
      </Section>

      <Section id="exp-math" title="3 · The mathematics underneath" subtitle="The matrices that produced the pictures.">
        <div className="matrix-eq" style={{ alignItems: 'flex-start', gap: 18 }}>
          <MatrixView M={[run.means]} title="x̄ᵀ" colLabels={names} digits={3} compact />
          <MatrixView M={[run.sds]} title="sᵀ" colLabels={names} digits={3} compact />
          <MatrixView M={pcaRes.S} title={scaling === 'standardize' ? 'R (correlation)' : scaling === 'center' ? 'S (covariance)' : 'XᵀX/(n−1)'} rowLabels={names} colLabels={names} digits={3} heat="diverging" compact />
          <MatrixView M={[pcaRes.eigenvalues]} title="Λ (eigenvalues)" colLabels={pcaRes.eigenvalues.map((_, j) => `λ${j + 1}`)} digits={3} compact />
          <MatrixView M={pcaRes.V} title="V (loadings, columns = PCs)" rowLabels={names} colLabels={pcaRes.eigenvalues.map((_, j) => `PC${j + 1}`)} digits={3} heat="diverging" highlightCols={Array.from({ length: kk }, (_, j) => j)} compact />
          <MatrixView M={[pcaRes.singularValues]} title="Σ (singular values)" colLabels={pcaRes.singularValues.map((_, j) => `σ${j + 1}`)} digits={3} compact />
        </div>
        <div className="small secondary" style={{ marginTop: 6 }}>
          Shapes: U is {ds.X.length} × {pcaRes.singularValues.length}, Σ is {pcaRes.singularValues.length} × {pcaRes.singularValues.length}, Vᵀ is {pcaRes.singularValues.length} × {p}. Eigenvalues of B (first {Math.min(8, mdsRes.eigenvalues.length)}): {mdsRes.eigenvalues.slice(0, 8).map((l) => fmt(l, 3)).join(', ')}.
        </div>
        {ldaRes && (
          <div className="matrix-eq" style={{ alignItems: 'flex-start', gap: 18, marginTop: 12 }}>
            <MatrixView M={ldaRes.SW} title="S_W" rowLabels={names} colLabels={names} digits={2} heat="diverging" compact />
            <MatrixView M={ldaRes.SB} title="S_B" rowLabels={names} colLabels={names} digits={2} heat="diverging" compact />
            <MatrixView M={ldaRes.W} title="W (discriminant directions)" rowLabels={names} colLabels={ldaRes.eigenvalues.map((_, j) => `LD${j + 1}`)} digits={3} heat="diverging" compact />
            <MatrixView M={ldaRes.classMeans} title="class means (rows)" rowLabels={ldaRes.classes.map((c) => ds.classNames?.[c] ?? `class ${c}`)} colLabels={names} digits={3} compact />
          </div>
        )}
      </Section>

      <Section id="exp-data" title="4 · The data" subtitle="The generated matrix, reproducible from the seed.">
        <div className="row" style={{ marginBottom: 8 }}>
          <Button small onClick={csv}>
            Download as CSV
          </Button>
          <Button
            small
            primary
            onClick={() => {
              setKind(kind);
              setParams(params);
              setPrep({ scaling, metric, k: kk });
              navigate('compare');
            }}
          >
            Use as the global dataset and open the four-way comparison
          </Button>
          <span className="small muted">
            Seed {params.seed}: the same parameters always regenerate this exact matrix.
          </span>
        </div>
        <DataTable X={ds.X} variableNames={names} y={ds.y} classNames={ds.classNames} maxRows={100} digits={3} />
      </Section>
    </div>
  );
}
