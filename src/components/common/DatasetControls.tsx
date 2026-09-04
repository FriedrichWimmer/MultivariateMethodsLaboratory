import { useRef, useState } from 'react';
import { useStore } from '../../state/store';
import { datasetCatalog, parseCSV, type DatasetKind, type DatasetParams } from '../../lib/datasets';
import { metricLabels, type Metric } from '../../lib/mds';
import { Slider, Select, Segmented, Button } from './Controls';
import { Card } from './Panels';
import type { Scaling } from '../../lib/pca';

const paramMeta: Partial<Record<keyof DatasetParams, { label: string; min: number; max: number; step: number }>> = {
  n: { label: 'Observations n', min: 10, max: 400, step: 10 },
  p: { label: 'Variables p', min: 2, max: 12, step: 1 },
  K: { label: 'Classes K', min: 2, max: 5, step: 1 },
  correlation: { label: 'Correlation ρ', min: -0.95, max: 0.95, step: 0.05 },
  variance: { label: 'Variance', min: 0.1, max: 5, step: 0.1 },
  noise: { label: 'Noise sd', min: 0, max: 2, step: 0.05 },
  separation: { label: 'Class separation', min: 0, max: 8, step: 0.25 },
  scaleFactor: { label: 'Scale factor for x₁', min: 1, max: 1000, step: 1 },
  outlierCount: { label: 'Number of outliers', min: 0, max: 20, step: 1 },
  outlierMagnitude: { label: 'Outlier magnitude', min: 1, max: 20, step: 0.5 },
};

/** Dataset selection and generator parameters, bound to the global store. */
export function DatasetControls({ compact = false, allowUpload = true }: { compact?: boolean; allowUpload?: boolean }) {
  const { kind, params, setKind, setParams, setUploaded, uploaded, dataset } = useStore();
  const desc = datasetCatalog.find((d) => d.kind === kind);
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onFile = async (f: File) => {
    try {
      const text = await f.text();
      const ds = parseCSV(text, f.name);
      setUploaded(ds);
      setKind('uploaded');
      setUploadError(null);
    } catch (e) {
      setUploadError((e as Error).message);
    }
  };

  const options = datasetCatalog.map((d) => ({ value: d.kind as DatasetKind, label: d.label }));
  if (uploaded) options.push({ value: 'uploaded', label: `Uploaded: ${uploaded.name}` });

  return (
    <div className="controls-panel">
      <Select label="Dataset" value={kind} options={options} onChange={(k) => setKind(k)} />
      {kind !== 'uploaded' && desc && (
        <>
          {!compact && <div className="small secondary">{desc.short}</div>}
          {desc.params
            .filter((k) => paramMeta[k])
            .map((k) => {
              const m = paramMeta[k]!;
              return <Slider key={k} label={m.label} value={params[k] as number} min={m.min} max={m.max} step={m.step} onChange={(v) => setParams({ [k]: v } as Partial<DatasetParams>)} />;
            })}
          {desc.params.includes('classProportions') && (
            <Slider
              label="Share of first class"
              value={params.classProportions[0]}
              min={0.05}
              max={0.9}
              step={0.05}
              onChange={(v) => {
                const K = Math.max(2, Math.round(params.K));
                const rest = (1 - v) / (K - 1);
                setParams({ classProportions: [v, ...new Array(K - 1).fill(rest)] });
              }}
              format={(v) => `${Math.round(v * 100)}%`}
            />
          )}
          <div className="row">
            <Slider label="Random seed" value={params.seed} min={1} max={999} step={1} onChange={(v) => setParams({ seed: v })} />
            <Button small onClick={() => setParams({ seed: Math.floor(Math.random() * 999) + 1 })}>
              New seed
            </Button>
          </div>
        </>
      )}
      {kind === 'uploaded' && uploaded && <div className="small secondary">{uploaded.description}</div>}
      {allowUpload && (
        <div className="row">
          <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" style={{ display: 'none' }} onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          <Button small onClick={() => fileRef.current?.click()}>
            Upload CSV…
          </Button>
          <span className="small muted">numeric columns → X; a text column → labels</span>
        </div>
      )}
      {uploadError && <div className="small" style={{ color: 'var(--danger)' }}>{uploadError}</div>}
      {!compact && (
        <div className="small muted">
          Current: <b>{dataset.name}</b> — n = {dataset.X.length}, p = {dataset.X[0].length}
          {dataset.y ? `, K = ${new Set(dataset.y).size}` : ', unlabelled'}
        </div>
      )}
    </div>
  );
}

/** Preprocessing controls (scaling, distance metric, retained dimension). */
export function PrepControls({ showMetric = true, showK = true, maxK }: { showMetric?: boolean; showK?: boolean; maxK?: number }) {
  const { prep, setPrep, dataset } = useStore();
  const p = dataset.X[0].length;
  const kmax = Math.max(1, Math.min(maxK ?? p, p, dataset.X.length - 1));
  return (
    <div className="controls-panel">
      <Segmented<Scaling>
        label="Preprocessing"
        value={prep.scaling}
        onChange={(v) => setPrep({ scaling: v })}
        options={[
          { value: 'none', label: 'Raw', hint: 'No centring — decomposes X itself' },
          { value: 'center', label: 'Centre', hint: 'Subtract column means: covariance PCA' },
          { value: 'standardize', label: 'Standardise', hint: 'Centre and divide by sd: correlation PCA' },
        ]}
      />
      {showMetric && (
        <Select<Metric>
          label="Distance metric (MDS)"
          value={prep.metric}
          onChange={(v) => setPrep({ metric: v })}
          options={(Object.keys(metricLabels) as Metric[]).map((m) => ({ value: m, label: metricLabels[m] }))}
        />
      )}
      {showK && <Slider label="Dimensions retained k" value={Math.min(prep.k, kmax)} min={1} max={kmax} step={1} onChange={(v) => setPrep({ k: v })} />}
    </div>
  );
}

/** Small card summarising the active dataset, for use at the top of analysis tabs. */
export function ActiveDatasetCard({ children }: { children?: React.ReactNode }) {
  const { dataset, prep } = useStore();
  return (
    <Card title="Active dataset" plane>
      <div className="stack" style={{ gap: 6 }}>
        <div>
          <b>{dataset.name}</b>
          <span className="muted small">
            {' '}
            · n = {dataset.X.length}, p = {dataset.X[0].length}
            {dataset.y ? `, K = ${new Set(dataset.y).size}` : ''} · {prep.scaling === 'none' ? 'raw' : prep.scaling === 'center' ? 'centred' : 'standardised'}
          </span>
        </div>
        <div className="small secondary">{dataset.description}</div>
        {children}
      </div>
    </Card>
  );
}
