import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { generateDataset, defaultParams, type Dataset, type DatasetKind, type DatasetParams } from '../lib/datasets';
import { pca, type PCAResult, type Scaling } from '../lib/pca';
import { distanceMatrix, classicalMDS, type Metric, type MDSResult } from '../lib/mds';
import { lda, type LDAResult } from '../lib/lda';
import { svd, type SVDResult } from '../lib/linalg';

export type TabId = 'concept' | 'data' | 'svd' | 'pca' | 'mds' | 'lda' | 'compare' | 'unified' | 'diagnostics' | 'wrong' | 'quiz' | 'experiment' | 'takeaway';

export interface Prep {
  scaling: Scaling;
  metric: Metric;
  k: number;
}

export interface Analysis {
  dataset: Dataset;
  X: number[][];
  y?: number[];
  n: number;
  p: number;
  K: number;
  prep: Prep;
  pca: PCAResult;
  /** thin SVD of the analysed (centred/scaled) matrix */
  svd: SVDResult;
  mds: MDSResult;
  D: number[][];
  lda: LDAResult | null;
}

interface Store {
  kind: DatasetKind;
  params: DatasetParams;
  uploaded: Dataset | null;
  dataset: Dataset;
  prep: Prep;
  analysis: Analysis;
  tab: TabId;
  anchor: string | null;
  setKind: (k: DatasetKind) => void;
  setParams: (p: Partial<DatasetParams>) => void;
  setUploaded: (d: Dataset | null) => void;
  setPrep: (p: Partial<Prep>) => void;
  navigate: (tab: TabId, anchor?: string) => void;
}

const Ctx = createContext<Store | null>(null);

export function runAnalysis(dataset: Dataset, prep: Prep): Analysis {
  const X = dataset.X;
  const y = dataset.y;
  const n = X.length;
  const p = X[0].length;
  const K = y ? new Set(y).size : 0;
  const pcaRes = pca(X, prep.scaling);
  const D = distanceMatrix(pcaRes.Xc, prep.metric);
  const mdsRes = classicalMDS(D, Math.min(prep.k, n - 1, p));
  const ldaRes = y && K >= 2 && n > K ? lda(pcaRes.scaling === 'none' ? X : pcaRes.Xc, y) : null;
  return { dataset, X, y, n, p, K, prep, pca: pcaRes, svd: pcaRes.svd, mds: mdsRes, D, lda: ldaRes };
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [kind, setKindState] = useState<DatasetKind>('iris');
  const [params, setParamsState] = useState<DatasetParams>(defaultParams);
  const [uploaded, setUploaded] = useState<Dataset | null>(null);
  const [prep, setPrepState] = useState<Prep>({ scaling: 'center', metric: 'euclidean', k: 2 });
  const [tab, setTab] = useState<TabId>('concept');
  const [anchor, setAnchor] = useState<string | null>(null);

  const dataset = useMemo(() => {
    if (kind === 'uploaded' && uploaded) return uploaded;
    try {
      return generateDataset(kind === 'uploaded' ? 'iris' : kind, params);
    } catch {
      return generateDataset('iris', defaultParams);
    }
  }, [kind, params, uploaded]);

  const analysis = useMemo(() => runAnalysis(dataset, prep), [dataset, prep]);

  const setKind = useCallback((k: DatasetKind) => setKindState(k), []);
  const setParams = useCallback((p: Partial<DatasetParams>) => setParamsState((prev) => ({ ...prev, ...p })), []);
  const setPrep = useCallback((p: Partial<Prep>) => setPrepState((prev) => ({ ...prev, ...p })), []);
  const navigate = useCallback((t: TabId, a?: string) => {
    setTab(t);
    setAnchor(a ?? null);
    if (!a) window.scrollTo({ top: 0 });
  }, []);

  useEffect(() => {
    if (!anchor) return;
    const id = window.setTimeout(() => {
      const el = document.getElementById(anchor);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 60);
    return () => window.clearTimeout(id);
  }, [anchor, tab]);

  const value: Store = { kind, params, uploaded, dataset, prep, analysis, tab, anchor, setKind, setParams, setUploaded, setPrep, navigate };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error('useStore outside provider');
  return s;
}

export function useAnalysis(): Analysis {
  return useStore().analysis;
}
