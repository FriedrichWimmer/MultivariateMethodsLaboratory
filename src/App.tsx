import { Suspense, lazy, type ComponentType } from 'react';
import { StoreProvider, useStore, type TabId } from './state/store';

const tabs: { id: TabId; label: string; group: string; load: () => Promise<{ default: ComponentType }> }[] = [
  { id: 'concept', label: 'Concept map', group: 'Foundations', load: () => import('./components/tabs/ConceptMap') },
  { id: 'data', label: 'Data laboratory', group: 'Foundations', load: () => import('./components/tabs/DataLab') },
  { id: 'svd', label: 'SVD laboratory', group: 'Methods', load: () => import('./components/tabs/SVDLab') },
  { id: 'pca', label: 'PCA laboratory', group: 'Methods', load: () => import('./components/tabs/PCALab') },
  { id: 'mds', label: 'MDS laboratory', group: 'Methods', load: () => import('./components/tabs/MDSLab') },
  { id: 'lda', label: 'LDA laboratory', group: 'Methods', load: () => import('./components/tabs/LDALab') },
  { id: 'compare', label: 'Four-way comparison', group: 'Connections', load: () => import('./components/tabs/FourWay') },
  { id: 'unified', label: 'One dataset, four questions', group: 'Connections', load: () => import('./components/tabs/Unified') },
  { id: 'diagnostics', label: 'Diagnostics & stability', group: 'Practice', load: () => import('./components/tabs/Diagnostics') },
  { id: 'wrong', label: 'What can go wrong?', group: 'Practice', load: () => import('./components/tabs/WhatCanGoWrong') },
  { id: 'quiz', label: 'Assessment mode', group: 'Practice', load: () => import('./components/tabs/Quiz') },
  { id: 'experiment', label: 'Experiment mode', group: 'Practice', load: () => import('./components/tabs/Experiment') },
  { id: 'takeaway', label: 'Master’s-level takeaway', group: 'Synthesis', load: () => import('./components/tabs/Takeaway') },
];

const lazyTabs = Object.fromEntries(tabs.map((t) => [t.id, lazy(t.load)])) as unknown as Record<TabId, ComponentType>;
const groups = Array.from(new Set(tabs.map((t) => t.group)));

function Sidebar() {
  const { tab, navigate, dataset } = useStore();
  return (
    <aside className="sidebar">
      <div className="brand">
        <h1>Multivariate Methods Laboratory</h1>
        <div className="sub">SVD · PCA · MDS · LDA — a Master’s-level interactive course</div>
      </div>
      {groups.map((g) => (
        <nav className="navgroup" key={g} aria-label={g}>
          <h4>{g}</h4>
          {tabs
            .filter((t) => t.group === g)
            .map((t) => (
              <button key={t.id} className={`navbtn ${tab === t.id ? 'active' : ''}`} onClick={() => navigate(t.id)}>
                <span className="num">{tabs.indexOf(t) + 1}</span>
                {t.label}
              </button>
            ))}
        </nav>
      ))}
      <div className="chip" style={{ marginTop: 'auto', alignSelf: 'flex-start' }} title="The dataset shared by every laboratory. Change it in the Data laboratory.">
        <span className="dot" />
        <span>
          Data: <b>{dataset.name}</b>
        </span>
      </div>
    </aside>
  );
}

function Main() {
  const { tab } = useStore();
  const Comp = lazyTabs[tab];
  return (
    <main className="main">
      <Suspense fallback={<div className="muted">Loading…</div>}>
        <Comp />
      </Suspense>
    </main>
  );
}

export function App() {
  return (
    <StoreProvider>
      <div className="app">
        <Sidebar />
        <Main />
      </div>
    </StoreProvider>
  );
}
