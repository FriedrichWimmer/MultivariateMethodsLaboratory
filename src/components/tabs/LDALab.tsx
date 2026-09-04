import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../../state/store';
import { defaultParams, generateDataset } from '../../lib/datasets';
import { lda } from '../../lib/lda';
import { ActiveDatasetCard } from '../common/DatasetControls';
import { Button } from '../common/Controls';
import { M } from '../common/Math';
import { Callout } from '../common/Panels';
import { buildTwoD, defaultClassNames, defaultPair, fromDataset, type LabelledData } from './lda/helpers';
import IntroSection from './lda/IntroSection';
import GeometrySection from './lda/GeometrySection';
import VsPcaSection from './lda/VsPcaSection';
import ConstraintsSection from './lda/ConstraintsSection';
import ClassifierSection from './lda/ClassifierSection';

/**
 * LDA laboratory. Lessons 1, 2 and 5 run on the global dataset when it carries labels
 * (the analysed matrix is the same one the store hands to lda()); otherwise they fall back
 * to a local labelled dataset. Lessons 3 and 4 always use lesson-specific local datasets.
 */
export default function LDALab() {
  const { analysis, dataset, prep, navigate } = useStore();

  const fallback = useMemo<LabelledData>(() => {
    const ds = generateDataset('clusters', { ...defaultParams, n: 150, p: 2, K: 3, separation: 3, seed: 3 });
    return fromDataset(ds, true) ?? { X: ds.X, y: ds.X.map(() => 0), classNames: ['Class A'], variableNames: ds.variableNames, name: ds.name, local: true };
  }, []);

  const globalLabelled = useMemo<LabelledData | null>(() => {
    if (!analysis.lda || !analysis.y) return null;
    const Xa = analysis.prep.scaling === 'none' ? analysis.X : analysis.pca.Xc;
    return { X: Xa, y: analysis.y, classNames: dataset.classNames ?? defaultClassNames(analysis.K), variableNames: dataset.variableNames, name: dataset.name, local: false };
  }, [analysis, dataset]);

  const source = globalLabelled ?? fallback;
  const res = useMemo(() => (globalLabelled && analysis.lda ? analysis.lda : lda(source.X, source.y)), [globalLabelled, analysis.lda, source]);

  // variable pair for the two-dimensional lessons (2 and 5); reset when the data changes
  const [pairState, setPairState] = useState<[number, number] | null>(null);
  useEffect(() => setPairState(null), [source]);
  const pair = useMemo<[number, number]>(() => {
    const p = source.variableNames.length;
    if (pairState && pairState[0] < p && pairState[1] < p && pairState[0] !== pairState[1]) return pairState;
    return defaultPair(res.SB, res.SW);
  }, [pairState, source, res]);
  const twoD = useMemo(() => buildTwoD(source, pair), [source, pair]);

  const scalingLabel = prep.scaling === 'none' ? 'raw' : prep.scaling === 'center' ? 'centred' : 'standardised';

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>LDA laboratory</h2>
          <div className="lede">
            Linear discriminant analysis is the supervised member of the family: it uses the class labels <M tex="y" /> together with <M tex="X" /> and looks for the directions along which the
            classes are most separated relative to their internal spread — not the directions of largest variance.
          </div>
        </div>
      </div>

      <ActiveDatasetCard>
        {analysis.lda ? (
          <div className="small muted">
            LDA runs on the {scalingLabel} matrix with <M tex={`K = ${analysis.K}`} /> classes; <M tex={`m = ${analysis.lda.maxDims}`} /> discriminant direction{analysis.lda.maxDims === 1 ? '' : 's'}{' '}
            available.
          </div>
        ) : null}
      </ActiveDatasetCard>

      {!analysis.lda && (
        <Callout kind="warning" title="LDA needs class labels">
          <div className="stack" style={{ gap: 8 }}>
            <div>
              The active dataset <b>{dataset.name}</b> is {dataset.y ? 'labelled but has too few observations per class' : 'unlabelled'}: LDA is supervised and needs the pair{' '}
              <M tex="(X, y)" /> with at least two classes. Lessons 1, 2 and 5 therefore run on a local stand-in (three spherical clusters, <M tex="n = 150" />, <M tex="p = 2" />);
              lessons 3 and 4 always use their own local datasets. Choose a labelled dataset (Iris-like, clusters, PCA vs LDA, …) to analyse your own data here.
            </div>
            <div>
              <Button primary small onClick={() => navigate('data')}>
                Choose a labelled dataset
              </Button>
            </div>
          </div>
        </Callout>
      )}

      <IntroSection source={source} res={res} scalingLabel={scalingLabel} />
      <GeometrySection source={source} res={res} twoD={twoD} pair={pair} setPair={setPairState} />
      <VsPcaSection />
      <ConstraintsSection />
      <ClassifierSection twoD={twoD} />
    </div>
  );
}
