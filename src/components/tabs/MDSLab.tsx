import { ActiveDatasetCard, PrepControls } from '../common/DatasetControls';
import { Card, Callout } from '../common/Panels';
import { M } from '../common/Math';
import IntroSection from './mds/IntroSection';
import ChainSection from './mds/ChainSection';
import GeometrySection from './mds/GeometrySection';
import VsPcaSection from './mds/VsPcaSection';

/**
 * MDS laboratory. Section ids: mds-intro · mds-chain · mds-geometry · mds-vs-pca
 */
export default function MDSLab() {
  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>MDS laboratory</h2>
          <div className="lede">
            Suppose you do <b>not</b> have the original variables. You only know how far the observations are from one another. Can you reconstruct a useful low-dimensional map? Classical multidimensional scaling answers yes — and, for Euclidean distances, the map is exactly the one PCA would have drawn.
          </div>
        </div>
      </div>
      <div className="grid side-r">
        <ActiveDatasetCard>
          <div className="small muted">
            All lessons use the distance matrix <M tex="D" /> of the analysed (centred or standardised) global data under the metric chosen on the right; lesson 2 additionally extracts a five-point worked example from it.
          </div>
        </ActiveDatasetCard>
        <Card title="Distances and dimension (global)" plane>
          <PrepControls />
        </Card>
      </div>
      <Callout kind="info" title="What MDS is given, and what it is not">
        The input is the <M tex="n \times n" /> matrix <M tex="D" /> of pairwise dissimilarities — nothing else. Variables, their units and their number <M tex="p" /> are invisible to the method. Everything MDS reconstructs about the observations must be encoded in the <M tex="n(n-1)/2" /> distances.
      </Callout>
      <IntroSection />
      <ChainSection />
      <GeometrySection />
      <VsPcaSection />
    </div>
  );
}
