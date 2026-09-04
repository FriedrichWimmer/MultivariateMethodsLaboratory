import { ActiveDatasetCard, PrepControls } from '../common/DatasetControls';
import { Card } from '../common/Panels';
import { M } from '../common/Math';
import { FromSVD } from './pca/FromSVD';
import { Geometry } from './pca/Geometry';
import { Scree } from './pca/Scree';
import { Reconstruction } from './pca/Reconstruction';
import { ScalingLesson } from './pca/ScalingLesson';

/**
 * PCA laboratory — five lessons that build principal component analysis as the
 * statistical reading of the SVD of the centred data matrix.
 *
 * Section ids (stable anchors for the concept map):
 *   pca-from-svd · pca-geometry · pca-scree · pca-reconstruction · pca-scaling
 */
export default function PCALab() {
  return (
    <div className="stack">
      <div className="topbar">
        <div className="title">
          <h2>PCA laboratory</h2>
          <div className="lede">
            Principal components are the right singular vectors of the centred data matrix: PCA is the SVD of <M tex="X_c" /> read as a statement about variance.
          </div>
        </div>
      </div>

      <div className="grid side-r">
        <ActiveDatasetCard>
          <div className="small muted">
            Lessons 1–4 use this shared dataset; change it in the Data laboratory. Lesson 5 builds its own dataset to isolate the effect of scaling.
          </div>
        </ActiveDatasetCard>
        <Card title="Preprocessing (global)" plane>
          <PrepControls showMetric={false} />
          <div className="small muted" style={{ marginTop: 6 }}>
            "Centre" gives covariance PCA, "Standardise" correlation PCA; "Raw" decomposes <M tex="X" /> itself and is shown for contrast.
          </div>
        </Card>
      </div>

      <FromSVD />
      <Geometry />
      <Scree />
      <Reconstruction />
      <ScalingLesson />
    </div>
  );
}
