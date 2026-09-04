import { useMemo, useState } from 'react';
import { useStore } from '../../../state/store';
import { Section, Card, Callout, Interpretation, StatTile, Badge, ClassLegend, Derivation } from '../../common/Panels';
import { ScatterSVG, type ExtraPoint } from '../../common/ScatterSVG';
import { Segmented } from '../../common/Controls';
import { M, MBlock } from '../../common/Math';
import { distanceMatrix, classicalMDS, procrustesAlign, metricLabels } from '../../../lib/mds';
import { fmt } from '../../../lib/linalg';
import { methodColor } from '../../../lib/theme';
import { classNamesOf, texNum } from './helpers';

type View = 'pca' | 'mds' | 'overlay';

/** Lesson 4 — classical MDS with Euclidean distances reproduces the PCA scores up to a rigid motion. */
export default function VsPcaSection() {
  const { analysis: a, dataset, prep } = useStore();
  const [view, setView] = useState<View>('overlay');
  const classNames = classNamesOf(dataset);
  const Xc = a.pca.Xc;
  const Z2 = useMemo(() => a.pca.scores.map((r) => [r[0] ?? 0, r[1] ?? 0]), [a.pca.scores]);

  const euclid = useMemo(() => classicalMDS(distanceMatrix(Xc, 'euclidean'), 2), [Xc]);
  const coordsE = useMemo(() => euclid.coords.map((r) => [r[0] ?? 0, r[1] ?? 0]), [euclid.coords]);
  const alignE = useMemo(() => procrustesAlign(coordsE, Z2), [coordsE, Z2]);

  const nonEuclidean = prep.metric !== 'euclidean';
  const coordsG = useMemo(() => a.mds.coords.map((r) => [r[0] ?? 0, r[1] ?? 0]), [a.mds.coords]);
  const alignG = useMemo(() => (nonEuclidean ? procrustesAlign(coordsG, Z2) : null), [nonEuclidean, coordsG, Z2]);

  const eigTable = useMemo(() => {
    const m = Math.min(6, a.svd.s.length);
    return Array.from({ length: m }, (_, j) => {
      const s2 = a.svd.s[j] ** 2;
      const lb = euclid.eigenvalues[j] ?? 0;
      return { j: j + 1, s2, lb, diff: Math.abs(s2 - lb) };
    });
  }, [a.svd.s, euclid.eigenvalues]);
  const maxDiff = Math.max(...eigTable.map((r) => r.diff), 0);
  const scaleRef = Math.max(eigTable[0]?.s2 ?? 1, 1e-12);

  const points = view === 'mds' ? coordsE : Z2;
  const extra: ExtraPoint[] = view === 'overlay' ? alignE.aligned.map((r) => ({ x: r[0], y: r[1], shape: 'ring' as const, r: 6, color: methodColor.MDS })) : [];

  return (
    <Section id="mds-vs-pca" title="4 · PCA versus classical MDS: the same configuration" subtitle="For Euclidean distances computed on the centred data, the classical MDS coordinates are the PCA scores — up to sign, rotation within repeated-eigenvalue subspaces, and translation." right={<Badge method="MDS" />}>
      <div className="grid side-r">
        <div className="stack">
          <div className="row between">
            <Segmented<View>
              value={view}
              onChange={setView}
              options={[
                { value: 'pca', label: 'PCA scores Z₂', hint: 'first two columns of X_cV' },
                { value: 'mds', label: 'Classical MDS (Euclidean)', hint: 'V₂Λ₂^{1/2} from B = −½JD⁽²⁾J' },
                { value: 'overlay', label: 'Overlay after Procrustes', hint: 'MDS configuration rotated/reflected onto the PCA scores' },
              ]}
            />
            {classNames && <ClassLegend classNames={classNames} extra={view === 'overlay' ? <span className="item"><svg width="14" height="14" viewBox="-7 -7 14 14"><circle r="5" fill="none" stroke={methodColor.MDS} strokeWidth="2" /></svg>aligned MDS</span> : undefined} />}
          </div>
          <ScatterSVG
            points={points}
            labels={dataset.y}
            classNames={classNames}
            extraPoints={extra}
            width={560}
            height={420}
            xLabel={view === 'mds' ? `MDS 1 (λ₁(B) = ${fmt(euclid.eigenvalues[0] ?? 0, 2)})` : `PC1 (σ₁² = ${fmt(a.svd.s[0] ** 2, 2)})`}
            yLabel={view === 'mds' ? `MDS 2 (λ₂(B) = ${fmt(euclid.eigenvalues[1] ?? 0, 2)})` : `PC2 (σ₂² = ${fmt((a.svd.s[1] ?? 0) ** 2, 2)})`}
            title={view === 'pca' ? 'PCA scores' : view === 'mds' ? 'Classical MDS coordinates from Euclidean distances' : 'PCA scores (filled) with the Procrustes-aligned MDS configuration (rings)'}
          />
        </div>
        <div className="stack">
          <Card title="Agreement (Euclidean MDS vs PCA)">
            <div className="stats">
              <StatTile label="Procrustes relative residual" value={fmt(alignE.relative, 8)} note="‖Z₂ − (X_MDS Q + 1tᵀ)‖_F / ‖Z₂‖_F" />
              <StatTile label="alignment used a reflection" value={alignE.reflection ? 'yes' : 'no'} note="det Q < 0: a sign flip of one axis" />
              <StatTile label="max |λ_j(B) − σ_j²|" value={fmt(maxDiff, 8)} note={`relative to σ₁² = ${fmt(scaleRef, 3)}`} />
              <StatTile label="negative eigenvalues of B" value={euclid.negative} note="exactly zero for Euclidean input (up to rounding)" />
            </div>
          </Card>
          <table className="data-table" style={{ fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>j</th>
                <th>σ_j² (SVD of X_c)</th>
                <th>λ_j(B) (MDS)</th>
                <th>|difference|</th>
              </tr>
            </thead>
            <tbody>
              {eigTable.map((r) => (
                <tr key={r.j}>
                  <td>{r.j}</td>
                  <td>{fmt(r.s2, 5)}</td>
                  <td>{fmt(r.lb, 5)}</td>
                  <td>{fmt(r.diff, 8)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid c2" style={{ marginTop: 14 }}>
        <Derivation
          title="Why the configurations coincide"
          initiallyRevealed={2}
          steps={[
            { title: 'Euclidean squared distances of centred rows', body: <MBlock tex={String.raw`d_{ij}^2 = \|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2\,x_i^Tx_j,\quad x_i \text{ the rows of } X_c.`} />, note: 'Only inner products enter, which is why an inner-product matrix can be recovered from distances.' },
            { title: 'Double centring recovers the Gram matrix', body: <MBlock tex={String.raw`B = -\tfrac12 J D^{(2)} J = J X_c X_c^T J = X_cX_c^T`} />, note: 'J kills the rank-one terms c1ᵀ and 1cᵀ (J1 = 0) and leaves X_c unchanged because its columns already have mean zero (JX_c = X_c).' },
            { title: 'Insert the SVD of the centred data', body: <MBlock tex={String.raw`X_cX_c^T = U\Sigma V^TV\Sigma U^T = U\Sigma^2U^T`} />, note: 'So the non-zero eigenvalues of B are σ_j² with eigenvectors u_j — the table above checks this numerically.' },
            { title: 'Read off the MDS coordinates', body: <MBlock tex={String.raw`X_k^{\text{MDS}} = V_k(B)\,\Lambda_k^{1/2} = U_k\Sigma_k = X_cV_k = Z_k`} />, note: 'The classical MDS coordinates are exactly the first k PCA scores.' },
            { title: 'Where the freedom comes from', body: <>Each eigenvector is determined only up to sign, so any axis may be reflected; if <M tex="\sigma_j = \sigma_{j+1}" /> the eigenvectors span a plane in which any rotation is admissible; and since distances are invariant under rigid motions, translating or rotating the whole configuration changes nothing that MDS can see. Hence the two solutions agree up to <b>rotation, reflection and translation</b> — the Procrustes residual measures what is left after removing exactly these.</>, note: 'The Procrustes rotation Q comes from the SVD of the cross-product of the two centred configurations — the SVD once more.' },
          ]}
        />
        <div className="stack">
          <Callout kind="theorem" title="Gower (1966): classical MDS of Euclidean distances is PCA">
            Let <M tex="D" /> contain the Euclidean distances between the rows of <M tex="X_c" /> (centred, possibly standardised). Then <M tex="B = X_cX_c^T" /> is positive semi-definite, its non-zero eigenvalues are the squared singular values <M tex="\sigma_j^2" /> of <M tex="X_c" />, and the classical MDS configuration in <M tex="k" /> dimensions equals the matrix of the first <M tex="k" /> principal component scores up to an orthogonal transformation of each eigenspace. Conversely, classical MDS on a non-Euclidean dissimilarity matrix has no PCA counterpart: it is a genuinely different analysis, and <M tex="B" /> may be indefinite.
          </Callout>
          <Card title={`When the equivalence fails: the current global metric is ${metricLabels[prep.metric]}`}>
            {nonEuclidean && alignG ? (
              <div className="stack" style={{ gap: 6 }}>
                <div className="stats">
                  <StatTile label="Procrustes residual vs PCA scores" value={fmt(alignG.relative, 4)} note={`${metricLabels[prep.metric]} MDS against Z₂`} />
                  <StatTile label="negative eigenvalues of B" value={a.mds.negative} note={`|negative| mass ${(a.mds.negativeMass * 100).toFixed(2)} %`} />
                  <StatTile label="stress-1 (global metric, k = 2)" value={fmt(classicalMDSStress2(a.mds.k, a.mds.stress1), 4)} />
                </div>
                <div className="small secondary">
                  With a non-Euclidean metric the double-centred matrix is no longer a Gram matrix of <M tex="X_c" />, so nothing forces its eigenvectors to be the left singular vectors of <M tex="X_c" />. The configuration differs from the PCA scores by more than a rigid motion (residual {fmt(alignG.relative, 4)} instead of ≈ 0), and B acquires {a.mds.negative} negative eigenvalue{a.mds.negative === 1 ? '' : 's'}.
                </div>
              </div>
            ) : (
              <div className="small secondary">
                The global metric is Euclidean, so the global MDS configuration coincides with the PCA scores (residual {fmt(alignE.relative, 6)}). Switch the metric to Manhattan or Chebyshev in the preprocessing controls to see the equivalence break: the Procrustes residual becomes clearly positive and <M tex="B" /> acquires negative eigenvalues.
              </div>
            )}
          </Card>
        </div>
      </div>

      <Interpretation
        items={{
          seeing: <>{view === 'overlay' ? 'Filled markers are the PCA scores; rings are the Euclidean-MDS configuration after the best rotation/reflection/translation. They coincide to ' + fmt(alignE.relative, 6) + ' relative residual.' : view === 'pca' ? 'The first two principal component scores Z₂ = X_cV₂.' : 'The two-dimensional classical MDS configuration computed from the Euclidean distance matrix alone.'}</>,
          why: (
            <>
              The distance matrix contains the inner products of the centred rows, and inner products determine the configuration up to a rigid motion. Here the alignment {alignE.reflection ? 'needed a reflection (one MDS axis had the opposite sign from the PCA axis)' : 'needed no reflection'}; the eigenvalues of <M tex="B" /> match <M tex="\sigma_j^2" /> to {fmt(maxDiff, 8)}.
            </>
          ),
          math: (
            <>
              <M tex={`\\lambda_1(B) = ${texNum(euclid.eigenvalues[0] ?? 0)} = \\sigma_1^2`} />, <M tex={`\\lambda_2(B) = ${texNum(euclid.eigenvalues[1] ?? 0)} = \\sigma_2^2`} />; the PCA eigenvalues are these divided by <M tex={`n - 1 = ${a.n - 1}`} />, i.e. <M tex={`\\lambda_1 = ${texNum(a.pca.eigenvalues[0] ?? 0)}`} />.
            </>
          ),
          stats: <>Statistically the two methods answer the same question here — which linear configuration reproduces the Euclidean geometry of the centred data best — so nothing is gained by running both. MDS becomes a different (and useful) tool exactly when the input is not a Euclidean distance matrix of observed variables: perceived dissimilarities, edit distances, correlations between variables, and so on.</>,
          careful: <>Sign flips and rotations within equal-eigenvalue subspaces mean that two correct implementations can print different coordinates for the same data; compare configurations only after Procrustes alignment. Standardisation must be identical on both sides — PCA on the correlation matrix corresponds to MDS on distances between standardised rows, not raw rows.</>,
        }}
      />
    </Section>
  );
}

function classicalMDSStress2(k: number, stress: number): number {
  void k;
  return stress;
}
