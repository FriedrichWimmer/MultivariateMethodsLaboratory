import { useMemo, useState } from 'react';
import { useAnalysis } from '../../state/store';
import { Section, Card, Callout, StatTile, Interpretation, Badge } from '../common/Panels';
import { M, MBlock } from '../common/Math';
import { MatrixView, MatrixEquation } from '../common/MatrixView';
import { ActiveDatasetCard } from '../common/DatasetControls';
import { DecompositionSection } from './svd/Decomposition';
import { GeometrySection } from './svd/Geometry';
import { LowRankSection } from './svd/LowRank';
import { StatisticianSection } from './svd/Statistician';
import { presets, cellsFromMatrix, matrixFromCells, type Cells } from './svd/util';
import { svdChecks } from '../../lib/svdlab';
import { svd, symmetricEigen, gram, determinant, fmt } from '../../lib/linalg';

/**
 * SVD laboratory. Lessons 1–2 work on a small editable matrix; lessons 3–4 on the
 * analysed global data matrix; lesson 5 is the quality-control panel.
 * Section ids: svd-decomposition · svd-geometry · svd-lowrank · svd-statistician · svd-checks
 */
export default function SVDLab() {
  const [cells, setCells] = useState<Cells>(() => cellsFromMatrix(presets[3].A));
  const A = useMemo(() => matrixFromCells(cells), [cells]);
  const a = useAnalysis();
  const r = a.svd.s.length;
  const [kState, setK] = useState(2);
  const k = Math.max(1, Math.min(kState, r));

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="topbar">
        <div className="title">
          <h2>SVD laboratory</h2>
          <div className="lede">
            Every real matrix factorises as <M tex="X = U\Sigma V^T" />: an orthogonal change of basis, a stretch along the axes, and a second orthogonal change of basis. Learn to read the three factors, watch them act on the plane, and see why cutting the sum short gives the best low-rank approximation.
          </div>
        </div>
      </div>
      <ActiveDatasetCard>
        <div className="small muted">
          Lessons 1–2 use the small matrix you edit below; lessons 3–5 decompose the analysed global matrix (n = {a.n}, p = {a.p}, rank {a.svd.rank}).
        </div>
      </ActiveDatasetCard>
      <DecompositionSection cells={cells} setCells={setCells} A={A} />
      <GeometrySection A={A} />
      <LowRankSection k={k} setK={setK} />
      <StatisticianSection k={k} />
      <ChecksSection />
    </div>
  );
}

function sci(x: number): string {
  if (x === 0) return '0';
  return x.toExponential(2).replace('-', '−').replace('e', ' × 10^');
}

function ChecksSection() {
  const a = useAnalysis();
  const checks = useMemo(() => svdChecks(a.pca.Xc, a.svd), [a.pca.Xc, a.svd]);
  const hand = useMemo(() => {
    const A = [
      [3, 0],
      [4, 5],
    ];
    const dec = svd(A);
    const G = gram(A);
    const eig = symmetricEigen(G);
    return { A, dec, G, eig, det: determinant(A) };
  }, []);
  const ok = checks.reconError < 1e-8 * Math.max(1, a.svd.s[0] ?? 1) && checks.uOrthoError < 1e-8 && checks.vOrthoError < 1e-8 && checks.ordered;

  return (
    <Section id="svd-checks" title="5 · Quality control: is the decomposition correct?" subtitle="Numerical verification of the properties every SVD must satisfy, on the matrix currently being analysed, plus a 2×2 example you can check by hand." right={<Badge method="SVD" />}>
      <div className="stats">
        <StatTile label="‖X_c − UΣVᵀ‖_F" value={sci(checks.reconError)} note="reconstruction residual (should be ≈ 0)" />
        <StatTile label="max |UᵀU − I|" value={sci(checks.uOrthoError)} note="orthonormality of the left singular vectors" />
        <StatTile label="max |VᵀV − I|" value={sci(checks.vOrthoError)} note="orthonormality of the right singular vectors" />
        <StatTile label="σ₁ ≥ σ₂ ≥ … ≥ σᵣ" value={checks.ordered ? 'yes' : 'NO'} note="ordering of singular values" />
        <StatTile label="numerical rank" value={`${a.svd.rank} of ${a.svd.s.length}`} note={`tolerance ${fmt(a.svd.tol, 3)}`} />
      </div>
      <Callout kind={ok ? 'good' : 'warning'} title={ok ? 'All checks pass' : 'A check is outside tolerance'}>
        {ok ? (
          <>
            Residuals of order <M tex="10^{-14}" /> to <M tex="10^{-12}" /> are rounding noise in double precision (unit round-off <M tex="\varepsilon \approx 2.2\times10^{-16}" />, amplified by roughly <M tex="\sqrt{np}" /> operations). They are not zero, and they would not be zero in any software.
          </>
        ) : (
          <>The decomposition of the current matrix violates a property; inspect the singular values and the conditioning of the data in the Diagnostics laboratory.</>
        )}
      </Callout>
      <div className="grid c2" style={{ marginTop: 12 }}>
        <div className="stack">
          <h3>Hand-verifiable example</h3>
          <MBlock tex={String.raw`A = \begin{bmatrix}3&0\\4&5\end{bmatrix},\qquad A^TA = \begin{bmatrix}25&20\\20&25\end{bmatrix}`} />
          <div className="prose small">
            <p>
              The eigenvalues of <M tex="A^TA" /> solve <M tex="(25-\mu)^2 = 400" />, hence <M tex="\mu_1 = 45" /> and <M tex="\mu_2 = 5" />, so <M tex="\sigma_1 = \sqrt{45} \approx 6.708" /> and <M tex="\sigma_2 = \sqrt5 \approx 2.236" />. Check: <M tex="\sigma_1\sigma_2 = \sqrt{225} = 15 = |\det A|" />, and the eigenvector of <M tex="\mu_1" /> is <M tex="(1,1)/\sqrt2" />, so <M tex="v_1 = (1,1)/\sqrt2" />.
            </p>
          </div>
        </div>
        <div className="stack">
          <MatrixEquation items={[<MatrixView key="a" M={hand.A} title="A" digits={0} />, <MatrixView key="g" M={hand.G} title="AᵀA (computed)" digits={0} />, <MatrixView key="v" M={hand.dec.V} title="V (computed)" digits={4} />]} />
          <div className="table-scroll">
          <table className="data-table wrap" style={{ fontSize: 12.5, width: 'auto' }}>
            <thead>
              <tr>
                <th>quantity</th>
                <th>by hand</th>
                <th>computed (one-sided Jacobi SVD)</th>
                <th>computed (eigenvalues of AᵀA)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>σ₁</td>
                <td>√45 = 6.7082</td>
                <td>{fmt(hand.dec.s[0], 6)}</td>
                <td>{fmt(Math.sqrt(hand.eig.values[0]), 6)}</td>
              </tr>
              <tr>
                <td>σ₂</td>
                <td>√5 = 2.2361</td>
                <td>{fmt(hand.dec.s[1], 6)}</td>
                <td>{fmt(Math.sqrt(hand.eig.values[1]), 6)}</td>
              </tr>
              <tr>
                <td>σ₁σ₂ = |det A|</td>
                <td>15</td>
                <td>{fmt(hand.dec.s[0] * hand.dec.s[1], 6)}</td>
                <td>{fmt(Math.abs(hand.det), 6)}</td>
              </tr>
            </tbody>
          </table>
          </div>
        </div>
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: <>Five numerical certificates for the SVD of the analysed matrix, and a tiny example whose singular values follow from a quadratic equation.</>,
          why: (
            <>
              A correct SVD must reproduce the matrix, have orthonormal <M tex="U" /> and <M tex="V" />, and list the singular values in decreasing order; the residuals shown ({sci(checks.reconError)}, {sci(checks.uOrthoError)}, {sci(checks.vOrthoError)}) are all at rounding level.
            </>
          ),
          math: (
            <>
              The singular values of <M tex="A" /> are the square roots of the eigenvalues of <M tex="A^TA" /> (or <M tex="AA^T" />), and <M tex="\prod_j\sigma_j = |\det A|" /> for square <M tex="A" />; the laboratory's SVD never forms <M tex="A^TA" /> — it orthogonalises the columns of <M tex="A" /> directly — which is why the two routes can disagree in the last digits when <M tex="A" /> is ill-conditioned.
            </>
          ),
          stats: <>Certifying the decomposition matters because every PCA quantity (eigenvalues, scores, reconstruction errors) is read off this factorisation; an unverified decomposition would propagate silently into every downstream interpretation.</>,
          careful: <>Rounding residuals scale with the size of the matrix entries and with the condition number — compare the residual for the "different scales" dataset (entries of order 100) with the Iris-like data.</>,
        }}
      />
      <Card plane>
        <div className="small muted">Related: the Diagnostics laboratory shows why forming <M tex="X^TX" /> squares the condition number and can lose small singular values entirely.</div>
      </Card>
    </Section>
  );
}
