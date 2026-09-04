import { useMemo, useState } from 'react';
import { useAnalysis } from '../../../state/store';
import { Section, Card, Callout, Interpretation, StatTile, Derivation } from '../../common/Panels';
import { MatrixView, MatrixEquation } from '../../common/MatrixView';
import { Segmented } from '../../common/Controls';
import { M, MBlock } from '../../common/Math';
import { metricLabels } from '../../../lib/mds';
import { fmt, type Matrix } from '../../../lib/linalg';
import { workedExample, texNum } from './helpers';

type ExampleMode = 'data' | 'rect';

/** A 3–4–5 rectangle: every distance is an integer (sides 3 and 4, diagonals 5), so the whole chain can be checked by hand. */
const RECT: number[][] = [
  [0, 0],
  [3, 0],
  [3, 4],
  [0, 4],
];
const RECT_LABELS = ['A', 'B', 'C', 'D'];

/** MatrixView with a KaTeX title. */
function LabeledMatrix({ tex, M: A, labels, colLabels, heat, digits = 2, caption }: { tex: string; M: Matrix; labels?: string[]; colLabels?: string[]; heat: 'none' | 'diverging' | 'sequential'; digits?: number; caption?: string }) {
  return (
    <div className="matrix-wrap">
      <div className="matrix-title">
        <M tex={tex} />
      </div>
      <MatrixView M={A} rowLabels={labels} colLabels={colLabels ?? labels} digits={digits} heat={heat} compact caption={caption} />
    </div>
  );
}

export default function ChainSection() {
  const a = useAnalysis();
  const { n, prep, mds } = a;
  const [mode, setMode] = useState<ExampleMode>('data');
  const m = Math.min(5, n);
  const ex = useMemo(() => {
    if (mode === 'rect') return workedExample(RECT, 'euclidean', RECT_LABELS);
    return workedExample(
      a.pca.Xc.slice(0, m),
      prep.metric,
      Array.from({ length: m }, (_, i) => `#${i + 1}`),
    );
  }, [mode, a, m, prep.metric]);

  const traceB = useMemo(() => mds.eigenvalues.reduce((s, v) => s + v, 0), [mds.eigenvalues]);
  const maxD = useMemo(() => {
    let mx = 0;
    for (const r of a.D) for (const x of r) if (x > mx) mx = x;
    return mx;
  }, [a.D]);

  const vLabels = ex.labels.map((_, j) => `v${j + 1}`);
  const lamLabels = ex.labels.map((_, j) => `λ${j + 1}`);
  const isEuclid = ex.metric === 'euclidean';
  const eigList = ex.eigen.values.map((v) => texNum(v, 3)).join(',\; ');
  const zeroEig = ex.eigen.values.filter((v) => Math.abs(v) <= Math.max(Math.abs(ex.eigen.values[0] ?? 0), 1e-300) * ex.m * 1e-12).length;
  const pDim = ex.X[0]?.length ?? 0;
  const rankBound = Math.min(ex.m - 1, pDim);
  const Lambda2sqrt = [
    [Math.sqrt(Math.max(ex.eigen.values[0] ?? 0, 0)), 0],
    [0, Math.sqrt(Math.max(ex.eigen.values[1] ?? 0, 0))],
  ];
  const V2 = ex.eigen.vectors.map((r) => [r[0] ?? 0, r[1] ?? 0]);
  const Vt = ex.eigen.vectors[0] ? ex.eigen.vectors[0].map((_, j) => ex.eigen.vectors.map((r) => r[j])) : [];

  return (
    <Section id="mds-chain" title="2 · The transformation chain, and a worked example you can check by hand" subtitle="D → D⁽²⁾ → B → eigendecomposition → coordinates. Each arrow is one line of algebra; each matrix below is computed live.">
      <div className="pipeline">
        <div className="stage">
          <div className="stage-title">1 · Distances</div>
          <M tex="D = (d_{ij})" />
          <div className="small muted">
            <M tex="n\times n" />, symmetric, zero diagonal. Here <M tex={`\\max d_{ij} = ${texNum(maxD, 2)}`} />.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="stage">
          <div className="stage-title">2 · Square entrywise</div>
          <M tex="D^{(2)} = (d_{ij}^2)" />
          <div className="small muted">
            Not <M tex="D\cdot D" />: each entry squared. <M tex={`\\max d_{ij}^2 = ${texNum(maxD * maxD, 2)}`} />.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="stage">
          <div className="stage-title">3 · Double-centre</div>
          <M tex="B = -\tfrac12\, J D^{(2)} J" />
          <div className="small muted">
            <M tex="J = I - \tfrac1n \mathbf 1\mathbf 1^T" />. <M tex={`\\operatorname{tr}(B) = \\sum_j \\lambda_j = ${texNum(traceB, 2)}`} />.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="stage">
          <div className="stage-title">4 · Eigendecompose</div>
          <M tex="B = V \Lambda V^T" />
          <div className="small muted">
            <M tex={`\\lambda_1 = ${texNum(mds.eigenvalues[0] ?? 0, 2)},\; \\lambda_2 = ${texNum(mds.eigenvalues[1] ?? 0, 2)}`} />; {mds.negative} negative.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="stage">
          <div className="stage-title">5 · Coordinates</div>
          <M tex="X_k = V_k \Lambda_k^{1/2}" />
          <div className="small muted">
            <M tex={`k = ${mds.k}`} />, stress-1 <M tex={`= ${texNum(mds.stress1, 3)}`} />.
          </div>
        </div>
      </div>
      <p className="small secondary" style={{ marginTop: 8 }}>
        Read the strip left to right: the live numbers refer to the full <M tex={`${n}\\times ${n}`} /> problem for the active dataset. The worked example below runs the same five steps on a handful of points so that every entry can be verified with a pocket calculator.
      </p>

      <div className="divider" />
      <div className="row between">
        <h3>Worked example</h3>
        <Segmented<ExampleMode>
          compact
          value={mode}
          onChange={setMode}
          options={[
            { value: 'data', label: `First ${m} observations (${metricLabels[prep.metric]})`, hint: 'The first rows of the analysed matrix under the global metric' },
            { value: 'rect', label: '3–4–5 rectangle (Euclidean)', hint: 'Four corners of a 3 × 4 rectangle: all distances are integers' },
          ]}
        />
      </div>
      <p className="small secondary">
        {mode === 'rect' ? (
          <>
            Four points <M tex="A=(0,0),\; B=(3,0),\; C=(3,4),\; D=(0,4)" />: sides 3 and 4, diagonals 5. The centring matrix uses <M tex="m = 4" />, so <M tex="J" /> has <M tex="3/4" /> on the diagonal and <M tex="-1/4" /> elsewhere. Expected eigenvalues of <M tex="B" />: <M tex="16, 9, 0, 0" /> (the centred coordinates are <M tex="\pm 1.5" /> and <M tex="\pm 2" />, and <M tex="4\times 2^2 = 16" />, <M tex="4\times 1.5^2 = 9" />).
          </>
        ) : (
          <>
            The first <M tex={`m = ${m}`} /> rows of the {prep.scaling === 'none' ? 'raw' : prep.scaling === 'center' ? 'centred' : 'standardised'} matrix, distances under the {metricLabels[prep.metric]} metric. Note that <M tex="J" /> centres these <M tex="m" /> points about <em>their own</em> mean, so this is a self-contained MDS of five points, not a sub-configuration of the full solution.
          </>
        )}
      </p>

      <div className="stack" style={{ gap: 16 }}>
        <Card title="Steps 1–2 · distances and their squares" plane>
          <MatrixEquation items={[<LabeledMatrix key="D" tex="D" M={ex.D} labels={ex.labels} heat="sequential" />, '→', <LabeledMatrix key="D2" tex="D^{(2)}" M={ex.D2} labels={ex.labels} heat="sequential" />]} />
        </Card>
        <Card title="Step 3 · double centring" plane>
          <MatrixEquation
            items={[
              <LabeledMatrix key="B" tex="B" M={ex.B} labels={ex.labels} heat="diverging" />,
              '=',
              <M key="half" tex="-\tfrac12" />,
              <LabeledMatrix key="J" tex={`J = I - \\tfrac1{${ex.m}}\\mathbf 1\\mathbf 1^T`} M={ex.J} labels={ex.labels} heat="diverging" />,
              '·',
              <LabeledMatrix key="D2b" tex="D^{(2)}" M={ex.D2} labels={ex.labels} heat="sequential" />,
              '·',
              <LabeledMatrix key="J2" tex="J" M={ex.J} labels={ex.labels} heat="diverging" />,
            ]}
          />
          <div className="small muted" style={{ marginTop: 6 }}>
            Two routes to the same <M tex="B" />: the explicit triple product and the library's row/column-mean formula <M tex="b_{ij} = -\tfrac12 (d_{ij}^2 - \bar d^2_{i\cdot} - \bar d^2_{\cdot j} + \bar d^2_{\cdot\cdot})" /> agree to <M tex={`${texNum(ex.centringCheck, 3)}`} />. Row sums of <M tex="B" /> are zero because <M tex="J\mathbf 1 = 0" />.
          </div>
        </Card>
        <Card title="Step 4 · eigendecomposition of B" plane>
          <MatrixEquation
            items={[
              <LabeledMatrix key="B2" tex="B" M={ex.B} labels={ex.labels} heat="diverging" />,
              '=',
              <LabeledMatrix key="V" tex="V" M={ex.eigen.vectors} labels={ex.labels} colLabels={vLabels} heat="diverging" digits={3} />,
              '·',
              <LabeledMatrix key="L" tex="\Lambda" M={ex.Lambda} labels={lamLabels} heat="diverging" />,
              '·',
              <LabeledMatrix key="Vt" tex="V^T" M={Vt} labels={vLabels} colLabels={ex.labels} heat="diverging" digits={3} />,
            ]}
          />
          <div className="small muted" style={{ marginTop: 6 }}>
            Eigenvalues in decreasing order: <M tex={`\\lambda = (${eigList})`} />. {ex.positive} positive, {zeroEig} numerically zero, {ex.negative} negative. Columns of <M tex="V" /> are unit eigenvectors, each determined only up to sign; those belonging to zero eigenvalues are an arbitrary orthonormal basis of the null space and carry no coordinate information.
          </div>
        </Card>
        <Card title="Step 5 · coordinates and reconstructed distances" plane>
          <MatrixEquation
            items={[
              <LabeledMatrix key="X2" tex="X_2" M={ex.X2} labels={ex.labels} colLabels={['MDS 1', 'MDS 2']} heat="diverging" digits={3} />,
              '=',
              <LabeledMatrix key="V2" tex="V_2" M={V2} labels={ex.labels} colLabels={['v1', 'v2']} heat="diverging" digits={3} />,
              '·',
              <LabeledMatrix key="L2" tex="\Lambda_2^{1/2}" M={Lambda2sqrt} labels={['λ1', 'λ2']} heat="diverging" digits={3} />,
            ]}
          />
          <div className="divider" />
          <MatrixEquation
            items={[
              <LabeledMatrix key="Dh" tex="\hat D = \bigl(\|x_i - x_j\|\bigr)\ \text{from } X_2" M={ex.Dhat} labels={ex.labels} heat="sequential" />,
              'vs',
              <LabeledMatrix key="Dorig" tex="D" M={ex.D} labels={ex.labels} heat="sequential" />,
              '→',
              <LabeledMatrix key="Ddiff" tex="D - \hat D" M={ex.Ddiff} labels={ex.labels} heat="diverging" digits={3} />,
            ]}
          />
          <div className="stats" style={{ marginTop: 10 }}>
            <StatTile label={<M tex="\max_{ij} |d_{ij} - \hat d_{ij}|" />} value={fmt(ex.maxDistErr, 4)} note={isEuclid ? (ex.positive <= 2 ? 'exact: rank(B) ≤ 2' : 'k = 2 < rank(B): truncation error') : 'non-Euclidean input'} />
            <StatTile label={<M tex="\max_{ij} |b_{ij} - x_i^T x_j|" />} value={fmt(ex.gramDiff, 4)} note={isEuclid ? 'B equals the Gram matrix of the centred points' : `not a Gram matrix under ${metricLabels[ex.metric]}`} />
            <StatTile label={<M tex="\operatorname{tr}(B) = \sum_j \lambda_j" />} value={fmt(ex.traceB, 3)} note="total scatter about the centroid" />
            <StatTile label="positive / zero / negative" value={`${ex.positive} / ${zeroEig} / ${ex.negative}`} note={<M tex={`\\operatorname{rank}(B) \\le \\min(m-1,p) = ${rankBound}`} />} />
          </div>
        </Card>
      </div>

      <Interpretation
        items={{
          seeing: (
            <>
              The five steps of classical scaling applied to <M tex={`m = ${ex.m}`} /> points{mode === 'rect' ? ' of a 3 × 4 rectangle' : ` (observations #1–#${ex.m})`} under the {metricLabels[ex.metric]} metric. The eigenvalues of <M tex="B" /> are <M tex={`(${eigList})`} />; the two leading eigenpairs give the coordinates <M tex="X_2" />, whose Euclidean distances <M tex="\hat D" /> differ from <M tex="D" /> by at most {fmt(ex.maxDistErr, 4)}.
            </>
          ),
          why: (
            <>
              {isEuclid ? (
                <>
                  For Euclidean input <M tex="B" /> is the Gram matrix of the centred points: <M tex={`\\max|b_{ij} - x_i^T x_j| = ${texNum(ex.gramDiff, 3)}`} />. Its rank is at most <M tex={`\\min(m-1, p) = ${rankBound}`} /> — the centring costs one dimension because <M tex="B\mathbf 1 = 0" />, so <M tex="\mathbf 1" /> is always an eigenvector with eigenvalue 0 — and <M tex="X_2" /> reproduces <M tex="D" />{' '}
                  {ex.positive <= 2 ? 'exactly, because no more than two eigenvalues are positive.' : `only approximately, because ${ex.positive} eigenvalues are positive and two axes cannot carry all of them.`}
                </>
              ) : (
                <>
                  Under the {metricLabels[ex.metric]} metric the double-centred matrix is still symmetric with zero row sums, but it is not the Gram matrix of any point set: <M tex={`\\max|b_{ij} - x_i^T x_j| = ${texNum(ex.gramDiff, 3)}`} /> against the actual centred coordinates, and {ex.negative > 0 ? `${ex.negative} eigenvalue${ex.negative > 1 ? 's are' : ' is'} negative` : 'the eigenvalues happen to be non-negative for these few points'}. The reconstruction error {fmt(ex.maxDistErr, 4)} is therefore not a truncation effect alone.
                </>
              )}
            </>
          ),
          math: (
            <>
              <MBlock tex="B = -\tfrac12\, J D^{(2)} J = X_c X_c^T,\qquad \hat d_{ij}^2 = b_{ii} + b_{jj} - 2 b_{ij} = d_{ij}^2 \quad\text{(Euclidean } D\text{)}" />
              The derivation below proves both identities. With <M tex="B = V\Lambda V^T" /> and <M tex="\Lambda \succeq 0" />, <M tex="X = V\Lambda^{1/2}" /> satisfies <M tex="XX^T = B" />, so its rows realise <M tex="D" />; keeping only the <M tex="k" /> leading eigenpairs is the best rank-<M tex="k" /> approximation of <M tex="B" /> in Frobenius norm (Eckart–Young).
            </>
          ),
          stats: (
            <>
              <M tex="B" /> is an inner-product matrix of centred observations: <M tex="b_{ii} = \|x_i\|^2" /> is the squared distance of observation <M tex="i" /> from the centroid and <M tex={`\\operatorname{tr}(B) = \\sum_i \\|x_i\\|^2 = ${texNum(ex.traceB, 3)}`} /> is the total scatter, which the eigenvalues partition across MDS axes exactly as eigenvalues of <M tex="S" /> partition variance across principal components. The first axis carries <M tex={`\\lambda_1/\\operatorname{tr}(B) = ${texNum(ex.traceB > 0 ? (ex.eigen.values[0] ?? 0) / ex.traceB : 0, 3)}`} /> of it.
            </>
          ),
          careful: (
            <>
              Signs of eigenvectors are arbitrary, so <M tex="X_2" /> may appear mirrored relative to the original coordinates; eigenvectors of zero eigenvalues are arbitrary altogether. The worked example centres its {ex.m} points about their own mean, so its <M tex="B" /> is not a sub-block of the full <M tex="B" />. And numerically "zero" means below <M tex="\lambda_1 \cdot m \cdot 10^{-12}" />: distinguishing a tiny positive eigenvalue from a rounding artefact requires such a tolerance.
            </>
          ),
        }}
      />

      <div className="divider" />
      <h3>Derivation: why double centring recovers inner products</h3>
      <p className="small secondary">
        Throughout, <M tex="x_i \in \mathbb R^p" /> denotes row <M tex="i" /> of the centred matrix <M tex="X_c" /> written as a column vector, and <M tex="D" /> is its Euclidean distance matrix.
      </p>
      <Derivation
        initiallyRevealed={1}
        steps={[
          {
            title: 'Expand one squared Euclidean distance',
            body: <MBlock tex="d_{ij}^2 = \|x_i - x_j\|^2 = (x_i - x_j)^T (x_i - x_j) = \|x_i\|^2 + \|x_j\|^2 - 2\,x_i^T x_j" />,
            note: 'A pure identity of inner products — centring plays no role yet, and it fails for every non-Euclidean metric (there is no inner product whose induced distance is, say, Manhattan).',
          },
          {
            title: 'Assemble all n² identities into one matrix equation',
            body: (
              <>
                <MBlock tex="D^{(2)} = c\,\mathbf 1^T + \mathbf 1\,c^T - 2\,X_c X_c^T,\qquad c = \bigl(\|x_1\|^2,\dots,\|x_n\|^2\bigr)^T" />
                <M tex="c\,\mathbf 1^T" /> has constant rows (entry <M tex="(i,j)" /> equals <M tex="c_i" />), <M tex="\mathbf 1\,c^T" /> has constant columns (entry <M tex="c_j" />); both are rank one. <M tex="X_c X_c^T" /> is the Gram matrix of inner products <M tex="x_i^T x_j" />.
              </>
            ),
          },
          {
            title: 'The centring matrix J and what it annihilates',
            body: (
              <>
                <MBlock tex="J = I - \tfrac1n \mathbf 1\mathbf 1^T,\qquad J^T = J,\qquad J^2 = J,\qquad J\mathbf 1 = \mathbf 1 - \tfrac1n\,\mathbf 1(\mathbf 1^T\mathbf 1) = \mathbf 1 - \mathbf 1 = 0" />
                <MBlock tex="J\,(c\,\mathbf 1^T)\,J = (Jc)\,(\mathbf 1^T J) = (Jc)\,(J\mathbf 1)^T = 0,\qquad J\,(\mathbf 1\,c^T)\,J = (J\mathbf 1)\,(c^T J) = 0" />
              </>
            ),
            note: 'Left-multiplication by J subtracts column means; right-multiplication subtracts row means. Both rank-one terms are constant along the direction J kills, so they vanish.',
          },
          {
            title: 'J leaves a centred matrix unchanged',
            body: <MBlock tex="\mathbf 1^T X_c = 0 \;\Longrightarrow\; J X_c = X_c - \tfrac1n\,\mathbf 1(\mathbf 1^T X_c) = X_c,\qquad X_c^T J = (J X_c)^T = X_c^T,\qquad\text{hence}\quad J X_c X_c^T J = X_c X_c^T" />,
            note: 'This is the only place centring enters. If X were not centred, J would centre it: B is always the Gram matrix of the centred coordinates, whatever the origin of the original data — distances cannot see translations.',
          },
          {
            title: 'Put the pieces together',
            body: <MBlock tex="B = -\tfrac12\, J D^{(2)} J = -\tfrac12\bigl(0 + 0 - 2\, J X_c X_c^T J\bigr) = X_c X_c^T" />,
            note: 'Every entry of B is an inner product b_ij = x_iᵀx_j of centred observation vectors, recovered from distances alone. The strip above computes exactly this matrix.',
          },
          {
            title: 'Torgerson–Gower theorem: Euclidean if and only if B is positive semi-definite',
            body: (
              <>
                Let <M tex="D" /> be symmetric with zero diagonal. There exist points <M tex="z_1,\dots,z_n" /> in some <M tex="\mathbb R^q" /> with <M tex="\|z_i - z_j\| = d_{ij}" /> if and only if <M tex="B = -\tfrac12 J D^{(2)} J \succeq 0" />. Necessity is step 5. For sufficiency, first note that double centring is undone by a fixed identity — with <M tex="\bar d^2_{i\cdot}, \bar d^2_{\cdot j}, \bar d^2_{\cdot\cdot}" /> the row, column and grand means of <M tex="D^{(2)}" /> and <M tex="d_{ii} = 0" />:
                <MBlock tex="b_{ij} = -\tfrac12\bigl(d_{ij}^2 - \bar d^{2}_{i\cdot} - \bar d^{2}_{\cdot j} + \bar d^{2}_{\cdot\cdot}\bigr) \;\Longrightarrow\; b_{ii} + b_{jj} - 2 b_{ij} = d_{ij}^2 ." />
                If <M tex="B \succeq 0" /> with <M tex="B = V\Lambda V^T" /> and <M tex="q = \operatorname{rank}(B)" />, put <M tex="X = V_q \Lambda_q^{1/2}" />. Then <M tex="XX^T = B" /> and
                <MBlock tex="\|x_i - x_j\|^2 = x_i^Tx_i + x_j^Tx_j - 2x_i^Tx_j = b_{ii} + b_{jj} - 2b_{ij} = d_{ij}^2 ," />
                so the rows of <M tex="X" /> realise <M tex="D" /> exactly, in <M tex="q" /> dimensions, and <M tex="q" /> is the smallest dimension in which this is possible.
              </>
            ),
            note: 'Torgerson (1952) gave the construction; Gower (1966) the characterisation and the link to PCA. Negative eigenvalues of B are therefore a certificate that no Euclidean configuration exists — for any k.',
          },
          {
            title: 'Truncation: what X_k = V_kΛ_k^{1/2} optimises when k < rank(B)',
            body: (
              <>
                <MBlock tex="X_k = V_k \Lambda_k^{1/2} = \arg\min_{X \in \mathbb R^{n\times k}} \|B - XX^T\|_F,\qquad \|B - X_kX_k^T\|_F^2 = \sum_{j>k} \lambda_j^2\quad (B \succeq 0)" />
                Classical MDS minimises the <em>strain</em> — the error in inner products — by Eckart–Young applied to the symmetric matrix <M tex="B" />. It does not minimise Kruskal's stress <M tex="\sqrt{\sum_{i<j}(d_{ij} - \hat d_{ij})^2 / \sum_{i<j} d_{ij}^2}" />, which is the objective of iterative (non-metric) MDS; stress is reported in Section 3 as a diagnostic only. For non-Euclidean <M tex="D" />, classical MDS keeps the leading positive eigenpairs and discards the negative ones, so even <M tex="k = n-1" /> cannot reproduce <M tex="D" />.
              </>
            ),
          },
        ]}
      />
      <Callout kind="definition" title="Vocabulary">
        <b>Classical (metric, Torgerson) MDS</b>: the spectral construction above, exact for Euclidean input. <b>Principal coordinates analysis</b> (Gower): the same thing, the name preferred in ecology. <b>Non-metric MDS</b> (Shepard, Kruskal): iterative minimisation of stress over monotone transformations of the dissimilarities — a different method that shares only the goal of preserving distances.
      </Callout>
    </Section>
  );
}
