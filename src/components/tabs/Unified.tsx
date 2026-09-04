/**
 * "One dataset, four questions": live answers from the four methods on the
 * current global dataset, the conceptual connection between them, and four
 * interactive step-by-step derivations (PCA, LDA, MDS, SVD with progressive
 * reconstruction).
 */
import { useMemo, useState, type ReactNode } from 'react';
import type { Data } from 'plotly.js';
import { useStore, type Analysis } from '../../state/store';
import { Section, Card, Callout, Interpretation, Badge, Derivation, StatTile, type DerivationStep } from '../common/Panels';
import { M, MBlock } from '../common/Math';
import { MatrixView, MatrixEquation } from '../common/MatrixView';
import { Slider } from '../common/Controls';
import { Plot } from '../common/Plot';
import { svd, fmt, frobenius, sub, truncatedReconstruction, firstColumns, column, gram, symmetricEigen, inverse, matvec, normalize, dot, vsub, outer, scale as scaleMatrix } from '../../lib/linalg';
import { metricLabels, procrustesAlign } from '../../lib/mds';
import { methodColor } from '../../lib/theme';
import { pct, fmtList, sumSquares, energyFraction, kForThreshold, texNamedVector, texNum, columnCorrelation, scalingWord, methodName } from './compare/helpers';

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Unified() {
  const { analysis } = useStore();
  const a = analysis;

  return (
    <>
      <div className="topbar">
        <div className="title">
          <h2>One dataset, four questions</h2>
          <div className="lede">
            The same matrix <M tex="X_c" /> answers four different questions because the four methods optimise four different notions of information. Below: the live answers for the current dataset, the connection between the methods, and
            each derivation step by step.
          </div>
        </div>
      </div>

      <Section
        id="unified-questions"
        title="Four questions about the current dataset"
        subtitle={
          <>
            <b>{a.dataset.name}</b> — {scalingWord(a.prep.scaling)}, <M tex={`n = ${a.n},\\ p = ${a.p}`} />
            {a.K > 0 ? (
              <>
                , <M tex={`K = ${a.K}`} />
              </>
            ) : (
              ', unlabelled'
            )}
            , metric {metricLabels[a.prep.metric]}, <M tex={`k = ${a.prep.k}`} />. Change any of these in the Data laboratory or the Four-way comparison and the answers update.
          </>
        }
      >
        <div className="grid c2">
          <SVDQuestion a={a} />
          <PCAQuestion a={a} />
          <MDSQuestion a={a} />
          <LDAQuestion a={a} />
        </div>
        <div style={{ marginTop: 12 }}>
          <QuestionsInterpretation a={a} />
        </div>
      </Section>

      <Section id="unified-connection" title="What connects the four methods — and what separates them" subtitle="Four lower-dimensional representations, four different quantities preserved.">
        <Connection a={a} />
      </Section>

      <Section id="derive-pca" title="Derivation: PCA as constrained variance maximisation" subtitle="From the variance of a projection to the eigenproblem of S and its SVD form." right={<Badge method="PCA" />}>
        <PCADerivation a={a} />
      </Section>

      <Section id="derive-lda" title="Derivation: Fisher's criterion as a generalised eigenproblem" subtitle="From the ratio of scatters to a symmetric eigenproblem by whitening." right={<Badge method="LDA" />}>
        <LDADerivation a={a} />
      </Section>

      <Section id="derive-mds" title="Derivation: classical MDS from distances to coordinates" subtitle="Torgerson–Gower: double centring recovers the Gram matrix; its eigendecomposition recovers the configuration." right={<Badge method="MDS" />}>
        <MDSDerivation a={a} />
      </Section>

      <Section id="derive-svd" title="Derivation: existence of the SVD and progressive reconstruction" subtitle="A constructive sketch, then the rank-k approximations built term by term on a block of the live data." right={<Badge method="SVD" />}>
        <SVDDerivation a={a} />
        <div style={{ marginTop: 14 }}>
          <ProgressiveReconstruction a={a} />
        </div>
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Question cards
// ---------------------------------------------------------------------------

function Question({ method, question, children }: { method: 'SVD' | 'PCA' | 'MDS' | 'LDA'; question: ReactNode; children: ReactNode }) {
  return (
    <Card
      title={
        <>
          <Badge method={method} /> {methodName[method]}
        </>
      }
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{question}</div>
      {children}
    </Card>
  );
}

function SVDQuestion({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const s = a.svd.s;
    const r = a.svd.rank;
    const kEff = Math.max(0, Math.min(a.prep.k, s.length));
    const kappa = r > 0 ? s[0] / s[r - 1] : NaN;
    return { s, r, kEff, energy: energyFraction(s, kEff), kappa, normF: Math.sqrt(sumSquares(s)) };
  }, [a]);
  const conditioning = !Number.isFinite(d.kappa) ? 'rank-deficient' : d.kappa < 10 ? 'well conditioned' : d.kappa < 1000 ? 'moderately conditioned' : 'ill conditioned';
  return (
    <Question method="SVD" question="What is the fundamental structure of this matrix?">
      <div className="stats">
        <StatTile label={<>numerical rank <M tex="r" /></>} value={d.r} note={`of min(n, p) = ${Math.min(a.n, a.p)}`} />
        <StatTile label={<M tex="\kappa_2 = \sigma_1/\sigma_r" />} value={fmt(d.kappa, 2)} note={conditioning} />
        <StatTile label={<M tex={`\\sum_{j\\le ${d.kEff}}\\sigma_j^2\\big/\\sum_j\\sigma_j^2`} />} value={pct(d.energy)} note={`k = ${d.kEff}`} />
      </div>
      <div className="small" style={{ marginTop: 8 }}>
        Singular values <M tex="\sigma_1\ge\sigma_2\ge\cdots" />: <span className="mono">{fmtList(d.s, 3, 8)}</span>
      </div>
      <div className="small secondary" style={{ marginTop: 6 }}>
        Answer: <M tex="X_c" /> is a sum of <M tex={`r = ${d.r}`} /> rank-one layers <M tex="\sigma_ju_jv_j^T" />; the first <M tex={`k = ${d.kEff}`} /> of them carry {pct(d.energy)} of <M tex={`\\|X_c\\|_F^2 = ${texNum(d.normF ** 2, 2)}`} />, and the matrix is{' '}
        {conditioning} (<M tex={`\\kappa_2 = ${texNum(d.kappa, 2)}`} />). No statistical assumption was used.
      </div>
    </Question>
  );
}

function PCAQuestion({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const P = a.pca;
    const v1 = column(P.V, 0);
    const e1 = P.explained[0] ?? 0;
    const e2 = P.explained[1] ?? 0;
    const k90 = kForThreshold(P.cumulative, 0.9);
    const top = v1.map((x, j) => ({ x, j })).sort((u, v) => Math.abs(v.x) - Math.abs(u.x))[0];
    return { v1, e1, e2, k90, top, reached: (P.cumulative[k90 - 1] ?? 0) >= 0.9 - 1e-12 };
  }, [a]);
  const names = a.dataset.variableNames;
  return (
    <Question method="PCA" question="What directions explain the greatest variation?">
      <MBlock tex={`v_1 = ${texNamedVector(d.v1, names, 3)}`} />
      <div className="stats">
        <StatTile label="PC1" value={pct(d.e1)} note={`λ₁ = ${fmt(a.pca.eigenvalues[0] ?? 0, 3)}`} />
        <StatTile label="PC2" value={pct(d.e2)} note={`λ₂ = ${fmt(a.pca.eigenvalues[1] ?? 0, 3)}`} />
        <StatTile label={<><M tex="k" /> for <M tex="\ge 90\%" /></>} value={d.k90} note={d.reached ? `cumulative ${pct(a.pca.cumulative[d.k90 - 1] ?? 0)}` : 'never reached'} />
      </div>
      <div className="small secondary" style={{ marginTop: 6 }}>
        Answer: the unit direction of greatest variance is <M tex="v_1" /> above, dominated by <em>{names[d.top?.j ?? 0]}</em> (loading <M tex={`${texNum(d.top?.x ?? 0, 3)}`} />); it explains {pct(d.e1)} of <M tex="\operatorname{tr}S" />, the first two together{' '}
        {pct(d.e1 + d.e2)}, and {d.k90} component{d.k90 === 1 ? '' : 's'} {d.reached ? 'are' : 'would be'} needed to reach 90%.{' '}
        {a.prep.scaling === 'none' && (
          <>
            With raw preprocessing <M tex="X_c" /> is uncentred, so “variance” means the second moment about the origin.
          </>
        )}
      </div>
    </Question>
  );
}

function MDSQuestion({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const Md = a.mds;
    const k = Md.k;
    const cols = a.pca.scores[0]?.length ?? 0;
    const proc = k > 0 && cols >= k ? procrustesAlign(Md.coords, firstColumns(a.pca.scores, k)) : null;
    return { k, proc, same: proc !== null && proc.relative < 1e-6 };
  }, [a]);
  const metric = metricLabels[a.prep.metric];
  return (
    <Question method="MDS" question="What low-dimensional configuration best represents the observed dissimilarities?">
      <div className="stats">
        <StatTile label="metric" value={<span style={{ fontSize: 15 }}>{metric}</span>} note={`D is ${a.n} × ${a.n}`} />
        <StatTile label={<>stress-1 at <M tex={`k = ${d.k}`} /></>} value={fmt(a.mds.stress1, 4)} note={`strain ${fmt(a.mds.strain, 4)}`} />
        <StatTile label={<>negative eigenvalues of <M tex="B" /></>} value={a.mds.negative} note={`negative mass ${pct(a.mds.negativeMass)}`} />
      </div>
      <div className="small" style={{ marginTop: 8 }}>
        Procrustes residual against the first {d.k} PCA scores:{' '}
        {d.proc ? (
          <>
            <M tex={`\\|X_k^{\\text{MDS}}Q + 1t^T - Z_k\\|_F = ${texNum(d.proc.residual, 4)}`} /> (relative <M tex={`${texNum(d.proc.relative, 3)}`} />, <M tex="Q" /> {d.proc.reflection ? 'includes a reflection' : 'is a rotation'}).
          </>
        ) : (
          <>not available (no positive eigenvalue).</>
        )}
      </div>
      <div className="small secondary" style={{ marginTop: 6 }}>
        Answer: the configuration <M tex="X_k = V_k\Lambda_k^{1/2}" /> in <M tex={`k = ${d.k}`} /> dimensions reproduces the {metric} distances with stress-1 <M tex={`${texNum(a.mds.stress1, 4)}`} />.{' '}
        {d.same ? (
          <>
            It coincides with the PCA scores up to a rigid motion — as it must for Euclidean distances of the centred data, since then <M tex="B = X_cX_c^T" />.
          </>
        ) : (
          <>
            It does <em>not</em> coincide with the PCA scores{a.prep.metric !== 'euclidean' ? <>: the {metric} metric makes <M tex="B" /> differ from <M tex="X_cX_c^T" /> ({a.mds.negative} negative eigenvalues).</> : a.prep.scaling === 'none' ? (
              <>
                : <M tex="J" /> centres the configuration while the raw PCA scores are uncentred projections.
              </>
            ) : (
              '.'
            )}
          </>
        )}
      </div>
    </Question>
  );
}

function LDAQuestion({ a }: { a: Analysis }) {
  const L = a.lda;
  const names = a.dataset.variableNames;
  const d = useMemo(() => {
    if (!L || L.maxDims === 0) return null;
    return { w1: column(L.W, 0), theoretical: Math.min(L.K - 1, L.swRank) };
  }, [L]);
  return (
    <Question method="LDA" question="What directions best separate the known groups?">
      {!L ? (
        <Callout kind="warning" title="Unlabelled data">
          The current dataset has no class labels; <M tex="S_B" /> and <M tex="S_W" /> are undefined and the question has no answer. The other three questions do not depend on labels.
        </Callout>
      ) : !d ? (
        <Callout kind="warning" title="Degenerate scatter">
          <M tex="\operatorname{rank}S_B" /> or <M tex="\operatorname{rank}S_W" /> is zero for this dataset, so no discriminant direction exists.
        </Callout>
      ) : (
        <>
          <MBlock tex={`w_1 = ${texNamedVector(d.w1, names, 3)}`} />
          <div className="stats">
            <StatTile label="Fisher eigenvalues" value={<span style={{ fontSize: 15 }}>{fmtList(L.eigenvalues, 3, 4)}</span>} note={<M tex="\lambda_j = J(w_j)" />} />
            <StatTile label="maxDims" value={L.maxDims} note={`min(K − 1, rank S_W) = min(${L.K - 1}, ${L.swRank}) = ${d.theoretical}`} />
            <StatTile label="LD1 share" value={pct(L.explained[0] ?? 0)} note={<>of <M tex="\sum_j\lambda_j" /></>} />
          </div>
          <div className="small secondary" style={{ marginTop: 6 }}>
            Answer: along <M tex="w_1" /> the between-class scatter is <M tex={`\\lambda_1 = ${texNum(L.eigenvalues[0], 3)}`} /> times the within-class scatter; {L.maxDims} discriminant direction{L.maxDims === 1 ? '' : 's'} exist{L.maxDims === 1 ? 's' : ''} for{' '}
            <M tex={`K = ${L.K}`} /> classes. Unlike PCA, this direction was chosen using the labels and is invariant to rescaling the variables.
          </div>
        </>
      )}
    </Question>
  );
}

function QuestionsInterpretation({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const P = a.pca;
    const L = a.lda;
    const v1 = column(P.V, 0);
    const cos = L && L.maxDims > 0 ? Math.abs(dot(normalize(v1), normalize(column(L.W, 0)))) : NaN;
    const kEff = Math.max(0, Math.min(a.prep.k, a.svd.s.length));
    return { cos, kEff, energy: energyFraction(a.svd.s, kEff), e1: P.explained[0] ?? 0 };
  }, [a]);
  return (
    <Interpretation
      title="Reading the four answers together"
      defaultOpen={false}
      items={{
        seeing: (
          <>
            Four summaries of one matrix: an algebraic one (rank {a.svd.rank}, {pct(d.energy)} of <M tex="\|X_c\|_F^2" /> in {d.kEff} terms), a variance one (PC1 explains {pct(d.e1)}), a distance one (stress-1 <M tex={`${texNum(a.mds.stress1, 4)}`} /> in{' '}
            {a.mds.k} dimensions) and{a.lda ? <> a class-separation one (<M tex={`\\lambda_1 = ${texNum(a.lda.eigenvalues[0] ?? 0, 3)}`} />).</> : ' — with no labels — no class-separation one.'}
          </>
        ),
        why: (
          <>
            {a.prep.scaling !== 'none' && a.prep.metric === 'euclidean' ? (
              <>
                SVD, PCA and MDS agree here because they decompose the same centred matrix: the SVD share {pct(d.energy)} equals the cumulative PCA proportion for <M tex={`k = ${d.kEff}`} /> ({pct(a.pca.cumulative[d.kEff - 1] ?? 0)}), and the MDS
                coordinates are the PCA scores.
              </>
            ) : (
              <>
                The three unsupervised answers separate here: {a.prep.scaling === 'none' ? 'the raw matrix is uncentred, so SVD/PCA include the mean while MDS centres through J' : `the ${metricLabels[a.prep.metric]} metric makes B differ from X_cX_c^T`}.
              </>
            )}
          </>
        ),
        math: (
          <>
            {Number.isFinite(d.cos) ? (
              <>
                The angle between <M tex="v_1" /> (maximal variance) and <M tex="w_1" /> (maximal separation) has <M tex={`|\\cos\\theta| = ${texNum(d.cos, 3)}`} /> — {d.cos > 0.95 ? 'nearly the same direction: here variance and separation happen to align.' : 'the two criteria point in different directions, as they are entitled to.'}
              </>
            ) : (
              <>
                <M tex="v_1" /> maximises <M tex="w^TSw" />; <M tex="w_1" /> would maximise <M tex="w^TS_Bw/w^TS_Ww" /> — a different quadratic form, hence in general a different direction.
              </>
            )}
          </>
        ),
        stats: (
          <>
            Only PCA's proportions and LDA's eigenvalues are statistics of a sample; the SVD numbers describe the matrix and the MDS stress describes the fit of a configuration. None of them is an estimate with a standard error until a model is
            specified.
          </>
        ),
        careful: (
          <>
            Comparing “% explained” across methods is meaningless: the PCA proportion is of <M tex="\operatorname{tr}S" />, the MDS proportion is of the positive eigenvalue mass of <M tex="B" />, and LDA's share is of <M tex="\sum_j\lambda_j" /> — three
            different totals.
          </>
        ),
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// The connection
// ---------------------------------------------------------------------------

function Connection({ a }: { a: Analysis }) {
  const euclidCentred = a.prep.metric === 'euclidean' && a.prep.scaling !== 'none';
  return (
    <div className="prose">
      <p>
        All four methods construct a lower-dimensional representation of the same <M tex="n\times p" /> matrix, and all four obtain it from the eigendecomposition or singular value decomposition of a symmetric matrix built from the data. They differ in{' '}
        <em>which information they promise to preserve</em> — and therefore in which matrix they decompose, which objective they optimise, and what they are entitled to claim about the result.
      </p>
      <p>
        <b>SVD.</b> The singular value decomposition is a statement about a matrix, not about a population. <M tex="X = U\Sigma V^T" /> exists for every real matrix, and the truncation <M tex="X_k = U_k\Sigma_kV_k^T" /> is the best rank-<M tex="k" /> approximation
        simultaneously in the Frobenius and in the spectral norm (Eckart–Young–Mirsky): <M tex="\|X - X_k\|_F = \sqrt{\textstyle\sum_{j>k}\sigma_j^2}" /> and <M tex="\|X - X_k\|_2 = \sigma_{k+1}" />. There is no random variable, no centring convention and
        no notion of distance in the theorem — all of those are imported by the user through the choice of <M tex="X" />.
      </p>
      <p>
        <b>PCA.</b> Principal component analysis adds a statistical reading. Centre the columns so that <M tex="S = X_c^TX_c/(n-1)" /> is a covariance matrix, and interpret the right singular vectors of <M tex="X_c" /> as directions of maximal variance: the
        unit <M tex="w" /> maximising <M tex="\operatorname{Var}(X_cw) = w^TSw" /> is <M tex="v_1" />, and the successive orthogonal maximisers are <M tex="v_2, v_3,\dots" />. Equivalently, <M tex="\hat X_k = Z_kV_k^T" /> minimises the reconstruction error{' '}
        <M tex="\|X_c - \hat X_k\|_F" /> among rank-<M tex="k" /> matrices — which is precisely the SVD statement applied to <M tex="X_c" />. PCA is the SVD of the centred data read through the variance, with <M tex="\lambda_j = \sigma_j^2/(n-1)" /> the variances of the
        scores.
      </p>
      <p>
        <b>MDS.</b> Classical multidimensional scaling starts one step later: it never sees coordinates, only a dissimilarity matrix <M tex="D" />. Double centring the squared dissimilarities gives <M tex="B = -\tfrac12 JD^{(2)}J" />. When <M tex="D" /> is the
        Euclidean distance matrix of some configuration, <M tex="B = X_cX_c^T" /> is the Gram matrix of that configuration after centring, and the top-<M tex="k" /> eigenvectors scaled by <M tex="\sqrt{\lambda_j}" /> minimise the strain{' '}
        <M tex="\|B - XX^T\|_F" />. Because <M tex="X_cX_c^T = U\Sigma^2U^T" />, the coordinates <M tex="X_k = U_k\Sigma_k" /> are exactly the PCA scores: MDS and PCA are the same decomposition seen from the <M tex="n\times n" /> side (Gower's duality). For a
        non-Euclidean <M tex="D" />, <M tex="B" /> has negative eigenvalues, the equivalence breaks, and what is preserved is the pairwise geometry only approximately — not the variance of anything.{' '}
        {euclidCentred ? (
          <>
            For the current settings ({metricLabels[a.prep.metric]}, {scalingWord(a.prep.scaling)}) the equivalence holds exactly; <M tex="B" /> has {a.mds.negative} negative eigenvalues.
          </>
        ) : (
          <>
            For the current settings ({metricLabels[a.prep.metric]}, {scalingWord(a.prep.scaling)}) the equivalence does not hold: <M tex="B" /> has {a.mds.negative} negative eigenvalues{a.prep.scaling === 'none' ? ' and the raw PCA scores are uncentred' : ''}.
          </>
        )}
      </p>
      <p>
        <b>LDA.</b> Fisher's linear discriminant analysis changes the question. It uses the labels to split the total scatter, <M tex="S_T = S_W + S_B = (n-1)S" />, and seeks directions along which between-class scatter is large relative to within-class
        scatter: <M tex="J(w) = w^TS_Bw/w^TS_Ww" />. This is the generalised eigenproblem <M tex="S_Bw = \lambda S_Ww" />, i.e. the eigenproblem of the non-symmetric <M tex="S_W^{-1}S_B" />; whitening with <M tex="S_W = LL^T" /> turns it into the symmetric problem{' '}
        <M tex="L^{-1}S_BL^{-T}u = \lambda u" /> with <M tex="w = L^{-T}u" />. What is preserved is class separation measured in the Mahalanobis geometry of <M tex="S_W" />; everything unrelated to the labels — including most of the variance — is discarded. It
        requires <M tex="S_W^{-1}" /> (so <M tex="n - K \ge p" />) and yields at most <M tex="K-1" /> directions because <M tex="\operatorname{rank}S_B\le K-1" />.
      </p>

      <div style={{ overflowX: 'auto', margin: '12px 0' }}>
        <table className="summary-table">
          <thead>
            <tr>
              <th />
              <th>
                <Badge method="SVD" />
              </th>
              <th>
                <Badge method="PCA" />
              </th>
              <th>
                <Badge method="MDS" />
              </th>
              <th>
                <Badge method="LDA" />
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>What is preserved</td>
              <td>
                The matrix itself, to within <M tex="\|X - X_k\|_F = \sqrt{\sum_{j>k}\sigma_j^2}" />: its dominant singular directions
              </td>
              <td>
                The variance along the retained directions, <M tex="\sum_{j\le k}\lambda_j" /> out of <M tex="\operatorname{tr}S" />
              </td>
              <td>
                The pairwise squared distances, through the inner products in <M tex="B" /> — up to a rigid motion
              </td>
              <td>The ratio of between- to within-class scatter along the retained directions</td>
            </tr>
            <tr>
              <td>What is discarded</td>
              <td>
                The trailing triplets <M tex="(\sigma_j, u_j, v_j)" />, <M tex="j>k" />
              </td>
              <td>Low-variance directions, whatever their relevance</td>
              <td>
                Small and negative eigendirections of <M tex="B" /> — the non-Euclidean part of <M tex="D" />
              </td>
              <td>
                All variation not aligned with differences between class means; anything outside the range of <M tex="S_W" />
              </td>
            </tr>
            <tr>
              <td>Which matrix is decomposed</td>
              <td>
                <M tex="X" /> (<M tex="n\times p" />) directly
              </td>
              <td>
                <M tex="S = X_c^TX_c/(n-1)" /> (<M tex="p\times p" />), or <M tex="X_c" /> by SVD
              </td>
              <td>
                <M tex="B = -\tfrac12 JD^{(2)}J" /> (<M tex="n\times n" />), equal to <M tex="X_cX_c^T" /> in the Euclidean case
              </td>
              <td>
                <M tex="S_W^{-1}S_B" /> (<M tex="p\times p" />), symmetrised as <M tex="L^{-1}S_BL^{-T}" />
              </td>
            </tr>
            <tr>
              <td>Which objective</td>
              <td>
                <M tex="\min_{\operatorname{rank}A\le k}\|X - A\|" /> in Frobenius or spectral norm
              </td>
              <td>
                <M tex="\max_{\|w\|=1}w^TSw" />, equivalently <M tex="\min\|X_c - \hat X_k\|_F" />
              </td>
              <td>
                <M tex="\min_{X}\|B - XX^T\|_F" /> (strain)
              </td>
              <td>
                <M tex="\max_w\, w^TS_Bw/w^TS_Ww" />
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <Callout kind="theorem" title="Shared mechanism">
        Each method is an eigenproblem of a symmetric — or symmetrised — matrix. PCA diagonalises <M tex="X_c^TX_c" /> (<M tex="p\times p" />); MDS diagonalises <M tex="X_cX_c^T = B" /> (<M tex="n\times n" />); the SVD of <M tex="X_c" /> delivers both at once,
        since <M tex="X_c^TX_c = V\Sigma^2V^T" /> and <M tex="X_cX_c^T = U\Sigma^2U^T" />; LDA diagonalises <M tex="S_W^{-1}S_B" />, which is not symmetric but is similar to the symmetric <M tex="L^{-1}S_BL^{-T}" /> after whitening. The spectral theorem is what
        guarantees real eigenvalues, orthogonal eigenvectors and nested optimal subspaces in every case.
      </Callout>
      <Callout kind="warning" title="Shared limitation: linearity">
        Every representation is a <em>linear</em> image of the data — scores <M tex="X_cV" />, <M tex="U_k\Sigma_k" />, <M tex="(X-1m^T)W" /> — and classical MDS reproduces distances by a configuration whose geometry is that of a linear inner-product space.
        Curved structure (a manifold, a ring of classes, an XOR-type pattern) is flattened by all four; the number of retained dimensions is a linear notion of dimension, not an intrinsic one. All four also rely on squared criteria and inherit their
        sensitivity to outliers.
      </Callout>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Derivation: PCA
// ---------------------------------------------------------------------------

function PCADerivation({ a }: { a: Analysis }) {
  const P = a.pca;
  const d = useMemo(() => {
    const l1 = P.eigenvalues[0] ?? 0;
    const l2 = P.eigenvalues[1] ?? 0;
    const s1 = P.singularValues[0] ?? 0;
    const eigDirect = P.eigen.values[0] ?? 0;
    const corr = (P.scores[0]?.length ?? 0) >= 2 ? columnCorrelation(P.scores, 0, 1) : NaN;
    return { l1, l2, s1, eigDirect, corr, fromSvd: (s1 * s1) / Math.max(a.n - 1, 1) };
  }, [P, a.n]);

  const steps: DerivationStep[] = [
    {
      title: 'Variance of a projection',
      body: (
        <>
          Project the centred data on a direction <M tex="w\in\mathbb R^p" />: <M tex="z = X_cw" />. Since <M tex="1^TX_c = 0" />, the sample mean of <M tex="z" /> is zero and
          <MBlock tex="\operatorname{Var}(z) = \frac{1}{n-1}z^Tz = \frac{1}{n-1}w^TX_c^TX_cw = w^TSw,\qquad S = \frac{1}{n-1}X_c^TX_c." />
        </>
      ),
      note: (
        <>
          Centring is what makes <M tex="w^TSw" /> a variance. For the current data <M tex={`\\operatorname{tr}S = ${texNum(P.totalVariance, 3)}`} /> is the total variance to be distributed among the components.
          {a.prep.scaling === 'none' && <> (The tab is currently in raw mode, where <M tex="X_c" /> is uncentred and <M tex="w^TSw" /> is a second moment about the origin.)</>}
        </>
      ),
    },
    {
      title: 'The constraint',
      body: (
        <>
          <M tex="w^TSw" /> scales as <M tex="c^2" /> under <M tex="w\mapsto cw" />, so without a constraint the supremum is infinite. The question is about a direction, hence
          <MBlock tex="\max_{w\in\mathbb R^p}\ w^TSw\qquad\text{subject to}\qquad w^Tw = 1." />
        </>
      ),
    },
    {
      title: 'Lagrangian and stationarity',
      body: <MBlock tex="\mathcal L(w,\lambda) = w^TSw - \lambda\,(w^Tw - 1),\qquad \nabla_w\mathcal L = 2Sw - 2\lambda w = 0." />,
      note: (
        <>
          The gradient of <M tex="w^TSw" /> is <M tex="2Sw" /> because <M tex="S" /> is symmetric; <M tex="\lambda" /> is the multiplier of the unit-norm constraint.
        </>
      ),
    },
    {
      title: 'An eigenvalue problem',
      body: (
        <>
          <MBlock tex="Sw = \lambda w,\qquad w^TSw = \lambda\,w^Tw = \lambda." />
          Every stationary point is a unit eigenvector of <M tex="S" /> and the variance attained is the corresponding eigenvalue. The maximum is therefore the largest eigenvalue <M tex="\lambda_1" />, attained at <M tex="w = v_1" />.
        </>
      ),
      note: (
        <>
          Here <M tex={`\\lambda_1 = ${texNum(d.l1, 4)}`} />, i.e. PC1 has standard deviation <M tex={`\\sqrt{\\lambda_1} = ${texNum(Math.sqrt(Math.max(d.l1, 0)), 3)}`} /> and explains <M tex={`\\lambda_1/\\operatorname{tr}S = ${pct(P.explained[0] ?? 0)}`} />.
        </>
      ),
    },
    {
      title: 'The Rayleigh quotient bound',
      body: (
        <>
          Expand <M tex="w = \sum_j c_jv_j" /> in the orthonormal eigenbasis of <M tex="S" />:
          <MBlock tex="R(w) = \frac{w^TSw}{w^Tw} = \frac{\sum_j\lambda_jc_j^2}{\sum_jc_j^2}\ \le\ \lambda_1," />
          with equality iff <M tex="w" /> lies in the eigenspace of <M tex="\lambda_1" />. This confirms that the stationary point <M tex="v_1" /> is the global maximum, not merely a critical point.
        </>
      ),
      note: (
        <>
          Courant–Fischer generalises the bound: <M tex="\lambda_j = \max_{\dim\mathcal U = j}\ \min_{w\in\mathcal U\setminus\{0\}} R(w)" />, which is why the optimal <M tex="k" />-dimensional subspace is spanned by <M tex="v_1,\dots,v_k" /> for every{' '}
          <M tex="k" /> (nested solutions).
        </>
      ),
    },
    {
      title: 'Second component: orthogonality to the first',
      body: (
        <>
          Maximise <M tex="w^TSw" /> subject to <M tex="w^Tw = 1" /> and <M tex="w^Tv_1 = 0" />:
          <MBlock tex="\mathcal L = w^TSw - \lambda(w^Tw - 1) - \mu\,w^Tv_1,\qquad 2Sw - 2\lambda w - \mu v_1 = 0." />
          Multiplying from the left by <M tex="v_1^T" /> gives <M tex="2v_1^TSw - \mu = 2\lambda_1v_1^Tw - \mu = -\mu" />, so <M tex="\mu = 0" /> and again <M tex="Sw = \lambda w" />. Among unit eigenvectors orthogonal to <M tex="v_1" /> the variance is maximised by{' '}
          <M tex="v_2" /> with value <M tex="\lambda_2" />.
        </>
      ),
      note: (
        <>
          Here <M tex={`\\lambda_2 = ${texNum(d.l2, 4)}`} /> ({pct(P.explained[1] ?? 0)} of the total). Uniqueness of <M tex="v_2" /> requires <M tex="\lambda_2 > \lambda_3" />; the sign of each <M tex="v_j" /> is never determined by the problem.
        </>
      ),
    },
    {
      title: 'Principal components are uncorrelated',
      body: <MBlock tex="\operatorname{Cov}(z_i, z_j) = \frac{1}{n-1}v_i^TX_c^TX_cv_j = v_i^TSv_j = \lambda_j\,v_i^Tv_j = 0\qquad (i\ne j)." />,
      note: (
        <>
          On the current scores the sample correlation between PC1 and PC2 is <M tex={`${texNum(d.corr, 4)}`} /> — zero to rounding, exactly as the algebra predicts.
        </>
      ),
    },
    {
      title: 'Link to the SVD',
      body: (
        <>
          <MBlock tex="X_c = U\Sigma V^T\ \Rightarrow\ S = \frac{1}{n-1}V\Sigma^2V^T,\qquad \lambda_j = \frac{\sigma_j^2}{n-1},\qquad Z = X_cV = U\Sigma." />
          The right singular vectors are the loadings and the left singular vectors are the scores scaled to unit norm. Computing PCA through the SVD avoids forming <M tex="S" /> and squaring the condition number.
        </>
      ),
      note: (
        <>
          Check on the live data: <M tex={`\\sigma_1^2/(n-1) = ${texNum(d.s1, 4)}^2/${a.n - 1} = ${texNum(d.fromSvd, 4)}`} />, while the largest eigenvalue of <M tex="S" /> computed directly by Jacobi rotations is <M tex={`${texNum(d.eigDirect, 4)}`} />; the two routes
          differ by <M tex={`${texNum(Math.abs(d.fromSvd - d.eigDirect), 2)}`} />.
        </>
      ),
    },
  ];
  return <Derivation steps={steps} />;
}

// ---------------------------------------------------------------------------
// Derivation: LDA
// ---------------------------------------------------------------------------

function LDADerivation({ a }: { a: Analysis }) {
  const L = a.lda;
  const d = useMemo(() => {
    if (!L) return null;
    const sbTol = 1e-10 * Math.max(Math.abs(L.sbEigenvalues[0] ?? 0), 1e-300);
    const sbRank = L.sbEigenvalues.filter((v) => v > sbTol).length;
    let twoClass: { cos: number; w: number[] } | null = null;
    if (L.K === 2 && L.maxDims >= 1) {
      const inv = inverse(L.SW);
      if (inv) {
        const w = normalize(matvec(inv, vsub(L.classMeans[0], L.classMeans[1])));
        twoClass = { cos: Math.abs(dot(w, column(L.W, 0))), w };
      }
    }
    return { sbRank, twoClass, l1: L.eigenvalues[0] ?? 0 };
  }, [L]);

  const steps: DerivationStep[] = [
    {
      title: "Fisher's criterion",
      body: (
        <>
          With class means <M tex="m_k" />, sizes <M tex="n_k" /> and grand mean <M tex="m" />, define the within- and between-class scatter matrices
          <MBlock tex="S_W = \sum_{k=1}^K\sum_{i\in k}(x_i - m_k)(x_i - m_k)^T,\qquad S_B = \sum_{k=1}^K n_k(m_k - m)(m_k - m)^T." />
          For a projection <M tex="z = Xw" /> the between- and within-class scatters of <M tex="z" /> are <M tex="w^TS_Bw" /> and <M tex="w^TS_Ww" />, and Fisher's criterion is their ratio
          <MBlock tex="J(w) = \frac{w^TS_Bw}{w^TS_Ww}." />
        </>
      ),
      note: (
        <>
          <M tex="S_T = S_W + S_B = (n-1)S" /> — the total scatter that PCA analyses is exactly what LDA splits.{' '}
          {L ? (
            <>
              Current data: <M tex={`K = ${L.K}`} />, class sizes {L.classSizes.join(', ')}.
            </>
          ) : (
            <>The current dataset is unlabelled, so the live checks below are unavailable.</>
          )}
        </>
      ),
    },
    {
      title: 'Scale invariance of J',
      body: (
        <>
          <MBlock tex="J(cw) = \frac{c^2\,w^TS_Bw}{c^2\,w^TS_Ww} = J(w)\qquad (c\ne 0)." />
          Only the direction of <M tex="w" /> matters, so the maximisation is over lines through the origin and we are free to impose a normalisation.
        </>
      ),
    },
    {
      title: 'Fix the within-class scatter',
      body: (
        <>
          Use the normalisation <M tex="w^TS_Ww = 1" /> — the natural one, since it turns the problem into
          <MBlock tex="\max_w\ w^TS_Bw\qquad\text{subject to}\qquad w^TS_Ww = 1." />
        </>
      ),
      note: (
        <>
          Equivalent formulations: fix <M tex="w^TS_Bw = 1" /> and minimise <M tex="w^TS_Ww" />, or maximise the ratio directly. Fixing <M tex="w^Tw = 1" /> instead would <em>not</em> lead to an ordinary eigenproblem.
        </>
      ),
    },
    {
      title: 'Lagrangian and stationarity',
      body: <MBlock tex="\mathcal L(w,\lambda) = w^TS_Bw - \lambda\,(w^TS_Ww - 1),\qquad \nabla_w\mathcal L = 2S_Bw - 2\lambda S_Ww = 0." />,
    },
    {
      title: 'A generalised eigenproblem',
      body: (
        <>
          <MBlock tex="S_Bw = \lambda\,S_Ww,\qquad J(w) = \frac{w^TS_Bw}{w^TS_Ww} = \lambda." />
          Every stationary direction is a generalised eigenvector of the pair <M tex="(S_B, S_W)" /> and the value of the criterion is its eigenvalue. If <M tex="S_W" /> is invertible this reads <M tex="S_W^{-1}S_Bw = \lambda w" /> — an ordinary eigenproblem, but for a{' '}
          <em>non-symmetric</em> matrix.
        </>
      ),
      note: L ? (
        <>
          The largest generalised eigenvalue on the current data is <M tex={`\\lambda_1 = J(w_1) = ${texNum(d?.l1 ?? 0, 4)}`} />.
        </>
      ) : undefined,
    },
    {
      title: 'Whitening turns it into a symmetric eigenproblem',
      body: (
        <>
          Since <M tex="S_W" /> is symmetric positive definite, factor <M tex="S_W = LL^T" /> (Cholesky). Substitute <M tex="w = L^{-T}u" />:
          <MBlock tex="S_BL^{-T}u = \lambda\,LL^TL^{-T}u = \lambda\,Lu\quad\Longrightarrow\quad \bigl(L^{-1}S_BL^{-T}\bigr)u = \lambda u." />
          The matrix <M tex="L^{-1}S_BL^{-T}" /> is symmetric positive semidefinite, so the spectral theorem applies: real eigenvalues <M tex="\lambda_1\ge\lambda_2\ge\dots\ge 0" />, orthonormal <M tex="u_j" />, and discriminants <M tex="w_j = L^{-T}u_j" /> that
          are <M tex="S_W" />-orthogonal: <M tex="w_i^TS_Ww_j = u_i^Tu_j = \delta_{ij}" />.
        </>
      ),
      note: (
        <>
          Geometrically, <M tex="x\mapsto L^{-1}x" /> makes the pooled within-class scatter spherical; in these whitened coordinates LDA is simply PCA of the class means. The implementation here uses the symmetric square root <M tex="S_W^{-1/2}" /> instead of{' '}
          <M tex="L^{-1}" /> — the eigenvalues are identical.
        </>
      ),
    },
    {
      title: 'Two classes: a closed form',
      body: (
        <>
          For <M tex="K = 2" /> the grand mean is <M tex="m = (n_1m_1 + n_2m_2)/n" />, so <M tex="m_1 - m = \tfrac{n_2}{n}(m_1 - m_2)" /> and <M tex="m_2 - m = -\tfrac{n_1}{n}(m_1 - m_2)" />, giving
          <MBlock tex="S_B = \frac{n_1n_2}{n}(m_1 - m_2)(m_1 - m_2)^T," />
          a rank-one matrix. Then <M tex="S_Bw = \lambda S_Ww" /> implies that <M tex="S_Ww" /> is proportional to <M tex="m_1 - m_2" />, hence
          <MBlock tex="w \propto S_W^{-1}(m_1 - m_2)." />
        </>
      ),
      note:
        d?.twoClass ? (
          <>
            Live check: normalising <M tex="S_W^{-1}(m_1 - m_2)" /> and comparing with the computed <M tex="w_1" /> gives <M tex={`|\\cos\\theta| = ${texNum(d.twoClass.cos, 6)}`} /> — the same direction up to sign.
          </>
        ) : L ? (
          <>
            The current dataset has <M tex={`K = ${L.K}`} /> classes, so <M tex="S_B" /> has rank up to <M tex={`${L.K - 1}`} /> and there is no single closed-form direction; the <M tex={`${L.maxDims}`} /> discriminants come from the symmetric eigenproblem above.
          </>
        ) : undefined,
    },
    {
      title: 'Why at most K − 1 discriminants',
      body: (
        <>
          <M tex="S_B" /> is a sum of <M tex="K" /> rank-one matrices built from the vectors <M tex="m_k - m" />. These satisfy one linear relation, <M tex="\sum_k n_k(m_k - m) = 0" />, so they span at most <M tex="K-1" /> dimensions and
          <MBlock tex="\operatorname{rank}(S_B)\le K-1." />
          Generalised eigenvectors with <M tex="\lambda > 0" /> must lie in the range of <M tex="S_B" />, and must also lie in the range of <M tex="S_W" /> for <M tex="J" /> to be defined; hence the number of non-trivial discriminants is at most{' '}
          <M tex="\min(K-1,\ \operatorname{rank}S_W)" />.
        </>
      ),
      note: L && d ? (
        <>
          Live: eigenvalues of <M tex="S_B" /> are <span className="mono">{fmtList(L.sbEigenvalues, 2, 6)}</span> — <M tex={`\\operatorname{rank}S_B = ${d.sbRank} \\le K-1 = ${L.K - 1}`} />; <M tex={`\\operatorname{rank}S_W = ${L.swRank}`} /> of <M tex={`p = ${L.p}`} />; hence{' '}
          <M tex={`${L.maxDims}`} /> discriminant{L.maxDims === 1 ? '' : 's'}. Equality <M tex="\operatorname{rank}S_B = K-1" /> holds iff the class means are affinely independent.
        </>
      ) : undefined,
    },
  ];
  return <Derivation steps={steps} />;
}

// ---------------------------------------------------------------------------
// Derivation: MDS
// ---------------------------------------------------------------------------

function MDSDerivation({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const Md = a.mds;
    const B = Md.B;
    const d12sq = a.n >= 2 ? a.D[0][1] ** 2 : NaN;
    const fromB = a.n >= 2 ? B[0][0] + B[1][1] - 2 * B[0][1] : NaN;
    const rowSum0 = a.n >= 1 ? B[0].reduce((s, x) => s + x, 0) : NaN;
    const l1 = Md.eigenvalues[0] ?? NaN;
    const lmin = Md.eigenvalues[Md.eigenvalues.length - 1] ?? NaN;
    const sigma1sq = (a.svd.s[0] ?? 0) ** 2;
    return { d12sq, fromB, rowSum0, l1, lmin, sigma1sq };
  }, [a]);
  const Md = a.mds;
  const metric = metricLabels[a.prep.metric];

  const steps: DerivationStep[] = [
    {
      title: 'From distances to squared distances, and the key identity',
      body: (
        <>
          Suppose the dissimilarities are Euclidean distances of unknown points <M tex="x_1,\dots,x_n" /> and write <M tex="b_{ij} = x_i^Tx_j" /> for their inner products (the Gram matrix <M tex="B = XX^T" />). Then
          <MBlock tex="d_{ij}^2 = \|x_i - x_j\|^2 = x_i^Tx_i + x_j^Tx_j - 2x_i^Tx_j = b_{ii} + b_{jj} - 2b_{ij}." />
          So <M tex="D^{(2)} = c1^T + 1c^T - 2B" /> with <M tex="c = \operatorname{diag}(B)" />: the squared distances are the inner products plus a rank-two “row + column” term. The task is to undo this and recover <M tex="B" />.
        </>
      ),
      note: (
        <>
          Live, for observations 1 and 2 with the {metric} metric: <M tex={`d_{12}^2 = ${texNum(d.d12sq, 4)}`} /> and <M tex={`b_{11} + b_{22} - 2b_{12} = ${texNum(d.fromB, 4)}`} />. The identity holds for the double-centred <M tex="B" /> of <em>any</em> symmetric
          zero-diagonal <M tex="D^{(2)}" /> — Euclideanity is a separate question (step 4).
        </>
      ),
    },
    {
      title: 'The centring matrix J',
      body: (
        <>
          <MBlock tex="J = I - \tfrac1n 11^T,\qquad J1 = 1 - \tfrac1n 1(1^T1) = 1 - 1 = 0,\qquad J^T = J,\quad J^2 = J." />
          <M tex="JA" /> subtracts from every column of <M tex="A" /> its mean (removes column means); <M tex="AJ" /> removes row means. Because <M tex="J1 = 0" />, the nuisance terms vanish: <M tex="J(c1^T)J = Jc\,(J1)^T = 0" /> and <M tex="J(1c^T)J = (J1)c^TJ = 0" />.
        </>
      ),
      note: (
        <>
          Distances are translation invariant, but inner products are not; <M tex="J" /> resolves this ambiguity by placing the centroid of the recovered configuration at the origin. Live: the first row of <M tex="B" /> sums to{' '}
          <M tex={`${texNum(d.rowSum0, 6)}`} />, i.e. <M tex="B1 = 0" />.
        </>
      ),
    },
    {
      title: 'Double centring recovers the Gram matrix',
      body: (
        <>
          <MBlock tex="-\tfrac12\,JD^{(2)}J = -\tfrac12\,J\bigl(c1^T + 1c^T - 2B\bigr)J = JBJ = B_c," />
          where <M tex="B_c = X_cX_c^T" /> is the Gram matrix of the <em>centred</em> configuration (<M tex="JX = X_c" />). Hence the definition
          <MBlock tex="B := -\tfrac12\,JD^{(2)}J." />
        </>
      ),
    },
    {
      title: 'Eigendecomposition; positive semidefiniteness ⇔ Euclidean',
      body: (
        <>
          <MBlock tex="B = V\Lambda V^T,\qquad \lambda_1\ge\lambda_2\ge\cdots\ge\lambda_n." />
          If <M tex="D" /> is Euclidean then <M tex="B = X_cX_c^T\succeq 0" />. Conversely (Schoenberg 1935; Young–Householder 1938), if <M tex="B\succeq 0" /> then <M tex="X = V\Lambda^{1/2}" /> satisfies <M tex="XX^T = B" /> and reproduces{' '}
          <M tex="D" /> exactly by the identity of step 1. Negative eigenvalues are therefore a proof that <M tex="D" /> is not a Euclidean distance matrix.
        </>
      ),
      note: (
        <>
          Live: <M tex={`\\lambda_1(B) = ${texNum(d.l1, 3)}`} />, smallest eigenvalue <M tex={`${texNum(d.lmin, 3)}`} />, {Md.positive} positive and {Md.negative} negative eigenvalues (negative mass {pct(Md.negativeMass)}).{' '}
          {a.prep.metric === 'euclidean' ? (
            <>
              Because the metric is Euclidean, <M tex="\lambda_j(B) = \sigma_j^2" /> of the centred data: <M tex={`\\sigma_1^2 = ${texNum(d.sigma1sq, 3)}`} />
              {a.prep.scaling === 'none' ? <> (differs here only because the raw <M tex="X_c" /> is uncentred while <M tex="J" /> centres).</> : '.'}
            </>
          ) : (
            <>
              The {metric} metric is not Euclidean in general, which is exactly what the negative eigenvalues report.
            </>
          )}
        </>
      ),
    },
    {
      title: 'Coordinates in k dimensions',
      body: (
        <>
          <MBlock tex="X_k = V_k\Lambda_k^{1/2}\qquad\text{so that}\qquad X_kX_k^T = V_k\Lambda_kV_k^T," />
          the truncated eigendecomposition of <M tex="B" /> keeping the <M tex="k" /> largest (positive) eigenvalues. If <M tex="B\succeq 0" /> has rank <M tex="\le k" />, then <M tex="X_kX_k^T = B" /> and all distances are reproduced exactly.
        </>
      ),
      note: (
        <>
          Live: <M tex={`k = ${Md.k}`} /> dimensions retained; they carry {pct(Md.cumulative[Md.k - 1] ?? 0)} of the positive eigenvalue mass. Stress-1 <M tex={`= ${texNum(Md.stress1, 4)}`} />.
        </>
      ),
    },
    {
      title: 'The strain criterion and Eckart–Young for symmetric matrices',
      body: (
        <>
          Classical MDS is the solution of
          <MBlock tex="\min_{X\in\mathbb R^{n\times k}}\ \|B - XX^T\|_F ." />
          As <M tex="X" /> ranges over <M tex="n\times k" /> matrices, <M tex="XX^T" /> ranges over all symmetric positive semidefinite matrices of rank <M tex="\le k" />. For a symmetric <M tex="B = V\Lambda V^T" /> the Eckart–Young theorem gives the minimiser
          by truncation: keep the <M tex="k" /> largest positive eigenvalues, <M tex="XX^T = V_k\Lambda_kV_k^T" />, with
          <MBlock tex="\min\|B - XX^T\|_F^2 = \sum_{j>k,\ \lambda_j>0}\lambda_j^2 + \sum_{\lambda_j<0}\lambda_j^2 ." />
          Negative eigenvalues cannot be represented by any <M tex="XX^T" /> and contribute to the residual in full.
        </>
      ),
      note: (
        <>
          Live relative strain <M tex={`\\|B - X_kX_k^T\\|_F/\\|B\\|_F = ${texNum(Md.strain, 4)}`} />. Note the distinction: classical MDS minimises <em>strain</em> (inner products); Kruskal's <em>stress</em> compares distances directly and is minimised only by
          iterative (non-classical) MDS.
        </>
      ),
    },
    {
      title: 'What is, and is not, determined',
      body: (
        <>
          For any orthogonal <M tex="Q" /> (<M tex="k\times k" />), <M tex="(X_kQ)(X_kQ)^T = X_kX_k^T" />: the configuration is determined only up to rotation and reflection, and — since <M tex="J" /> fixed the centroid — up to translation as well. Axes of
          an MDS plot therefore carry no intrinsic meaning; only inter-point distances do.
        </>
      ),
      note: (
        <>
          This is why the comparison with PCA scores in the questions above uses a Procrustes alignment: the two configurations agree exactly for Euclidean centred data, but only after the best rotation/reflection has been applied.
        </>
      ),
    },
  ];
  return <Derivation steps={steps} />;
}

// ---------------------------------------------------------------------------
// Derivation: SVD (existence) + progressive reconstruction
// ---------------------------------------------------------------------------

function useBlock(a: Analysis) {
  const block = useMemo(() => a.pca.Xc.slice(0, Math.min(6, a.n)).map((r) => r.slice()), [a]);
  const dec = useMemo(() => svd(block), [block]);
  return { block, dec };
}

function SVDDerivation({ a }: { a: Analysis }) {
  const { block, dec } = useBlock(a);
  const d = useMemo(() => {
    const gramEig = symmetricEigen(gram(block)).values;
    const s2 = dec.s.map((x) => x * x);
    const maxDiff = Math.max(...s2.map((x, j) => Math.abs(x - (gramEig[j] ?? 0))), 0);
    const u1 = column(dec.U, 0);
    const u2 = dec.s.length > 1 ? column(dec.U, 1) : null;
    const orth = u2 ? Math.abs(dot(u1, u2)) : 0;
    return { gramEig, s2, maxDiff, orth, m: block.length };
  }, [block, dec]);

  const steps: DerivationStep[] = [
    {
      title: 'Start from the symmetric matrix XᵀX',
      body: (
        <>
          For any real <M tex="X\in\mathbb R^{n\times p}" />, <M tex="X^TX" /> is symmetric and positive semidefinite (<M tex="w^TX^TXw = \|Xw\|^2\ge 0" />). By the spectral theorem
          <MBlock tex="X^TX = V\Lambda V^T,\qquad V^TV = I_p,\qquad \lambda_1\ge\cdots\ge\lambda_p\ge 0." />
          Define <M tex="\sigma_j = \sqrt{\lambda_j}" />. Then <M tex="\|Xv_j\|^2 = v_j^TX^TXv_j = \lambda_j = \sigma_j^2" />.
        </>
      ),
      note: (
        <>
          On the <M tex={`${d.m}\\times ${a.p}`} /> block of the live data used below: eigenvalues of <M tex="X^TX" /> are <span className="mono">{fmtList(d.gramEig, 3, 6)}</span> and the squared singular values <M tex="\sigma_j^2" /> are{' '}
          <span className="mono">{fmtList(d.s2, 3, 6)}</span> (largest difference <M tex={`${texNum(d.maxDiff, 2)}`} />).
        </>
      ),
    },
    {
      title: 'Construct the left singular vectors',
      body: (
        <>
          For every <M tex="\sigma_j > 0" /> set <M tex="u_j = Xv_j/\sigma_j" />. These are orthonormal:
          <MBlock tex="u_i^Tu_j = \frac{v_i^TX^TXv_j}{\sigma_i\sigma_j} = \frac{\lambda_j\,v_i^Tv_j}{\sigma_i\sigma_j} = \delta_{ij}." />
          For <M tex="\sigma_j = 0" /> we have <M tex="Xv_j = 0" /> (its norm is <M tex="\sigma_j" />); complete <M tex="\{u_j\}" /> to an orthonormal set by any Gram–Schmidt procedure.
        </>
      ),
      note: (
        <>
          Live: <M tex={`|u_1^Tu_2| = ${texNum(d.orth, 2)}`} /> on the block.
        </>
      ),
    },
    {
      title: 'Assemble the factorisation',
      body: (
        <>
          <MBlock tex="Xv_j = \sigma_ju_j\ \ \forall j\quad\Longleftrightarrow\quad XV = U\Sigma\quad\Longleftrightarrow\quad X = U\Sigma V^T = \sum_{j=1}^{r}\sigma_ju_jv_j^T," />
          using <M tex="VV^T = I" /> and <M tex="r = \operatorname{rank}X" /> (the number of positive <M tex="\sigma_j" />). This is the thin SVD; padding <M tex="U" /> and <M tex="V" /> to square orthogonal matrices gives the full SVD.
        </>
      ),
      note: (
        <>
          The singular values are unique; <M tex="u_j, v_j" /> are unique up to a common sign when <M tex="\sigma_j" /> is simple, and only the singular subspaces are determined when singular values repeat. The dashboard fixes signs so that the largest
          entry of each <M tex="v_j" /> is positive.
        </>
      ),
    },
    {
      title: 'Eckart–Young–Mirsky: the best rank-k approximation',
      body: (
        <>
          Among all matrices <M tex="A" /> with <M tex="\operatorname{rank}A\le k" />,
          <MBlock tex="X_k = \sum_{j\le k}\sigma_ju_jv_j^T\quad\text{minimises}\quad\|X - A\|_F,\qquad \|X - X_k\|_F = \sqrt{\textstyle\sum_{j>k}\sigma_j^2}," />
          and the same <M tex="X_k" /> minimises the spectral norm, with <M tex="\|X - X_k\|_2 = \sigma_{k+1}" /> (Mirsky: any unitarily invariant norm).
        </>
      ),
      note: (
        <>
          The interactive block below builds <M tex="X_1, X_2, \dots" /> term by term and reports <M tex="\|X - X_k\|_F" /> next to the formula.
        </>
      ),
    },
  ];
  return <Derivation steps={steps} />;
}

function ProgressiveReconstruction({ a }: { a: Analysis }) {
  const { block, dec } = useBlock(a);
  const r = dec.s.length;
  const [kState, setK] = useState(1);
  const k = Math.max(0, Math.min(kState, r));
  const names = a.dataset.variableNames;

  const d = useMemo(() => {
    const partials = Array.from({ length: k }, (_, j) => truncatedReconstruction(dec, j + 1));
    const Xk = k === 0 ? truncatedReconstruction(dec, 0) : partials[k - 1];
    const layer = k > 0 ? scaleMatrix(outer(column(dec.U, k - 1), column(dec.V, k - 1)), dec.s[k - 1]) : null;
    const residual = sub(block, Xk);
    const err = frobenius(residual);
    const tail = Math.sqrt(Math.max(0, sumSquares(dec.s.slice(k))));
    const errors = Array.from({ length: r + 1 }, (_, j) => Math.sqrt(Math.max(0, sumSquares(dec.s.slice(j)))));
    const normX = frobenius(block);
    let heatMax = 1e-12;
    for (const row of block) for (const x of row) heatMax = Math.max(heatMax, Math.abs(x));
    return { partials, Xk, layer, residual, err, tail, errors, normX, heatMax, energy: energyFraction(dec.s, k) };
  }, [block, dec, k, r]);

  const rowLabels = block.map((_, i) => String(i + 1));
  const traces: Data[] = [
    {
      type: 'scatter',
      mode: 'lines+markers',
      x: d.errors.map((_, j) => j),
      y: d.errors,
      name: '‖X − X<sub>k</sub>‖<sub>F</sub> = √Σ<sub>j>k</sub> σ<sub>j</sub>²',
      line: { color: methodColor.SVD, width: 1.5 },
      marker: { color: methodColor.SVD, size: 6 },
      hovertemplate: 'k = %{x}: %{y:.4g}<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'markers',
      x: [k],
      y: [d.err],
      name: 'current k (computed directly)',
      marker: { color: '#0b0b0b', size: 12, symbol: 'circle-open', line: { width: 2, color: '#0b0b0b' } },
      hovertemplate: 'k = %{x}: %{y:.4g}<extra></extra>',
    },
  ];

  return (
    <Card
      title={
        <>
          <Badge method="SVD" /> Progressive reconstruction of the first {block.length} rows of <M tex="X_c" />
        </>
      }
    >
      <div className="grid side">
        <div className="stack">
          <Slider label={<>terms retained <M tex="k" /></>} value={k} min={0} max={r} step={1} onChange={setK} />
          <div className="stats">
            <StatTile label={<M tex={`\\|X - X_{${k}}\\|_F`} />} value={fmt(d.err, 4)} note="computed from the residual" />
            <StatTile label={<M tex={`\\sqrt{\\sum_{j>${k}}\\sigma_j^2}`} />} value={fmt(d.tail, 4)} note="Eckart–Young value" />
            <StatTile label="relative error" value={pct(d.normX > 0 ? d.err / d.normX : 0)} note={`${pct(d.energy)} of ‖X‖²_F retained`} />
          </div>
          <div className="small muted">
            The block is decomposed on its own: <M tex={`X\\in\\mathbb R^{${block.length}\\times ${a.p}}`} />, <M tex={`\\operatorname{rank}X = ${dec.rank}`} />, singular values <span className="mono">{fmtList(dec.s, 3, 6)}</span>. Cells are coloured on a
            common diverging scale (<M tex={`\\pm ${texNum(d.heatMax, 2)}`} />) so that the residual visibly shrinks.
          </div>
        </div>
        <div className="stack">
          <MatrixEquation
            items={[
              <MatrixView key="X" M={block} title="X (target block)" rowLabels={rowLabels} colLabels={names} heat="diverging" heatMax={d.heatMax} digits={2} compact />,
              '=',
              <MatrixView key="Xk" M={d.Xk} title={`X_${k} = Σ_{j≤${k}} σ_j u_j v_jᵀ`} rowLabels={rowLabels} colLabels={names} heat="diverging" heatMax={d.heatMax} digits={2} compact />,
              '+',
              <MatrixView key="R" M={d.residual} title={`R_${k} = X − X_${k}`} rowLabels={rowLabels} colLabels={names} heat="diverging" heatMax={d.heatMax} digits={2} compact caption={`‖R_${k}‖_F = ${fmt(d.err, 4)}`} />,
            ]}
          />
          {k > 0 && (
            <div className="row" style={{ alignItems: 'flex-start', overflowX: 'auto' }}>
              {d.layer && (
                <MatrixView M={d.layer} title={`layer ${k}: σ_${k} u_${k} v_${k}ᵀ (rank one)`} heat="diverging" heatMax={d.heatMax} digits={2} compact caption={`σ_${k} = ${fmt(dec.s[k - 1], 4)}`} />
              )}
              {d.partials.slice(0, Math.max(0, k - 1)).map((Xj, j) => (
                <MatrixView key={j} M={Xj} title={`X_${j + 1}`} heat="diverging" heatMax={d.heatMax} digits={2} compact caption={`‖X − X_${j + 1}‖_F = ${fmt(d.errors[j + 1], 4)}`} />
              ))}
            </div>
          )}
          <Plot
            data={traces}
            height={220}
            layout={{
              showlegend: true,
              xaxis: { title: { text: 'k' }, dtick: 1 },
              yaxis: { title: { text: '‖X − X<sub>k</sub>‖<sub>F</sub>' }, rangemode: 'tozero' },
              margin: { l: 56, r: 10, t: 36, b: 40 },
            }}
          />
        </div>
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              The first {block.length} rows of the analysed matrix (left), their rank-<M tex={`${k}`} /> approximation <M tex={`X_{${k}}`} /> built from the {k} leading singular triplets, and the residual. <M tex={`\\|X - X_{${k}}\\|_F = ${texNum(d.err, 4)}`} /> out of{' '}
              <M tex={`\\|X\\|_F = ${texNum(d.normX, 4)}`} />.
            </>
          ),
          why: (
            <>
              Each added layer <M tex="\sigma_ju_jv_j^T" /> is a rank-one matrix — every row is a multiple of the same <M tex="v_j^T" /> — and the layers are mutually orthogonal in the Frobenius inner product, so the squared error drops by exactly{' '}
              <M tex="\sigma_j^2" /> at each step: from <M tex={`${texNum(d.errors[Math.max(0, k - 1)] ** 2, 3)}`} /> to <M tex={`${texNum(d.err ** 2, 3)}`} /> when the {k > 0 ? `${k}-th` : 'first'} layer is added
              {k > 0 && (
                <>
                  , and <M tex={`\\sigma_{${k}}^2 = ${texNum(dec.s[k - 1] ** 2, 3)}`} />
                </>
              )}
              .
            </>
          ),
          math: (
            <>
              Eckart–Young: no matrix of rank <M tex={`\\le ${k}`} /> is closer to <M tex="X" /> than <M tex={`X_{${k}}`} />. The directly computed residual norm <M tex={`${texNum(d.err, 5)}`} /> and the formula <M tex={`\\sqrt{\\sum_{j>${k}}\\sigma_j^2} = ${texNum(d.tail, 5)}`} />{' '}
              agree to <M tex={`${texNum(Math.abs(d.err - d.tail), 1)}`} />. With <M tex={`k = ${dec.rank}`} /> the residual vanishes because the block has rank {dec.rank}.
            </>
          ),
          stats: (
            <>
              The share <M tex={`\\sum_{j\\le ${k}}\\sigma_j^2/\\sum_j\\sigma_j^2 = ${pct(d.energy)}`} /> is the “energy” or, for a centred matrix, the proportion of variance retained. For this small block it is a property of these {block.length} rows only — not of
              the full dataset, whose spectrum appears in the SVD question above.
            </>
          ),
          careful: (
            <>
              A small block has rank at most <M tex={`\\min(${block.length}, ${a.p}) = ${Math.min(block.length, a.p)}`} />, so its singular values are not those of the full matrix. The approximation is optimal in the Frobenius and spectral norms only; other norms have
              other minimisers.
            </>
          ),
        }}
      />
    </Card>
  );
}
