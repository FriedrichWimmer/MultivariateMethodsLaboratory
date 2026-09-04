import { useMemo } from 'react';
import { useAnalysis, useStore } from '../../state/store';
import { Section, Card, Callout, Badge } from '../common/Panels';
import { M, MBlock } from '../common/Math';
import { fmt } from '../../lib/linalg';

export default function Takeaway() {
  const a = useAnalysis();
  const { navigate } = useStore();

  const live = useMemo(() => {
    const pc1 = a.pca.explained[0] ?? 0;
    const k90 = a.pca.cumulative.findIndex((c) => c >= 0.9) + 1;
    const kappa = a.svd.s.length > 1 && a.svd.s[a.svd.s.length - 1] > 0 ? a.svd.s[0] / a.svd.s[a.svd.s.length - 1] : Infinity;
    return { pc1, k90: k90 > 0 ? k90 : a.p, kappa, stress: a.mds.stress1, neg: a.mds.negative, rank: a.svd.rank, lda: a.lda };
  }, [a]);

  return (
    <>
      <div className="topbar">
        <div className="title">
          <h2>If you remember only one thing</h2>
          <div className="lede">A synthesis of the whole laboratory, ending with the question a statistician should ask before reducing any dataset.</div>
        </div>
      </div>

      <Section id="takeaway-four" title="Four methods, four notions of information">
        <div className="grid c4">
          <Card>
            <Badge method="SVD" />
            <p style={{ marginTop: 8 }}>
              <b>SVD</b> is fundamentally a <i>matrix factorisation</i>: <M tex="X = U\Sigma V^T" />. It has no statistical model. Its statistical power comes from the Eckart–Young theorem: truncating the factorisation gives the best rank-<M tex="k" /> approximation of the matrix, and everything else here is a special case of that.
            </p>
          </Card>
          <Card>
            <Badge method="PCA" />
            <p style={{ marginTop: 8 }}>
              <b>PCA</b> uses the structure of the <i>centred</i> data matrix to find orthogonal directions of maximum variance and minimum reconstruction error. It is the SVD of <M tex="X_c" /> read statistically: <M tex="\lambda_j = \sigma_j^2/(n-1)" />, and the scores are <M tex="Z = U\Sigma" />.
            </p>
          </Card>
          <Card>
            <Badge method="MDS" />
            <p style={{ marginTop: 8 }}>
              <b>MDS</b> starts from dissimilarities or distances and seeks a geometric representation that preserves them as faithfully as possible. Classical MDS double-centres the squared distances into an inner-product matrix <M tex="B" /> and diagonalises it; for Euclidean distances the result is PCA.
            </p>
          </Card>
          <Card>
            <Badge method="LDA" />
            <p style={{ marginTop: 8 }}>
              <b>LDA</b> uses class information to find directions that maximise between-class separation relative to within-class variation: <M tex="S_Bw = \lambda S_Ww" />. It yields at most <M tex="K-1" /> directions and needs <M tex="S_W" /> to be invertible (or regularised).
            </p>
          </Card>
        </div>
      </Section>

      <Section id="takeaway-lesson" title="The deeper statistical lesson">
        <Callout kind="theorem" title="Dimensionality reduction is not a single problem">
          It depends on <b>what information you want the lower-dimensional representation to preserve</b>. Variance, pairwise distances, class separation and matrix approximation error are different functionals of the data; a representation that is optimal for one is in general not optimal for another. The PCA-versus-LDA laboratory shows a dataset where the variance-maximising direction is almost orthogonal to the class-separating one.
        </Callout>
        <div className="grid c2" style={{ marginTop: 12 }}>
          <div className="prose">
            <p>
              The methods look alike because each one ends in an eigenproblem of a symmetric matrix: <M tex="X_c^TX_c" /> for PCA, <M tex="X_cX_c^T" /> (that is, <M tex="B" />) for classical MDS, and <M tex="S_W^{-1}S_B" /> — symmetrised by whitening — for LDA. The SVD sits underneath all three as the numerically preferred way to obtain those eigenvectors without forming products that square the condition number.
            </p>
            <p>
              They differ because each maximises a different Rayleigh quotient. PCA maximises <M tex="w^TSw/w^Tw" />, LDA maximises <M tex="w^TS_Bw/w^TS_Ww" />, and MDS minimises the strain <M tex="\|B - XX^T\|_F" />. Change the quotient and you change what "important" means.
            </p>
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>If you want to preserve…</th>
                <th>use</th>
                <th>and you are assuming</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>the matrix itself (in Frobenius / spectral norm)</td>
                <td>
                  <Badge method="SVD" />
                </td>
                <td>nothing statistical; only that low rank is a useful description</td>
              </tr>
              <tr>
                <td>the dispersion of the observations (variance, covariance)</td>
                <td>
                  <Badge method="PCA" />
                </td>
                <td>linear structure; variance is meaningful on the chosen scale; centring; no labels</td>
              </tr>
              <tr>
                <td>the pairwise distances / dissimilarities</td>
                <td>
                  <Badge method="MDS" />
                </td>
                <td>the dissimilarities are (approximately) Euclidean; distant pairs matter as much as close ones</td>
              </tr>
              <tr>
                <td>the separation of known groups</td>
                <td>
                  <Badge method="LDA" />
                </td>
                <td>labels are correct; classes share a covariance matrix; <M tex="S_W" /> is well conditioned; at most <M tex="K-1" /> directions</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="takeaway-question" title="The question to ask of every new dataset" subtitle="The conceptual conclusion of the entire laboratory.">
        <MBlock tex={String.raw`\textbf{What structure am I trying to preserve, and what assumptions am I willing to make?}`} />
        <div className="grid c2" style={{ marginTop: 10 }}>
          <div className="prose">
            <h3>A reasoning checklist</h3>
            <ol>
              <li>
                <b>What structure exists in this data?</b> Look at scales, correlations, rank, outliers and — if present — labels before any decomposition.
              </li>
              <li>
                <b>What notion of information should I preserve?</b> Variance, distances, class separation or the matrix itself.
              </li>
              <li>
                <b>Which mathematical method corresponds to that objective?</b> SVD, PCA, MDS or LDA — or something nonlinear if the structure is curved.
              </li>
              <li>
                <b>What assumptions does it require?</b> Centring and scaling decisions, Euclidean distances, equal class covariances, invertible <M tex="S_W" />.
              </li>
              <li>
                <b>How should I interpret the resulting representation?</b> Signs are arbitrary, "variance explained" is not importance, training accuracy is optimistic, and rotations within equal-eigenvalue subspaces are undetermined.
              </li>
            </ol>
          </div>
          <Card title="The checklist applied to the active dataset" plane>
            <div className="stack" style={{ gap: 6 }}>
              <div>
                <b>{a.dataset.name}</b> — n = {a.n}, p = {a.p}
                {a.y ? `, K = ${a.K}` : ', no labels'}; preprocessing: {a.prep.scaling === 'none' ? 'raw' : a.prep.scaling === 'center' ? 'centred' : 'standardised'}.
              </div>
              <div>
                <Badge method="SVD" /> numerical rank {live.rank} of min(n, p) = {Math.min(a.n, a.p)}; condition number κ₂ = {live.kappa === Infinity ? '∞' : fmt(live.kappa, 1)}.
              </div>
              <div>
                <Badge method="PCA" /> PC1 explains {(live.pc1 * 100).toFixed(1)} % of the total variance; {live.k90} component{live.k90 === 1 ? '' : 's'} reach 90 %.
              </div>
              <div>
                <Badge method="MDS" /> stress-1 = {fmt(live.stress, 3)} at k = {a.mds.k} with the {a.prep.metric} metric; {live.neg} negative eigenvalue{live.neg === 1 ? '' : 's'} of B{live.neg > 0 ? ' — the distances are not exactly Euclidean' : ''}.
              </div>
              <div>
                <Badge method="LDA" />{' '}
                {live.lda ? (
                  <>
                    largest Fisher ratio λ₁ = {fmt(live.lda.eigenvalues[0] ?? 0, 2)}; {live.lda.maxDims} discriminant direction{live.lda.maxDims === 1 ? '' : 's'} (K − 1 = {a.K - 1}){live.lda.swSingular ? '; S_W is singular — regularisation needed' : ''}.
                  </>
                ) : (
                  'not applicable without labels.'
                )}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="btn small" onClick={() => navigate('compare')}>
                  Compare all four on this dataset
                </button>
                <button className="btn small" onClick={() => navigate('wrong')}>
                  Try to break them
                </button>
              </div>
            </div>
          </Card>
        </div>
      </Section>
    </>
  );
}
