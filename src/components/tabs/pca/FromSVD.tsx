import { useMemo } from 'react';
import { useAnalysis, useStore } from '../../../state/store';
import { M, MBlock } from '../../common/Math';
import { MatrixView, MatrixEquation } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile, Badge } from '../../common/Panels';
import { Button } from '../../common/Controls';
import { matmul, transpose, sub, maxAbs, fmt, dot, column, diag } from '../../../lib/linalg';
import { pct, sci, texNum, pcLabels } from './util';

/** Lesson 1 — PCA as the SVD of the centred data matrix, verified live by two independent algorithms. */
export function FromSVD() {
  const a = useAnalysis();
  const { prep, navigate } = useStore();
  const { pca: P, n, p, dataset } = a;
  const names = dataset.variableNames;

  const c = useMemo(() => {
    const sigma = P.singularValues;
    const r = sigma.length;
    const denom = Math.max(n - 1, 1);
    const lam = P.eigenvalues; // σ_j²/(n−1) from the SVD route
    const lamEig = P.eigen.values; // Jacobi eigendecomposition of S (length p)
    const Vsvd = P.V; // p × r
    const Veig = P.eigen.vectors; // p × p
    const VeigR = Veig.map((row) => row.slice(0, r));
    const tol = P.svd.tol;
    const rows = sigma.map((s, j) => {
      const le = lamEig[j] ?? 0;
      const cos = Math.min(1, Math.abs(dot(column(Vsvd, j), column(Veig, j))));
      return { j, sigma: s, lam: lam[j], lamEig: le, diff: Math.abs(lam[j] - le), cos, zero: s <= tol };
    });
    const live = rows.filter((x) => !x.zero);
    const keep = rows.map((x) => !x.zero);
    // Z = X_c V versus U Σ
    const US = P.svd.U.map((row) => row.map((u, j) => u * sigma[j]));
    const scoreGap = maxAbs(sub(P.scores, US));
    // S versus V Λ Vᵀ with Λ = Σ²/(n−1)
    const VL = Vsvd.map((row) => row.map((v, j) => v * lam[j]));
    const sGap = maxAbs(sub(P.S, matmul(VL, transpose(Vsvd))));
    const maxLamDiff = rows.reduce((m, x) => Math.max(m, x.diff), 0);
    const minCos = live.reduce((m, x) => Math.min(m, x.cos), 1);
    const vGap = live.length ? maxAbs(sub(VeigR.map((row) => row.filter((_, j) => keep[j])), Vsvd.map((row) => row.filter((_, j) => keep[j])))) : 0;
    const trailing = lamEig.slice(r);
    const maxTrailing = trailing.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    let minRelGap = Infinity;
    for (let j = 0; j + 1 < live.length; j++) minRelGap = Math.min(minRelGap, (lam[j] - lam[j + 1]) / (lam[0] || 1));
    const degenerate = minCos < 0.999;
    const trS = P.S.reduce((s, row, i) => s + row[i], 0);
    const cond = sigma[r - 1] > tol ? sigma[0] / sigma[r - 1] : Infinity;
    return { sigma, r, denom, lam, lamEig, Vsvd, VeigR, rows, scoreGap, sGap, maxLamDiff, minCos, vGap, trailing, maxTrailing, minRelGap, degenerate, trS, cond, Lambda: diag(lam), zeroCount: rows.length - live.length };
  }, [P, n]);

  const pcs = pcLabels(c.r);
  const raw = prep.scaling === 'none';
  const std = prep.scaling === 'standardize';
  const sName = std ? 'S = R (correlation matrix)' : raw ? 'XᵀX/(n−1) (second moments, uncentred)' : 'S (covariance matrix)';
  const sigmaList = c.sigma
    .slice(0, 4)
    .map((s) => fmt(s, 3))
    .join(', ');

  return (
    <Section
      id="pca-from-svd"
      title="1 · From the SVD to PCA"
      subtitle="The principal directions are the right singular vectors of the centred data matrix; the eigenvalues of the covariance matrix are the scaled squared singular values."
      right={<Badge method="PCA" />}
    >
      <div className="prose">
        <p>
          Let <M tex="X" /> be the <M tex="n \times p" /> data matrix with column means <M tex="\bar x" />. PCA is defined on the centred matrix and on its sample covariance, which the spectral theorem
          diagonalises in an orthonormal eigenbasis:
        </p>
        <MBlock tex={String.raw`X_c = X - \mathbf{1}\,\bar{x}^{T}, \qquad S = \frac{1}{n-1}\,X_c^{T}X_c = V\Lambda V^{T}, \qquad \Lambda = \operatorname{diag}(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0).`} />
        <p>
          Now take the thin SVD <M tex="X_c = U\Sigma V^{T}" /> with <M tex="U^{T}U = I_r" /> and <M tex="V^{T}V = I_r" />, <M tex="r = \min(n,p)" />. Substituting,
        </p>
        <MBlock tex={String.raw`S = \frac{1}{n-1}\,V\Sigma U^{T}U\Sigma V^{T} = V\,\frac{\Sigma^{2}}{n-1}\,V^{T} \quad\Longrightarrow\quad \lambda_j = \frac{\sigma_j^{2}}{n-1},`} />
        <p>
          so the right singular vectors of <M tex="X_c" /> <em>are</em> the eigenvectors of <M tex="S" /> (the loadings), and the singular values fix the eigenvalues. The scores follow at once:
        </p>
        <MBlock tex={String.raw`Z = X_c V = U\Sigma V^{T}V = U\Sigma .`} />
        <p>
          Everything below is computed live from the active dataset by two independent routes — a one-sided Jacobi SVD of <M tex="X_c" /> (which never forms <M tex="X_c^{T}X_c" />) and a cyclic Jacobi
          eigendecomposition of <M tex="S" /> — so that the identities can be checked to rounding error.
          {std && (
            <>
              {' '}
              Under standardisation the analysed matrix is <M tex="X_s = X_c D_s^{-1}" /> (each column divided by its standard deviation) and <M tex="S" /> becomes the correlation matrix <M tex="R" />;
              the algebra is unchanged, so we keep writing <M tex="X_c" /> and <M tex="S" />.
            </>
          )}
        </p>
      </div>

      {raw && (
        <Callout kind="warning" title="Raw preprocessing: this is the SVD of X, not of X_c">
          With no centring the decomposition is of <M tex="X" /> itself and the matrix being diagonalised is the second-moment matrix <M tex="\tfrac{1}{n-1}X^{T}X = S + \tfrac{n}{n-1}\bar x\bar x^{T}" />, not the
          covariance. The leading direction then maximises the <em>mean square</em> of the projection, which includes the squared mean, and the fitted "components" are lines through the origin rather
          than through <M tex="\bar x" />. The identities below still hold algebraically, but their statistical reading as variances is lost. Switch to "Centre" for PCA proper.
        </Callout>
      )}

      <div className="grid side-r">
        <Card title="Singular values of X_c versus eigenvalues of S">
          <div className="table-wrap">
            <table className="summary-table">
              <thead>
                <tr>
                  <th>
                    <M tex="j" />
                  </th>
                  <th>
                    <M tex="\sigma_j" /> (SVD of <M tex="X_c" />)
                  </th>
                  <th>
                    <M tex="\sigma_j^{2}/(n-1)" />
                  </th>
                  <th>
                    <M tex="\lambda_j" /> (eigen of <M tex="S" />)
                  </th>
                  <th>|difference|</th>
                  <th>
                    <M tex="|\cos\angle(v_j^{\mathrm{svd}}, v_j^{\mathrm{eig}})|" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {c.rows.map((x) => (
                  <tr key={x.j}>
                    <td>{x.j + 1}</td>
                    <td>{fmt(x.sigma, 4)}</td>
                    <td>{fmt(x.lam, 5)}</td>
                    <td>{fmt(x.lamEig, 5)}</td>
                    <td className="mono">{sci(x.diff)}</td>
                    <td>{x.zero ? <span className="muted">— (σ ≈ 0, null-space vector arbitrary)</span> : fmt(x.cos, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            n − 1 = {c.denom}. {c.trailing.length > 0 && <>S has p − r = {c.trailing.length} further eigenvalues beyond the {c.r} singular values; all are numerically zero (max |λ| = {sci(c.maxTrailing)}) because rank X_c ≤ min(n − 1, p). </>}
            {c.zeroCount > 0 && <>{c.zeroCount} singular value{c.zeroCount > 1 ? 's are' : ' is'} below the rank tolerance {sci(P.svd.tol)}: centring removes one dimension, so a centred matrix has rank at most n − 1.</>}
          </div>
        </Card>
        <div className="stats">
          <StatTile label={<M tex="\max|X_cV - U\Sigma|" />} value={sci(c.scoreGap)} note="scores identity Z = X_c V = UΣ" />
          <StatTile label={<M tex="\max|S - V\Lambda V^{T}|" />} value={sci(c.sGap)} note="Λ = Σ²/(n−1) from the SVD" />
          <StatTile label={<M tex="\max_j|\lambda_j - \sigma_j^{2}/(n-1)|" />} value={sci(c.maxLamDiff)} note="two independent algorithms" />
          <StatTile label={<M tex="\min_j |\cos\angle(v_j)|" />} value={fmt(c.minCos, 6)} note="1 = same axis up to sign" />
          <StatTile label="numerical rank" value={String(P.rank)} note={`r = min(n, p) = ${c.r}`} />
          <StatTile label={<M tex="\kappa(X_c) = \sigma_1/\sigma_r" />} value={Number.isFinite(c.cond) ? fmt(c.cond, 1) : '∞'} note="κ(S) = κ(X_c)²" />
        </div>
      </div>

      {p <= 6 ? (
        <MatrixEquation
          items={[
            <MatrixView M={P.S} title={sName} rowLabels={names} colLabels={names} digits={2} heat="diverging" compact />,
            '=',
            <MatrixView M={c.Vsvd} title="V (right singular vectors of X_c)" rowLabels={names} colLabels={pcs} digits={2} heat="diverging" highlightCols={[0]} compact />,
            '·',
            <MatrixView M={c.Lambda} title="Λ = Σ²/(n−1)" rowLabels={pcs} colLabels={pcs} digits={3} heat="sequential" compact />,
            '·',
            <MatrixView M={transpose(c.Vsvd)} title="Vᵀ" rowLabels={pcs} colLabels={names} digits={2} heat="diverging" compact />,
          ]}
        />
      ) : (
        <div className="grid c2">
          <MatrixView M={P.S} title={sName} rowLabels={names} colLabels={names} digits={2} heat="diverging" compact caption="Signed entries → diverging colour scale; the diagonal holds the variances (or 1 under standardisation)." />
          <MatrixView M={c.Lambda} title="Λ = Σ²/(n−1)" rowLabels={pcs} colLabels={pcs} digits={3} heat="sequential" compact />
        </div>
      )}

      <div className="grid c2">
        <MatrixView
          M={c.Vsvd}
          title="V from the SVD of X_c"
          rowLabels={names}
          colLabels={pcs}
          digits={3}
          heat="diverging"
          highlightCols={[0]}
          caption="Columns are the loadings v_j: unit vectors, mutually orthogonal. Column 1 (highlighted) is the first principal direction."
        />
        <MatrixView
          M={c.VeigR}
          title="V from the eigendecomposition of S (first r eigenvectors)"
          rowLabels={names}
          colLabels={pcs}
          digits={3}
          heat="diverging"
          highlightCols={[0]}
          caption={`Both matrices are sign-normalised (largest-magnitude entry of each column positive). Entrywise gap on the non-null columns: ${sci(c.vGap)}.`}
        />
      </div>

      <Callout kind="info" title="Sign indeterminacy — and, with repeated eigenvalues, rotation indeterminacy">
        If <M tex="Sv = \lambda v" /> then also <M tex="S(-v) = \lambda(-v)" />: an eigenvector, and likewise a singular vector pair <M tex="(u_j, v_j) \to (-u_j, -v_j)" />, is defined only up to sign. Different software
        will therefore report loadings and scores with different signs; the library fixes the convention that the largest-magnitude entry of each <M tex="v_j" /> is positive, which is why the two routes agree
        entrywise and not merely up to sign. Flipping <M tex="v_j" /> flips the score column <M tex="z_j = X_c v_j" />; nothing statistical changes. When two eigenvalues coincide, <M tex="\lambda_j = \lambda_{j+1}" />,
        any orthonormal basis of the two-dimensional eigenspace is valid and the individual vectors <M tex="v_j, v_{j+1}" /> are not identifiable — only the subspace is.
      </Callout>
      {c.degenerate && (
        <Callout kind="warning" title="Repeated (or nearly repeated) eigenvalues in this dataset">
          The smallest relative gap between consecutive non-zero eigenvalues is <M tex={`(\\lambda_j - \\lambda_{j+1})/\\lambda_1 = ${texNum(c.minRelGap, 4)}`} />, and the smallest
          <M tex="|\cos\angle(v_j^{\mathrm{svd}}, v_j^{\mathrm{eig}})|" /> is {fmt(c.minCos, 4)}: the two algorithms returned different bases of the same eigenspace. The eigen<em>values</em> still agree to{' '}
          {sci(c.maxLamDiff)}, and the span of the affected columns is identical — check the "Correlated Gaussian variables" dataset, whose <M tex="p-1" /> trailing eigenvalues are all equal to{' '}
          <M tex="1-\rho" /> in the population.
        </Callout>
      )}

      <Interpretation
        items={{
          seeing: (
            <>
              For <b>{dataset.name}</b> (n = {n}, p = {p}, {prep.scaling === 'none' ? 'no centring' : prep.scaling === 'center' ? 'mean-centred' : 'standardised'}) the {c.r} singular values of <M tex="X_c" /> start σ = ({sigmaList}
              {c.r > 4 ? ', …' : ''}). Dividing <M tex="\sigma_j^{2}" /> by n − 1 = {c.denom} reproduces the eigenvalues of <M tex="S" /> computed independently by Jacobi rotations: the largest discrepancy is {sci(c.maxLamDiff)}, and the
              loadings from the two routes differ by at most {sci(c.vGap)} entrywise{c.degenerate ? ' on the columns with well-separated eigenvalues (see the note above for the repeated ones)' : ''}. The scores identity{' '}
              <M tex="X_cV = U\Sigma" /> holds to {sci(c.scoreGap)}.
            </>
          ),
          why: (
            <>
              Both routes diagonalise the same matrix, <M tex="X_c^{T}X_c/(n-1) = V\Sigma^{2}V^{T}/(n-1)" />. The SVD works on <M tex="X_c" /> directly, the eigen-route first forms <M tex="S" />. Forming{' '}
              <M tex="S" /> squares the condition number — here <M tex={`\\kappa(X_c) = ${Number.isFinite(c.cond) ? texNum(c.cond, 1) : '\\infty'}`} /> so{' '}
              <M tex={`\\kappa(S) ${Number.isFinite(c.cond) ? `\\approx ${texNum(c.cond * c.cond, 0)}` : '= \\infty'}`} /> — which is why numerically careful implementations compute PCA through the SVD; with only{' '}
              {p} variables and backward-stable Jacobi methods both routes are accurate here.
            </>
          ),
          math: (
            <>
              <MBlock tex={String.raw`\lambda_j = \frac{\sigma_j^{2}}{n-1}, \qquad V_{\mathrm{eig}} = V_{\mathrm{svd}} \ (\text{up to sign}), \qquad Z = X_cV = U\Sigma, \qquad \operatorname{tr} S = \sum_j \lambda_j = \frac{\|X_c\|_F^{2}}{n-1}.`} />
              The trace identity gives the total variance <M tex={`\\operatorname{tr} S = ${texNum(c.trS, 4)}`} /> against <M tex={`\\sum_j \\lambda_j = ${texNum(P.totalVariance, 4)}`} />.
            </>
          ),
          stats: (
            <>
              <M tex={`\\lambda_1 = ${texNum(c.lam[0], 4)}`} /> is the sample variance of the first principal component <M tex="z_1 = X_c v_1" />; it accounts for {pct(P.explained[0])} of the total variance{' '}
              <M tex={`\\operatorname{tr} S = ${texNum(c.trS, 3)}`} />.{' '}
              {std ? (
                <>
                  Because the variables are standardised, <M tex="\operatorname{tr} R = p" /> = {p} and each eigenvalue is measured in units of "one standardised variable": <M tex="\lambda_1" /> = {fmt(c.lam[0], 3)} means PC1 carries the
                  variance of {fmt(c.lam[0], 2)} original variables.
                </>
              ) : raw ? (
                <>Without centring these "variances" are mean squares about the origin and inherit the location of the data — they are not dispersion measures.</>
              ) : (
                <>
                  The eigenvalues are in the squared units of the variables, so they are only comparable across variables that share a unit — this is the covariance-versus-correlation question of lesson 5.
                </>
              )}
            </>
          ),
          careful: (
            <>
              Signs of loadings and scores are conventions, not findings — never interpret "PC1 is negative on {names[0]}" without checking the convention. Only <em>ratios</em> and <em>angles</em> are invariant.
              {c.degenerate ? ' With (nearly) repeated eigenvalues the individual directions are not identifiable — only the subspace they span is. ' : ' '}
              PCA summarises second moments only: it needs no normality assumption to be computed, but the variance-ellipsoid reading of <M tex="\lambda_j" /> as "spread" is a Gaussian intuition.
              <div style={{ marginTop: 8 }}>
                <Button small onClick={() => navigate('svd')}>
                  Revisit the SVD laboratory
                </Button>
              </div>
            </>
          ),
        }}
      />
    </Section>
  );
}
