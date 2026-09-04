/**
 * Four-way comparison laboratory: one dataset analysed simultaneously by
 * SVD | PCA | MDS | LDA, four synchronised panels, ten-property method cards
 * and the summary matrix.
 */
import { useMemo, type ReactNode } from 'react';
import type { Data } from 'plotly.js';
import { useStore, type Analysis } from '../../state/store';
import { DatasetControls, PrepControls } from '../common/DatasetControls';
import { Section, Card, Callout, Interpretation, Badge, ClassLegend, StatTile } from '../common/Panels';
import { M, MBlock } from '../common/Math';
import { Plot } from '../common/Plot';
import { ScatterSVG } from '../common/ScatterSVG';
import { fmt, frobenius, sub, truncatedReconstruction, firstColumns } from '../../lib/linalg';
import { metricLabels, procrustesAlign } from '../../lib/mds';
import { methodColor, classColor } from '../../lib/theme';
import { firstTwoColumns, pct, fmtList, sumSquares, columnCorrelation, scalingWord, methodName, classNamesFor, texNum, type Method } from './compare/helpers';

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function FourWay() {
  const { analysis } = useStore();
  const a = analysis;
  const classNames = useMemo(() => (a.y ? classNamesFor(a.K, a.dataset.classNames) : undefined), [a]);

  return (
    <>
      <div className="topbar">
        <div className="title">
          <h2>Four-way comparison laboratory</h2>
          <div className="lede">
            One dataset, one analysed matrix <M tex="X_c" />, four decompositions side by side — every panel recomputes from the same matrix when the data or the preprocessing changes.
          </div>
        </div>
      </div>

      <Section
        id="compare-controls"
        title="Synchronised analysis"
        subtitle={
          <>
            The controls act on the shared matrix <M tex="X_c" /> — currently <b>{a.dataset.name}</b>, {scalingWord(a.prep.scaling)}, <M tex={`n = ${a.n},\\ p = ${a.p}`} />
            {a.K > 0 ? (
              <>
                , <M tex={`K = ${a.K}`} /> classes
              </>
            ) : (
              ', unlabelled'
            )}
            . The SVD, PCA and MDS panels are computed from <M tex="X_c" />; LDA additionally reads the labels.
          </>
        }
      >
        <div className="grid side-wide">
          <div className="stack">
            <Card title="Dataset">
              <DatasetControls compact />
            </Card>
            <Card title="Preprocessing">
              <PrepControls />
            </Card>
            <Callout kind="info" title="What is shared">
              <div className="small">
                <M tex="X_c" /> is the matrix after the chosen preprocessing (raw <M tex="X" />, centred <M tex="X - 1\bar X^T" /> or standardised). SVD decomposes <M tex="X_c" /> itself; PCA reads its right singular vectors as
                loadings; MDS starts from the {metricLabels[a.prep.metric]} distance matrix <M tex="D" /> of its rows; LDA splits its scatter by class. The slider <M tex="k" /> is the retained dimension used by the SVD truncation and by
                MDS.
              </div>
            </Callout>
          </div>
          <div id="compare-panels" className="grid c2">
            <SVDPanel a={a} />
            <PCAPanel a={a} classNames={classNames} />
            <MDSPanel a={a} classNames={classNames} />
            <LDAPanel a={a} classNames={classNames} />
          </div>
        </div>
      </Section>

      <Section id="compare-cards" title="Method cards" subtitle="The same ten properties, stated for each method in the same order — so that any two cells can be compared line by line.">
        <div className="grid c4">
          {methodCards.map((c) => (
            <Card key={c.method} title={<><Badge method={c.method} /> {methodName[c.method]}</>}>
              <div className="stack" style={{ gap: 8 }}>
                {c.rows.map(([label, body]) => (
                  <div key={label}>
                    <div className="small secondary" style={{ fontWeight: 600 }}>
                      {label}
                    </div>
                    <div className="small">{body}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </Section>

      <Section id="compare-matrix" title="Summary matrix" subtitle="Property by method. The first five rows are the canonical comparison; the remaining rows make the mathematical differences explicit.">
        <SummaryMatrix a={a} />
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// SVD panel
// ---------------------------------------------------------------------------

function SVDPanel({ a }: { a: Analysis }) {
  const d = useMemo(() => {
    const dec = a.svd;
    const s = dec.s;
    const rThin = s.length;
    const r = dec.rank;
    const kEff = Math.max(0, Math.min(a.prep.k, rThin));
    const Xk = truncatedReconstruction(dec, kEff);
    const err = frobenius(sub(a.pca.Xc, Xk));
    const total = sumSquares(s);
    const tail = Math.sqrt(Math.max(0, sumSquares(s.slice(kEff))));
    const energy = total > 0 ? sumSquares(s.slice(0, kEff)) / total : 0;
    const kappa = r > 0 ? s[0] / s[r - 1] : NaN;
    const normX = frobenius(a.pca.Xc);
    const ratio21 = s.length > 1 && s[0] > 0 ? s[1] / s[0] : NaN;
    return { s, rThin, r, kEff, err, tail, energy, kappa, normX, ratio21, tol: dec.tol };
  }, [a]);

  const traces: Data[] = [
    {
      type: 'bar',
      x: d.s.map((_, j) => j + 1),
      y: d.s,
      marker: { color: methodColor.SVD, opacity: d.s.map((_, j) => (j < d.r ? 0.9 : 0.3)) },
      hovertemplate: 'σ<sub>%{x}</sub> = %{y:.4g}<extra></extra>',
    },
  ];
  const raw = a.prep.scaling === 'none';

  return (
    <Card
      title={
        <>
          <Badge method="SVD" /> Singular value spectrum of <M tex="X_c" />
        </>
      }
    >
      <Plot
        data={traces}
        height={220}
        layout={{
          xaxis: { title: { text: 'index j' }, dtick: 1 },
          yaxis: { title: { text: 'σ<sub>j</sub>' }, rangemode: 'tozero' },
          margin: { l: 48, r: 10, t: 10, b: 40 },
        }}
      />
      <div className="plot-caption">
        Bars beyond the numerical rank (<M tex={`j > ${d.r}`} />) are drawn faint: their <M tex="\sigma_j" /> lies below the tolerance <M tex={`${texNum(d.tol, 2)}`} />.
      </div>
      <div className="stats" style={{ marginTop: 10 }}>
        <StatTile label={<>numerical rank <M tex="r" /></>} value={d.r} note={`of ${d.rThin} singular values`} />
        <StatTile label={<M tex="\kappa_2 = \sigma_1/\sigma_r" />} value={fmt(d.kappa, 2)} note={`σ₁ = ${fmt(d.s[0] ?? NaN, 3)}, σ_r = ${fmt(d.s[d.r - 1] ?? NaN, 3)}`} />
        <StatTile label={<M tex={`\\|X_c - X_k\\|_F,\\ k = ${d.kEff}`} />} value={fmt(d.err, 3)} note={`Eckart–Young value √Σ_{j>k}σ_j² = ${fmt(d.tail, 3)}`} />
      </div>
      <MBlock
        tex={`X_c = U\\Sigma V^T,\\qquad U\\in\\mathbb R^{${a.n}\\times ${d.r}},\\quad \\Sigma\\in\\mathbb R^{${d.r}\\times ${d.r}},\\quad V^T\\in\\mathbb R^{${d.r}\\times ${a.p}}`}
      />
      <div className="small muted">
        Compact form with <M tex={`r = \\operatorname{rank}(X_c) = ${d.r}`} />. The thin SVD carries <M tex={`\\min(n,p) = ${d.rThin}`} /> columns; the {d.rThin - d.r} trailing ones belong to numerically zero singular values.
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              The {d.rThin} singular values of the {scalingWord(a.prep.scaling)} matrix <M tex="X_c" /> (<M tex={`${a.n}\\times ${a.p}`} />). {d.r} of them exceed the tolerance, so the numerical rank is {d.r}; <M tex={`\\sigma_1 = ${texNum(d.s[0] ?? NaN, 3)}`} />,{' '}
              <M tex={`\\sigma_r = ${texNum(d.s[d.r - 1] ?? NaN, 3)}`} /> and <M tex={`\\kappa_2 = ${texNum(d.kappa, 2)}`} />.
            </>
          ),
          why: (
            <>
              The spectrum decays because the columns of <M tex="X_c" /> are linearly related: <M tex={`\\sigma_2/\\sigma_1 = ${texNum(d.ratio21, 3)}`} /> measures how far the matrix is from rank one.{' '}
              {raw ? (
                <>
                  Without centring the column means enter the decomposition, so <M tex="\sigma_1" /> is inflated and <M tex="u_1" /> points roughly towards the mean row — this is the reason PCA insists on centring.
                </>
              ) : (
                <>
                  After centring, <M tex="\sigma_j = \sqrt{(n-1)\lambda_j}" /> is proportional to the standard deviation of the data along the <M tex="j" />-th principal axis.
                </>
              )}
            </>
          ),
          math: (
            <>
              By Eckart–Young–Mirsky, <M tex="X_k = U_k\Sigma_kV_k^T" /> is the best rank-<M tex="k" /> approximation and <M tex="\|X_c - X_k\|_F = \sqrt{\sum_{j>k}\sigma_j^2}" />. Here the directly computed error{' '}
              <M tex={`${texNum(d.err, 4)}`} /> agrees with the formula <M tex={`${texNum(d.tail, 4)}`} /> to <M tex={`${texNum(Math.abs(d.err - d.tail), 2)}`} />.
            </>
          ),
          stats: (
            <>
              The first <M tex={`k = ${d.kEff}`} /> terms retain <M tex={`\\sum_{j\\le k}\\sigma_j^2/\\sum_j\\sigma_j^2 = ${pct(d.energy)}`} /> of <M tex="\|X_c\|_F^2" />.{' '}
              {raw ? (
                <>Because <M tex="X_c" /> is uncentred this is a share of the total second moment about the origin, not of the variance.</>
              ) : (
                <>Because <M tex="X_c" /> is centred, this is exactly the cumulative proportion of variance explained by the first {d.kEff} principal components.</>
              )}
            </>
          ),
          careful: (
            <>
              The numerical rank depends on a tolerance (<M tex={`${texNum(d.tol, 2)}`} />, relative to <M tex="\sigma_1" />); a large <M tex="\kappa_2" /> signals near-collinearity long before the rank drops. The SVD knows nothing about labels or about
              the distance metric — those enter only in LDA and MDS.
            </>
          ),
        }}
      />
    </Card>
  );
}

// ---------------------------------------------------------------------------
// PCA panel
// ---------------------------------------------------------------------------

function PCAPanel({ a, classNames }: { a: Analysis; classNames?: string[] }) {
  const d = useMemo(() => {
    const P = a.pca;
    const cols = P.scores[0]?.length ?? 0;
    const has2 = cols >= 2;
    const pts = firstTwoColumns(P.scores);
    const e1 = P.explained[0] ?? 0;
    const e2 = has2 ? P.explained[1] ?? 0 : 0;
    const l1 = P.eigenvalues[0] ?? 0;
    const l2 = has2 ? P.eigenvalues[1] ?? 0 : 0;
    const corr12 = has2 ? columnCorrelation(P.scores, 0, 1) : NaN;
    const eigDirect = P.eigen.values[0] ?? 0;
    const sd1 = Math.sqrt(Math.max(l1, 0));
    const sd2 = Math.sqrt(Math.max(l2, 0));
    return { pts, has2, e1, e2, l1, l2, corr12, eigDirect, sd1, sd2, cum2: e1 + e2 };
  }, [a]);
  const raw = a.prep.scaling === 'none';

  return (
    <Card
      title={
        <>
          <Badge method="PCA" /> Scores on the first two principal components
        </>
      }
    >
      <ScatterSVG
        points={d.pts}
        labels={a.y}
        classNames={classNames}
        width={480}
        height={340}
        xLabel={`PC1 (${pct(d.e1)})`}
        yLabel={d.has2 ? `PC2 (${pct(d.e2)})` : 'PC2 (not available: r = 1)'}
        caption={
          <>
            <M tex="Z = X_cV" />; axes labelled with the proportion of variance <M tex="\lambda_j/\sum_i\lambda_i" />. Equal aspect ratio, so spread along PC1 versus PC2 is visually faithful.
          </>
        }
      />
      {a.y && classNames && <ClassLegend classNames={classNames} />}
      <div className="stats" style={{ marginTop: 10 }}>
        <StatTile label={<M tex="\lambda_1" />} value={fmt(d.l1, 3)} note={`PC1 explains ${pct(d.e1)}`} />
        <StatTile label={<M tex="\lambda_2" />} value={fmt(d.l2, 3)} note={`PC2 explains ${pct(d.e2)}`} />
        <StatTile label="PC1 + PC2" value={pct(d.cum2)} note="cumulative proportion" />
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              Each observation projected onto the first two principal directions of <M tex="X_c" />. PC1 explains {pct(d.e1)} and PC2 {pct(d.e2)} of the total variance <M tex={`\\operatorname{tr} S = ${texNum(a.pca.totalVariance, 3)}`} />; together{' '}
              {pct(d.cum2)}. {a.y ? `Colours and marker shapes are the ${a.K} class labels — which PCA never used.` : 'The dataset is unlabelled, so all points share one colour.'}
            </>
          ),
          why: (
            <>
              <M tex="v_1" /> is the unit direction maximising <M tex="\operatorname{Var}(X_cw) = w^TSw" />, so the cloud is widest along PC1 (standard deviation <M tex={`\\sqrt{\\lambda_1} = ${texNum(d.sd1, 3)}`} />) and next widest along the orthogonal PC2 (
              <M tex={`\\sqrt{\\lambda_2} = ${texNum(d.sd2, 3)}`} />). {raw && <>With raw preprocessing the matrix is not centred, so “variance” here is the second moment about the origin and PC1 largely encodes the mean.</>}
            </>
          ),
          math: (
            <>
              <M tex="Sv_j = \lambda_jv_j" /> with <M tex="S = X_c^TX_c/(n-1)" />; via the SVD <M tex="X_c = U\Sigma V^T" /> one has <M tex="\lambda_j = \sigma_j^2/(n-1)" />: here <M tex={`\\sigma_1^2/(n-1) = ${texNum(d.l1, 4)}`} /> while the
              eigenvalue of <M tex="S" /> computed directly is <M tex={`${texNum(d.eigDirect, 4)}`} />.
            </>
          ),
          stats: (
            <>
              Scores on distinct components are uncorrelated: the sample correlation between PC1 and PC2 is <M tex={`${texNum(d.corr12, 3)}`} />, since <M tex="\operatorname{Cov}(z_1,z_2) = v_1^TSv_2 = \lambda_2v_1^Tv_2 = 0" />. The explained proportions
              are descriptive statistics of this sample, not estimates with a sampling distribution unless a model is assumed.
            </>
          ),
          careful: (
            <>
              Signs of <M tex="v_j" /> are arbitrary (here fixed so the largest loading is positive); eigenvalues are scale-dependent, so Centre versus Standardise gives different components. {a.y && <>Any class separation visible here is incidental — compare with the LDA panel, whose objective is separation.</>}
            </>
          ),
        }}
      />
    </Card>
  );
}

// ---------------------------------------------------------------------------
// MDS panel
// ---------------------------------------------------------------------------

function MDSPanel({ a, classNames }: { a: Analysis; classNames?: string[] }) {
  const d = useMemo(() => {
    const Md = a.mds;
    const k = Md.k;
    const pts = k >= 2 ? firstTwoColumns(Md.coords) : Md.coords.map((r) => [r[0] ?? 0, 0]);
    const l1 = Md.eigenvalues[0] ?? NaN;
    const l2 = Md.eigenvalues[1] ?? NaN;
    const lmin = Md.eigenvalues[Md.eigenvalues.length - 1] ?? NaN;
    const scoresCols = a.pca.scores[0]?.length ?? 0;
    const proc = k > 0 && scoresCols >= k ? procrustesAlign(Md.coords, firstColumns(a.pca.scores, k)) : null;
    const sigma1sq = (a.svd.s[0] ?? 0) ** 2;
    const euclid = a.prep.metric === 'euclidean';
    return { k, pts, l1, l2, lmin, proc, sigma1sq, euclid, requested: Math.min(a.prep.k, a.n - 1, a.p) };
  }, [a]);
  const metric = metricLabels[a.prep.metric];
  const raw = a.prep.scaling === 'none';
  const same = d.proc !== null && d.proc.relative < 1e-6;

  return (
    <Card
      title={
        <>
          <Badge method="MDS" /> Classical MDS configuration ({metric})
        </>
      }
    >
      {d.k === 0 ? (
        <Callout kind="warning" title="No positive eigenvalue">
          <M tex="B = -\tfrac12 JD^{(2)}J" /> has no positive eigenvalue for this distance matrix, so no configuration can be drawn.
        </Callout>
      ) : (
        <ScatterSVG
          points={d.pts}
          labels={a.y}
          classNames={classNames}
          width={480}
          height={340}
          xLabel={`MDS 1 (λ₁ of B = ${fmt(d.l1, 2)})`}
          yLabel={d.k >= 2 ? `MDS 2 (λ₂ of B = ${fmt(d.l2, 2)})` : '1-D configuration (y = 0)'}
          caption={
            <>
              <M tex="X_k = V_k\Lambda_k^{1/2}" /> from the eigendecomposition of <M tex="B" />; <M tex={`k = ${d.k}`} /> {d.k < d.requested ? `(requested ${d.requested}, limited by positive eigenvalues)` : ''}.
              {d.k === 1 && ' A one-dimensional configuration is shown as a strip.'}
            </>
          }
        />
      )}
      {a.y && classNames && <ClassLegend classNames={classNames} />}
      <div className="stats" style={{ marginTop: 10 }}>
        <StatTile label="stress-1" value={fmt(a.mds.stress1, 4)} note={`between D and D̂ in k = ${d.k}`} />
        <StatTile label={<>negative eigenvalues of <M tex="B" /></>} value={a.mds.negative} note={`negative mass ${pct(a.mds.negativeMass)}`} />
        <StatTile label="metric" value={<span style={{ fontSize: 15 }}>{metric}</span>} note={`${a.mds.positive} positive eigenvalues`} />
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              A {d.k}-dimensional configuration of the {a.n} rows of <M tex="X_c" /> whose Euclidean distances approximate the {metric} distances in <M tex="D" />. Kruskal's stress-1 is <M tex={`${texNum(a.mds.stress1, 4)}`} />; <M tex="B" /> has{' '}
              {a.mds.positive} positive and {a.mds.negative} negative eigenvalues (smallest <M tex={`${texNum(d.lmin, 3)}`} />).
            </>
          ),
          why: (
            <>
              {d.euclid ? (
                <>
                  For Euclidean distances <M tex="B = X_cX_c^T" /> (after centring) is positive semidefinite, so the eigenvalues of <M tex="B" /> are the squared singular values: <M tex={`\\lambda_1(B) = ${texNum(d.l1, 3)}`} /> versus{' '}
                  <M tex={`\\sigma_1^2 = ${texNum(d.sigma1sq, 3)}`} />
                  {raw ? <> — they differ here only because the raw <M tex="X_c" /> is uncentred while <M tex="J" /> always centres.</> : '.'}
                </>
              ) : (
                <>
                  The {metric} metric is not Euclidean in general, so <M tex="B" /> acquires negative eigenvalues (negative mass {pct(a.mds.negativeMass)}) and the configuration can only approximate <M tex="D" />; the equality with PCA is lost.
                </>
              )}
            </>
          ),
          math: (
            <>
              <M tex="D \to D^{(2)} \to B = -\tfrac12 JD^{(2)}J = V\Lambda V^T \to X_k = V_k\Lambda_k^{1/2}" />. <M tex="X_k" /> minimises the strain <M tex="\|B - XX^T\|_F" /> over <M tex="n\times k" /> matrices (relative strain{' '}
              <M tex={`${texNum(a.mds.strain, 4)}`} />).{' '}
              {d.proc && (
                <>
                  Procrustes alignment to the first {d.k} PCA scores leaves a relative residual of <M tex={`${texNum(d.proc.relative, 3)}`} /> — {same ? 'the two configurations coincide up to a rigid motion.' : 'the configurations differ.'}
                </>
              )}
            </>
          ),
          stats: (
            <>
              The first {d.k} axes carry {pct(a.mds.cumulative[d.k - 1] ?? 0)} of the positive eigenvalue mass of <M tex="B" />. Stress-1 compares distances, strain compares inner products; classical MDS optimises the latter, so stress is a
              diagnostic here, not the objective.
            </>
          ),
          careful: (
            <>
              The configuration is determined only up to translation, rotation and reflection — axis directions and signs carry no meaning. <M tex="k" /> is capped by the number of positive eigenvalues ({a.mds.positive}); requested{' '}
              <M tex={`k = ${d.requested}`} />, used <M tex={`k = ${d.k}`} />.
            </>
          ),
        }}
      />
    </Card>
  );
}

// ---------------------------------------------------------------------------
// LDA panel
// ---------------------------------------------------------------------------

function LDAPanel({ a, classNames }: { a: Analysis; classNames?: string[] }) {
  const L = a.lda;
  const y = a.y;
  const d = useMemo(() => {
    if (!L || !y) return null;
    const m = L.maxDims;
    const theoretical = Math.max(0, Math.min(L.K - 1, L.swRank));
    const pts = m >= 2 ? firstTwoColumns(L.scores) : null;
    const ld1 = L.scores.map((r) => r[0] ?? 0);
    const names = classNames ?? L.classes.map((c) => `class ${c + 1}`);
    const hist: Data[] = L.classes.map((c) => ({
      type: 'histogram',
      x: ld1.filter((_, i) => y[i] === c),
      name: names[c] ?? `class ${c + 1}`,
      marker: { color: classColor(c) },
      opacity: 0.65,
      nbinsx: 24,
    }));
    return { m, theoretical, pts, ld1, hist };
  }, [L, y, classNames]);

  const title = (
    <>
      <Badge method="LDA" /> Fisher discriminant scores
    </>
  );

  if (!L || !d || !y) {
    return (
      <Card title={title}>
        <Callout kind="warning" title="LDA needs labels">
          The current dataset carries no class labels, so <M tex="S_W" /> and <M tex="S_B" /> cannot be formed and the criterion <M tex="J(w) = w^TS_Bw / w^TS_Ww" /> is undefined. Choose a labelled dataset (or upload a CSV with a text column)
          to populate this panel. The other three panels do not need labels.
        </Callout>
      </Card>
    );
  }

  const lam = L.eigenvalues;
  return (
    <Card title={title}>
      {d.m >= 2 && d.pts ? (
        <ScatterSVG
          points={d.pts}
          labels={y}
          classNames={classNames}
          width={480}
          height={340}
          xLabel={`LD1 (λ₁ = ${fmt(lam[0], 2)})`}
          yLabel={`LD2 (λ₂ = ${fmt(lam[1], 2)})`}
          caption={
            <>
              <M tex="Z = (X - 1m^T)W" /> with unit-norm columns <M tex="w_j" /> solving <M tex="S_Bw = \lambda S_Ww" />; <M tex="K \ge 3" /> gives at least two discriminants.
            </>
          }
        />
      ) : d.m === 1 ? (
        <Plot
          data={d.hist}
          height={300}
          layout={{
            barmode: 'overlay',
            showlegend: true,
            xaxis: { title: { text: `LD1 (λ₁ = ${fmt(lam[0], 3)})` } },
            yaxis: { title: { text: 'count' } },
            margin: { l: 48, r: 10, t: 36, b: 44 },
          }}
        />
      ) : (
        <Callout kind="warning" title="No discriminant direction">
          <M tex={`\\operatorname{rank}(S_B) = ${L.sbEigenvalues.filter((v) => v > 1e-10 * Math.max(Math.abs(L.sbEigenvalues[0] ?? 0), 1e-300)).length}`} /> or <M tex={`\\operatorname{rank}(S_W) = ${L.swRank}`} /> is zero, so no Fisher direction exists.
        </Callout>
      )}
      {d.m >= 2 && classNames && <ClassLegend classNames={classNames} />}
      <div className="stats" style={{ marginTop: 10 }}>
        <StatTile label="Fisher eigenvalues" value={<span style={{ fontSize: 15 }}>{fmtList(lam, 3, 4)}</span>} note={<M tex="\lambda_j = J(w_j)" />} />
        <StatTile label={<M tex="\text{maxDims} = \min(K-1,\ \operatorname{rank}S_W)" />} value={d.m} note={`min(${L.K - 1}, ${L.swRank}) = ${d.theoretical}${d.m !== d.theoretical ? `; reduced to ${d.m} because rank(S_B) < K − 1` : ''}`} />
        <StatTile label={<M tex="\kappa(S_W)" />} value={fmt(L.swCondition, 1)} note={L.swSingular ? 'S_W singular — pseudo-inverse used' : 'S_W invertible'} />
      </div>
      <Interpretation
        defaultOpen={false}
        items={{
          seeing: (
            <>
              {d.m >= 2 ? (
                <>
                  The {a.n} observations projected onto the first two Fisher directions; the {L.K} classes are as separated as any linear projection allows in the sense of <M tex="J" />. Fisher eigenvalues <M tex={`\\lambda_1 = ${texNum(lam[0], 3)}`} />,{' '}
                  <M tex={`\\lambda_2 = ${texNum(lam[1] ?? 0, 3)}`} />.
                </>
              ) : (
                <>
                  With <M tex={`K = ${L.K}`} /> there is a single discriminant, so the projection is one-dimensional: per-class histograms of <M tex="z = w_1^T(x - m)" />, with <M tex={`\\lambda_1 = ${texNum(lam[0], 3)}`} />.
                </>
              )}
            </>
          ),
          why: (
            <>
              <M tex="w_1" /> maximises <M tex="J(w) = w^TS_Bw/w^TS_Ww" />; its value <M tex={`J(w_1) = \\lambda_1 = ${texNum(lam[0], 3)}`} /> says that along LD1 the between-class scatter is {fmt(lam[0], 2)} times the within-class scatter.{' '}
              {d.m >= 2 && (
                <>
                  LD2 is the best direction that is <M tex="S_W" />-orthogonal to LD1 (<M tex="w_1^TS_Ww_2 = 0" />), with ratio <M tex={`${texNum(lam[1] ?? 0, 3)}`} />.
                </>
              )}
            </>
          ),
          math: (
            <>
              Generalised eigenproblem <M tex="S_Bw = \lambda S_Ww" />, solved by whitening with <M tex="S_W^{-1/2}" /> and diagonalising the symmetric <M tex="S_W^{-1/2}S_BS_W^{-1/2}" />. Because <M tex="\operatorname{rank}S_B \le K-1" /> and directions must lie in the
              range of <M tex="S_W" />, at most <M tex={`\\min(K-1,\\operatorname{rank}S_W) = \\min(${L.K - 1}, ${L.swRank}) = ${d.theoretical}`} /> discriminants exist.
            </>
          ),
          stats: (
            <>
              LD1 carries {pct(L.explained[0] ?? 0)} of the discriminant trace <M tex="\sum_j\lambda_j" />
              {d.m >= 2 && <>, LD2 {pct(L.explained[1] ?? 0)}</>}. Fisher's criterion uses only class means and pooled scatter; turning it into a classifier requires the additional Gaussian, equal-covariance model. <M tex={`\\kappa(S_W) = ${texNum(L.swCondition, 1)}`} />{' '}
              indicates how well conditioned the inversion is.
            </>
          ),
          careful: (
            <>
              The eigenvalues do not change under Centre or Standardise — LDA is invariant to invertible affine maps — only the unit-norm <M tex="w_j" /> and the score scale change. Signs of <M tex="w_j" /> are arbitrary. With <M tex="p" /> close to{' '}
              <M tex="n - K" /> the criterion overfits: separation seen here is in-sample.
            </>
          ),
        }}
      />
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Method cards — ten properties each
// ---------------------------------------------------------------------------

type CardRows = [string, ReactNode][];

const methodCards: { method: Method; rows: CardRows }[] = [
  {
    method: 'SVD',
    rows: [
      ['Objective', <>Factorise a real <M tex="n\times p" /> matrix into orthonormal bases and non-negative singular values; truncation gives the best rank-<M tex="k" /> approximation.</>],
      [
        'Mathematical formulation',
        <>
          <MBlock tex="X = U\Sigma V^T,\quad U^TU = V^TV = I_r,\quad \sigma_1\ge\cdots\ge\sigma_r>0" />
          <MBlock tex="X_k = \sum_{j\le k}\sigma_ju_jv_j^T = \arg\min_{\operatorname{rank}A\le k}\|X-A\|_F" />
        </>,
      ],
      ['Required input', <>Any real matrix <M tex="X" />. No centring, labels or distances are presupposed.</>],
      ['Supervised / unsupervised', <>Unsupervised — more precisely, a purely algebraic decomposition with no statistical model at all.</>],
      [
        'Geometric interpretation',
        <>
          <M tex="X" /> maps the unit sphere to a hyper-ellipsoid with semi-axes <M tex="\sigma_ju_j" />: <M tex="V^T" /> rotates to the principal axes in the domain, <M tex="\Sigma" /> stretches, <M tex="U" /> rotates in the codomain.
        </>,
      ],
      ['Output', <><M tex="U" /> (<M tex="n\times r" />), <M tex="\sigma_1,\dots,\sigma_r" />, <M tex="V" /> (<M tex="p\times r" />); numerical rank, condition number <M tex="\kappa_2 = \sigma_1/\sigma_r" />, truncations <M tex="X_k" />.</>],
      ['Optimisation / decomposition', <>Matrix factorisation (one-sided Jacobi here, Golub–Kahan in LAPACK); equivalently the eigendecompositions of <M tex="X^TX" /> and <M tex="XX^T" />. Optimality of <M tex="X_k" /> is the Eckart–Young–Mirsky theorem.</>],
      ['Main assumptions', <>None beyond real entries. What the factors mean depends entirely on how the columns were scaled and centred beforehand.</>],
      ['Strengths', <>Always exists; singular values are unique; numerically stable; yields rank, null space and conditioning; the computational engine behind PCA and MDS.</>],
      ['Limitations', <>No inference; an uncentred <M tex="X" /> lets the mean dominate <M tex="\sigma_1" />; scale-dependent; linear; the sign of each pair <M tex="(u_j, v_j)" /> is arbitrary and subspaces are indeterminate for repeated <M tex="\sigma_j" />.</>],
    ],
  },
  {
    method: 'PCA',
    rows: [
      ['Objective', <>Find orthonormal directions of maximal variance of the centred data; equivalently, minimise the reconstruction error of a rank-<M tex="k" /> linear approximation.</>],
      [
        'Mathematical formulation',
        <>
          <MBlock tex="v_1 = \arg\max_{\|w\|=1} w^TSw,\qquad Sv_j = \lambda_jv_j" />
          <MBlock tex="Z = X_cV,\quad \hat X_k = Z_kV_k^T,\quad \lambda_j = \sigma_j^2/(n-1)" />
        </>,
      ],
      ['Required input', <>An <M tex="n\times p" /> data matrix. Columns must be centred; when units differ they should be standardised (correlation PCA).</>],
      ['Supervised / unsupervised', <>Unsupervised — labels, if present, are ignored.</>],
      ['Geometric interpretation', <>Rotates the axes to the principal axes of the covariance ellipsoid; scores are orthogonal projections onto <M tex="v_j" />; <M tex="\hat X_k" /> is the projection onto the best-fitting <M tex="k" />-flat through the centroid.</>],
      ['Output', <>Loadings <M tex="V" /> (<M tex="p\times r" />), scores <M tex="Z" /> (<M tex="n\times r" />), eigenvalues <M tex="\lambda_j" />, explained proportions <M tex="\lambda_j/\sum_i\lambda_i" />, reconstruction <M tex="\hat X_k" />.</>],
      ['Optimisation / decomposition', <>Constrained maximisation (Lagrange multipliers) leading to the eigendecomposition of <M tex="S" />; computed as the SVD of <M tex="X_c" />. Optimality by the Rayleigh quotient and Courant–Fischer.</>],
      ['Main assumptions', <>Linear structure; variance is the relevant information; second moments suffice (Gaussianity only for inference); variables on comparable scales.</>],
      ['Strengths', <>Optimal linear compression in least squares; components uncorrelated; nested solutions in <M tex="k" />; closed form; interpretable loadings.</>],
      ['Limitations', <>Scale-dependent; sensitive to outliers; blind to labels; linear; large variance is not the same as relevance; the sign of each <M tex="v_j" /> is arbitrary.</>],
    ],
  },
  {
    method: 'MDS',
    rows: [
      ['Objective', <>Find <M tex="n" /> points in <M tex="\mathbb R^k" /> whose Euclidean distances reproduce a given dissimilarity matrix as well as possible (strain criterion).</>],
      [
        'Mathematical formulation',
        <>
          <MBlock tex="B = -\tfrac12 JD^{(2)}J,\quad J = I - \tfrac1n 11^T,\quad B = V\Lambda V^T" />
          <MBlock tex="X_k = V_k\Lambda_k^{1/2} = \arg\min_{X\in\mathbb R^{n\times k}}\|B - XX^T\|_F" />
        </>,
      ],
      ['Required input', <>A symmetric <M tex="n\times n" /> dissimilarity matrix <M tex="D" /> with zero diagonal — no coordinates, no labels.</>],
      ['Supervised / unsupervised', <>Unsupervised.</>],
      ['Geometric interpretation', <>Recovers a configuration from its distances up to rigid motion; positive eigenvalues of <M tex="B" /> count Euclidean dimensions, negative ones measure how non-Euclidean <M tex="D" /> is.</>],
      ['Output', <>Coordinates <M tex="X_k" /> (<M tex="n\times k" />), eigenvalues of <M tex="B" />, strain, stress-1, number of negative eigenvalues.</>],
      ['Optimisation / decomposition', <>Eigendecomposition of the double-centred matrix <M tex="B" /> (Torgerson–Gower); truncation is optimal by Eckart–Young for symmetric positive semidefinite matrices.</>],
      ['Main assumptions', <>Dissimilarities are (approximately) Euclidean distances, i.e. <M tex="B \succeq 0" />; the metric scale is meaningful (classical, not ordinal, MDS).</>],
      ['Strengths', <>Works from distances alone; exact for Euclidean <M tex="D" /> and then identical to the PCA scores; applicable to non-coordinate data such as judged similarities or edit distances.</>],
      ['Limitations', <><M tex="O(n^3)" /> in <M tex="n" />; configuration defined only up to rotation and reflection; a non-Euclidean <M tex="D" /> loses information in negative eigenvalues; minimises strain, not stress; no out-of-sample mapping without extensions.</>],
    ],
  },
  {
    method: 'LDA',
    rows: [
      ['Objective', <>Find directions along which between-class scatter is large relative to within-class scatter — directions that separate the known groups.</>],
      [
        'Mathematical formulation',
        <>
          <MBlock tex="J(w) = \frac{w^TS_Bw}{w^TS_Ww},\qquad S_Bw = \lambda S_Ww" />
          <MBlock tex="S_W = \sum_k\sum_{i\in k}(x_i-m_k)(x_i-m_k)^T,\quad S_B = \sum_k n_k(m_k-m)(m_k-m)^T" />
        </>,
      ],
      ['Required input', <>Data <M tex="X" /> and class labels <M tex="y" /> with <M tex="K\ge 2" />; <M tex="S_W" /> invertible (<M tex="n-K\ge p" />) or regularised.</>],
      ['Supervised / unsupervised', <>Supervised — the labels define the objective.</>],
      ['Geometric interpretation', <>After whitening by <M tex="S_W^{-1/2}" /> (spherical within-class scatter) the discriminants are the principal axes of the class means; LD1 is the line along which the Mahalanobis distance between class means is largest.</>],
      ['Output', <>Up to <M tex="\min(K-1,\operatorname{rank}S_W)" /> directions <M tex="W" />, discriminant scores <M tex="Z = (X-1m^T)W" />, Fisher eigenvalues <M tex="\lambda_j = J(w_j)" />; with a Gaussian model, a classification rule.</>],
      ['Optimisation / decomposition', <>Generalised symmetric-definite eigenproblem; solved by whitening <M tex="S_W = LL^T" /> and a symmetric eigendecomposition of <M tex="L^{-1}S_BL^{-T}" />.</>],
      ['Main assumptions', <>Fisher's criterion: only class means and pooled scatter (no distribution). The Gaussian LDA classifier adds normal classes with a common covariance.</>],
      ['Strengths', <>Uses labels; invariant to invertible affine transformations of <M tex="X" />; <M tex="K-1" /> dimensions summarise all linear class separation; closed form.</>],
      ['Limitations', <>At most <M tex="K-1" /> dimensions; undefined or unstable when <M tex="S_W" /> is singular (<M tex="p > n-K" />); non-robust to outliers; linear boundaries only; dimensions collapse when class means are collinear.</>],
    ],
  },
];

// ---------------------------------------------------------------------------
// Summary matrix
// ---------------------------------------------------------------------------

function SummaryMatrix({ a }: { a: Analysis }) {
  const L = a.lda;
  const rows: [ReactNode, ReactNode, ReactNode, ReactNode, ReactNode][] = [
    ['Uses labels?', 'No', 'No', 'No', 'Yes'],
    ['Works from distances alone?', 'No', 'No', 'Yes', 'No'],
    ['Main objective', 'Decompose', 'Maximise variance', 'Preserve distances', 'Separate classes'],
    ['Main mathematical tool', 'Matrix factorisation', 'Eigen/SVD', 'Eigendecomposition', 'Generalised eigenproblem'],
    [
      'Maximum dimensions',
      <>Rank-dependent¹</>,
      <>
        <M tex="\min(n-1,\,p)" />²
      </>,
      <>
        Data-dependent (number of positive eigenvalues of <M tex="B" />)³
      </>,
      <>
        <M tex="K-1" />⁴
      </>,
    ],
    [
      'Objective function',
      <M tex="\min_{\operatorname{rank}A\le k}\|X - A\|_F" />,
      <>
        <M tex="\max_{\|w\|=1} w^TSw" /> (equivalently <M tex="\min\|X_c - \hat X_k\|_F" />)
      </>,
      <M tex="\min_{X\in\mathbb R^{n\times k}}\|B - XX^T\|_F" />,
      <M tex="\max_w \dfrac{w^TS_Bw}{w^TS_Ww}" />,
    ],
    [
      'Centring required?',
      <>No — decomposes <M tex="X" /> exactly as given</>,
      <>Yes — otherwise the mean dominates <M tex="\sigma_1" /></>,
      <>Built in — <M tex="J" /> double-centres <M tex="D^{(2)}" /></>,
      <>No — translation-invariant; means enter through <M tex="S_B" /></>,
    ],
    [
      'Invariances',
      <>
        Singular values invariant under orthogonal equivalence <M tex="X\mapsto PXQ" /> (<M tex="P,Q" /> orthogonal); not under scaling or centring
      </>,
      <>
        Scores and <M tex="\lambda_j" /> invariant under a rotation of variable space <M tex="X_c\mapsto X_cQ" />; not under rescaling of variables
      </>,
      <>Configuration determined only up to rigid motion (translation, rotation, reflection); <M tex="D" /> itself is invariant to rigid motions of the data</>,
      <>
        <M tex="\lambda_j" /> and class separation invariant under any invertible affine map <M tex="X\mapsto XA + 1b^T" />
      </>,
    ],
    [
      'Sensitivity to outliers',
      <>High — the Frobenius criterion squares residuals</>,
      <>High — variance is a squared criterion</>,
      <>High — squared distances enter <M tex="B" /></>,
      <>High — means and pooled scatter are non-robust</>,
    ],
  ];

  const sbRank = L ? L.sbEigenvalues.filter((v) => v > 1e-10 * Math.max(Math.abs(L.sbEigenvalues[0] ?? 0), 1e-300)).length : null;

  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <table className="summary-table">
          <thead>
            <tr>
              <th>Property</th>
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
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="small muted stack" style={{ gap: 4, marginTop: 10 }}>
        <div>
          ¹ The SVD has exactly <M tex="\operatorname{rank}(X)" /> non-zero singular values; for the current <M tex="X_c" /> this is <M tex={`${a.svd.rank}`} /> of <M tex={`\\min(n,p) = ${Math.min(a.n, a.p)}`} />.
        </div>
        <div>
          ² <M tex="\min(n-1,p)" /> is the largest possible rank of a centred matrix; the number of non-zero <M tex="\lambda_j" /> equals <M tex="\operatorname{rank}(X_c)" />, here <M tex={`${a.pca.rank}`} />.
        </div>
        <div>
          ³ For a Euclidean <M tex="D" /> the number of positive eigenvalues of <M tex="B" /> equals <M tex="\operatorname{rank}(X_c)" />; currently <M tex="B" /> has <M tex={`${a.mds.positive}`} /> positive and <M tex={`${a.mds.negative}`} /> negative eigenvalues.
        </div>
        <div>
          ⁴ Precisely, LDA yields at most <M tex="\min(K-1,\ \operatorname{rank}S_W)" /> non-trivial discriminants, because <M tex="\operatorname{rank}S_B\le K-1" /> (with equality iff the class means are affinely independent) and every direction must lie in the range
          of <M tex="S_W" />.{' '}
          {L && sbRank !== null ? (
            <>
              Currently <M tex={`\\min(${L.K - 1},\\ ${L.swRank}) = ${Math.min(L.K - 1, L.swRank)}`} />, <M tex={`\\operatorname{rank}S_B = ${sbRank}`} />, and <M tex={`${L.maxDims}`} /> discriminant{L.maxDims === 1 ? '' : 's'} are computed.
            </>
          ) : (
            <>The current dataset is unlabelled, so LDA is not defined.</>
          )}
        </div>
      </div>
    </>
  );
}
