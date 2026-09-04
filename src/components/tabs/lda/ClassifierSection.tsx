import { useMemo, useState } from 'react';
import { column, dot, fmt, matvec, scale, vadd, vsub } from '../../../lib/linalg';
import { lda, ldaClassify } from '../../../lib/lda';
import { classColor, ink, methodColor, neutralMark } from '../../../lib/theme';
import { M, MBlock, texVector } from '../../common/Math';
import { Select, Toggle } from '../../common/Controls';
import { Badge, Callout, Card, ClassLegend, Derivation, Interpretation, Section, StatTile } from '../../common/Panels';
import { MatrixView } from '../../common/MatrixView';
import { ScatterSVG } from '../../common/ScatterSVG';
import { axialAngle, axialAngleBetween, confusionMatrix, invertSymmetric, normalCdf, pct, unitFromAngle, type TwoD } from './helpers';

interface Props {
  twoD: TwoD;
}

/** Lesson 5 — Fisher's discriminant versus Gaussian generative LDA classification. */
export default function ClassifierSection({ twoD }: Props) {
  const { X2, y, classNames, res: resAll } = twoD;
  const classes = resAll.classes;

  const pairs = useMemo(() => {
    const out: [number, number][] = [];
    for (let i = 0; i < classes.length; i++) for (let j = i + 1; j < classes.length; j++) out.push([classes[i], classes[j]]);
    return out;
  }, [classes]);
  const [pairKey, setPairKey] = useState('');
  const pairSel = useMemo(() => pairs.find((p) => `${p[0]}-${p[1]}` === pairKey) ?? pairs[0] ?? [classes[0], classes[0]], [pairs, pairKey, classes]);
  const [equalPriors, setEqualPriors] = useState(false);

  const subset = useMemo(() => {
    const idx: number[] = [];
    for (let i = 0; i < y.length; i++) if (y[i] === pairSel[0] || y[i] === pairSel[1]) idx.push(i);
    return { idx, X: idx.map((i) => X2[i]), y: idx.map((i) => y[i]) };
  }, [X2, y, pairSel]);
  const res = useMemo(() => lda(subset.X, subset.y), [subset]);

  const geo = useMemo(() => {
    const nS = res.n;
    const Sigma = scale(res.SW, 1 / Math.max(nS - res.K, 1));
    const SigInv = invertSymmetric(Sigma);
    const m1 = res.classMeans[0];
    const m2 = res.classMeans[1] ?? res.classMeans[0];
    const n1 = res.classSizes[0];
    const n2 = res.classSizes[1] ?? 0;
    const d = vsub(m1, m2);
    const a = matvec(SigInv, d);
    const pi1 = equalPriors ? 0.5 : n1 / nS;
    const pi2 = equalPriors ? 0.5 : n2 / nS;
    const logPrior = Math.log(pi1 / pi2);
    const b = -0.5 * dot(vadd(m1, m2), a) + logPrior;
    const a2 = dot(a, a) || 1;
    const x0: [number, number] = [(-b * a[0]) / a2, (-b * a[1]) / a2];
    const boundaryAngle = Math.atan2(-a[0], a[1]);
    const aAngle = axialAngle(Math.atan2(a[1], a[0]));
    const mid: [number, number] = [(m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2];
    const delta2 = dot(d, matvec(SigInv, d));
    const delta = Math.sqrt(Math.max(delta2, 0));
    const w1 = res.maxDims > 0 ? column(res.W, 0) : [1, 0];
    const angleAW = axialAngleBetween(a, w1);
    const lambdaFromDelta = (n1 * n2 * delta2) / (nS * Math.max(nS - 2, 1));
    const cls = ldaClassify(res, subset.X, equalPriors ? [0.5, 0.5] : undefined);
    const pred = cls.predictions;
    const mis: number[] = [];
    for (let i = 0; i < pred.length; i++) if (pred[i] !== subset.y[i]) mis.push(i);
    const acc = nS ? 1 - mis.length / nS : NaN;
    const conf = confusionMatrix(subset.y, pred, res.classes);
    const gaussErr = normalCdf(-delta / 2);
    const aUnit = unitFromAngle(aAngle);
    return { nS, n1, n2, m1, m2, a, b, pi1, pi2, logPrior, x0, boundaryAngle, aAngle, aUnit, mid, delta, w1, angleAW, lambdaFromDelta, pred, mis, acc, conf, gaussErr, Sigma };
  }, [res, subset, equalPriors]);

  const all = useMemo(() => {
    const cls = ldaClassify(resAll, X2);
    const conf = confusionMatrix(y, cls.predictions, classes);
    const correct = cls.predictions.reduce((s, p, i) => s + (p === y[i] ? 1 : 0), 0);
    return { conf, acc: y.length ? correct / y.length : NaN, correct };
  }, [resAll, X2, y, classes]);

  const name = (c: number) => classNames[c] ?? `class ${c}`;
  const halfRange = twoD.halfRange;
  const L = halfRange * 0.4;
  const pairNames = res.classes.map(name);

  return (
    <Section
      id="lda-classifier"
      title="5 · Fisher's discriminant versus Gaussian LDA classification"
      subtitle="Two different objects share one name; with two classes they point in the same direction"
      right={<Badge method="LDA" />}
    >
      <div className="grid c2">
        <Callout kind="definition" title="Fisher's linear discriminant (a projection)">
          <p>
            No distributional assumption. Choose <M tex="w" /> to maximise <M tex="J(w) = w^{\mathsf T} S_B w / w^{\mathsf T} S_W w" />; report the scores <M tex="z = X_c w" />. It reduces
            dimension to at most <M tex="K-1" /> axes and makes no decision by itself. A threshold on <M tex="z" /> (lesson 3) is an add-on, not part of the method.
          </p>
        </Callout>
        <Callout kind="definition" title="Gaussian generative LDA (a classifier)">
          <p>
            Assume <M tex="x \mid y = k \sim N(\mu_k, \Sigma)" /> with a <b>common</b> covariance <M tex="\Sigma" /> and prior class probabilities <M tex="\pi_k" />. Estimate{' '}
            <M tex="\hat\mu_k = m_k" />, <M tex="\hat\pi_k = n_k/n" /> and the pooled <M tex="\hat\Sigma = S_W/(n-K)" /> (here <M tex="\hat\Sigma" /> is this pooled covariance, not the
            singular-value matrix of the SVD). Bayes' rule assigns <M tex="x" /> to the class with the largest posterior, which after dropping common terms is the class with the largest linear
            discriminant function
          </p>
          <MBlock tex="\delta_k(x) = x^{\mathsf T}\hat\Sigma^{-1} m_k - \tfrac12\, m_k^{\mathsf T}\hat\Sigma^{-1} m_k + \log \pi_k ." />
        </Callout>
      </div>

      <Derivation
        title="Why the two-class Bayes direction is Fisher's direction"
        initiallyRevealed={1}
        steps={[
          {
            title: 'The boundary between two classes is linear',
            body: (
              <>
                <M tex="\delta_1(x) = \delta_2(x)" /> is equivalent to <M tex="a^{\mathsf T} x + b = 0" /> with
                <MBlock tex="a = \hat\Sigma^{-1}(m_1 - m_2), \qquad b = -\tfrac12 (m_1 + m_2)^{\mathsf T} a + \log\frac{\pi_1}{\pi_2} ." />
                The quadratic terms <M tex="x^{\mathsf T}\hat\Sigma^{-1}x" /> cancel because <M tex="\Sigma" /> is shared; that is exactly what makes LDA <em>linear</em>. The boundary is
                perpendicular (in the Euclidean sense) to <M tex="a" />, not to <M tex="m_1 - m_2" />.
              </>
            ),
          },
          {
            title: 'With two classes S_B has rank one',
            body: (
              <>
                From <M tex="m = (n_1 m_1 + n_2 m_2)/n" /> we get <M tex="m_1 - m = \tfrac{n_2}{n}(m_1 - m_2)" /> and <M tex="m_2 - m = -\tfrac{n_1}{n}(m_1 - m_2)" />, hence
                <MBlock tex="S_B = n_1 (m_1 - m)(m_1 - m)^{\mathsf T} + n_2 (m_2 - m)(m_2 - m)^{\mathsf T} = \frac{n_1 n_2}{n}\,(m_1 - m_2)(m_1 - m_2)^{\mathsf T} ." />
              </>
            ),
          },
          {
            title: 'The generalised eigenproblem collapses to one direction',
            body: (
              <>
                <M tex="S_B w = \lambda S_W w" /> with <M tex="S_B w = \tfrac{n_1 n_2}{n}(m_1 - m_2)\,[(m_1 - m_2)^{\mathsf T} w]" /> forces <M tex="S_W w \propto m_1 - m_2" />, i.e.
                <MBlock tex="w_1 \propto S_W^{-1}(m_1 - m_2) = (n-2)^{-1}\,\hat\Sigma^{-1}(m_1 - m_2) \propto a ." />
                Substituting back gives the eigenvalue in closed form, <M tex="\lambda_1 = \tfrac{n_1 n_2}{n}\,(m_1 - m_2)^{\mathsf T} S_W^{-1}(m_1 - m_2) = \tfrac{n_1 n_2}{n(n-2)}\,\Delta^2" />{' '}
                with <M tex="\Delta^2 = (m_1 - m_2)^{\mathsf T}\hat\Sigma^{-1}(m_1 - m_2)" /> the squared Mahalanobis distance between the class means.
              </>
            ),
            note: (
              <>
                Live check on the selected pair: angle between <M tex="a" /> and <M tex="w_1" /> = {geo.angleAW.toFixed(3)}°; <M tex={`\\lambda_1 = ${fmt(res.eigenvalues[0] ?? 0, 4)}`} /> versus{' '}
                <M tex={`\\tfrac{n_1 n_2}{n(n-2)}\\Delta^2 = ${fmt(geo.lambdaFromDelta, 4)}`} />.
              </>
            ),
          },
        ]}
      />

      {twoD.local && (
        <Callout kind="info" title="Local stand-in data">
          The active dataset is unlabelled; the classifier runs on the same two-dimensional local data as lesson 2 ({twoD.sourceName}).
        </Callout>
      )}

      <div className="grid side-r">
        <div>
          <ScatterSVG
            points={subset.X}
            labels={subset.y}
            classNames={classNames}
            width={520}
            height={440}
            xLabel={`${twoD.names[0]} (centred)`}
            yLabel={`${twoD.names[1]} (centred)`}
            title={`Decision boundary δ₁(x) = δ₂(x) for ${pairNames.join(' vs ')}`}
            lines={[
              { angle: geo.boundaryAngle, through: geo.x0, color: ink.primary, width: 2 },
              { angle: geo.aAngle, through: geo.mid, color: methodColor.LDA, dashed: true, opacity: 0.55 },
            ]}
            vectors={[{ x: L * geo.aUnit[0], y: L * geo.aUnit[1], from: geo.mid, color: methodColor.LDA, label: 'a' }]}
            extraPoints={[
              { x: geo.m1[0], y: geo.m1[1], shape: 'class', classIndex: res.classes[0], color: classColor(res.classes[0]), r: 7, label: 'm₁' },
              { x: geo.m2[0], y: geo.m2[1], shape: 'class', classIndex: res.classes[1] ?? res.classes[0], color: classColor(res.classes[1] ?? res.classes[0]), r: 7, label: 'm₂' },
              { x: geo.mid[0], y: geo.mid[1], shape: 'diamond', color: neutralMark, r: 5, label: 'midpoint', labelPosition: 'below' },
            ]}
            highlight={geo.mis}
            hoverInfo={(i) => `#${subset.idx[i] + 1}  true: ${name(subset.y[i])}\npredicted: ${name(geo.pred[i])}${geo.pred[i] !== subset.y[i] ? '  (misclassified)' : ''}`}
            caption={
              <>
                Solid black line: the boundary <M tex="a^{\mathsf T}x + b = 0" />. Dashed orange line and arrow: the direction <M tex="a = \hat\Sigma^{-1}(m_1 - m_2)" /> through the midpoint
                of the class means — the same direction as Fisher's <M tex="w_1" />. Misclassified training points are drawn at full opacity, correctly classified ones are faded
                {geo.mis.length === 0 ? ' (none are misclassified here)' : ''}.
              </>
            }
          />
          <ClassLegend classNames={pairNames} />
        </div>
        <div className="stack">
          {pairs.length > 1 && (
            <Select<string>
              label="Class pair"
              value={`${pairSel[0]}-${pairSel[1]}`}
              options={pairs.map((p) => ({ value: `${p[0]}-${p[1]}`, label: `${name(p[0])} vs ${name(p[1])}` }))}
              onChange={setPairKey}
            />
          )}
          <Toggle label="Equal priors π₁ = π₂ (instead of n_k / n)" checked={equalPriors} onChange={setEqualPriors} />
          <div className="stats">
            <StatTile label="training accuracy" value={pct(geo.acc)} note={`${geo.nS - geo.mis.length} of ${geo.nS} correct, ${geo.mis.length} misclassified`} />
            <StatTile label={<M tex="\Delta" />} value={fmt(geo.delta, 3)} note="Mahalanobis distance between the class means" />
            <StatTile label={<M tex="\Phi(-\Delta/2)" />} value={pct(geo.gaussErr)} note="model-implied error rate with equal priors" />
            <StatTile label={<M tex="\angle(a, w_1)" />} value={`${geo.angleAW.toFixed(3)}°`} note="Bayes direction vs Fisher direction" />
            <StatTile label={<M tex="\log(\pi_1/\pi_2)" />} value={fmt(geo.logPrior, 3)} note={`π₁ = ${fmt(geo.pi1, 3)}, π₂ = ${fmt(geo.pi2, 3)}`} />
            <StatTile label={<M tex="b" />} value={fmt(geo.b, 3)} note="intercept of the boundary" />
          </div>
          <Card title="Confusion counts (training data)" plane>
            <div className="table-wrap">
              <table className="summary-table">
                <thead>
                  <tr>
                    <th>true \ predicted</th>
                    {res.classes.map((c) => (
                      <th key={c}>{name(c)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {res.classes.map((c, i) => (
                    <tr key={c}>
                      <td>{name(c)}</td>
                      {res.classes.map((c2, j) => (
                        <td key={c2} style={{ fontWeight: i === j ? 600 : 400 }}>
                          {geo.conf[i][j]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <div className="small secondary">
            <M tex={`a = ${texVector(geo.a, 3, false)}^{\\mathsf T}`} />, <M tex={`b = ${fmt(geo.b, 3)}`} />, <M tex={`\\hat\\Sigma = S_W/(n-K)`} /> with <M tex={`n - K = ${geo.nS - res.K}`} />.
          </div>
        </div>
      </div>

      {classes.length > 2 && (
        <Card title={`All K = ${classes.length} classes: argmax_k δ_k(x) on the same two variables`}>
          <div className="grid side-r">
            <div className="prose small">
              <p>
                With more than two classes each pair <M tex="(k, l)" /> has its own linear boundary <M tex="\delta_k(x) = \delta_l(x)" />; the decision regions are convex polygons formed by
                the pieces of those lines that matter. The rule below uses all <M tex={`K = ${classes.length}`} /> classes with priors <M tex="n_k/n" /> and the pooled{' '}
                <M tex="\hat\Sigma = S_W/(n-K)" />; it classifies {all.correct} of {y.length} training points correctly ({pct(all.acc)}). Off-diagonal cells show which classes are confused.
              </p>
            </div>
            <MatrixView M={all.conf} title="Confusion matrix (rows: true, columns: predicted)" rowLabels={classes.map(name)} colLabels={classes.map(name)} digits={0} heat="sequential" compact />
          </div>
        </Card>
      )}

      <Callout kind="warning" title="Training accuracy is optimistic">
        The same {geo.nS} observations were used to estimate <M tex="m_1, m_2, \hat\Sigma" /> and to evaluate the rule, so {pct(geo.acc)} over-states what the classifier would achieve on new
        data from the same population; the optimism grows with <M tex="p/n" /> and is extreme in the <M tex="p > n" /> setting of lesson 4. The plug-in Gaussian estimate{' '}
        <M tex={`\\Phi(-\\hat\\Delta/2) = ${pct(geo.gaussErr)}`} /> is also optimistic, because <M tex="\hat\Delta" /> is biased upwards. Report cross-validated or hold-out error rates.
      </Callout>

      <Interpretation
        items={{
          seeing: (
            <>
              The {geo.nS} observations of {pairNames.join(' and ')} in the plane of lesson 2, the two class means <M tex="m_1, m_2" />, and the Gaussian LDA boundary. Points on the{' '}
              <M tex="m_1" /> side of the line are assigned to {name(res.classes[0])}, the others to {name(res.classes[1] ?? res.classes[0])}; {geo.mis.length} training point
              {geo.mis.length === 1 ? ' is' : 's are'} on the wrong side ({pct(geo.acc)} accuracy). The confusion table splits those errors by true class.
            </>
          ),
          why: (
            <>
              The boundary is the set where the two discriminant functions tie. Its normal vector is <M tex={`a = \\hat\\Sigma^{-1}(m_1 - m_2) = ${texVector(geo.a, 2, false)}^{\\mathsf T}`} />
              , which is tilted away from <M tex="m_1 - m_2" /> by the pooled covariance: the boundary is perpendicular to <M tex="a" />, not to the segment joining the means. It passes through
              the midpoint only when <M tex="\pi_1 = \pi_2" />; with the current priors the intercept is shifted by <M tex={`\\log(\\pi_1/\\pi_2) = ${fmt(geo.logPrior, 3)}`} />, moving the
              line towards the smaller class.
            </>
          ),
          math: (
            <>
              <M tex="a" /> and Fisher's <M tex="w_1" /> differ by an angle of {geo.angleAW.toFixed(3)}°, confirming <M tex="w_1 \propto \hat\Sigma^{-1}(m_1 - m_2)" /> numerically. The
              closed form <M tex={`\\lambda_1 = \\tfrac{n_1 n_2}{n(n-2)}\\Delta^2 = ${fmt(geo.lambdaFromDelta, 4)}`} /> matches the generalised eigenvalue{' '}
              <M tex={`${fmt(res.eigenvalues[0] ?? 0, 4)}`} /> with <M tex={`\\Delta = ${fmt(geo.delta, 3)}`} />. The projection method and the classifier therefore use the same axis;
              they differ in what they do with it: Fisher stops at the scores, Bayes adds the model, the priors and a threshold at <M tex="a^{\mathsf T}x + b = 0" />.
            </>
          ),
          stats: (
            <>
              Under the fitted Gaussian model with equal priors the misclassification probability is <M tex={`\\Phi(-\\Delta/2) = ${pct(geo.gaussErr)}`} />; the observed training error is{' '}
              {pct(1 - geo.acc)}. Both are estimates from the same {geo.nS} points. The linear rule is optimal only if the classes are Gaussian with a shared covariance; otherwise QDA
              (class-specific covariances) or a non-parametric rule may be needed, at the price of many more parameters.
            </>
          ),
          careful: (
            <>
              Two things are called LDA. Fisher's discriminant is distribution-free dimension reduction; Gaussian LDA is a classifier that inherits its direction but also its assumptions.
              Priors matter: switching to equal priors moves the boundary by <M tex={`${fmt(Math.log((geo.n1 / geo.nS) / (geo.n2 / geo.nS)), 3)}`} /> in units of{' '}
              <M tex="a^{\mathsf T}x" /> relative to the empirical priors. With <M tex="K > 2" /> the direction <M tex="\hat\Sigma^{-1}(m_k - m_l)" /> differs per pair and Fisher's{' '}
              <M tex="K-1" /> axes span, but do not equal, the set of pairwise Bayes directions.
            </>
          ),
        }}
      />
    </Section>
  );
}
