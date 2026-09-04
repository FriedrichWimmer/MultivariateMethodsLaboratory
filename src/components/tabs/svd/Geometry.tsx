import { useMemo, useState } from 'react';
import { transpose, matmul, identity, determinant, fmt, type Matrix } from '../../../lib/linalg';
import { svd2d, transformPoints, unitCircle } from '../../../lib/svdlab';
import { accent, accent2, neutralMark, methodColor, withAlpha, ink } from '../../../lib/theme';
import { M, MBlock } from '../../common/Math';
import { ScatterSVG } from '../../common/ScatterSVG';
import { MatrixView, MatrixEquation } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile } from '../../common/Panels';
import { Slider, Toggle, Button } from '../../common/Controls';
import { presets, orthogonal2, housePolygon, toDeg, toRad, fmtDeg, angleDiffModPi, texNum } from './util';

interface GeomState {
  thetaUdeg: number;
  thetaVdeg: number;
  reflectU: boolean;
  reflectV: boolean;
  s1: number;
  s2: number;
}

function stateFromMatrix(A: Matrix): GeomState {
  const d = svd2d(A);
  return { thetaUdeg: toDeg(d.thetaU), thetaVdeg: toDeg(d.thetaV), reflectU: d.reflectionU, reflectV: d.reflectionV, s1: d.s[0], s2: d.s[1] };
}

const PANEL_W = 300;
const PANEL_H = 290; // inner plotting area 240 × 240 → equal aspect with a symmetric domain

const panelTitles = ['1 · Unit circle and basis', '2 · After Vᵀ', '3 · After ΣVᵀ', '4 · After UΣVᵀ = A'];
const markLabels = [
  ['v₁', 'v₂'],
  ['Vᵀv₁', 'Vᵀv₂'],
  ['ΣVᵀv₁', 'ΣVᵀv₂'],
  ['σ₁u₁', 'σ₂u₂'],
];

/** Lesson 2 — rotation · scaling · rotation on the unit circle. */
export function GeometrySection({ A }: { A: Matrix }) {
  const is2x2 = A.length === 2 && A[0].length === 2;
  const [g, setG] = useState<GeomState>(() => stateFromMatrix(presets[0].A));
  const patch = (x: Partial<GeomState>) => setG((prev) => ({ ...prev, ...x }));
  const sMax = Math.max(4, Math.ceil(g.s1) + 1);

  const geo = useMemo(() => {
    const thetaU = toRad(g.thetaUdeg);
    const thetaV = toRad(g.thetaVdeg);
    const U = orthogonal2(thetaU, g.reflectU);
    const V = orthogonal2(thetaV, g.reflectV);
    const sig = [g.s1, Math.min(g.s2, g.s1)];
    const Sigma: Matrix = [
      [sig[0], 0],
      [0, sig[1]],
    ];
    const Vt = transpose(V);
    const SVt = matmul(Sigma, Vt);
    const Am = matmul(U, SVt);
    const stages: Matrix[] = [identity(2), Vt, SVt, Am];
    const circle = unitCircle(96);
    const v1: [number, number] = [V[0][0], V[1][0]];
    const v2: [number, number] = [V[0][1], V[1][1]];
    const panels = stages.map((Mx) => ({
      M: Mx,
      circle: transformPoints(Mx, circle),
      house: transformPoints(Mx, housePolygon),
      e1: transformPoints(Mx, [[1, 0]])[0],
      e2: transformPoints(Mx, [[0, 1]])[0],
      v1: transformPoints(Mx, [v1])[0],
      v2: transformPoints(Mx, [v2])[0],
    }));
    const recovered = svd2d(Am);
    const L = Math.max(1, sig[0]) * 1.25;
    const detU = determinant(U);
    const detV = determinant(V);
    const detA = determinant(Am);
    return { thetaU, thetaV, U, V, sig, Sigma, Vt, SVt, A: Am, panels, recovered, L, detU, detV, detA, v1, v2 };
  }, [g]);

  const { sig, recovered } = geo;
  const dV = angleDiffModPi(geo.thetaV, recovered.thetaV);
  const dU = angleDiffModPi(geo.thetaU, recovered.thetaU);
  const degenerate = sig[0] - sig[1] < 1e-6;
  const collapsed = sig[1] < 1e-9;
  const signTex = (x: number) => (x >= 0 ? '+1' : '-1');
  const captions = [
    <>
      Arrows: <M tex="e_1, e_2" />. Marks: <M tex="v_1" /> at <M tex={`\\theta_V = ${texNum(toDeg(geo.thetaV), 1)}^\\circ`} /> and <M tex="v_2 \perp v_1" /> — the pre-images of
      the ellipse axes.
    </>,
    <>
      <M tex="V^T" /> sends <M tex="v_1 \mapsto e_1" />, <M tex="v_2 \mapsto e_2" />. <M tex={`\\det V = ${signTex(geo.detV)}`} />: {geo.detV > 0 ? 'a rotation' : 'a reflection'}.
    </>,
    <>
      <M tex="\Sigma" /> stretches <M tex="e_1" /> by <M tex={`\\sigma_1 = ${texNum(sig[0], 2)}`} /> and <M tex="e_2" /> by <M tex={`\\sigma_2 = ${texNum(sig[1], 2)}`} />. Dashed:
      the analytic ellipse with semi-axes <M tex="\sigma_1, \sigma_2" />.
    </>,
    <>
      <M tex="U" /> sends <M tex="e_1 \mapsto u_1" /> at <M tex={`\\theta_U = ${texNum(toDeg(geo.thetaU), 1)}^\\circ`} />; <M tex={`\\det U = ${signTex(geo.detU)}`} />:{' '}
      {geo.detU > 0 ? 'a rotation' : 'a reflection'}. Dashed: ellipse with semi-axes <M tex="\sigma_1 u_1, \sigma_2 u_2" />.
    </>,
  ];

  return (
    <Section
      id="svd-geometry"
      title="2 · Geometry: rotate, scale, rotate"
      subtitle="Any 2×2 matrix acts on the plane as Vᵀ (rotation or reflection), then Σ (axis-aligned scaling), then U (rotation or reflection). Move the sliders and follow the unit circle through the three stages."
    >
      <div className="grid c3">
        <Card title={<>Right factor <M tex="V" /></>} plane>
          <div className="controls-panel">
            <Slider label={<M tex="\theta_V \text{ (angle of } v_1)" />} value={g.thetaVdeg} min={-180} max={180} step={1} onChange={(v) => patch({ thetaVdeg: v })} format={(v) => `${v.toFixed(0)}°`} />
            <Toggle label={<>reflection: <M tex="\det V = -1" /></>} checked={g.reflectV} onChange={(v) => patch({ reflectV: v })} />
          </div>
        </Card>
        <Card title={<>Gains <M tex="\Sigma = \operatorname{diag}(\sigma_1, \sigma_2)" /></>} plane>
          <div className="controls-panel">
            <Slider label={<M tex="\sigma_1" />} value={g.s1} min={0} max={sMax} step={0.05} onChange={(v) => patch({ s1: v, s2: Math.min(g.s2, v) })} />
            <Slider label={<M tex="\sigma_2 \le \sigma_1" />} value={Math.min(g.s2, g.s1)} min={0} max={Math.max(g.s1, 0.05)} step={0.05} onChange={(v) => patch({ s2: Math.min(v, g.s1) })} />
          </div>
        </Card>
        <Card title={<>Left factor <M tex="U" /></>} plane>
          <div className="controls-panel">
            <Slider label={<M tex="\theta_U \text{ (angle of } u_1)" />} value={g.thetaUdeg} min={-180} max={180} step={1} onChange={(v) => patch({ thetaUdeg: v })} format={(v) => `${v.toFixed(0)}°`} />
            <Toggle label={<>reflection: <M tex="\det U = -1" /></>} checked={g.reflectU} onChange={(v) => patch({ reflectU: v })} />
            <div className="row">
              <Button small primary disabled={!is2x2} onClick={() => setG(stateFromMatrix(A))} title={is2x2 ? 'Decompose the matrix edited in lesson 1' : 'Lesson 1 matrix must be 2×2'}>
                Load U, Σ, V from lesson 1
              </Button>
              <Button small onClick={() => setG(stateFromMatrix(presets[0].A))}>
                Reset (shear)
              </Button>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid c4" style={{ marginTop: 14 }}>
        {geo.panels.map((pn, idx) => (
          <ScatterSVG
            key={idx}
            points={[]}
            width={PANEL_W}
            height={PANEL_H}
            domain={{ x: [-geo.L, geo.L], y: [-geo.L, geo.L] }}
            title={panelTitles[idx]}
            vectors={[
              { x: pn.e1[0], y: pn.e1[1], label: 'e₁', color: ink.primary },
              { x: pn.e2[0], y: pn.e2[1], label: 'e₂', color: ink.secondary },
            ]}
            extraPoints={[
              { x: pn.v1[0], y: pn.v1[1], color: methodColor.SVD, r: 4.5, shape: 'diamond', label: markLabels[idx][0], labelPosition: 'right' },
              { x: pn.v2[0], y: pn.v2[1], color: accent2, r: 4.5, shape: 'diamond', label: markLabels[idx][1], labelPosition: 'right' },
            ]}
            ellipses={idx >= 2 ? [{ cx: 0, cy: 0, rx: sig[0], ry: sig[1], angle: idx === 2 ? 0 : geo.thetaU, color: accent2, dashed: true, width: 1 }] : []}
            render={(sx, sy) => (
              <g>
                <polyline points={pn.circle.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} fill="none" stroke={accent} strokeWidth={1.8} />
                <polygon points={pn.house.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} fill={withAlpha(neutralMark, 0.12)} stroke={neutralMark} strokeWidth={1.2} strokeLinejoin="round" />
              </g>
            )}
            caption={captions[idx]}
          />
        ))}
      </div>

      <div className="grid side-r" style={{ marginTop: 14 }}>
        <div className="stack">
          <MatrixEquation
            items={[
              <MatrixView M={geo.A} title="A" digits={3} heat="diverging" />,
              '=',
              <MatrixView M={geo.U} title="U" digits={3} heat="diverging" />,
              '·',
              <MatrixView M={geo.Sigma} title="Σ" digits={3} heat="sequential" />,
              '·',
              <MatrixView M={geo.Vt} title="Vᵀ" digits={3} heat="diverging" />,
            ]}
          />
          <div className="stats">
            <StatTile label={<M tex="\theta_V" />} value={fmtDeg(geo.thetaV)} note={geo.detV > 0 ? 'V is a rotation' : 'V is a reflection'} />
            <StatTile label={<M tex="\theta_U" />} value={fmtDeg(geo.thetaU)} note={geo.detU > 0 ? 'U is a rotation' : 'U is a reflection'} />
            <StatTile label={<M tex="\det A = \det U \cdot \sigma_1\sigma_2 \cdot \det V" />} value={fmt(geo.detA, 3)} note={`= (${signTex(geo.detU)})·${fmt(sig[0] * sig[1], 3)}·(${signTex(geo.detV)})`} />
            <StatTile label={<M tex="\kappa_2 = \sigma_1/\sigma_2" />} value={fmt(collapsed ? Infinity : sig[0] / sig[1], 3)} note="eccentricity of the ellipse" />
            <StatTile label={<M tex="\text{area}(A\,\mathbb{D}) = \pi\sigma_1\sigma_2" />} value={fmt(Math.PI * sig[0] * sig[1], 3)} note={`= π·|det A| = ${fmt(Math.PI * Math.abs(geo.detA), 3)}`} />
          </div>
        </div>
        <Card title="Round trip: does svd2d(A) recover the sliders?">
          <table className="summary-table">
            <thead>
              <tr>
                <th>quantity</th>
                <th>set</th>
                <th>recovered</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <M tex="\sigma_1" />
                </td>
                <td className="mono">{fmt(sig[0], 4)}</td>
                <td className="mono">{fmt(recovered.s[0], 4)}</td>
              </tr>
              <tr>
                <td>
                  <M tex="\sigma_2" />
                </td>
                <td className="mono">{fmt(sig[1], 4)}</td>
                <td className="mono">{fmt(recovered.s[1], 4)}</td>
              </tr>
              <tr>
                <td>
                  <M tex="\theta_V \bmod \pi" />
                </td>
                <td className="mono">{fmtDeg(geo.thetaV)}</td>
                <td className="mono">
                  {fmtDeg(recovered.thetaV)} (gap {fmtDeg(dV, 2)})
                </td>
              </tr>
              <tr>
                <td>
                  <M tex="\theta_U \bmod \pi" />
                </td>
                <td className="mono">{fmtDeg(geo.thetaU)}</td>
                <td className="mono">
                  {fmtDeg(recovered.thetaU)} (gap {fmtDeg(dU, 2)})
                </td>
              </tr>
              <tr>
                <td>
                  <M tex="\det U \cdot \det V" />
                </td>
                <td className="mono">{signTex(geo.detU * geo.detV)}</td>
                <td className="mono">{signTex((recovered.reflectionU ? -1 : 1) * (recovered.reflectionV ? -1 : 1))}</td>
              </tr>
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 8 }}>
            Angles agree modulo <M tex="\pi" /> because <M tex="(u_j, v_j) \mapsto (-u_j, -v_j)" /> changes nothing; only the product <M tex="\det U \det V = \operatorname{sign}\det A" /> is
            determined, not the two reflection flags separately.
            {degenerate && (
              <>
                {' '}
                With <M tex="\sigma_1 = \sigma_2" /> the singular vectors are not unique at all, so the recovered angles may differ arbitrarily.
              </>
            )}
          </div>
        </Card>
      </div>

      <div style={{ marginTop: 14 }}>
        <Callout kind="theorem" title="The image of the unit circle is an ellipse with semi-axes σ₁u₁ and σ₂u₂">
          Let <M tex="A = U\Sigma V^T \in \mathbb{R}^{2\times 2}" />. Every unit vector is <M tex="x = \cos t\, v_1 + \sin t\, v_2" /> for some <M tex="t" />, because{' '}
          <M tex="\{v_1, v_2\}" /> is an orthonormal basis. Since <M tex="A v_j = \sigma_j u_j" />,
          <MBlock tex="A x = \sigma_1 \cos t\; u_1 + \sigma_2 \sin t\; u_2 ,\qquad t \in [0, 2\pi)," />
          which is the ellipse centred at the origin with semi-axes <M tex="\sigma_1 u_1" /> and <M tex="\sigma_2 u_2" /> (a segment if <M tex="\sigma_2 = 0" />, a circle if{' '}
          <M tex="\sigma_1 = \sigma_2" />). The pre-images of the semi-axes are <M tex="\pm v_1" /> and <M tex="\pm v_2" />, and <M tex="\|Ax\|" /> is maximised (<M tex="= \sigma_1" />) at{' '}
          <M tex="x = \pm v_1" /> and minimised (<M tex="= \sigma_2" />) at <M tex="x = \pm v_2" />. In the panels the transformed circle (solid) lies on the analytic ellipse (dashed).
        </Callout>
      </div>

      <div style={{ marginTop: 14 }}>
        <Interpretation
          items={{
            seeing: (
              <>
                Panel 1 is the unit circle with a house inside it. Panel 2 applies <M tex="V^T" />, which turns the picture by <M tex={`${texNum(-toDeg(geo.thetaV), 1)}^\\circ`} />
                {geo.detV < 0 ? ' and mirrors it' : ''} so that <M tex="v_1" /> lands on <M tex="e_1" />. Panel 3 stretches the axes by <M tex={`\\sigma_1 = ${texNum(sig[0], 2)}`} /> and{' '}
                <M tex={`\\sigma_2 = ${texNum(sig[1], 2)}`} />, producing an axis-aligned ellipse. Panel 4 applies <M tex="U" />, turning the ellipse to angle{' '}
                <M tex={`\\theta_U = ${texNum(toDeg(geo.thetaU), 1)}^\\circ`} />
                {geo.detU < 0 ? ' with a mirror image' : ''}.
              </>
            ),
            why: (
              <>
                Orthogonal matrices preserve lengths and angles, so panels 1 → 2 and 3 → 4 are rigid motions (the house keeps its shape{geo.detV < 0 || geo.detU < 0 ? ', though its handedness flips at a reflection' : ''}); all
                deformation happens in panel 3, where the circle becomes an ellipse with axis ratio <M tex={`\\kappa_2 = ${texNum(collapsed ? Infinity : sig[0] / sig[1], 3)}`} />.{' '}
                {collapsed ? (
                  <>
                    With <M tex="\sigma_2 = 0" /> the plane is flattened onto a line: <M tex="A" /> has rank one.
                  </>
                ) : degenerate ? (
                  <>
                    With <M tex="\sigma_1 = \sigma_2" /> the ellipse is a circle: <M tex="A" /> is a scaled orthogonal matrix.
                  </>
                ) : (
                  <>
                    The area of the house is multiplied by <M tex={`|\\det A| = \\sigma_1\\sigma_2 = ${texNum(sig[0] * sig[1], 3)}`} />.
                  </>
                )}
              </>
            ),
            math: (
              <>
                <M tex="A = U\Sigma V^T" /> is read right to left on a vector: <M tex="x \mapsto V^Tx \mapsto \Sigma V^T x \mapsto U\Sigma V^T x" />. The coordinates{' '}
                <M tex="V^T x = (v_1^Tx, v_2^Tx)" /> are the coordinates of <M tex="x" /> in the basis <M tex="\{v_1, v_2\}" />; <M tex="\Sigma" /> scales them; <M tex="U" /> places them
                onto the basis <M tex="\{u_1, u_2\}" />. Hence <M tex="\det A = \det U \cdot \sigma_1\sigma_2 \cdot \det V = " /> <M tex={texNum(geo.detA, 3)} />.
              </>
            ),
            stats: (
              <>
                For a centred <M tex="2\times 2" /> covariance-type problem this is PCA in miniature: <M tex="v_1" /> is the direction of largest gain (the first principal axis),{' '}
                <M tex="\sigma_1^2/\sigma_2^2 = " /> <M tex={texNum(collapsed ? Infinity : (sig[0] / sig[1]) ** 2, 3)} /> is the ratio of the variances along the two axes, and a covariance
                ellipse is exactly the image of a circle under the matrix square root of the covariance.
              </>
            ),
            careful: (
              <>
                The angles are defined only modulo <M tex="\pi" /> (sign indeterminacy of singular vectors), and the reflection flags only jointly: the round-trip table shows{' '}
                <M tex={`\\det U\\det V = ${signTex(geo.detU * geo.detV)}`} /> is recovered, not each factor. When <M tex="\sigma_1 - \sigma_2" /> is small (currently{' '}
                <M tex={texNum(sig[0] - sig[1], 3)} />) the singular vectors become ill-determined: tiny perturbations of <M tex="A" /> rotate them by large angles even though the
                singular values move little.
              </>
            ),
          }}
        />
      </div>
    </Section>
  );
}
