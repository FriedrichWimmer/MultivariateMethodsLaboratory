import { useMemo } from 'react';
import { svdFull, svd, matmul, transpose, sub, frobenius, column, matvec, vsub, vscale, norm, maxAbs, identity, fmt, type Matrix } from '../../../lib/linalg';
import { M, MBlock } from '../../common/Math';
import { MatrixView, MatrixEquation } from '../../common/MatrixView';
import { Section, Card, Callout, Interpretation, StatTile, Badge } from '../../common/Panels';
import { Segmented, Button } from '../../common/Controls';
import { presets, cellsFromMatrix, resizeCells, presetIdOf, texNum, subscript, type Cells } from './util';

interface Props {
  cells: Cells;
  setCells: (c: Cells) => void;
  A: Matrix;
}

const dimOptions = [2, 3, 4].map((v) => ({ value: v, label: String(v) }));

const inputStyle: React.CSSProperties = {
  width: 70,
  padding: '4px 6px',
  border: '1px solid var(--border-2)',
  borderRadius: 4,
  fontSize: 13,
  background: 'var(--surface)',
  textAlign: 'right',
};

/** Lesson 1 — X = UΣVᵀ on a small editable matrix. */
export function DecompositionSection({ cells, setCells, A }: Props) {
  const n = A.length;
  const p = A[0].length;
  const r = Math.min(n, p);
  const activePreset = presetIdOf(cells);

  const d = useMemo(() => {
    const full = svdFull(A);
    const thin = svd(A);
    const recon = matmul(matmul(full.U, full.Sigma), transpose(full.V));
    const residual = frobenius(sub(A, recon));
    const kappa = thin.rank < r ? Infinity : full.s[0] / full.s[r - 1];
    const UtU = matmul(transpose(full.U), full.U);
    const VtV = matmul(transpose(full.V), full.V);
    const uOrtho = maxAbs(sub(UtU, identity(n)));
    const vOrtho = maxAbs(sub(VtV, identity(p)));
    const defining = full.s.map((sig, j) => norm(vsub(matvec(A, column(full.V, j)), vscale(column(full.U, j), sig))));
    const frobX = frobenius(A);
    const sumSq = full.s.reduce((acc, x) => acc + x * x, 0);
    return { full, thin, residual, kappa, UtU, VtV, uOrtho, vOrtho, defining, frobX, sumSq, Vt: transpose(full.V) };
  }, [A, n, p, r]);

  const { full, thin } = d;
  const s = full.s;
  const rankDeficient = thin.rank < r;

  const update = (i: number, j: number, v: string) => {
    const next = cells.map((row) => row.slice());
    next[i][j] = v;
    setCells(next);
  };

  const sigmaTex = s.map((x, j) => `\\sigma_{${j + 1}} = ${texNum(x, 3)}`).join(',\\quad ');
  const uLabels = Array.from({ length: n }, (_, j) => `u${subscript(j + 1)}`);
  const vtLabels = Array.from({ length: p }, (_, j) => `v${subscript(j + 1)}ᵀ`);

  return (
    <Section
      id="svd-decomposition"
      title={
        <>
          1 · The decomposition <M tex="X = U\Sigma V^T" />
        </>
      }
      subtitle="Edit a small matrix and watch its factorisation update: the orthogonal factors U and V carry directions, Σ carries the gains."
      right={<Badge method="SVD" />}
    >
      <Callout kind="definition" title="Singular value decomposition">
        For every <M tex="X \in \mathbb{R}^{n \times p}" /> there exist an orthogonal <M tex="U \in \mathbb{R}^{n \times n}" /> (<M tex="U^T U = I_n" />), an orthogonal{' '}
        <M tex="V \in \mathbb{R}^{p \times p}" /> and a rectangular diagonal <M tex="\Sigma \in \mathbb{R}^{n \times p}" /> with <M tex="\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0" />,{' '}
        <M tex="r = \min(n, p)" />, such that
        <MBlock tex="X = U \Sigma V^T = \sum_{j=1}^{r} \sigma_j\, u_j v_j^T, \qquad X v_j = \sigma_j u_j, \qquad X^T u_j = \sigma_j v_j ." />
        The singular values are unique. A pair <M tex="(u_j, v_j)" /> is unique up to a joint sign change when <M tex="\sigma_j" /> is simple; for repeated singular values only the
        subspace spanned by the corresponding vectors is determined.
      </Callout>

      <div className="grid side" style={{ marginTop: 14 }}>
        <div className="controls-panel">
          <Segmented<string>
            label="Preset"
            value={activePreset}
            options={presets.map((pr) => ({ value: pr.id, label: pr.label, hint: pr.hint }))}
            onChange={(id) => {
              const pr = presets.find((x) => x.id === id);
              if (pr) setCells(cellsFromMatrix(pr.A));
            }}
            compact
          />
          <div className="row">
            <Segmented<number> label="rows n" value={n} options={dimOptions} onChange={(v) => setCells(resizeCells(cells, v, p))} compact />
            <Segmented<number> label="columns p" value={p} options={dimOptions} onChange={(v) => setCells(resizeCells(cells, n, v))} compact />
          </div>
          <div>
            <div className="small secondary" style={{ marginBottom: 6 }}>
              Entries of <M tex="X" /> (editable)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${p}, 70px)`, gap: 6 }}>
              {cells.map((row, i) =>
                row.map((v, j) => (
                  <input
                    key={`${i}-${j}`}
                    type="number"
                    step={0.5}
                    className="mono"
                    aria-label={`entry ${i + 1},${j + 1}`}
                    value={v}
                    onChange={(e) => update(i, j, e.target.value)}
                    style={inputStyle}
                  />
                )),
              )}
            </div>
          </div>
          <div className="row">
            <Button small onClick={() => setCells(cellsFromMatrix(presets[0].A))}>
              Reset to shear
            </Button>
            <span className="kbd-hint">Try a zero column, or make two columns proportional.</span>
          </div>
        </div>

        <div className="stack">
          <MatrixEquation
            items={[
              <MatrixView M={A} title="X" digits={2} heat="diverging" />,
              '=',
              <MatrixView M={full.U} title="U" digits={3} heat="diverging" colLabels={uLabels} highlightCols={[0]} />,
              '·',
              <MatrixView M={full.Sigma} title="Σ" digits={3} heat="sequential" />,
              '·',
              <MatrixView M={d.Vt} title="Vᵀ" digits={3} heat="diverging" rowLabels={vtLabels} highlightRows={[0]} />,
            ]}
          />
          <div className="small muted">
            Highlighted: the leading pair <M tex="(u_1, v_1)" /> that multiplies <M tex="\sigma_1" />. Shapes: <M tex={`U:\\ ${n}\\times${n}`} />, <M tex={`\\Sigma:\\ ${n}\\times${p}`} />,{' '}
            <M tex={`V^T:\\ ${p}\\times${p}`} />.
          </div>
          <div className="stats">
            <StatTile label="Singular values" value={<M tex={sigmaTex} />} note="decreasing by construction" />
            <StatTile label="Rank" value={`${thin.rank} of ${r}`} note={`tolerance ${fmt(thin.tol, 2)}`} />
            <StatTile label={<M tex="\kappa_2 = \sigma_1 / \sigma_r" />} value={fmt(d.kappa, 3)} note={rankDeficient ? 'rank deficient: unbounded' : 'sensitivity to perturbations'} />
            <StatTile label={<M tex="\|X - U\Sigma V^T\|_F" />} value={fmt(d.residual, 2)} note="reconstruction residual" />
            <StatTile label={<M tex="\|X\|_F^2 \text{ vs } \sum_j \sigma_j^2" />} value={`${fmt(d.frobX * d.frobX, 4)} · ${fmt(d.sumSq, 4)}`} note="Frobenius norm is orthogonally invariant" />
          </div>
        </div>
      </div>

      <div className="grid c3" style={{ marginTop: 14 }}>
        <Card title={<>Left singular vectors <M tex="U" /></>} plane>
          <div className="prose small">
            <p>
              The columns <M tex="u_1, \dots, u_n \in \mathbb{R}^n" /> are orthonormal: <M tex="U^T U = I_n" /> holds to <M tex={`\\max|U^TU - I| = ${texNum(d.uOrtho, 2)}`} />. Each
              column of <M tex="X" /> is a vector in observation space <M tex="\mathbb{R}^n" />; <M tex="u_1, \dots, u_{\operatorname{rank} X}" /> form an orthonormal basis of that
              column space, and the remaining <M tex={`${n - thin.rank}`} /> columns span the left null space (<M tex="u_j^T X = 0" />).
            </p>
          </div>
          <MatrixView M={d.UtU} title="UᵀU" digits={2} compact />
        </Card>
        <Card title={<>Singular values <M tex="\Sigma" /></>} plane>
          <div className="prose small">
            <p>
              <M tex="\sigma_j = \|X v_j\|" /> is the gain of <M tex="X" /> along <M tex="v_j" />. Hence <M tex={`\\|X\\|_2 = \\sigma_1 = ${texNum(s[0], 3)}`} />,{' '}
              <M tex={`\\|X\\|_F = \\sqrt{\\sum_j \\sigma_j^2} = ${texNum(d.frobX, 3)}`} />, <M tex={`\\operatorname{rank} X = \\#\\{\\sigma_j > \\text{tol}\\} = ${thin.rank}`} />. The{' '}
              <M tex="\sigma_j^2" /> are the eigenvalues of both <M tex="X^T X" /> and <M tex="X X^T" />; the SVD computes them without forming either product.
            </p>
          </div>
          <table className="summary-table">
            <thead>
              <tr>
                <th>j</th>
                <th>
                  <M tex="\sigma_j" />
                </th>
                <th>
                  <M tex="\|X v_j - \sigma_j u_j\|" />
                </th>
              </tr>
            </thead>
            <tbody>
              {s.map((x, j) => (
                <tr key={j}>
                  <td>{j + 1}</td>
                  <td className="mono">{fmt(x, 4)}</td>
                  <td className="mono">{fmt(d.defining[j], 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title={<>Right singular vectors <M tex="V" /></>} plane>
          <div className="prose small">
            <p>
              The columns <M tex="v_1, \dots, v_p \in \mathbb{R}^p" /> are orthonormal (<M tex={`\\max|V^TV - I| = ${texNum(d.vOrtho, 2)}`} />) and live in variable space: a{' '}
              <M tex="v_j" /> is a weighting of the <M tex="p" /> columns. <M tex="v_1, \dots, v_{\operatorname{rank} X}" /> span the row space; any <M tex="v_j" /> with{' '}
              <M tex="\sigma_j = 0" /> is a null direction, <M tex="X v_j = 0" />
              {rankDeficient ? (
                <>
                  {' '}
                  — here <M tex={`v_{${r}}`} /> is one such direction, since <M tex={`\\sigma_{${r}} = ${texNum(s[r - 1], 2)}`} /> is below the tolerance.
                </>
              ) : (
                <> — none here, since all singular values exceed the tolerance.</>
              )}
            </p>
          </div>
          <MatrixView M={d.VtV} title="VᵀV" digits={2} compact />
        </Card>
      </div>

      <div style={{ marginTop: 14 }}>
        <Interpretation
          items={{
            seeing: (
              <>
                A <M tex={`${n} \\times ${p}`} /> matrix and its three factors. The singular values are <M tex={sigmaTex} />; the numerical rank is {thin.rank} of{' '}
                <M tex={`\\min(n,p) = ${r}`} />; multiplying the factors back reproduces <M tex="X" /> to <M tex={`\\|X - U\\Sigma V^T\\|_F = ${texNum(d.residual, 2)}`} />,
                i.e. rounding error.
              </>
            ),
            why: (
              <>
                <M tex="U" /> and <M tex="V" /> are orthogonal, so they neither stretch nor shrink anything (their <M tex="U^TU" /> and <M tex="V^TV" /> panels are identities to{' '}
                {fmt(Math.max(d.uOrtho, d.vOrtho), 1)}); every gain sits in <M tex="\Sigma" />. The highlighted pair says: the unit direction <M tex="v_1" /> in{' '}
                <M tex="\mathbb{R}^p" /> is sent to <M tex={`\\sigma_1 u_1`} /> in <M tex="\mathbb{R}^n" />, a vector of length <M tex={`\\sigma_1 = ${texNum(s[0], 3)}`} />.
                {rankDeficient ? (
                  <>
                    {' '}
                    Since <M tex={`\\sigma_{${r}} \\approx 0`} />, one direction is annihilated: the columns of <M tex="X" /> are linearly dependent and <M tex="\kappa_2 = \infty" />.
                  </>
                ) : (
                  <>
                    {' '}
                    The condition number <M tex={`\\kappa_2 = \\sigma_1/\\sigma_{${r}} = ${texNum(d.kappa, 3)}`} /> says how unequally <M tex="X" /> treats different directions.
                  </>
                )}
              </>
            ),
            math: (
              <>
                From <M tex="X = U\Sigma V^T" /> and orthogonality, <M tex="X^T X = V (\Sigma^T\Sigma) V^T" /> and <M tex="X X^T = U (\Sigma\Sigma^T) U^T" />: the right singular vectors
                are eigenvectors of <M tex="X^TX" />, the left ones of <M tex="XX^T" />, both with eigenvalues <M tex="\sigma_j^2" />. Orthogonal invariance gives{' '}
                <M tex={`\\|X\\|_F^2 = \\|\\Sigma\\|_F^2 = \\sum_j \\sigma_j^2`} />: <M tex={`${texNum(d.frobX * d.frobX, 4)} = ${texNum(d.sumSq, 4)}`} />.
              </>
            ),
            stats: (
              <>
                Read <M tex="X" /> as a data matrix with <M tex={`n = ${n}`} /> observations and <M tex={`p = ${p}`} /> variables. Then <M tex="v_j" /> is a pattern across
                variables (a loading vector), <M tex="u_j" /> the corresponding pattern across observations (a standardised score vector), and <M tex="\sigma_j" /> the strength of
                that pattern; for a centred matrix <M tex="\sigma_j^2/(n-1)" /> is the variance of the <M tex="j" />-th principal component.
              </>
            ),
            careful: (
              <>
                Signs are a convention: <M tex="(u_j, v_j) \mapsto (-u_j, -v_j)" /> leaves <M tex="X" /> unchanged (this implementation makes the largest entry of each{' '}
                <M tex="v_j" /> positive). If two singular values coincide the individual vectors are not determined, only their span. The rank decision depends on a tolerance
                (here <M tex={`${texNum(thin.tol, 2)}`} />): entries of <M tex="X" /> perturbed at that level can change the rank.
              </>
            ),
          }}
        />
      </div>
    </Section>
  );
}
