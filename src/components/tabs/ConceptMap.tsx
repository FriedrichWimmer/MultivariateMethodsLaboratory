import { useState } from 'react';
import { useStore, type TabId } from '../../state/store';
import { Section, Card, Callout, Badge } from '../common/Panels';
import { M, MBlock } from '../common/Math';
import { methodColor, withAlpha } from '../../lib/theme';

type Method = 'SVD' | 'PCA' | 'MDS' | 'LDA';

interface Node {
  id: string;
  x: number;
  y: number;
  w?: number;
  label: string;
  sub?: string;
  tab: TabId;
  anchor?: string;
  method?: Method;
}

interface Edge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
  bidirectional?: boolean;
  /** label offset perpendicular to the edge (px) */
  offset?: number;
}

const NODE_W = 150;
const NODE_H = 46;

const nodes: Node[] = [
  { id: 'X', x: 620, y: 36, label: 'Data matrix X', sub: 'n observations × p variables', tab: 'data', anchor: 'data-matrix' },
  { id: 'y', x: 185, y: 36, label: 'Class labels y', sub: 'K groups — supervised only', tab: 'lda', anchor: 'lda-intro' },
  { id: 'Xc', x: 620, y: 132, label: 'Centring & scaling', sub: 'X_c = X − 1x̄ᵀ, optionally X_s', tab: 'data', anchor: 'data-preprocess' },
  { id: 'S', x: 620, y: 240, label: 'Covariance matrix S', sub: 'S = X_cᵀ X_c / (n−1)', tab: 'pca', anchor: 'pca-from-svd' },
  { id: 'SVD', x: 448, y: 240, label: 'SVD', sub: 'X_c = U Σ Vᵀ', tab: 'svd', anchor: 'svd-decomposition', method: 'SVD' },
  { id: 'EIG', x: 620, y: 346, label: 'Eigenvalues / eigenvectors', sub: 'S = V Λ Vᵀ,  λ_j = σ_j² / (n−1)', tab: 'pca', anchor: 'pca-from-svd', w: 178 },
  { id: 'PCA', x: 620, y: 452, label: 'PCA', sub: 'max wᵀSw,  scores Z = X_c V', tab: 'pca', anchor: 'pca-geometry', method: 'PCA' },
  { id: 'LR', x: 445, y: 346, label: 'Low-rank approximation', sub: 'X_k = U_k Σ_k V_kᵀ', tab: 'svd', anchor: 'svd-lowrank', w: 166 },
  { id: 'D', x: 870, y: 240, label: 'Pairwise distances D', sub: 'd_ij = ‖x_i − x_j‖', tab: 'mds', anchor: 'mds-intro' },
  { id: 'MDS', x: 870, y: 452, label: 'Classical MDS', sub: 'B = −½ J D⁽²⁾ J = V Λ Vᵀ', tab: 'mds', anchor: 'mds-chain', method: 'MDS' },
  { id: 'SW', x: 95, y: 240, label: 'Within-class scatter S_W', sub: 'spread around the class means', tab: 'lda', anchor: 'lda-intro', w: 170 },
  { id: 'SB', x: 278, y: 240, label: 'Between-class scatter S_B', sub: 'spread of the class means', tab: 'lda', anchor: 'lda-intro', w: 170 },
  { id: 'LDA', x: 185, y: 452, label: 'LDA', sub: 'S_B w = λ S_W w', tab: 'lda', anchor: 'lda-geometry', method: 'LDA' },
];

const edges: Edge[] = [
  { from: 'X', to: 'Xc', label: 'subtract means, optionally divide by sd' },
  { from: 'Xc', to: 'S', label: 'S = X_cᵀX_c/(n−1)' },
  { from: 'Xc', to: 'SVD', label: 'factorise X_c directly', offset: -12 },
  { from: 'S', to: 'EIG', label: 'diagonalise' },
  { from: 'SVD', to: 'EIG', label: 'same V;  λ_j = σ_j²/(n−1)', bidirectional: true, offset: 12 },
  { from: 'EIG', to: 'PCA', label: 'principal directions v_j' },
  { from: 'SVD', to: 'LR', label: 'keep k terms' },
  { from: 'LR', to: 'PCA', label: 'X̂_k = Z_k V_kᵀ = X_k', bidirectional: true, offset: 12 },
  { from: 'Xc', to: 'D', label: 'all n(n−1)/2 distances', offset: -12 },
  { from: 'D', to: 'MDS', label: 'square, double-centre, diagonalise' },
  { from: 'MDS', to: 'PCA', label: 'Euclidean D ⇒ same coordinates', bidirectional: true, offset: 36 },
  { from: 'Xc', to: 'SW', label: 'X_c and y', offset: 12 },
  { from: 'Xc', to: 'SB', label: '' },
  { from: 'y', to: 'SW', label: 'group by class', offset: -10 },
  { from: 'y', to: 'SB', label: '' },
  { from: 'SW', to: 'LDA', label: 'generalised eigenproblem', offset: -12 },
  { from: 'SB', to: 'LDA', label: '' },
  { from: 'PCA', to: 'LDA', label: 'same X, different objective: no labels vs labels', dashed: true, bidirectional: true, offset: 36 },
];

function nodeBox(n: Node) {
  const w = n.w ?? NODE_W;
  return { x: n.x - w / 2, y: n.y - NODE_H / 2, w, h: NODE_H };
}

/** Point on the boundary of node `a` in the direction of node `b`. */
function boundaryPoint(a: Node, b: Node): [number, number] {
  const { x, y, w, h } = nodeBox(a);
  const cx = a.x;
  const cy = a.y;
  const dx = b.x - cx;
  const dy = b.y - cy;
  if (dx === 0 && dy === 0) return [cx, cy];
  const sx = dx !== 0 ? (w / 2) / Math.abs(dx) : Infinity;
  const sy = dy !== 0 ? (h / 2) / Math.abs(dy) : Infinity;
  const s = Math.min(sx, sy);
  void x;
  void y;
  return [cx + dx * s, cy + dy * s];
}

export default function ConceptMap() {
  const { navigate } = useStore();
  const [hover, setHover] = useState<string | null>(null);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

  return (
    <>
      <div className="topbar">
        <div className="title">
          <h2>Concept map</h2>
          <div className="lede">
            See the mathematics, manipulate the data, observe the geometry, and connect the methods. Every node below is a live laboratory — click it to jump there. Arrows show how one object is computed from another; double-headed arrows mark exact mathematical
            identities.
          </div>
        </div>
      </div>

      <Section id="concept-map" title="How the four methods are connected" subtitle="Everything starts from the data matrix X. The four methods differ in which derived object they decompose and which notion of information they preserve.">
        <svg viewBox="0 0 980 500" className="svgplot" style={{ maxHeight: 560 }} role="img" aria-label="Concept map of SVD, PCA, MDS and LDA">
          <defs>
            <marker id="cm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#898781" />
            </marker>
          </defs>
          {edges.map((e, i) => {
            const a = byId[e.from];
            const b = byId[e.to];
            const [x1, y1] = boundaryPoint(a, b);
            const [x2, y2] = boundaryPoint(b, a);
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;
            const len = Math.hypot(x2 - x1, y2 - y1) || 1;
            const nx = -(y2 - y1) / len;
            const ny = (x2 - x1) / len;
            const off = e.offset ?? -10;
            const active = hover === e.from || hover === e.to;
            return (
              <g key={i}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={active ? '#0b0b0b' : '#b5b3ab'} strokeWidth={active ? 1.8 : 1.2} strokeDasharray={e.dashed ? '6 4' : undefined} markerEnd="url(#cm-arrow)" markerStart={e.bidirectional ? 'url(#cm-arrow)' : undefined} />
                {e.label && (
                  <text x={mx + nx * off} y={my + ny * off} fontSize={10.5} fill={active ? '#0b0b0b' : '#52514e'} textAnchor="middle" stroke="#fcfcfb" strokeWidth={3} paintOrder="stroke">
                    {e.label}
                  </text>
                )}
              </g>
            );
          })}
          {nodes.map((n) => {
            const { x, y, w, h } = nodeBox(n);
            const color = n.method ? methodColor[n.method] : '#52514e';
            const fill = n.method ? withAlpha(color, hover === n.id ? 0.2 : 0.12) : hover === n.id ? '#ebe9e2' : '#f2f1ec';
            return (
              <g key={n.id} className="concept-node" onClick={() => navigate(n.tab, n.anchor)} onPointerEnter={() => setHover(n.id)} onPointerLeave={() => setHover(null)} role="link" tabIndex={0} onKeyDown={(ev) => ev.key === 'Enter' && navigate(n.tab, n.anchor)}>
                <rect x={x} y={y} width={w} height={h} rx={7} fill={fill} stroke={color} strokeWidth={n.method ? 1.8 : 1} />
                <text x={n.x} y={n.y - 4} fontSize={12.5} fontWeight={600} fill="#0b0b0b" textAnchor="middle">
                  {n.label}
                </text>
                {n.sub && (
                  <text x={n.x} y={n.y + 12} fontSize={9.8} fill="#52514e" textAnchor="middle" fontFamily="ui-monospace, Menlo, monospace">
                    {n.sub}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        <div className="row" style={{ marginTop: 10, justifyContent: 'space-between' }}>
          <div className="legend">
            <span className="item">
              <Badge method="SVD" /> matrix factorisation
            </span>
            <span className="item">
              <Badge method="PCA" /> variance-maximising projection
            </span>
            <span className="item">
              <Badge method="MDS" /> distance-preserving configuration
            </span>
            <span className="item">
              <Badge method="LDA" /> class-separating projection
            </span>
          </div>
          <span className="kbd-hint">Click any node to open the corresponding laboratory section.</span>
        </div>
        <div className="grid c2" style={{ marginTop: 14 }}>
          <Callout kind="definition" title="The chain SVD → PCA → low-rank approximation">
            The thin SVD <M tex="X_c = U\Sigma V^T" /> already contains PCA: the right singular vectors <M tex="V" /> are the principal directions, the scores are <M tex="Z = X_cV = U\Sigma" />, and <M tex="\lambda_j = \sigma_j^2/(n-1)" />. Truncating the sum after <M tex="k" /> terms gives both the best rank-<M tex="k" /> approximation of <M tex="X_c" /> and the PCA reconstruction <M tex="\hat X_k = Z_kV_k^T" />.
          </Callout>
          <Callout kind="definition" title="The two departures: distances and labels">
            Classical MDS forgets the variables and keeps only the distance matrix <M tex="D" />; for Euclidean distances it recovers exactly the PCA scores (up to rotation, reflection and translation). LDA keeps the variables but adds labels <M tex="y" />, replacing the single covariance <M tex="S" /> by the pair <M tex="(S_W, S_B)" /> and the ordinary eigenproblem by a generalised one.
          </Callout>
        </div>
      </Section>

      <Section id="concept-problems" title="What problem is each method solving?" subtitle="Four different objectives — not four variations of one algorithm.">
        <table className="summary-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Core objective</th>
              <th>Input</th>
              <th>Output</th>
              <th>Supervised?</th>
              <th>Open the laboratory</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Badge method="SVD" />
              </td>
              <td>
                Matrix decomposition and optimal low-rank approximation: <M tex="\min_{\mathrm{rank}(B)\le k}\|X-B\|_F" />
              </td>
              <td>Any matrix (n × p)</td>
              <td>
                <M tex="U,\ \Sigma,\ V^T" />
              </td>
              <td>No</td>
              <td>
                <button className="btn small" onClick={() => navigate('svd')}>
                  SVD laboratory
                </button>
              </td>
            </tr>
            <tr>
              <td>
                <Badge method="PCA" />
              </td>
              <td>
                Maximum variance / minimum reconstruction error: <M tex="\max_{\|w\|=1} w^TSw" />
              </td>
              <td>Observations × variables (centred, possibly standardised)</td>
              <td>Principal components (directions, scores, eigenvalues)</td>
              <td>No</td>
              <td>
                <button className="btn small" onClick={() => navigate('pca')}>
                  PCA laboratory
                </button>
              </td>
            </tr>
            <tr>
              <td>
                <Badge method="MDS" />
              </td>
              <td>
                Preserve pairwise distances: <M tex="\min_X \|B - XX^T\|_F" /> (strain)
              </td>
              <td>Distance / dissimilarity matrix (n × n)</td>
              <td>Coordinates in k dimensions</td>
              <td>No</td>
              <td>
                <button className="btn small" onClick={() => navigate('mds')}>
                  MDS laboratory
                </button>
              </td>
            </tr>
            <tr>
              <td>
                <Badge method="LDA" />
              </td>
              <td>
                Maximise class separation relative to within-class spread: <M tex="\max_w \frac{w^TS_Bw}{w^TS_Ww}" />
              </td>
              <td>Labelled observations (X, y)</td>
              <td>Discriminant directions (at most K − 1)</td>
              <td>Yes</td>
              <td>
                <button className="btn small" onClick={() => navigate('lda')}>
                  LDA laboratory
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section id="concept-layers" title="Four layers for every concept" subtitle="Each laboratory presents its method through the same four complementary lenses.">
        <div className="grid c4">
          <Card title="Intuition">What is the method trying to accomplish? Which question about the data does it answer?</Card>
          <Card title="Mathematics">Which quantity is optimised or decomposed, and why does the optimum take the form of an eigen- or singular-value problem?</Card>
          <Card title="Geometry">What happens to points, vectors, distances, projections and subspaces? Every laboratory lets you drag a direction and watch the numbers respond.</Card>
          <Card title="Statistics">Which assumptions are needed, which quantities are estimated, and how should the low-dimensional result be interpreted (and not over-interpreted)?</Card>
        </div>
        <div className="divider" />
        <div className="grid c2">
          <div className="prose">
            <h3>Suggested path through the laboratory</h3>
            <ol>
              <li>
                <b>Data laboratory</b> — pick a dataset, look at the raw matrix and decide how to centre and scale it.
              </li>
              <li>
                <b>SVD</b> — see a matrix as rotation · scaling · rotation and understand truncation.
              </li>
              <li>
                <b>PCA</b> — connect eigenvalues of <M tex="S" /> to singular values of <M tex="X_c" />, then drag a direction to feel the variance criterion.
              </li>
              <li>
                <b>MDS</b> — start from distances only and rediscover the PCA configuration.
              </li>
              <li>
                <b>LDA</b> — add labels and watch the optimal direction move away from the high-variance axis.
              </li>
              <li>
                <b>Comparisons, diagnostics, failure modes</b> — one dataset, four answers; then break the methods on purpose.
              </li>
              <li>
                <b>Assessment and experiment</b> — predict before you reveal; design your own experiment.
              </li>
            </ol>
          </div>
          <div>
            <h3>Notation used throughout</h3>
            <MBlock tex={String.raw`\begin{aligned}
&n\ \text{observations},\quad p\ \text{variables},\quad K\ \text{classes}\\
&X\ (n\times p),\quad X_c = X - \mathbf 1\bar x^T,\quad S = \tfrac{1}{n-1}X_c^TX_c\\
&X_c = U\Sigma V^T,\qquad S = V\Lambda V^T,\qquad \lambda_j = \sigma_j^2/(n-1)\\
&D\ (n\times n),\quad J = I - \tfrac1n\mathbf 1\mathbf 1^T,\quad B = -\tfrac12 J D^{(2)} J\\
&S_W,\ S_B,\quad S_W + S_B = S_T = (n-1)S
\end{aligned}`} />
            <div className="small muted">Symbols are never silently reused: <M tex="V,\Lambda" /> denote eigenvectors and eigenvalues of whichever symmetric matrix is being decomposed, and the text always says which.</div>
          </div>
        </div>
      </Section>
    </>
  );
}
