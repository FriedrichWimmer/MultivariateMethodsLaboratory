# Multivariate Methods Laboratory — component & library API (for tab authors)

The app is Vite + React 18 + TypeScript (strict). Plotting: Plotly (via `Plot` wrapper) for
standard charts, `ScatterSVG` for interactive geometry. Math: KaTeX via `M`/`MBlock`.
All numerical work is done in `src/lib/*` — **never hard-code results; always compute**.

Run `npx tsc --noEmit -p tsconfig.json` to typecheck. Run `npx vitest run` for the numerical tests.

## Notation (use consistently in all text and TeX)
- `n` observations, `p` variables, `K` classes
- `X` data matrix (n×p), `X_c` centred matrix, `\bar X` column means, `S = \frac{1}{n-1}X_c^T X_c` covariance
- `X_c = U\Sigma V^T` (thin SVD), `\sigma_j` singular values, `\lambda_j = \sigma_j^2/(n-1)` eigenvalues of S
- `V, \Lambda` eigenvectors/eigenvalues (of S, or of B in MDS — say which)
- `Z = X_c V` scores; `\hat X_k = Z_k V_k^T`; `X_k = U_k\Sigma_k V_k^T`
- `D` distance matrix, `D^{(2)}` squared distances, `J = I - \frac1n 11^T`, `B = -\frac12 J D^{(2)} J`
- `S_W`, `S_B`, `S_T = S_W + S_B = (n-1)S` scatter matrices; `w` a projection direction; `J(w)` Fisher criterion
- Do **not** reuse a symbol for a different object without saying so.

## Global state — `src/state/store.tsx`
```ts
const { dataset, prep, analysis, setKind, setParams, setPrep, navigate, kind, params } = useStore();
const a = useAnalysis();   // same as useStore().analysis, memoised
// Analysis = { dataset, X, y?, n, p, K, prep, pca: PCAResult, svd: SVDResult, mds: MDSResult, D, lda: LDAResult | null }
// prep = { scaling: 'none'|'center'|'standardize', metric: Metric, k: number }
// navigate(tab: TabId, anchor?: string)  — TabIds: 'concept'|'data'|'svd'|'pca'|'mds'|'lda'|'compare'|'unified'|'diagnostics'|'wrong'|'quiz'|'experiment'|'takeaway'
```
`analysis.pca.Xc` is the analysed matrix (centred/standardised per `prep.scaling`); `analysis.D` is the
distance matrix of `Xc` under `prep.metric`; `analysis.mds` uses `prep.k`; `analysis.lda` is computed on `Xc`
(null when the dataset has no labels). Labels: `dataset.y` (0..K-1) and `dataset.classNames`.

A tab may ALSO build its own local dataset for a specific lesson:
```ts
import { generateDataset, defaultParams } from '../../lib/datasets';
const ds = useMemo(() => generateDataset('pcaVsLda', { ...defaultParams, n: 200, separation: 3, seed }), [seed]);
```
Dataset kinds: 'gaussian2d' | 'correlated' | 'clusters' | 'iris' | 'separated' | 'overlapping' | 'scales' | 'outliers' |
'pcaVsLda' | 'pGreaterN' | 'manifold' | 'unequalCov' | 'imbalanced' | 'collinear'. Params: n, p, K, correlation,
variance, noise, separation, classProportions, seed, scaleFactor, outlierCount, outlierMagnitude.
Use `<DatasetControls compact />` and `<PrepControls />` (from `common/DatasetControls`) to expose global controls,
`<ActiveDatasetCard />` for a summary card.

## Numerical library — `src/lib/`
`linalg.ts`: `Matrix = number[][]` (row-major). `zeros, identity, transpose, matmul, matvec, add, sub, scale, outer,
dot, norm, normalize, frobenius, trace, quadForm, colMeans, colStds, centerColumns(X)->{Xc,means},
standardizeColumns(X)->{Xs,means,stds}, covariance(X), correlation(X), gram(A)=AᵀA,
symmetricEigen(A)->{values (desc), vectors (COLUMNS)}, symmetricEigenAuto, svd(X)->{U,s,V,rank,tol}
(thin, s descending, U n×r, V p×r), svdFull(X)->{U n×n, s, V p×p, Sigma n×p}, truncatedReconstruction(res,k),
conditionNumber, cholesky(A)|null, inverse(A)|null, determinant, symmetricPower(A,power), rotation2(theta),
column(A,j), firstColumns(A,k), fromColumns(cols), fmt(x,digits), fmtMatrix, angleOf`.

`pca.ts`: `pca(X, scaling)` -> `{ n,p,scaling,means,stds,Xc,S,eigen,svd,eigenvalues,singularValues,V (p×r loadings),
scores (n×r),explained,cumulative,totalVariance,rank }`; `pcaReconstruct(res,k)->{analysed, original}`;
`reconstructionErrors(res)` (k=0..r); `projectionVariance(S,w)`; `projectOnDirection(Xc,w)`; `sampleVariance(z)`.

`mds.ts`: `distanceMatrix(X, metric)`, `metricLabels`, `pointDistance`, `doubleCenter(A)`, `classicalMDS(D,k)` ->
`{ n,D,D2,B,eigen,eigenvalues (all, desc, may be negative),positive,negative,negativeMass,k (actual),coords (n×k),
Dhat,stress1,explained,cumulative,strain }`, `procrustesAlign(A,B)->{aligned,residual,relative,Q,reflection}`.

`lda.ts`: `lda(X,y,{regularization?})` -> `{ n,p,K,classes,classSizes,classMeans (K×p),grandMean,SW,SB,ST,
eigenvalues,W (p×m unit-norm directions),scores (n×m),maxDims,swRank,swSingular,swCondition,regularization,
swEigenvalues,sbEigenvalues,explained }`; `scatterMatrices(X,y)`; `fisherCriterion(SB,SW,w)->{between,within,J}`;
`ldaClassify(res,X)->{predictions, accuracy(y)}`; `classStats(X,y)`.

`svdlab.ts`: `svd2d(A)->{U,V,s,Sigma,thetaU,thetaV,reflectionU,reflectionV,rank,det,conditionNumber}`,
`transformPoints(M, pts)`, `unitCircle(m)`, `lowRankSummaries(X,res)` (k=0..r: Xk,error,relativeError,eckartYoung,energy),
`svdChecks(X,res)->{reconError,uOrthoError,vOrthoError,ordered}`.

`random.ts`: `makeRNG(seed)` -> `{uniform,normal,int,categorical,shuffle}`.
`theme.ts`: `categorical[]`, `classColor(k)`, `classSymbol(k)`, `methodColor.{SVD,PCA,MDS,LDA}`, `accent`, `accent2`,
`neutralMark`, `sequential[]`, `diverging[]`, `rampColor(ramp,t)`, `withAlpha(hex,a)`, `plotlySequential`, `plotlyDiverging`, `ink`.

## UI components — `src/components/common/`
- `Math.tsx`: `<M tex="\lambda_j" />` inline, `<MBlock tex="S = V\Lambda V^T" />` display. Helpers `texMatrix(A,digits)`,
  `texVector(v,digits)`, `texDiag(v,digits)` return TeX strings for numeric matrices (use inside tex).
- `Plot.tsx`: `<Plot data={[...]} layout={{...}} height={320} title="..." />` — Plotly with lab styling merged in
  (axes recessive, legend horizontal at top, no logo). Only cartesian traces are available (scatter, bar, heatmap,
  histogram, box, contour). Set `layout.showlegend: true` when ≥2 series. Use `plotlySequential` for magnitude heatmaps
  (`zmin: 0`) and `plotlyDiverging` for signed matrices (`zmid: 0`, symmetric zmin/zmax).
- `MatrixView.tsx`: `<MatrixView M={A} title="S" rowLabels colLabels digits={2} heat="diverging"|"sequential"|"none"
  highlightCols={[0]} highlightRows dimOthers caption maxRows maxCols compact />`, `<MatrixEquation items={[<MatrixView/>, '=', <MatrixView/>, '·', ...]} />`.
- `Controls.tsx`: `<Slider label value min max step onChange format />`, `<Select label value options={[{value,label}]} onChange />`,
  `<Segmented label value options={[{value,label,hint}]} onChange compact />`, `<Toggle label checked onChange />`,
  `<NumberField />`, `<Button primary small onClick>`.
- `Panels.tsx`: `<Section id title subtitle right>` (a card with anchor id — one per numbered lesson), `<Card title plane>`,
  `<Callout kind="info|warning|danger|theorem|definition|good" title>`, `<Interpretation items={{seeing, why, math, stats, careful}} />`
  (REQUIRED next to every visualisation; contents must be computed from the current numbers, not static),
  `<Derivation steps={[{title, body, note}]} title initiallyRevealed />` (step-by-step reveal), `<Accordion items={[{title, body}]} />`,
  `<StatTile label value note />` inside `<div className="stats">`, `<Badge method="PCA" />`, `<ClassLegend classNames />`, `<MarkerShape k r />`.
- `DataTable.tsx`: `<DataTable X variableNames y classNames maxRows digits highlight onRowClick />`.
- `ScatterSVG.tsx`: interactive SVG scatter (equal aspect by default):
  ```tsx
  <ScatterSVG points={n×2} labels={y} classNames={names} width={480} height={400} xLabel yLabel title caption
     vectors={[{x,y,from?,color,label,dashed}]}           // arrows from origin (or `from`)
     segments={[{from:[x,y], to:[x,y], color, dashed, opacity}]}   // e.g. projection foot lines
     extraPoints={[{x,y,color,r,label,shape:'circle'|'square'|'diamond'|'cross'|'ring'|'class', classIndex, labelPosition}]}
     ellipses={[{cx,cy,rx,ry,angle,color,fill,dashed}]}   // covariance ellipses (angle in radians)
     lines={[{angle, through?, color, dashed, label}]}     // infinite lines
     direction={{ angle, onChange: setAngle, color, label:'w', radius?, through?, axial:true }}  // DRAGGABLE direction handle; clicking in the plot also re-orients it
     highlight={[i,...]} selected={[i,...]} onPointClick={(i)=>...} hoverInfo={(i)=>string} include={[[x,y]]}
     render={(sx, sy) => <g>...</g>}                      // custom SVG in data coords
  />
  ```
  Helper for a covariance ellipse: eigen-decompose the 2×2 covariance `S`; `rx = c·sqrt(λ1)`, `ry = c·sqrt(λ2)`, `angle = atan2(v1[1], v1[0])`.

## Layout classes (CSS in `src/styles/global.css`)
`.grid.c2/.c3/.c4`, `.grid.side` (300px controls | plot), `.grid.side-r` (plot | 320px), `.grid.side-wide`,
`.stack`, `.row`, `.row.between`, `.stats`, `.muted`, `.small`, `.secondary`, `.mono`, `.divider`, `.prose`,
`.summary-table` (comparison tables), `.pipeline > .stage / .arrow` (D → D² → B → … chains), `.plot-caption`, `.kbd-hint`.
Page header pattern at the top of each tab:
```tsx
<div className="topbar"><div className="title"><h2>PCA laboratory</h2><div className="lede">one-sentence framing</div></div></div>
```

## Design rules (non-negotiable)
- Graduate laboratory look: neutral, thin marks, hairline solid grid (already in `Plot`/`ScatterSVG`), no decoration, no emoji.
- Colours: class identity = `classColor(k)` (fixed order, never re-assigned when filtering). Methods = `methodColor`. Neutral marks `#52514e`.
  Signed matrices → diverging heat; magnitudes → sequential. Never rainbow. Text is never coloured with a series colour.
- Every chart with ≥2 series needs a legend (`ClassLegend` or Plotly legend). Tooltips are provided by `ScatterSVG`/Plotly.
- Every visualisation gets an `<Interpretation>` whose sentences use the live computed numbers (e.g. "PC1 explains 72.4%").
- Prefer `useMemo` for all computations; keep n ≤ 400 for MDS-type n×n work.
- Each lesson is a `<Section id="…">` with stable ids so the concept map can `navigate(tab, id)`.
- Mathematical rigour: state theorems precisely (Eckart–Young, Rayleigh quotient, Torgerson, generalised eigenproblem),
  distinguish Fisher's discriminant from Gaussian LDA classification, and mention sign/rotation indeterminacy where relevant.
- Prose: short paragraphs, precise, no marketing tone. Use `<M>` for every symbol in prose (never raw Unicode math for formulas).
