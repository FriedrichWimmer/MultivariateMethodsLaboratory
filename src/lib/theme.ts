/** Design tokens shared by CSS, SVG widgets and Plotly layouts. */
export const ink = {
  primary: '#0b0b0b',
  secondary: '#52514e',
  muted: '#898781',
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  surface: '#fcfcfb',
  plane: '#f9f9f7',
  border: 'rgba(11,11,11,0.10)',
};

/** Validated categorical palette (fixed order, never cycled). */
export const categorical = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];

/** Marker shapes give a second identity channel for classes beyond the first three colours. */
export const markerSymbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star', 'hexagon'];

export const methodColor = {
  SVD: '#4a3aa7',
  PCA: '#2a78d6',
  MDS: '#1baf7a',
  LDA: '#eb6834',
} as const;

export const accent = '#2a78d6';
export const accent2 = '#eb6834';
export const neutralMark = '#52514e';

/** Single-hue sequential ramp (blue, light → dark). */
export const sequential = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];

/** Diverging ramp blue ↔ neutral ↔ red, equal steps per arm. */
export const diverging = ['#184f95', '#2a78d6', '#6da7ec', '#b7d3f6', '#f0efec', '#f6c1c0', '#ee8a89', '#e34948', '#a52a2a'];

export function classColor(k: number): string {
  return categorical[k % categorical.length];
}

export function classSymbol(k: number): string {
  return markerSymbols[k % markerSymbols.length];
}

/** Interpolate through a list of hex colours for t ∈ [0,1]. */
export function rampColor(ramp: string[], t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  if (i >= ramp.length - 1) return ramp[ramp.length - 1];
  return mixHex(ramp[i], ramp[i + 1], f);
}

export function mixHex(a: string, b: string, t: number): string {
  const pa = hexToRgb(a);
  const pb = hexToRgb(b);
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function hexToRgb(h: string): [number, number, number] {
  const s = h.replace('#', '');
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Plotly colourscale arrays. */
export const plotlySequential = sequential.map((c, i) => [i / (sequential.length - 1), c] as [number, string]);
export const plotlyDiverging = diverging.map((c, i) => [i / (diverging.length - 1), c] as [number, string]);
