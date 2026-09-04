import Plotly from 'plotly.js-cartesian-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import type { PlotParams } from 'react-plotly.js';
import type { Layout, Config } from 'plotly.js';
import { ink } from '../../lib/theme';

const PlotlyChart = createPlotlyComponent(Plotly);

export const baseLayout: Partial<Layout> = {
  paper_bgcolor: ink.surface,
  plot_bgcolor: ink.surface,
  font: { family: 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size: 12, color: ink.secondary },
  margin: { l: 52, r: 16, t: 28, b: 44 },
  hovermode: 'closest',
  hoverlabel: { bgcolor: '#0b0b0b', bordercolor: '#0b0b0b', font: { color: '#fff', size: 12 } },
  xaxis: { gridcolor: ink.grid, zerolinecolor: ink.axis, linecolor: ink.axis, tickfont: { color: ink.muted }, title: { font: { color: ink.secondary, size: 12 } } },
  yaxis: { gridcolor: ink.grid, zerolinecolor: ink.axis, linecolor: ink.axis, tickfont: { color: ink.muted }, title: { font: { color: ink.secondary, size: 12 } } },
  legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 12, color: ink.secondary }, bgcolor: 'rgba(0,0,0,0)' },
  showlegend: false,
};

export const baseConfig: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
  displayModeBar: 'hover',
};

interface Props extends Omit<Partial<PlotParams>, 'layout'> {
  data: PlotParams['data'];
  /** Plotly layout. Typed loosely because @types/plotly.js omits valid keys (barmode, xref: 'x', …). */
  layout?: Partial<Layout> | Record<string, unknown>;
  height?: number;
  title?: string;
}

function mergeLayout(base: Partial<Layout>, extra?: Partial<Layout>): Partial<Layout> {
  if (!extra) return base;
  const out: Partial<Layout> = { ...base, ...extra };
  for (const k of ['xaxis', 'yaxis', 'xaxis2', 'yaxis2', 'legend', 'font', 'margin'] as const) {
    const b = (base as Record<string, unknown>)[k] as Record<string, unknown> | undefined;
    const e = (extra as Record<string, unknown>)[k] as Record<string, unknown> | undefined;
    if (b && e) (out as Record<string, unknown>)[k] = { ...b, ...e };
    else if (k === 'xaxis2' || k === 'yaxis2') {
      if (e) (out as Record<string, unknown>)[k] = { ...(base.xaxis as Record<string, unknown>), ...e };
    }
  }
  return out;
}

/** Thin wrapper around react-plotly with the laboratory's default styling. */
export function Plot({ data, layout, config, height = 320, title, style, ...rest }: Props) {
  const merged = mergeLayout(baseLayout, layout as Partial<Layout> | undefined);
  if (title) merged.title = { text: title, font: { size: 13, color: ink.secondary }, x: 0, xanchor: 'left' };
  return (
    <PlotlyChart
      data={data}
      layout={{ ...merged, autosize: true }}
      config={{ ...baseConfig, ...config }}
      useResizeHandler
      style={{ width: '100%', height, ...style }}
      {...rest}
    />
  );
}
