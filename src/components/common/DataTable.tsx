import { fmt } from '../../lib/linalg';

interface Props {
  X: number[][];
  variableNames: string[];
  y?: number[];
  classNames?: string[];
  maxRows?: number;
  digits?: number;
  highlight?: number[];
  rowLabel?: string;
  onRowClick?: (i: number) => void;
}

export function DataTable({ X, variableNames, y, classNames, maxRows = 200, digits = 2, highlight, rowLabel = 'i', onRowClick }: Props) {
  const rows = Math.min(X.length, maxRows);
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>{rowLabel}</th>
            {variableNames.map((v) => (
              <th key={v}>{v}</th>
            ))}
            {y && <th>class</th>}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }, (_, i) => (
            <tr key={i} className={highlight?.includes(i) ? 'hl' : ''} onClick={onRowClick ? () => onRowClick(i) : undefined} style={onRowClick ? { cursor: 'pointer' } : undefined}>
              <td>{i + 1}</td>
              {X[i].map((x, j) => (
                <td key={j}>{fmt(x, digits)}</td>
              ))}
              {y && <td style={{ textAlign: 'left' }}>{classNames ? classNames[y[i]] : y[i]}</td>}
            </tr>
          ))}
          {X.length > rows && (
            <tr>
              <td colSpan={variableNames.length + 1 + (y ? 1 : 0)} className="muted">
                … {X.length - rows} more rows
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
