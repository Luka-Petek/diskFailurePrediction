import { ShieldCheck, AlertCircle, XCircle } from 'lucide-react';
import Skeleton from './Skeleton';

const verdictToPill = {
  HEALTHY: { cls: 'healthy', label: 'HEALTHY', Icon: ShieldCheck },
  AT_RISK: { cls: 'warning', label: 'AT RISK', Icon: AlertCircle },
  FAILURE: { cls: 'critical', label: 'FAILURE', Icon: XCircle },
};

const scoreToStatusColor = (score) => {
  if (score >= 0.70) return 'var(--status-critical)';
  if (score >= 0.40) return 'var(--status-warning)';
  return 'var(--status-healthy)';
};

const StatusWidget = ({ result, loading, driveInfo }) => {
  if (loading) {
    return (
      <div className="card widget-status">
        <div className="card-title">Verdict</div>
        <div className="donut-container">
          <Skeleton width="120px" height="120px" radius="50%" />
          <div className="status-legend" style={{ flex: 1 }}>
            <Skeleton width="80%" height="16px" />
            <Skeleton width="60%" height="16px" />
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card widget-status">
        <div className="card-title">Verdict</div>
        <div className="donut-container">
          <div className="donut" style={{ background: 'conic-gradient(var(--status-neutral) 0%, #2a2d38 0)' }}>
            <div className="donut-text">
              <h2>—</h2>
              <span>No data</span>
            </div>
          </div>
          <div className="status-legend">
            <div className="legend-item">
              <span className="status-pill neutral">PENDING</span>
            </div>
            <div className="legend-item">Upload a SMART report to begin</div>
          </div>
        </div>
      </div>
    );
  }

  const score = result.disk_health_score;
  const pct = Math.round(score * 100);
  const verdict = result.verdict || 'HEALTHY';
  const pill = verdictToPill[verdict] || verdictToPill.HEALTHY;
  const { Icon } = pill;
  const color = scoreToStatusColor(score);

  return (
    <div className="card widget-status">
      <div className="card-title">
        Verdict
        <span className={`status-pill ${pill.cls}`}>
          <Icon size={11} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
          {pill.label}
        </span>
      </div>
      <div className="donut-container">
        <div
          className="donut"
          style={{ background: `conic-gradient(${color} ${pct}%, #2a2d38 0)` }}
          aria-live="polite"
        >
          <div className="donut-text">
            <h2>{pct}%</h2>
            <span>Risk</span>
          </div>
        </div>
        <div className="status-legend">
          <div className="legend-item">
            Confidence
            <b style={{ textTransform: 'capitalize' }}>{result.confidence}</b>
          </div>
          {driveInfo && (
            <div className="legend-item">
              Drive
              <b>{driveInfo.model}</b>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatusWidget;