import { ShieldCheck, AlertCircle, XCircle } from 'lucide-react';
import Skeleton from './Skeleton';

const verdictToPill = {
  HEALTHY: { cls: 'healthy', label: 'HEALTHY', Icon: ShieldCheck },
  WARNING: { cls: 'warning', label: 'WARNING', Icon: AlertCircle },
  CRITICAL: { cls: 'critical', label: 'CRITICAL', Icon: XCircle },
};

const scoreToStatusColor = (score) => {
  if (score >= 75) return 'var(--status-critical)';
  if (score >= 40) return 'var(--status-warning)';
  return 'var(--status-healthy)';
};

//final % circle:
const RING_SIZE = 160;
const RING_STROKE = 12;
const RING_RADIUS = (RING_SIZE - RING_STROKE) / 2;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

const VerdictRing = ({ pct, color }) => {
  const offset = RING_CIRCUMFERENCE - (pct / 100) * RING_CIRCUMFERENCE;
  return (
    <svg width={RING_SIZE} height={RING_SIZE} className="verdict-ring">
      <defs>
        <filter id="ringGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        fill="none"
        stroke="#2a2d38"
        strokeWidth={RING_STROKE}
      />
      <circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        fill="none"
        stroke={color}
        strokeWidth={RING_STROKE}
        strokeLinecap="round"
        strokeDasharray={RING_CIRCUMFERENCE}
        strokeDashoffset={offset}
        transform={`rotate(-90 ${RING_SIZE / 2} ${RING_SIZE / 2})`}
        filter="url(#ringGlow)"
        className="verdict-ring-progress"
      />
    </svg>
  );
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
          <div className="verdict-ring-wrap">
            <VerdictRing pct={0} color="var(--status-neutral)" />
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
  const pct = Math.round(score);
  const verdict = result.verdict || 'HEALTHY';
  const pill = verdictToPill[verdict] || verdictToPill.HEALTHY;
  const { Icon } = pill;
  const color = scoreToStatusColor(score);

  return (
    <div className="card widget-status">
      <div className="card-title">
        Verdict
        <span className={`status-pill ${pill.cls}`}>
          <Icon size={13} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
          {pill.label}
        </span>
      </div>
      <div className="donut-container">
        <div className="verdict-ring-wrap">
          <VerdictRing pct={pct} color={color} />
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