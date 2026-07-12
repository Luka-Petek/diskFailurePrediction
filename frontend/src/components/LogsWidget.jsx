import { Clock } from 'lucide-react';
import Skeleton from './Skeleton';

const scoreToColor = (score) => {
  if (score >= 75) return 'var(--status-critical)';
  if (score >= 40) return 'var(--status-warning)';
  return 'var(--status-healthy)';
};

const formatTime = (iso) => {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const LogsWidget = ({ history, loading }) => {
  if (loading && history.length === 0) {
    return (
      <div className="card widget-logs">
        <div className="card-title">Recent Scans</div>
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} width="100%" height="32px" />
        ))}
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="card widget-logs">
        <div className="card-title">Recent Scans</div>
        <div className="empty-state">No scans yet this session</div>
      </div>
    );
  }

  return (
    <div className="card widget-logs">
      <div className="card-title">Recent Scans</div>
      {history.map((scan) => {
        const score = scan.result?.disk_health_score ?? 0;
        const pct = Math.round(score);
        const verdict = scan.result?.verdict || '—';
        return (
          <div key={scan.id} className="scan-item">
            <div>
              <strong>{scan.filename}</strong>
              <br />
              <span style={{ fontSize: '13px', opacity: 0.7 }}>
                <Clock size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '3px' }} />
                {formatTime(scan.timestamp)} · {verdict}
              </span>
            </div>
            <div className="scan-risk" style={{ color: scoreToColor(score) }}>
              {pct}%
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default LogsWidget;