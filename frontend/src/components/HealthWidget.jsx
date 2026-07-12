import { HardDrive } from 'lucide-react';
import Skeleton from './Skeleton';
import { MODEL_METADATA, reliabilityColor } from '../api/modelMetadata';

const MODEL_META = {
  tf_classification: { label: 'Bottleneck Clf', scoreKey: 'failure_probability' },
  tf_anomaly: { label: 'Anomaly AE', scoreKey: 'anomaly_score' },
  clustering: { label: 'HDBSCAN', scoreKey: 'cluster_score' },
  sklearn: { label: 'Random Forest', scoreKey: 'failure_probability' },
};

const scoreToColor = (score) => {
  if (score >= 0.70) return 'var(--status-critical)';
  if (score >= 0.40) return 'var(--status-warning)';
  return 'var(--status-healthy)';
};

const formatBytes = (bytes) => {
  if (!bytes) return '—';
  const gb = bytes / (1024 ** 3);
  if (gb >= 1000) return `${(gb / 1000).toFixed(1)} TB`;
  return `${gb.toFixed(0)} GB`;
};

const HealthWidget = ({ result, loading, driveInfo }) => {
  if (loading) {
    return (
      <div className="card widget-health">
        <div className="card-title">Model Consensus</div>
        <Skeleton width="100%" height="20px" />
        <div className="mini-donuts">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="mini-donut">
              <Skeleton width="50px" height="50px" radius="50%" />
              <Skeleton width="60px" height="12px" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card widget-health">
        <div className="card-title">Model Consensus</div>
        <div className="empty-state">No scan yet — upload a SMART report above</div>
      </div>
    );
  }

  const scores = result.model_scores || {};
  const consensus = result.consensus || {};

  return (
    <div className="card widget-health">
      <div className="card-title">
        Model Consensus
        <span style={{ fontSize: '11px', fontWeight: '400', color: 'var(--text-muted)' }}>
          {consensus.models_predicting_failure}/{consensus.models_total} flag failure
        </span>
      </div>
      {driveInfo && (
        <div className="drive-identity">
          <HardDrive size={12} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle', opacity: 0.6 }} />
          <strong>{driveInfo.model}</strong>
          {' · '}
          {formatBytes(driveInfo.capacity)}
          {driveInfo.serial && <span style={{ opacity: 0.6 }}> · S/N {driveInfo.serial.slice(-8)}</span>}
        </div>
      )}
      <div className="mini-donuts">
        {Object.entries(MODEL_META).map(([key, meta]) => {
          const data = scores[key];
          if (!data) {
            return (
              <div key={key} className="mini-donut">
                <div className="md-circle neutral-circle">
                  <div className="md-inner">—</div>
                </div>
                <span className="subsystem-label">{meta.label}</span>
                <span className="weight">N/A</span>
              </div>
            );
          }
          const score = data[meta.scoreKey] ?? 0;
          const pct = Math.round(score * 100);
          const color = scoreToColor(score);
          return (
            <div key={key} className="mini-donut">
              <div
                className="md-circle"
                style={{ background: `conic-gradient(${color} ${pct}%, #2a2d38 0)` }}
              >
                <div className="md-inner">{pct}%</div>
              </div>
              <span className="subsystem-label">{meta.label}</span>
              <span className="weight">w{data.weight?.toFixed(2) ?? '—'}</span>
              <div
                className="reliability-bar"
                style={{ background: reliabilityColor(MODEL_METADATA[key]?.rocAuc) }}
                title={MODEL_METADATA[key]?.rocAuc ? `ROC-AUC: ${MODEL_METADATA[key].rocAuc.toFixed(3)}` : 'No ROC-AUC'}
              />
            </div>
          );
        })}
      </div>
      <div className="consensus-breakdown">
        {consensus.models_predicting_failure}/{consensus.models_total} models flag failure
        {' · '}
        weights: Clf {MODEL_METADATA.tf_classification.weight.toFixed(2)}
        {' · '}HDBSCAN {MODEL_METADATA.clustering.weight.toFixed(2)}
        {' · '}RF {MODEL_METADATA.sklearn.weight.toFixed(2)}
        {' · '}AE {MODEL_METADATA.tf_anomaly.weight.toFixed(2)}
      </div>
    </div>
  );
};

export default HealthWidget;