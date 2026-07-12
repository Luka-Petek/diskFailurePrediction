import { AlertTriangle } from 'lucide-react';
import Skeleton from './Skeleton';
import {
  FEATURE_IMPORTANCE,
  featureLabel,
  featureToSmartId,
  CRITICAL_FEATURES,
} from '../api/modelMetadata';

const TOP_N = 10;

// Look up a SMART attribute value from the parsed smartctl JSON
function getDriveValue(featureName, smartData) {
  if (!smartData) return null;

  // Engineered features — compute from smartctl JSON
  if (featureName === 'any_critical_error') {
    const attrs = smartData?.ata_smart_attributes?.table || [];
    return CRITICAL_FEATURES.some((f) => {
      const id = featureToSmartId(f);
      const attr = attrs.find((a) => a.id === id);
      return attr && (attr.raw?.value ?? 0) > 0;
    }) ? 1 : 0;
  }
  if (featureName === 'total_error_count') {
    return smartData?.ata_smart_attributes?.table?.find((a) => a.id === 1)?.raw?.value ?? 0;
  }
  if (featureName === 'error_per_gb') {
    const capacity = smartData?.user_capacity?.bytes || 1;
    const errorCount = smartData?.ata_smart_attributes?.table?.find((a) => a.id === 1)?.raw?.value ?? 0;
    return errorCount / (capacity / (1024 ** 3));
  }
  if (featureName === 'capacity_gigabytes') {
    return Math.round((smartData?.user_capacity?.bytes || 0) / (1024 ** 3));
  }
  if (featureName === 'capacity_tb') {
    return (smartData?.user_capacity?.bytes || 0) / (1024 ** 4);
  }
  if (featureName === 'jeSSD') {
    const model = (smartData?.model_name || smartData?.model || '').toLowerCase();
    return model.includes('ssd') ? 1 : 0;
  }

  // SMART raw values
  const smartId = featureToSmartId(featureName);
  if (smartId == null) return null;
  const attrs = smartData?.ata_smart_attributes?.table || [];
  const attr = attrs.find((a) => a.id === smartId);
  return attr?.raw?.value ?? 0;
}

function formatValue(val, featureName) {
  if (val == null) return '—';
  if (featureName === 'error_per_gb') return val.toFixed(4);
  if (featureName === 'capacity_tb') return `${val.toFixed(2)} TB`;
  if (featureName === 'capacity_gigabytes') return `${val} GB`;
  if (featureName === 'jeSSD') return val ? 'Yes' : 'No';
  if (featureName === 'any_critical_error') return val ? 'Yes' : 'No';
  return val.toLocaleString();
}

const ShapWidget = ({ result, smartData, loading }) => {
  const topFeatures = FEATURE_IMPORTANCE.slice(0, TOP_N);
  const maxImportance = topFeatures[0]?.importance || 1;
  const sklearnScore = result?.model_scores?.sklearn;

  if (loading) {
    return (
      <div className="card widget-shap">
        <div className="card-title">Feature Importance</div>
        <div className="feature-bars">
          {[...Array(TOP_N)].map((_, i) => (
            <div key={i} className="feature-bar-row">
              <Skeleton width="100px" height="12px" />
              <Skeleton width="100%" height="8px" />
              <Skeleton width="40px" height="12px" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card widget-shap">
      <div className="card-title">
        Feature Importance
        <span style={{ fontSize: '11px', fontWeight: '400', color: 'var(--text-muted)' }}>
          Random Forest
        </span>
      </div>

      <div className="feature-bars">
        {topFeatures.map((f) => {
          const pct = (f.importance / maxImportance) * 100;
          const isCritical = CRITICAL_FEATURES.includes(f.feature);
          const isAge = f.feature === 'smart_9_raw';
          const barColor = isCritical
            ? 'var(--status-critical)'
            : isAge
              ? 'var(--status-warning)'
              : 'var(--accent-blue)';

          const driveVal = smartData ? getDriveValue(f.feature, smartData) : null;
          const hasCriticalValue = isCritical && driveVal != null && driveVal > 0;

          return (
            <div
              key={f.feature}
              className={`feature-bar-row${hasCriticalValue ? ' feature-flagged' : ''}`}
            >
              <div className="feature-label">
                {hasCriticalValue && <AlertTriangle size={11} className="feature-flag-icon" />}
                {featureLabel(f.feature)}
              </div>
              <div className="feature-bar-track">
                <div
                  className="feature-bar-fill"
                  style={{ width: `${pct}%`, background: barColor }}
                />
              </div>
              <div className="feature-importance-val tabular-nums">
                {(f.importance * 100).toFixed(1)}%
              </div>
              {smartData && (
                <div className={`feature-drive-val tabular-nums${hasCriticalValue ? ' text-critical' : ''}`}>
                  {formatValue(driveVal, f.feature)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {smartData && sklearnScore && (
        <div className="rf-verdict-detail">
          <div className="rf-verdict-row">
            <span className="rf-verdict-label">RF Vote</span>
            <span className={`status-pill ${sklearnScore.classification_fail ? 'critical' : 'healthy'}`}>
              {sklearnScore.classification_fail ? 'FAIL' : 'HEALTHY'}
            </span>
          </div>
          <div className="rf-verdict-row">
            <span className="rf-verdict-label">HIR Risk Score</span>
            <span className="tabular-nums">
              {sklearnScore.hir_risk_score != null
                ? `${Math.round(sklearnScore.hir_risk_score * 100)}%`
                : '—'}
            </span>
          </div>
        </div>
      )}

      {!smartData && (
        <div className="empty-state" style={{ padding: 'var(--space-4)', fontSize: '12px' }}>
          Upload a scan to see this drive's values for each feature
        </div>
      )}
    </div>
  );
};

export default ShapWidget;