import { useState } from 'react';
import { Image as ImageIcon, AlertTriangle, Info } from 'lucide-react';
import Skeleton from './Skeleton';
import GraphGallery from './GraphGallery';
import {
  MODEL_METADATA,
  reliabilityColor,
  fmt,
  fmtPct,
} from '../api/modelMetadata';

const scoreToColor = (score) => {
  if (score == null) return 'var(--status-neutral)';
  if (score >= 0.70) return 'var(--status-critical)';
  if (score >= 0.40) return 'var(--status-warning)';
  return 'var(--status-healthy)';
};

const verdictToPill = (verdict) => {
  if (!verdict) return { cls: 'neutral', label: '—' };
  const v = verdict.toUpperCase();
  if (v === 'FAILURE' || v === 'ANOMALY_DETECTED' || v === 'CRITICAL') return { cls: 'critical', label: v.replace(/_/g, ' ') };
  if (v === 'AT_RISK' || v === 'WARNING' || v === 'ELEVATED_RISK' || v === 'HIGH_RISK') return { cls: 'warning', label: v.replace(/_/g, ' ') };
  if (v === 'HEALTHY' || v === 'LOW_RISK') return { cls: 'healthy', label: v.replace(/_/g, ' ') };
  return { cls: 'neutral', label: v };
};

function MetricRow({ label, value, bar }) {
  return (
    <div className="mp-metric">
      <span className="mp-metric-label">{label}</span>
      <span className="mp-metric-val tabular-nums">{value}</span>
      {bar != null && (
        <div className="mp-metric-bar-track">
          <div className="mp-metric-bar-fill" style={{ width: `${bar * 100}%`, background: reliabilityColor(bar) }} />
        </div>
      )}
    </div>
  );
}

function ModelCard({ modelKey, modelData, scanData }) {
  const meta = MODEL_METADATA[modelKey];
  if (!meta) return null;

  const score = scanData ? (
    modelKey === 'tf_classification' ? scanData.failure_probability
    : modelKey === 'tf_anomaly' ? scanData.anomaly_score
    : modelKey === 'clustering' ? scanData.cluster_score
    : scanData.failure_probability
  ) : null;
  const verdict = scanData?.verdict || (modelKey === 'clustering' ? null : null);
  const pill = verdictToPill(verdict);

  return (
    <div className="mp-card">
      <div className="mp-card-header">
        <span className="mp-card-title">{meta.label}</span>
        <span className="mp-weight-badge">w{meta.weight.toFixed(2)}</span>
      </div>
      <div className="mp-card-role">{meta.role}</div>

      {/* Static reliability metrics */}
      <div className="mp-metrics-section">
        <div className="mp-section-label">Reliability</div>
        {meta.rocAuc != null && <MetricRow label="ROC-AUC" value={fmt(meta.rocAuc)} bar={meta.rocAuc} />}
        {meta.prAuc != null && <MetricRow label="PR-AUC" value={fmt(meta.prAuc)} />}
        {meta.failureRecall != null && <MetricRow label="Recall" value={fmtPct(meta.failureRecall)} bar={meta.failureRecall} />}
        {meta.failurePrecision != null && <MetricRow label="Precision" value={fmtPct(meta.failurePrecision)} />}
        {meta.failureF1 != null && <MetricRow label="F1" value={fmtPct(meta.failureF1)} bar={meta.failureF1} />}
        {meta.nClusters != null && <MetricRow label="Clusters" value={meta.nClusters} />}
        {meta.outlierRatio != null && <MetricRow label="Outliers" value={fmtPct(meta.outlierRatio)} />}
      </div>

      {/* This scan's results */}
      {scanData && (
        <div className="mp-scan-section">
          <div className="mp-section-label">This Scan</div>
          <div className="mp-scan-score" style={{ color: scoreToColor(score) }}>
            {score != null ? `${Math.round(score * 100)}%` : '—'}
          </div>
          {verdict && (
            <span className={`status-pill ${pill.cls}`}>{pill.label}</span>
          )}

          {/* Model-specific extras */}
          {modelKey === 'tf_classification' && scanData.bottleneck_features && (
            <div className="mp-extra">
              <div className="mp-extra-label">Bottleneck (8-dim)</div>
              <div className="bottleneck-sparkline">
                {scanData.bottleneck_features.map((v, i) => {
                  const max = Math.max(...scanData.bottleneck_features.map(Math.abs));
                  const h = max > 0 ? Math.abs(v) / max : 0;
                  return (
                    <div
                      key={i}
                      className="bottleneck-bar"
                      style={{
                        height: `${Math.max(h * 100, 4)}%`,
                        background: v >= 0 ? 'var(--accent-blue)' : 'var(--accent-pink)',
                      }}
                      title={`dim ${i + 1}: ${v.toFixed(4)}`}
                    />
                  );
                })}
              </div>
              <div className="mp-threshold-gauge">
                <span className="mp-gauge-label">vs threshold</span>
                <div className="mp-gauge-track">
                  <div className="mp-gauge-threshold" style={{ left: `${(scanData.threshold || 0) * 100}%` }} />
                  <div className="mp-gauge-highrisk" style={{ left: `${(scanData.high_risk_threshold || 0) * 100}%` }} />
                  <div className="mp-gauge-marker" style={{ left: `${(scanData.failure_probability || 0) * 100}%` }} />
                </div>
                <span className="mp-gauge-val tabular-nums">{fmt(scanData.failure_probability)}</span>
              </div>
            </div>
          )}

          {modelKey === 'tf_anomaly' && scanData.reconstruction_error != null && (
            <div className="mp-extra">
              <div className="mp-threshold-gauge">
                <span className="mp-gauge-label">Recon error</span>
                <div className="mp-gauge-track">
                  <div className="mp-gauge-threshold" style={{ left: '50%' }} />
                  <div className="mp-gauge-marker" style={{
                    left: `${Math.min((scanData.reconstruction_error / (scanData.threshold * 3 || 1)) * 50, 100)}%`,
                    background: scanData.is_anomaly ? 'var(--status-critical)' : 'var(--status-healthy)',
                  }} />
                </div>
                <span className="mp-gauge-val tabular-nums">{scanData.reconstruction_error.toFixed(6)}</span>
              </div>
              <div className="mp-threshold-detail">
                threshold: <span className="tabular-nums">{scanData.threshold?.toFixed(6)}</span>
                {' · '}
                <span className={scanData.is_anomaly ? 'text-critical' : 'text-healthy'}>
                  {scanData.is_anomaly ? 'ANOMALY' : 'normal'}
                </span>
              </div>
            </div>
          )}

          {modelKey === 'clustering' && (
            <div className="mp-extra">
              <div className="mp-cluster-detail">
                <span className="mp-cluster-id">Cluster {scanData.cluster_id ?? '—'}</span>
                {scanData.is_outlier && (
                  <span className="status-pill critical" style={{ marginLeft: '4px' }}>OUTLIER</span>
                )}
              </div>
              <div className="mp-cluster-stats">
                <span>Failure rate: <strong className="tabular-nums">{fmtPct(scanData.cluster_failure_rate)}</strong></span>
                <span>Strength: <strong className="tabular-nums">{fmt(scanData.cluster_strength)}</strong></span>
              </div>
              {scanData.is_outlier && (
                <div className="mp-outlier-warning">
                  <AlertTriangle size={11} />
                  Outliers have 66.9% failure rate in training data
                </div>
              )}
            </div>
          )}

          {modelKey === 'sklearn' && (
            <div className="mp-extra">
              <div className="mp-extra-row">
                <span>HIR Score</span>
                <strong className="tabular-nums">
                  {scanData.hir_risk_score != null ? `${Math.round(scanData.hir_risk_score * 100)}%` : '—'}
                </strong>
              </div>
              <div className="mp-extra-row">
                <span>RF Vote</span>
                <span className={`status-pill ${scanData.classification_fail ? 'critical' : 'healthy'}`}>
                  {scanData.classification_fail ? 'FAIL' : 'OK'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mp-card-footer">
        <Info size={10} />
        <span>{meta.description}</span>
      </div>
    </div>
  );
}

const TrendWidget = ({ result, loading }) => {
  const [showGallery, setShowGallery] = useState(false);

  if (loading) {
    return (
      <div className="card widget-trend">
        <div className="card-title">Model Performance</div>
        <div className="mp-grid">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="mp-card">
              <Skeleton width="80%" height="14px" />
              <Skeleton width="60%" height="10px" />
              <div style={{ marginTop: 'var(--space-3)' }}>
                {[...Array(5)].map((_, j) => (
                  <Skeleton key={j} width="100%" height="12px" />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const scores = result?.model_scores || {};

  return (
    <div className="card widget-trend">
      <div className="card-title">
        Model Performance
        <button className="mp-gallery-btn" onClick={() => setShowGallery(true)}>
          <ImageIcon size={13} />
          Training Graphs
        </button>
      </div>

      <div className="mp-grid">
        <ModelCard modelKey="tf_classification" scanData={scores.tf_classification} />
        <ModelCard modelKey="tf_anomaly" scanData={scores.tf_anomaly} />
        <ModelCard modelKey="sklearn" scanData={scores.sklearn} />
        <ModelCard modelKey="clustering" scanData={scores.clustering} />
      </div>

      {showGallery && <GraphGallery onClose={() => setShowGallery(false)} />}
    </div>
  );
};

export default TrendWidget;