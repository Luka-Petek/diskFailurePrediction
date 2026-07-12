// Static model metadata — bundled at build time, no backend calls needed.
// Used by Explainability Panel, Model Performance Panel, and Model Consensus.

// --- Metadata JSONs via import.meta.glob (eager, raw JSON) ---
const clfMetaMod = import.meta.glob('../../srcML/tensorflow_classification/bottleneck_metadata.json', { eager: true });
const clfAeMetaMod = import.meta.glob('../../srcML/tensorflow_classification/clf_ae_metadata.json', { eager: true });
const anomalyMetaMod = import.meta.glob('../../srcML/tensorflow_anomaly/tf_metadata.json', { eager: true });
const clusteringMetaMod = import.meta.glob('../../srcML/tensorflow_clustering/hdbscan_metadata.json', { eager: true });
const sweepMetaMod = import.meta.glob('../../DiskJson/bottleneck_sweep_results.json', { eager: true });

function extract(mod) {
  const entries = Object.entries(mod);
  if (entries.length === 0) return null;
  const val = entries[0][1];
  return val.default || val;
}

const clfMeta = extract(clfMetaMod);
const clfAeMeta = extract(clfAeMetaMod);
const anomalyMeta = extract(anomalyMetaMod);
const clusteringMeta = extract(clusteringMetaMod);
const sweepMeta = extract(sweepMetaMod);

// --- Feature importance (hardcoded — CSV is gitignored/dockerignored) ---
export const FEATURE_IMPORTANCE = [
  { feature: 'error_per_gb',       importance: 0.16783 },
  { feature: 'any_critical_error', importance: 0.14843 },
  { feature: 'total_error_count',  importance: 0.10308 },
  { feature: 'smart_9_raw',        importance: 0.07306 },
  { feature: 'smart_197_raw',      importance: 0.07089 },
  { feature: 'smart_193_raw',      importance: 0.06267 },
  { feature: 'smart_3_raw',        importance: 0.05774 },
  { feature: 'smart_5_raw',        importance: 0.05571 },
  { feature: 'smart_192_raw',      importance: 0.04257 },
  { feature: 'smart_4_raw',        importance: 0.04101 },
  { feature: 'smart_12_raw',       importance: 0.03996 },
  { feature: 'smart_198_raw',      importance: 0.02346 },
  { feature: 'smart_191_raw',      importance: 0.02051 },
  { feature: 'smart_1_raw',        importance: 0.02044 },
  { feature: 'smart_7_raw',        importance: 0.02004 },
  { feature: 'smart_187_raw',      importance: 0.01823 },
  { feature: 'capacity_tb',        importance: 0.01542 },
  { feature: 'capacity_gigabytes', importance: 0.01250 },
  { feature: 'smart_188_raw',      importance: 0.00631 },
  { feature: 'jeSSD',              importance: 0.00012 },
];

// Human-readable labels for SMART features
const SMART_NAMES = {
  smart_1_raw:   'Read Error Rate',
  smart_3_raw:   'Spin-Up Time',
  smart_4_raw:   'Start/Stop Count',
  smart_5_raw:   'Reallocated Sectors',
  smart_7_raw:   'Seek Error Rate',
  smart_9_raw:   'Power-On Hours',
  smart_12_raw:  'Power Cycle Count',
  smart_187_raw: 'Reported Uncorrect',
  smart_188_raw: 'Command Timeout',
  smart_191_raw: 'G-Sense Error Rate',
  smart_192_raw: 'Power-Off Retract',
  smart_193_raw: 'Load Cycle Count',
  smart_197_raw: 'Current Pending Sectors',
  smart_198_raw: 'Offline Uncorrectable',
};

export function featureLabel(featureName) {
  if (SMART_NAMES[featureName]) return SMART_NAMES[featureName];
  if (featureName === 'error_per_gb') return 'Errors per GB';
  if (featureName === 'any_critical_error') return 'Any Critical Error';
  if (featureName === 'total_error_count') return 'Total Error Count';
  if (featureName === 'capacity_gigabytes') return 'Capacity (GB)';
  if (featureName === 'capacity_tb') return 'Capacity (TB)';
  if (featureName === 'jeSSD') return 'Is SSD';
  return featureName;
}

// Feature → SMART attr ID mapping (for looking up this drive's values from smartctl JSON)
const FEATURE_TO_SMART_ID = {
  smart_1_raw: 1,
  smart_3_raw: 3,
  smart_4_raw: 4,
  smart_5_raw: 5,
  smart_7_raw: 7,
  smart_9_raw: 9,
  smart_12_raw: 12,
  smart_187_raw: 187,
  smart_188_raw: 188,
  smart_191_raw: 191,
  smart_192_raw: 192,
  smart_193_raw: 193,
  smart_197_raw: 197,
  smart_198_raw: 198,
};

export function featureToSmartId(featureName) {
  return FEATURE_TO_SMART_ID[featureName] ?? null;
}

// Critical SMART features that indicate physical degradation when non-zero
export const CRITICAL_FEATURES = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw'];

// --- Structured model metadata for the Model Performance Panel ---
export const MODEL_METADATA = {
  sklearn: {
    label: 'Random Forest',
    short: 'RF',
    role: 'Supervised classification',
    weight: 0.20,
    rocAuc: null,
    prAuc: null,
    failureRecall: 0.86,
    failurePrecision: null,
    failureF1: 0.88,
    trainingRows: null,
    epochs: null,
    description: 'Classical ML on 19 SMART attributes + manufacturer encoding. Strong interpretable baseline.',
  },
  tf_classification: {
    label: 'Bottleneck Classifier',
    short: 'Clf',
    role: 'Supervised (2-stage)',
    weight: 0.50,
    rocAuc: clfMeta?.evaluation?.roc_auc ?? 0.929,
    prAuc: clfMeta?.evaluation?.pr_auc ?? 0.934,
    failureRecall: clfMeta?.evaluation?.failure_recall ?? 0.891,
    failurePrecision: clfMeta?.evaluation?.failure_precision ?? 0.883,
    failureF1: clfMeta?.evaluation?.failure_f1 ?? 0.887,
    bottleneckDim: clfMeta?.bottleneck_dim ?? 8,
    threshold: clfMeta?.threshold ?? 0.283,
    trainingRows: clfMeta?.training ?? null,
    epochs: clfMeta?.training ? `${clfMeta.training.epochs_converged}/${clfMeta.training.epochs_requested}` : null,
    description: 'AE encoder compresses 19 features → 8-dim bottleneck. Supervised classifier on distilled features. Best performer.',
  },
  tf_anomaly: {
    label: 'Anomaly Autoencoder',
    short: 'AE',
    role: 'Unsupervised anomaly',
    weight: 0.10,
    rocAuc: anomalyMeta?.evaluation?.roc_auc ?? 0.901,
    prAuc: anomalyMeta?.evaluation?.pr_auc ?? 0.600,
    failureRecall: anomalyMeta?.evaluation?.classification_report?.failure?.recall ?? 0.447,
    failurePrecision: anomalyMeta?.evaluation?.classification_report?.failure?.precision ?? 0.730,
    failureF1: anomalyMeta?.evaluation?.classification_report?.failure?.['f1-score'] ?? 0.555,
    bottleneckDim: anomalyMeta?.bottleneck_dim ?? 12,
    threshold: anomalyMeta?.threshold ?? 0.00408,
    trainingRows: anomalyMeta?.training ?? null,
    epochs: anomalyMeta?.training ? `${anomalyMeta.training.epochs_converged}/${anomalyMeta.training.epochs_requested}` : null,
    description: 'Trained on 292k healthy rows only. Flags disks with high reconstruction error. Conservative — low false positive rate.',
  },
  clustering: {
    label: 'HDBSCAN Clustering',
    short: 'HDBSCAN',
    role: 'Unsupervised clustering',
    weight: 0.20,
    rocAuc: null,
    prAuc: null,
    failureRecall: null,
    failurePrecision: null,
    failureF1: null,
    nClusters: clusteringMeta?.n_clusters ?? 18,
    outlierRatio: clusteringMeta?.outlier_ratio ?? 0.1367,
    bottleneckDim: clusteringMeta?.bottleneck_dim ?? 8,
    trainingRows: null,
    epochs: null,
    description: 'UMAP + density clustering on 8-dim bottleneck. Assigns disk to a cluster with empirical failure rate. Outliers have 66.9% failure rate.',
  },
};

// --- Graph labels for the gallery ---
export const GRAPH_LABELS = {
  'nn_classification.png': 'Bottleneck Classifier — Training Curves',
  'nn_autoencoder.png': 'Autoencoder — Training & Reconstruction Error',
  'classification.png': 'Random Forest — Classification Results',
  'regression.png': 'Random Forest — HIR Regression Results',
  'clustering.png': 'Clustering — UMAP + HDBSCAN Visualization',
  'umap_hdbscan.png': 'UMAP Projection with HDBSCAN Clusters',
  'bottleneck_kmeans_clusters.png': 'Bottleneck Space — K-means Clusters',
  'kmeans_elbow.png': 'K-means Elbow Plot',
  'hir_formula.png': 'HIR Formula Diagram',
};

// --- Sweep results (optional: show why dim=8 was chosen) ---
export const SWEEP_RESULTS = sweepMeta;

// --- Helper: get ROC-AUC reliability color ---
export function reliabilityColor(rocAuc) {
  if (rocAuc == null) return 'var(--status-neutral)';
  if (rocAuc >= 0.90) return 'var(--status-healthy)';
  if (rocAuc >= 0.80) return 'var(--status-warning)';
  return 'var(--status-critical)';
}

// --- Helper: format a number with specified decimals, tabular-nums friendly ---
export function fmt(val, decimals = 3) {
  if (val == null) return '—';
  return Number(val).toFixed(decimals);
}

// --- Helper: format a percentage ---
export function fmtPct(val, decimals = 1) {
  if (val == null) return '—';
  return `${(val * 100).toFixed(decimals)}%`;
}
