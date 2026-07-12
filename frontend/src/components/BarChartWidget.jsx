import Skeleton from './Skeleton';

const FLAGGED_IDS = [5, 187, 188, 197];

const BarChartWidget = ({ smartData, loading }) => {
  if (loading) {
    return (
      <div className="card widget-bar">
        <div className="card-title">SMART Attributes</div>
        <div className="smart-attrs">
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} width="100%" height="36px" />
          ))}
        </div>
      </div>
    );
  }

  if (!smartData) {
    return (
      <div className="card widget-bar">
        <div className="card-title">SMART Attributes</div>
        <div className="empty-state">No scan yet — upload a SMART report above</div>
      </div>
    );
  }

  const attrs = smartData?.ata_smart_attributes?.table || [];
  // Show all attributes, sorted by id
  const sorted = [...attrs].sort((a, b) => a.id - b.id);

  // Find max raw value for bar scaling (use normalized value 0-100 if available, else raw)
  const maxValue = Math.max(...sorted.map((a) => a.value || 0), 100);

  return (
    <div className="card widget-bar">
      <div className="card-title">
        SMART Attributes
        <span style={{ fontSize: '11px', fontWeight: '400', color: 'var(--text-muted)' }}>
          {sorted.length} attributes
        </span>
      </div>
      <div className="smart-attrs">
        {sorted.map((attr) => {
          const isFlagged = FLAGGED_IDS.includes(attr.id);
          const rawValue = attr.raw?.value ?? 0;
          const normalized = attr.value ?? 0;
          const barPct = Math.min((normalized / maxValue) * 100, 100);
          const flagClass = isFlagged
            ? rawValue > 0
              ? 'flagged-critical'
              : 'flagged-warning'
            : '';
          const barColor = isFlagged && rawValue > 0
            ? 'var(--status-critical)'
            : isFlagged
              ? 'var(--status-warning)'
              : 'var(--accent-blue)';
          return (
            <div key={attr.id} className={`smart-attr-row ${flagClass}`}>
              <div className="smart-attr-name">
                <span className="smart-attr-id">{attr.id}</span>
                {attr.name}
              </div>
              <div className="smart-attr-bar">
                <div
                  className="smart-attr-bar-fill"
                  style={{ width: `${barPct}%`, background: barColor }}
                />
              </div>
              <div className="smart-attr-value">{rawValue.toLocaleString()}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BarChartWidget;