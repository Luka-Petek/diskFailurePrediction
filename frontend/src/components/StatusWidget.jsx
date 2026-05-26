const StatusWidget = () => (
  <div className="card widget-status">
    <div className="card-title">ML Verdict Today</div>
    <div className="donut-container">
      <div className="donut status-donut">
        <div className="donut-text">
          <h2>0.96</h2>
          <span>Risk</span>
        </div>
      </div>
      <div className="status-legend">
        <div className="legend-item"><span className="dot bg-yellow"></span>Critical (DOA) <b>WD-68N02A0</b></div>
        <div className="legend-item"><span className="dot bg-yellow"></span>Safe Drives <b>394 units</b></div>
      </div>
    </div>
  </div>
);

export default StatusWidget;