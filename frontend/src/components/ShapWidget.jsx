const ShapWidget = () => (
  <div className="card widget-shap">
    <div className="card-title">SHAP Value Influence <span className="dropdown-text">Last 48h ⌄</span></div>
    <div className="shap-circles">
      <div className="shap-donut shap-pink"><div className="shap-inner">+0.48</div></div>
      <div className="shap-donut shap-blue"><div className="shap-inner">-0.12</div></div>
    </div>
    <div className="shap-stats">
      <div><div className="stat-label">error_per_gb</div><b className="stat-value">Critical</b></div>
      <div><div className="stat-label">smart_9_raw</div><b className="stat-value">New Drive</b></div>
    </div>
    <div className="shap-matrix-section">
      <div className="card-title">SHAP Matrix View</div>
      <div className="shap-matrix-grid"></div>
    </div>
  </div>
);
export default ShapWidget;