import { Cpu } from 'lucide-react';

const ShapWidget = () => (
  <div className="card widget-shap widget-stub">
    <div className="stub-content">
      <div className="stub-icon">
        <Cpu size={32} />
      </div>
      <div className="stub-title">Coming soon</div>
      <div className="stub-desc">
        Per-feature SHAP contributions showing which SMART attributes drive the prediction.
      </div>
    </div>
  </div>
);

export default ShapWidget;