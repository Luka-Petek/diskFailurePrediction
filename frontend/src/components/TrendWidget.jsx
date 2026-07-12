import { TrendingUp } from 'lucide-react';

const TrendWidget = () => (
  <div className="card widget-trend widget-stub">
    <div className="stub-content">
      <div className="stub-icon">
        <TrendingUp size={32} />
      </div>
      <div className="stub-title">Coming soon</div>
      <div className="stub-desc">
        Anomaly trend over time — requires multiple scans to build a history.
      </div>
    </div>
  </div>
);

export default TrendWidget;