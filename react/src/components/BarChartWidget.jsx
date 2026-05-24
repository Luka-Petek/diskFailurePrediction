const BarChartWidget = () => (
  <div className="card widget-bar">
    <div className="card-title">SMART Parameters Variance <span className="dropdown-text">2026 ⌄</span></div>
    <div className="bars">
      {[40,30, 60,20, 100,40, 70,50, 30,80, 50,20].map((h, i) => (
        <div key={i} className="bar-group">
          <div className={`bar bg-pink h-${h}`}></div>
        </div>
      ))}
    </div>
  </div>
);
export default BarChartWidget;