const LogsWidget = () => (
  <div className="card widget-logs">
    <div className="card-title">Most Critical Logs <span className="dropdown-text">1 Month ⌄</span></div>
    {[ {text: 'DOA Detected (0 hrs)', drive: 'WDC-WD20EFAX', risk: '98%'}, {text: 'Sector Failure', drive: 'Seagate-ST4000', risk: '82%'}, {text: 'Age Degradation', drive: 'Toshiba-HDWE140', risk: '64%'} ].map((l, i) => (
      <div key={i} className="log-item">
        <div><strong>{l.text}</strong><br />{l.drive}</div>
        <div>{l.risk} Risk</div>
      </div>
    ))}
    <div className="btn-see-more">See More</div>
  </div>
);
export default LogsWidget;