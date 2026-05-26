const HealthWidget = () => (
  <div className="card widget-health">
    <div className="card-title">Drive Health Subsystems</div>
    <div className="mini-donuts">
      {[ {label: 'Mechanics', val: '90%', cls: 'safe-circle'}, {label: 'Sectors', val: '20%', cls: 'danger-circle'}, {label: 'Age', val: '60%', cls: 'warning-circle'} ].map((item, i) => (
        <div key={i} className="mini-donut">
          <div className={`md-circle ${item.cls}`}><div className="md-inner">{item.val}</div></div>
          <span className="subsystem-label">{item.label}</span>
        </div>
      ))}
    </div>
  </div>
);

export default HealthWidget;