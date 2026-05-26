import React from 'react';
import Navbar from './components/Navbar';
import StatusWidget from './components/StatusWidget';
import HealthWidget from './components/HealthWidget';
import ShapWidget from './components/ShapWidget';
import TrendWidget from './components/TrendWidget';
import BarChartWidget from './components/BarChartWidget';
import LogsWidget from './components/LogsWidget';

const Dashboard = () => {
  return (
    <div className="app-body">
      <div className="dashboard-container">
        <Navbar />
        <StatusWidget />
        <HealthWidget />
        <ShapWidget />
        <TrendWidget />
        <BarChartWidget />
        <LogsWidget />
      </div>
    </div>
  );
};

export default Dashboard;