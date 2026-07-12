import { useState, useCallback } from 'react';
import { AlertCircle, X } from 'lucide-react';
import Navbar from './components/Navbar';
import StatusWidget from './components/StatusWidget';
import HealthWidget from './components/HealthWidget';
import ShapWidget from './components/ShapWidget';
import TrendWidget from './components/TrendWidget';
import BarChartWidget from './components/BarChartWidget';
import LogsWidget from './components/LogsWidget';
import { useDiskAnalysis } from './hooks/useDiskAnalysis';

const Dashboard = () => {
  const { loading, error, result, history, analyze, dismissError } = useDiskAnalysis();
  const [smartData, setSmartData] = useState(null);
  const [driveInfo, setDriveInfo] = useState(null);

  const handleAnalyze = useCallback(async (file) => {
    // Parse the JSON file client-side for SMART attributes and drive info
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      setSmartData(parsed);
      setDriveInfo({
        model: parsed.model_name || parsed.model || 'Unknown',
        serial: parsed.serial_number || '',
        capacity: parsed.user_capacity?.bytes || 0,
        firmware: parsed.firmware_version || '',
      });
    } catch {
      setSmartData(null);
      setDriveInfo(null);
    }
    analyze(file);
  }, [analyze]);

  return (
    <div className="app-body">
      <div className="dashboard-container">
        <Navbar onAnalyze={handleAnalyze} loading={loading} />
        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{error}</span>
            <button className="dismiss-btn" onClick={dismissError}>
              <X size={16} />
            </button>
          </div>
        )}
        <StatusWidget result={result} loading={loading} driveInfo={driveInfo} />
        <HealthWidget result={result} loading={loading} driveInfo={driveInfo} />
        <ShapWidget />
        <TrendWidget />
        <BarChartWidget smartData={smartData} loading={loading} />
        <LogsWidget history={history} loading={loading} />
      </div>
    </div>
  );
};

export default Dashboard;