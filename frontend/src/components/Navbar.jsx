import { useState, useRef, useEffect } from 'react';
import { Upload, FileUp } from 'lucide-react';
import { loadSampleAsFile, SAMPLE_FILES } from '../api/client';
import logoWordmark from '../assets/logo-wordmark.svg';

const Navbar = ({ onAnalyze, loading, activeView, onViewChange }) => {
  const [open, setOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const popoverRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) setSelectedFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setSelectedFile(f);
  };

  const handleAnalyze = async () => {
    let fileToAnalyze = selectedFile;
    if (!fileToAnalyze && selectedSample) {
      try {
        fileToAnalyze = await loadSampleAsFile(selectedSample);
      } catch {
        return;
      }
    }
    if (fileToAnalyze) {
      onAnalyze(fileToAnalyze);
      setOpen(false);
      setSelectedFile(null);
      setSelectedSample('');
    }
  };

  return (
    <nav className="navbar" style={{ position: 'relative' }}>
      <img src={logoWordmark} alt="DiskGuard" className="logo" style={{ height: 26, display: 'block' }} />
      <div className="nav-links">
        <span
          className={activeView === 'dashboard' ? 'active' : ''}
          onClick={() => onViewChange?.('dashboard')}
        >Dashboard</span>
        <span
          className={activeView === 'models' ? 'active' : ''}
          onClick={() => onViewChange?.('models')}
        >Model Performance</span>
        <span>Settings</span>
      </div>
      <div style={{ position: 'relative' }} ref={popoverRef}>
        <button
          className="upload-btn"
          onClick={() => setOpen((v) => !v)}
          disabled={loading}
        >
          <Upload size={16} />
          {loading ? 'Analyzing…' : 'Upload Scan'}
        </button>
        {open && (
          <div className="upload-popover">
            <div
              className={`dropzone ${dragOver ? 'drag-over' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <div className="dz-icon">
                <FileUp size={28} style={{ opacity: 0.6 }} />
              </div>
              {selectedFile ? selectedFile.name : 'Drop smartctl JSON here or click to browse'}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <select
              className="sample-picker"
              value={selectedSample}
              onChange={(e) => setSelectedSample(e.target.value)}
            >
              <option value="">— or pick a sample —</option>
              {SAMPLE_FILES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={loading || (!selectedFile && !selectedSample)}
            >
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;