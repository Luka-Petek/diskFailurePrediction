import { useState, useCallback, useRef } from 'react';
import { analyzeDisk } from '../api/client';

/**
 * Hook that manages the disk analysis flow: file selection, API call,
 * loading/error state, the latest result, and a session-local scan history.
 */
export function useDiskAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const fileInputRef = useRef(null);

  const analyze = useCallback(async (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeDisk(selectedFile);
      setResult(data);
      setHistory((prev) => [
        {
          id: Date.now(),
          filename: selectedFile.name,
          timestamp: new Date().toISOString(),
          result: data,
        },
        ...prev,
      ]);
    } catch (err) {
      setError(err.message || 'Analysis failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  return {
    file,
    loading,
    error,
    result,
    history,
    analyze,
    dismissError,
    fileInputRef,
  };
}
