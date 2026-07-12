const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Posts a smartctl JSON file to the backend's /api/predict/combined endpoint.
 * @param {File} file - The smartctl -j JSON file to analyze.
 * @returns {Promise<object>} The parsed JSON response from the backend.
 */
export async function analyzeDisk(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/api/predict/combined`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let detail;
    try {
      const body = await res.json();
      detail = body.detail || `HTTP ${res.status}`;
    } catch {
      detail = `HTTP ${res.status}`;
    }
    throw new Error(detail);
  }

  return res.json();
}

// Import all JSON files from DiskJson at build time (no duplication needed)
const sampleModules = import.meta.glob('../../DiskJson/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});

const SMARTCTL_FILES = Object.entries(sampleModules)
  .map(([path, content]) => ({
    name: path.split('/').pop(),
    content,
  }))
  .filter((f) => !f.name.includes('_results') && !f.name.includes('sweep'))
  .sort((a, b) => a.name.localeCompare(b.name));

export const SAMPLE_FILES = SMARTCTL_FILES.map((f) => f.name);

/**
 * Loads a sample smartctl JSON file (bundled from DiskJson/) as a File object.
 * @param {string} sampleName - The filename (e.g. "disk_data_sda.json").
 * @returns {Promise<File>} A File object suitable for analyzeDisk().
 */
export async function loadSampleAsFile(sampleName) {
  const sample = SMARTCTL_FILES.find((f) => f.name === sampleName);
  if (!sample) throw new Error(`Sample not found: ${sampleName}`);
  return new File([sample.content], sampleName, { type: 'application/json' });
}

export { API_BASE_URL };
