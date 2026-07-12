import { useState } from 'react';
import { X } from 'lucide-react';
import { GRAPH_LABELS } from '../api/modelMetadata';

const graphModules = import.meta.glob('../../Graphs/*.png', {
  query: '?url',
  import: 'default',
  eager: true,
});

const GRAPHS = Object.entries(graphModules)
  .map(([path, url]) => ({
    filename: path.split('/').pop(),
    url,
    label: GRAPH_LABELS[path.split('/').pop()] || path.split('/').pop(),
  }))
  .sort((a, b) => a.filename.localeCompare(b.filename));

const GraphGallery = ({ onClose }) => {
  const [lightbox, setLightbox] = useState(null);

  if (GRAPHS.length === 0) {
    return (
      <div className="gallery-overlay" onClick={onClose}>
        <div className="gallery-modal" onClick={(e) => e.stopPropagation()}>
          <div className="gallery-header">
            <span>Training Graphs</span>
            <button className="gallery-close" onClick={onClose}><X size={18} /></button>
          </div>
          <div className="empty-state">No graphs found in build context.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="gallery-overlay" onClick={onClose}>
      <div className="gallery-modal" onClick={(e) => e.stopPropagation()}>
        <div className="gallery-header">
          <span>Training Graphs</span>
          <button className="gallery-close" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="gallery-grid">
          {GRAPHS.map((g) => (
            <div key={g.filename} className="gallery-item" onClick={() => setLightbox(g)}>
              <img src={g.url} alt={g.label} loading="lazy" />
              <span className="gallery-item-label">{g.label}</span>
            </div>
          ))}
        </div>
      </div>

      {lightbox && (
        <div className="gallery-lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox.url} alt={lightbox.label} />
          <span className="gallery-lightbox-label">{lightbox.label}</span>
          <button className="gallery-lightbox-close" onClick={() => setLightbox(null)}><X size={24} /></button>
        </div>
      )}
    </div>
  );
};

export default GraphGallery;
