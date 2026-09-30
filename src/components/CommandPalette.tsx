import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Layers, FileText, ArrowRight, CornerDownLeft } from 'lucide-react';
import { siteData, Project, Post } from '../data/siteData';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectProject: (project: Project) => void;
  onSelectPost: (post: Post) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectProject,
  onSelectPost
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
        else setQuery('');
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Filter items
  const matchedProjects = siteData.projects.filter(p => 
    p.name.toLowerCase().includes(query.toLowerCase()) ||
    p.description.toLowerCase().includes(query.toLowerCase()) ||
    p.tags.some(t => t.toLowerCase().includes(query.toLowerCase())) ||
    p.organization.toLowerCase().includes(query.toLowerCase())
  ).map(p => ({ type: 'project' as const, data: p }));

  const matchedPosts = siteData.posts.filter(p =>
    p.title.toLowerCase().includes(query.toLowerCase()) ||
    p.description.toLowerCase().includes(query.toLowerCase()) ||
    p.tags.some(t => t.toLowerCase().includes(query.toLowerCase()))
  ).map(p => ({ type: 'post' as const, data: p }));

  const allResults = [...matchedProjects, ...matchedPosts];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, allResults.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + allResults.length) % Math.max(1, allResults.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const current = allResults[selectedIndex];
      if (current) {
        if (current.type === 'project') {
          onSelectProject(current.data);
        } else {
          onSelectPost(current.data);
        }
        onClose();
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 px-4 bg-black/70 backdrop-blur-xs">
      <div 
        className="w-full max-w-2xl bg-[var(--bg-canvas)] border border-[var(--border-strong)] shadow-2xl overflow-hidden font-swiss-mono text-xs animate-in fade-in zoom-in-95 duration-100"
        role="dialog"
        aria-modal="true"
      >
        {/* Search Input Bar */}
        <div className="p-3 hairline-b bg-[var(--bg-surface)] flex items-center gap-3">
          <Search className="w-4 h-4 text-[var(--accent-swiss)]" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Type command or search keyword (e.g. TensorRT, Lockers, AdamW, Mask R-CNN)..."
            className="flex-1 bg-transparent border-0 text-sm text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:outline-hidden font-swiss-mono"
          />
          <button 
            onClick={onClose}
            className="p-1 hover:text-[var(--accent-swiss)] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-[60vh] overflow-y-auto divide-y divide-[var(--border-hairline)]">
          {allResults.length === 0 ? (
            <div className="p-8 text-center text-[var(--text-tertiary)]">
              NO MATCHES FOUND FOR "{query.toUpperCase()}".
            </div>
          ) : (
            allResults.map((item, index) => {
              const isSelected = index === selectedIndex;
              return (
                <div
                  key={item.type === 'project' ? item.data.id : item.data.slug}
                  onClick={() => {
                    if (item.type === 'project') onSelectProject(item.data);
                    else onSelectPost(item.data);
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={`p-3.5 flex items-center justify-between cursor-pointer transition-colors ${
                    isSelected ? 'bg-[var(--bg-subtle)] text-[var(--text-primary)]' : 'hover:bg-[var(--bg-subtle)]'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="p-1 bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[var(--accent-swiss)] shrink-0">
                      {item.type === 'project' ? <Layers className="w-3.5 h-3.5" /> : <FileText className="w-3.5 h-3.5" />}
                    </span>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-sm text-[var(--text-primary)] truncate font-swiss-sans">
                          {item.type === 'project' ? item.data.name : item.data.title}
                        </span>
                        <span className="text-[10px] text-[var(--text-tertiary)] uppercase border border-[var(--border-hairline)] px-1 py-0.2">
                          {item.type === 'project' ? item.data.id : 'PAPER'}
                        </span>
                      </div>
                      <div className="text-[11px] text-[var(--text-secondary)] truncate">
                        {item.type === 'project' 
                          ? `${item.data.organization || 'Independent'} · ${item.data.metric || item.data.year}` 
                          : `${item.data.date} · ${item.data.readTime}`}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 text-[var(--text-tertiary)] shrink-0 ml-3">
                    {isSelected && (
                      <span className="flex items-center gap-1 text-[var(--accent-swiss)] font-bold text-[10px]">
                        <span>SELECT</span>
                        <CornerDownLeft className="w-3 h-3" />
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="p-2.5 hairline-t bg-[var(--bg-subtle)] flex items-center justify-between text-[10px] text-[var(--text-tertiary)]">
          <div className="flex items-center gap-3">
            <span>↑↓ NAVIGATE</span>
            <span>↵ OPEN</span>
            <span>ESC CLOSE</span>
          </div>
          <span>TOTAL INDEX: {siteData.projects.length + siteData.posts.length} RECORDS</span>
        </div>
      </div>
    </div>
  );
};
