import React, { useState } from 'react';
import { 
  ExternalLink, 
  Layers, 
  LayoutGrid, 
  Table, 
  ArrowUpRight, 
  Tag, 
  Building, 
  Calendar,
  SlidersHorizontal,
  ChevronRight
} from 'lucide-react';
import { siteData, Project } from '../data/siteData';

interface ProjectMatrixProps {
  onSelectProject: (project: Project) => void;
  onOpenArticleBySlug: (slug: string) => void;
}

export const ProjectMatrix: React.FC<ProjectMatrixProps> = ({ 
  onSelectProject,
  onOpenArticleBySlug 
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');

  const categories = [
    { id: 'all', name: 'ALL SYSTEMS', count: siteData.projects.length },
    { id: 'computer_vision_ai', name: 'CV & AI', count: 4 },
    { id: 'iot', name: 'IOT & RETAIL', count: 1 },
    { id: 'big_data', name: 'CLOUD & DATA', count: 2 },
    { id: 'robotics', name: 'ROBOTICS & CONTROL', count: 2 },
    { id: 'app_development', name: 'APPS & SOFTWARE', count: 2 },
    { id: 'creative', name: 'VENTURES & CREATIVE', count: 4 },
  ];

  const filteredProjects = activeCategory === 'all'
    ? siteData.projects
    : siteData.projects.filter(p => p.categoryId === activeCategory);

  return (
    <section id="projects" className="hairline-b bg-[var(--bg-canvas)]">
      {/* Section Header Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-3 flex flex-wrap items-center justify-between border-b border-[var(--border-hairline)] text-xs font-swiss-mono text-[var(--text-tertiary)] gap-2">
        <h2 className="flex items-center gap-2 text-xs font-swiss-mono font-normal">
          <span className="text-[var(--accent-swiss)] font-bold">SECTION // 02</span>
          <span>SELECTED ENGINEERING SYSTEMS MATRIX</span>
        </h2>
        <div className="flex items-center gap-3">
          <span>INDEX: PRJ-01 → PRJ-15</span>
          <span>·</span>
          <span>FILTERED: {filteredProjects.length}</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-8">
        {/* Controls Toolbar: Categories and View Switcher */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border border-[var(--border-hairline)] p-2 bg-[var(--bg-surface)]">
          {/* Category Filter Pills */}
          <div className="flex flex-wrap items-center gap-1 font-swiss-mono text-xs">
            <span className="text-[var(--text-tertiary)] px-2 py-1 hidden sm:inline">
              <SlidersHorizontal className="w-3.5 h-3.5 inline mr-1" />
              DOMAIN:
            </span>
            {categories.map((cat) => {
              const isSelected = activeCategory === cat.id;
              return (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  className={`px-2.5 py-1 transition-all border ${
                    isSelected
                      ? 'bg-[var(--text-primary)] text-[var(--bg-canvas)] border-[var(--text-primary)] font-bold'
                      : 'bg-transparent text-[var(--text-secondary)] border-transparent hover:border-[var(--border-hairline)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  <span>{cat.name}</span>
                  <span className={`ml-1.5 text-[10px] ${isSelected ? 'text-[var(--accent-swiss)]' : 'text-[var(--text-tertiary)]'}`}>
                    [{cat.count}]
                  </span>
                </button>
              );
            })}
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-1 font-swiss-mono text-xs border-t md:border-t-0 pt-2 md:pt-0 border-[var(--border-hairline)]">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-2.5 py-1 flex items-center gap-1.5 border ${
                viewMode === 'grid'
                  ? 'border-[var(--text-primary)] bg-[var(--bg-subtle)] text-[var(--text-primary)] font-bold'
                  : 'border-transparent text-[var(--text-tertiary)] hover:text-[var(--text-primary)]'
              }`}
              title="Grid Card View"
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">CARDS</span>
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`px-2.5 py-1 flex items-center gap-1.5 border ${
                viewMode === 'table'
                  ? 'border-[var(--text-primary)] bg-[var(--bg-subtle)] text-[var(--text-primary)] font-bold'
                  : 'border-transparent text-[var(--text-tertiary)] hover:text-[var(--text-primary)]'
              }`}
              title="Engineering Spec Sheet Table"
            >
              <Table className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">SHEET</span>
            </button>
          </div>
        </div>

        {/* View Mode 1: Grid Cards */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProjects.map((project) => (
              <div
                key={project.id}
                className="group border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] flex flex-col justify-between transition-all duration-150 relative"
              >
                {/* Card Top Banner */}
                <div className="p-3.5 hairline-b bg-[var(--bg-subtle)] flex items-center justify-between text-xs font-swiss-mono">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 bg-[var(--text-primary)] text-[var(--bg-canvas)] font-bold text-[10px]">
                      {project.id}
                    </span>
                    <span className="text-[var(--text-tertiary)] text-[10px] uppercase truncate max-w-[120px]">
                      {project.categoryName}
                    </span>
                  </div>
                  <span className="font-medium text-[var(--text-secondary)] text-[11px]">
                    {project.year}
                  </span>
                </div>

                {/* Card Media Preview (if available) */}
                {(project.portfolioImage || project.icon) && (
                  <div 
                    onClick={() => onSelectProject(project)}
                    className="cursor-pointer overflow-hidden border-b border-[var(--border-hairline)] bg-black/5 dark:bg-white/5 relative aspect-video flex items-center justify-center group-hover:opacity-95 transition-opacity"
                  >
                    <img 
                      src={project.portfolioImage || project.icon} 
                      alt={project.name}
                      className={`w-full h-full ${project.portfolioImageContain ? 'object-contain p-3' : 'object-cover'} grayscale contrast-110 group-hover:grayscale-0 transition-all duration-300`} 
                    />
                    {project.metric && (
                      <div className="absolute top-2 right-2 bg-black/85 text-white font-swiss-mono text-[10px] px-2 py-0.5 border border-white/20">
                        {project.metric}
                      </div>
                    )}
                  </div>
                )}

                {/* Card Content Area */}
                <div className="p-4 sm:p-5 flex-1 flex flex-col justify-between space-y-4">
                  <div>
                    <h3 
                      onClick={() => onSelectProject(project)}
                      className="font-swiss-sans font-bold text-lg text-[var(--text-primary)] group-hover:text-[var(--accent-swiss)] transition-colors cursor-pointer flex items-center justify-between"
                    >
                      <span>{project.name}</span>
                      <ArrowUpRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity text-[var(--accent-swiss)]" />
                    </h3>

                    {project.subtitle && (
                      <p className="font-swiss-mono text-xs text-[var(--text-secondary)] mt-1">
                        {project.subtitle}
                      </p>
                    )}

                    {project.organization && (
                      <div className="mt-2 text-xs font-swiss-mono text-[var(--text-tertiary)] flex items-center gap-1.5">
                        <Building className="w-3 h-3" />
                        <span>ORG: {project.organization}</span>
                      </div>
                    )}

                    <p className="mt-3 text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed line-clamp-3">
                      {project.description}
                    </p>
                  </div>

                  {/* Tags */}
                  <div className="pt-2 border-t border-[var(--border-hairline)] flex flex-wrap gap-1">
                    {project.tags.slice(0, 4).map((tag, tIdx) => (
                      <span 
                        key={tIdx}
                        className="px-1.5 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[10px] font-swiss-mono text-[var(--text-tertiary)]"
                      >
                        {tag}
                      </span>
                    ))}
                    {project.tags.length > 4 && (
                      <span className="px-1.5 py-0.5 text-[10px] font-swiss-mono text-[var(--text-tertiary)]">
                        +{project.tags.length - 4}
                      </span>
                    )}
                  </div>
                </div>

                {/* Card Bottom Actions */}
                <div className="p-3 hairline-t bg-[var(--bg-subtle)] flex items-center justify-between text-xs font-swiss-mono">
                  <button
                    onClick={() => onSelectProject(project)}
                    className="font-bold text-[var(--text-primary)] hover:text-[var(--accent-swiss)] flex items-center gap-1 transition-colors"
                  >
                    <span>[ SPECIFICATIONS ]</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>

                  {project.url && (
                    project.url.startsWith('/blog/') ? (
                      <button
                        onClick={() => onOpenArticleBySlug(project.url.replace('/blog/', '').replace('.html', ''))}
                        className="text-[var(--accent-swiss)] hover:underline flex items-center gap-0.5"
                      >
                        <span>CASE STUDY</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    ) : (
                      <a
                        href={project.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:underline flex items-center gap-0.5"
                      >
                        <span>LINK</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* View Mode 2: Engineering Raw Table */
          <div className="overflow-x-auto border border-[var(--border-hairline)] bg-[var(--bg-surface)]">
            <table className="w-full text-left font-swiss-mono text-xs border-collapse">
              <thead>
                <tr className="bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)] text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider">
                  <th className="py-2.5 px-3">SYS.ID</th>
                  <th className="py-2.5 px-3">SYSTEM TITLE</th>
                  <th className="py-2.5 px-3 hidden md:table-cell">DOMAIN</th>
                  <th className="py-2.5 px-3 hidden sm:table-cell">ORGANIZATION</th>
                  <th className="py-2.5 px-3">METRIC / TELEMETRY</th>
                  <th className="py-2.5 px-3 hidden lg:table-cell">TIMELINE</th>
                  <th className="py-2.5 px-3 text-right">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-hairline)]">
                {filteredProjects.map((p) => (
                  <tr 
                    key={p.id}
                    onClick={() => onSelectProject(p)}
                    className="hover:bg-[var(--bg-subtle)] cursor-pointer transition-colors"
                  >
                    <td className="py-3 px-3 font-bold text-[var(--accent-swiss)]">
                      {p.id}
                    </td>
                    <td className="py-3 px-3">
                      <span className="font-bold text-[var(--text-primary)] hover:text-[var(--accent-swiss)] block">
                        {p.name}
                      </span>
                      {p.subtitle && (
                        <span className="text-[10px] text-[var(--text-secondary)] block">
                          {p.subtitle}
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 hidden md:table-cell text-[var(--text-secondary)]">
                      {p.categoryName}
                    </td>
                    <td className="py-3 px-3 hidden sm:table-cell text-[var(--text-primary)]">
                      {p.organization || 'Independent'}
                    </td>
                    <td className="py-3 px-3">
                      <span className="px-1.5 py-0.5 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-semibold border border-[var(--accent-swiss)]/20">
                        {p.metric || 'PROD'}
                      </span>
                    </td>
                    <td className="py-3 px-3 hidden lg:table-cell text-[var(--text-secondary)]">
                      {p.year}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectProject(p);
                        }}
                        className="px-2 py-1 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] text-[10px] font-bold"
                      >
                        [ VIEW ]
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};
