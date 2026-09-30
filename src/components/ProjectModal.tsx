import React, { useEffect } from 'react';
import { X, ExternalLink, FileText, Presentation, Tag, Calendar, Building, CheckCircle2 } from 'lucide-react';
import { Project } from '../data/siteData';

interface ProjectModalProps {
  project: Project | null;
  onClose: () => void;
  onOpenArticleBySlug?: (slug: string) => void;
}

export const ProjectModal: React.FC<ProjectModalProps> = ({ 
  project, 
  onClose,
  onOpenArticleBySlug 
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (project) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [project, onClose]);

  if (!project) return null;

  const isCaseStudyLink = project.url && project.url.startsWith('/blog/');
  const blogSlug = isCaseStudyLink ? project.url.replace('/blog/', '').replace('.html', '') : '';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/70 backdrop-blur-xs">
      <div 
        className="relative w-full max-w-4xl max-h-[92vh] flex flex-col bg-[var(--bg-canvas)] border border-[var(--border-strong)] shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
        role="dialog"
        aria-modal="true"
      >
        {/* Modal Header Bar */}
        <div className="hairline-b px-4 sm:px-6 py-3.5 bg-[var(--bg-subtle)] flex items-center justify-between text-xs font-swiss-mono">
          <div className="flex items-center gap-3">
            <span className="px-1.5 py-0.5 bg-[var(--accent-swiss)] text-white font-bold">
              {project.id}
            </span>
            <span className="font-bold text-[var(--text-primary)]">
              SPECIFICATION DOSSIER // {project.name.toUpperCase()}
            </span>
            <span className="hidden sm:inline text-[var(--text-tertiary)]">
              [{project.categoryName}]
            </span>
          </div>

          <button
            onClick={onClose}
            className="p-1 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Close modal (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Scrollable Body */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-6">
          {/* Main Title & Organization Banner */}
          <div className="space-y-2 border-b border-[var(--border-hairline)] pb-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-swiss-sans font-black text-2xl sm:text-3xl text-[var(--text-primary)] tracking-tight">
                {project.name}
              </h2>
              <div className="font-swiss-mono text-xs px-2.5 py-1 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-bold border border-[var(--accent-swiss)]/20">
                {project.metric || project.year}
              </div>
            </div>

            {project.subtitle && (
              <p className="font-swiss-mono text-xs sm:text-sm text-[var(--text-secondary)]">
                {project.subtitle}
              </p>
            )}

            {/* Metadata strip */}
            <div className="pt-2 flex flex-wrap gap-4 text-xs font-swiss-mono text-[var(--text-secondary)]">
              <div className="flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />
                <span>TIMELINE: {project.year}</span>
              </div>
              {project.organization && (
                <div className="flex items-center gap-1.5">
                  <Building className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />
                  <span>ORGANIZATION: </span>
                  {project.organizationLink ? (
                    <a 
                      href={project.organizationLink} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-[var(--text-primary)] hover:text-[var(--accent-swiss)] underline flex items-center gap-0.5"
                    >
                      {project.organization}
                      <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  ) : (
                    <span className="text-[var(--text-primary)]">{project.organization}</span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Project Media / Technical Schematic */}
          {(project.portfolioImage || project.icon) && (
            <div className="border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 relative">
              <div className="max-h-[380px] overflow-hidden flex items-center justify-center bg-black/5 dark:bg-white/5">
                <img 
                  src={project.portfolioImage || project.icon} 
                  alt={project.name} 
                  className={`max-h-[360px] w-auto ${project.portfolioImageContain ? 'object-contain' : 'object-cover'}`}
                />
              </div>
              <div className="mt-2 px-2 py-1 bg-[var(--bg-surface)] border border-[var(--border-hairline)] flex items-center justify-between text-[10px] font-swiss-mono text-[var(--text-tertiary)]">
                <span>SYSTEM ARTIFACT // {project.name.toUpperCase()}</span>
                <span>STATUS: VERIFIED IN PRODUCTION</span>
              </div>
            </div>
          )}

          {/* Technical Detailed Breakdown */}
          <div className="space-y-3">
            <div className="text-xs font-swiss-mono font-bold text-[var(--text-tertiary)] uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-[var(--accent-swiss)] inline-block"></span>
              <span>TECHNICAL SPECIFICATION & SCOPE</span>
            </div>
            <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-sm leading-relaxed text-[var(--text-primary)] font-swiss-sans space-y-3">
              {project.detailedDescription.split(/\n\s*\n/).map((paragraph, pIdx) => (
                <p key={pIdx}>
                  {paragraph}
                </p>
              ))}
            </div>
          </div>

          {/* Technical Stack Tags */}
          <div className="space-y-2">
            <div className="text-xs font-swiss-mono font-bold text-[var(--text-tertiary)] uppercase tracking-wider flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5" />
              <span>STACK & ARCHITECTURE COMPONENTS</span>
            </div>
            <div className="flex flex-wrap gap-1.5 font-swiss-mono text-xs">
              {project.tags.map((tag, i) => (
                <span 
                  key={i}
                  className="px-2.5 py-1 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Action Trigger Buttons */}
          <div className="pt-4 border-t border-[var(--border-hairline)] flex flex-wrap gap-3 font-swiss-mono text-xs">
            {project.url && (
              isCaseStudyLink && onOpenArticleBySlug ? (
                <button
                  onClick={() => {
                    onClose();
                    onOpenArticleBySlug(blogSlug);
                  }}
                  className="px-4 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] flex items-center gap-2 font-semibold transition-colors"
                >
                  <FileText className="w-4 h-4" />
                  <span>READ FULL CASE STUDY DISPATCH →</span>
                </button>
              ) : (
                <a
                  href={project.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-4 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] flex items-center gap-2 font-semibold transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>{project.buttonLabel || 'OPEN EXTERNAL SPEC'}</span>
                </a>
              )
            )}

            {project.secondaryButtonUrl && (
              <a
                href={project.secondaryButtonUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] flex items-center gap-2 font-medium transition-colors"
              >
                <Presentation className="w-4 h-4 text-[var(--text-tertiary)]" />
                <span>{project.secondaryButtonLabel || 'VIEW SECONDARY ARTIFACT'}</span>
              </a>
            )}

            <button
              onClick={onClose}
              className="ml-auto px-4 py-2.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              CLOSE [ESC]
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
