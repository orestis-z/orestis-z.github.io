import React, { useEffect, useState, useMemo } from 'react';
import { ArrowLeft, Clock, Calendar, Tag, Share2, Check, BookOpen, Layers } from 'lucide-react';
import { Post } from '../data/siteData';
import { renderMarkdownWithMath } from '../utils/markdown';

interface ArticleReaderProps {
  post: Post | null;
  onClose: () => void;
}

export const ArticleReader: React.FC<ArticleReaderProps> = ({ post, onClose }) => {
  const [copiedLink, setCopiedLink] = useState(false);

  // Compute rendered HTML synchronously with useMemo to avoid blank content flash
  const renderedHtml = useMemo(() => {
    return post ? renderMarkdownWithMath(post.content) : '';
  }, [post]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (post) {
      window.scrollTo(0, 0);
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [post, onClose]);

  if (!post) return null;

  const copyShareLink = () => {
    navigator.clipboard.writeText(window.location.origin + '#' + post.slug);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      {/* Sticky Reader Top Bar */}
      <div className="sticky top-0 z-50 hairline-b bg-[var(--bg-canvas)]/95 backdrop-blur-md px-4 sm:px-6 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between font-swiss-mono text-xs">
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-3 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] transition-colors group"
          >
            <ArrowLeft className="w-3.5 h-3.5 group-hover:-translate-x-0.5 transition-transform" />
            <span>RETURN TO ARCHIVE [ESC]</span>
          </button>

          <div className="hidden sm:flex items-center gap-3 text-[var(--text-tertiary)]">
            <span className="text-[var(--accent-swiss)] font-bold">DOC // {post.slug.toUpperCase()}</span>
            <span>·</span>
            <span>{post.readTime}</span>
          </div>

          <button
            onClick={copyShareLink}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Copy direct link"
          >
            {copiedLink ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Share2 className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{copiedLink ? 'LINK COPIED' : 'SHARE'}</span>
          </button>
        </div>
      </div>

      {/* Reader Main Content Container */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-10 sm:py-16">
        <article className="space-y-8">
          {/* Header Metadata */}
          <header className="space-y-4 border-b border-[var(--border-hairline)] pb-8">
            <div className="flex flex-wrap items-center gap-2 font-swiss-mono text-xs">
              <span className="px-2 py-0.5 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-bold border border-[var(--accent-swiss)]/20">
                TECHNICAL PUBLICATION
              </span>
              {post.categories.map((cat, i) => (
                <span 
                  key={i} 
                  className="px-2 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]"
                >
                  {cat}
                </span>
              ))}
            </div>

            <h1 className="font-swiss-sans font-black text-3xl sm:text-5xl tracking-tight text-[var(--text-primary)] leading-[1.15]">
              {post.title}
            </h1>

            {post.description && (
              <p className="font-swiss-mono text-sm sm:text-base text-[var(--text-secondary)] leading-relaxed pt-2">
                {post.description}
              </p>
            )}

            <div className="pt-4 flex flex-wrap items-center gap-4 sm:gap-6 font-swiss-mono text-xs text-[var(--text-tertiary)] border-t border-[var(--border-hairline)]">
              <div className="flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-[var(--accent-swiss)]" />
                <span>PUBLISHED: {post.date}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-[var(--accent-swiss)]" />
                <span>LENGTH: {post.readTime} ({post.wordCount} WORDS)</span>
              </div>
              <div>
                <span>AUTHOR: {post.author.toUpperCase()}</span>
              </div>
            </div>
          </header>

          {/* Hero Image if available */}
          {post.image && (
            <div className="border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 my-6">
              <img 
                src={post.image} 
                alt={post.title} 
                className="w-full max-h-[460px] object-cover"
              />
              <div className="mt-2 px-2 py-1 bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[10px] font-swiss-mono text-[var(--text-tertiary)] flex justify-between">
                <span>FIGURE 01 // OVERVIEW GRAPHIC</span>
                <span>STATUS: DOCUMENTED</span>
              </div>
            </div>
          )}

          {/* Rendered Prose Content with Math */}
          <div 
            className="prose-swiss max-w-none text-base sm:text-lg font-swiss-sans pt-4"
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
          />

          {/* Footer Tags & Navigation */}
          <footer className="pt-10 border-t border-[var(--border-hairline)] space-y-6">
            <div className="space-y-2">
              <div className="text-xs font-swiss-mono text-[var(--text-tertiary)] uppercase flex items-center gap-1.5">
                <Tag className="w-3.5 h-3.5" />
                <span>TAXONOMY / TAGS</span>
              </div>
              <div className="flex flex-wrap gap-1.5 font-swiss-mono text-xs">
                {post.tags.map((t, idx) => (
                  <span 
                    key={idx}
                    className="px-2 py-1 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]"
                  >
                    #{t}
                  </span>
                ))}
              </div>
            </div>

            <div className="hairline-t pt-6 flex justify-between items-center font-swiss-mono text-xs">
              <button
                onClick={onClose}
                className="px-4 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] font-bold flex items-center gap-2 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>RETURN TO ARCHIVE</span>
              </button>

              <button
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="px-3 py-2 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              >
                TOP ↑
              </button>
            </div>
          </footer>
        </article>
      </main>
    </div>
  );
};
