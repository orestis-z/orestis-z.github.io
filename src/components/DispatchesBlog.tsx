import React, { useState } from 'react';
import { 
  FileText, 
  Calendar, 
  Clock, 
  Tag, 
  ArrowRight, 
  Search,
  BookOpen,
  Filter
} from 'lucide-react';
import { siteData, Post } from '../data/siteData';

interface DispatchesBlogProps {
  onSelectPost: (post: Post) => void;
}

export const DispatchesBlog: React.FC<DispatchesBlogProps> = ({ onSelectPost }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string>('all');
  const { posts } = siteData;

  // Extract all unique tags
  const allTags = Array.from(new Set(posts.flatMap(p => p.tags))).slice(0, 8);

  const filteredPosts = posts.filter(post => {
    const matchesSearch = 
      post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesTag = selectedTag === 'all' || post.tags.includes(selectedTag);

    return matchesSearch && matchesTag;
  });

  return (
    <section id="dispatches" className="hairline-b bg-[var(--bg-canvas)]">
      {/* Section Header Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-3 flex flex-wrap items-center justify-between border-b border-[var(--border-hairline)] text-xs font-swiss-mono text-[var(--text-tertiary)] gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[var(--accent-swiss)] font-bold">SECTION // 04</span>
          <span>TECHNICAL PAPERS & RESEARCH DISPATCHES</span>
        </div>
        <div className="flex items-center gap-3">
          <span>ARCHIVE: 6 MONOGRAPHS</span>
          <span>·</span>
          <span>PEER-LEVEL RIGOR</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 lg:py-16 space-y-8">
        {/* Dispatches Filter Bar */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 p-2 bg-[var(--bg-surface)] border border-[var(--border-hairline)]">
          {/* Search Input */}
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-tertiary)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search papers by keyword, concept, formula (e.g. quantization, adam-w, ranking)..."
              className="w-full pl-9 pr-3 py-1.5 bg-transparent border-0 font-swiss-mono text-xs text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:outline-hidden"
            />
          </div>

          {/* Quick Tag Pills */}
          <div className="flex flex-wrap items-center gap-1 font-swiss-mono text-xs border-t md:border-t-0 pt-2 md:pt-0 border-[var(--border-hairline)]">
            <button
              onClick={() => setSelectedTag('all')}
              className={`px-2 py-1 border ${
                selectedTag === 'all'
                  ? 'border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] font-bold'
                  : 'border-transparent text-[var(--text-secondary)] hover:border-[var(--border-hairline)]'
              }`}
            >
              ALL
            </button>
            {allTags.map((t) => (
              <button
                key={t}
                onClick={() => setSelectedTag(t)}
                className={`px-2 py-1 border ${
                  selectedTag === t
                    ? 'border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] font-bold'
                    : 'border-transparent text-[var(--text-secondary)] hover:border-[var(--border-hairline)]'
                }`}
              >
                #{t}
              </button>
            ))}
          </div>
        </div>

        {/* Posts List */}
        <div className="space-y-4">
          {filteredPosts.map((post, idx) => (
            <article
              key={post.slug}
              onClick={() => onSelectPost(post)}
              className="group border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] p-5 sm:p-6 transition-all duration-150 cursor-pointer relative"
            >
              {/* Corner mark */}
              <span className="absolute top-2 right-2 text-[10px] font-swiss-mono text-[var(--text-tertiary)] group-hover:text-[var(--accent-swiss)] transition-colors">
                DOC-0{idx + 1}
              </span>

              <div className="space-y-3">
                {/* Meta info */}
                <div className="flex flex-wrap items-center gap-3 font-swiss-mono text-xs text-[var(--text-tertiary)]">
                  <span className="text-[var(--accent-swiss)] font-semibold">
                    {post.date}
                  </span>
                  <span>·</span>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{post.readTime}</span>
                  </div>
                  <span>·</span>
                  <span>{post.categories.join(' / ')}</span>
                </div>

                {/* Title */}
                <h3 className="font-swiss-sans font-extrabold text-xl sm:text-2xl text-[var(--text-primary)] group-hover:text-[var(--accent-swiss)] transition-colors leading-tight">
                  {post.title}
                </h3>

                {/* Excerpt / Description */}
                <p className="text-xs sm:text-sm text-[var(--text-secondary)] font-swiss-sans leading-relaxed line-clamp-2">
                  {post.description || post.content.slice(0, 260) + '...'}
                </p>

                {/* Tags and Action Bar */}
                <div className="pt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border-hairline)]">
                  <div className="flex flex-wrap gap-1 font-swiss-mono text-[10px]">
                    {post.tags.map((tag, tIdx) => (
                      <span 
                        key={tIdx}
                        className="px-1.5 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-tertiary)]"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>

                  <div className="font-swiss-mono text-xs font-bold text-[var(--text-primary)] group-hover:text-[var(--accent-swiss)] flex items-center gap-1 transition-colors">
                    <span>[ READ COMPLETE PAPER ]</span>
                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </div>
            </article>
          ))}

          {filteredPosts.length === 0 && (
            <div className="p-8 border border-dashed border-[var(--border-hairline)] text-center font-swiss-mono text-xs text-[var(--text-secondary)]">
              NO PUBLICATIONS MATCHED SEARCH QUERY "{searchQuery}".
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
