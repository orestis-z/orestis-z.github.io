import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { HeroDossier } from './components/HeroDossier';
import { ProjectMatrix } from './components/ProjectMatrix';
import { ShopAutomationArchitecture } from './components/ShopAutomationArchitecture';
import { DispatchesBlog } from './components/DispatchesBlog';
import { TransmitContact } from './components/TransmitContact';
import { Footer } from './components/Footer';
import { ProjectModal } from './components/ProjectModal';
import { ArticleReader } from './components/ArticleReader';
import { CommandPalette } from './components/CommandPalette';
import { siteData, Project, Post } from './data/siteData';

export const App: React.FC = () => {
  const [isDark, setIsDark] = useState<boolean>(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [activeSection, setActiveSection] = useState<string>('dossier');
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);

  // Apply dark mode class to html element
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  // Handle URL pathname and hash on initial load or navigation
  useEffect(() => {
    const handleRoute = () => {
      const pathname = window.location.pathname.toLowerCase();
      const hash = window.location.hash.replace('#', '').toLowerCase();

      // 1. Check pathname for blog route (/blog/:slug)
      const blogMatch = pathname.match(/\/blog\/([^\/\.]+)/);
      if (blogMatch) {
        const foundPost = siteData.posts.find(p => p.slug.toLowerCase() === blogMatch[1]);
        if (foundPost) {
          setSelectedPost(foundPost);
          return;
        }
      }

      // Helper to close modals and scroll to target section once mounted
      const scrollToSection = (targetId: string) => {
        setSelectedPost(null);
        setSelectedProject(null);
        setTimeout(() => {
          const el = document.getElementById(targetId);
          if (el) {
            el.scrollIntoView({ behavior: 'smooth' });
          }
        }, 50);
      };

      // 2. Check pathname for section pages
      if (pathname.includes('/portfolio')) {
        scrollToSection('projects');
        return;
      }
      if (pathname.includes('/shop-automation')) {
        scrollToSection('systems');
        return;
      }
      if (pathname.includes('/impressum')) {
        scrollToSection('contact');
        return;
      }

      if (!hash) {
        setSelectedPost(null);
        setSelectedProject(null);
        return;
      }

      // 3. Check if hash matches an article slug
      const foundPost = siteData.posts.find(p => p.slug.toLowerCase() === hash);
      if (foundPost) {
        setSelectedPost(foundPost);
        return;
      }

      // 4. Check if hash matches a project key/id
      const foundProject = siteData.projects.find(
        p => p.id.toLowerCase() === hash || p.key.toLowerCase().replace(/\s+/g, '-') === hash
      );
      if (foundProject) {
        setSelectedProject(foundProject);
        return;
      }

      // 5. Check if hash matches a section or alias
      const sectionAliases: Record<string, string> = {
        'shop-automation': 'systems',
        'systems': 'systems',
        'projects': 'projects',
        'portfolio': 'projects',
        'dossier': 'dossier',
        'about': 'dossier',
        'dispatches': 'dispatches',
        'blog': 'dispatches',
        'contact': 'contact',
        'impressum': 'contact',
      };

      const targetSection = sectionAliases[hash] || hash;
      scrollToSection(targetSection);
    };

    handleRoute();
    window.addEventListener('hashchange', handleRoute);
    window.addEventListener('popstate', handleRoute);
    return () => {
      window.removeEventListener('hashchange', handleRoute);
      window.removeEventListener('popstate', handleRoute);
    };
  }, []);

  // Update URL when opening or closing reader
  useEffect(() => {
    if (selectedPost) {
      if (!window.location.pathname.includes('/blog/')) {
        window.location.hash = selectedPost.slug;
      }
    } else {
      if (window.location.pathname.includes('/blog/')) {
        history.pushState(null, '', '/');
      } else if (window.location.hash.startsWith('#') && siteData.posts.some(p => p.slug === window.location.hash.replace('#', ''))) {
        history.replaceState(null, '', window.location.pathname);
      }
    }
  }, [selectedPost]);

  // Track active section for navigation highlighting
  useEffect(() => {
    const sections = ['dossier', 'projects', 'systems', 'dispatches', 'contact'];
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 200;
      for (const sectionId of sections) {
        const el = document.getElementById(sectionId);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPosition >= top && scrollPosition < top + height) {
            setActiveSection(sectionId);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Keyboard shortcut listener for search (Cmd+K or /)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(prev => !prev);
      } else if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setIsSearchOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const openArticleBySlug = (slug: string) => {
    const post = siteData.posts.find(p => p.slug === slug);
    if (post) {
      setSelectedPost(post);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg-canvas)] text-[var(--text-primary)] flex flex-col selection:bg-[var(--text-primary)] selection:text-[var(--bg-canvas)]">
      {/* If an article is open, render ArticleReader full-screen */}
      {selectedPost ? (
        <ArticleReader
          post={selectedPost}
          onClose={() => setSelectedPost(null)}
        />
      ) : (
        <>
          {/* Global Header */}
          <Header
            onOpenSearch={() => setIsSearchOpen(true)}
            isDark={isDark}
            onToggleTheme={() => setIsDark(!isDark)}
            activeSection={activeSection}
          />

          {/* Main Dossier Content */}
          <main className="flex-1">
            <HeroDossier />

            <ProjectMatrix
              onSelectProject={(project) => setSelectedProject(project)}
              onOpenArticleBySlug={openArticleBySlug}
            />

            <ShopAutomationArchitecture
              onOpenArticleBySlug={openArticleBySlug}
            />

            <DispatchesBlog
              onSelectPost={(post) => setSelectedPost(post)}
            />

            <TransmitContact />
          </main>

          {/* Global Engineering Footer */}
          <Footer />

          {/* Project Details Modal */}
          <ProjectModal
            project={selectedProject}
            onClose={() => setSelectedProject(null)}
            onOpenArticleBySlug={openArticleBySlug}
          />

          {/* Global Command Palette Search */}
          <CommandPalette
            isOpen={isSearchOpen}
            onClose={() => setIsSearchOpen(false)}
            onSelectProject={(project) => setSelectedProject(project)}
            onSelectPost={(post) => setSelectedPost(post)}
          />
        </>
      )}
    </div>
  );
};

export default App;
