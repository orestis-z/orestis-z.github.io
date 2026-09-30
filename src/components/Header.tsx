import React, { useState, useEffect } from 'react';
import { Search, Moon, Sun, Download, Terminal, Wifi } from 'lucide-react';
import { siteData } from '../data/siteData';
import { downloadVCard } from '../utils/vcard';

interface HeaderProps {
  onOpenSearch: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
  activeSection: string;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenSearch,
  isDark,
  onToggleTheme,
  activeSection
}) => {
  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      // Format Zurich/CET time
      const timeFormatter = new Intl.DateTimeFormat('en-GB', {
        timeZone: 'Europe/Zurich',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
      setTimeStr(`${timeFormatter.format(now)} CET`);
    };

    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'dossier', num: '01', label: 'DOSSIER' },
    { id: 'projects', num: '02', label: 'PROJECTS' },
    { id: 'systems', num: '03', label: 'SYSTEMS' },
    { id: 'dispatches', num: '04', label: 'PAPERS' },
    { id: 'contact', num: '05', label: 'TRANSMIT' }
  ];

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="sticky top-0 z-40 bg-[var(--bg-canvas)]/95 backdrop-blur-md hairline-b">
      {/* Top Telemetry Ticker */}
      <div className="hairline-b bg-[var(--bg-subtle)] text-[11px] font-swiss-mono px-4 py-1.5 flex flex-wrap items-center justify-between text-[var(--text-secondary)]">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 font-medium text-[var(--text-primary)]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span>SYS: OZ-NODE // ONLINE</span>
          </div>
          <span className="hidden sm:inline opacity-40">|</span>
          <span className="hidden sm:inline">LOC: 46°31'N 6°38'E</span>
          <span className="hidden md:inline opacity-40">|</span>
          <span className="hidden md:inline">NODE: LAUSANNE (CH)</span>
        </div>

        <div className="flex items-center gap-4">
          <span className="font-semibold text-[var(--text-primary)] tracking-wider">
            {timeStr || '10:00:00 CET'}
          </span>
          <span className="opacity-40">|</span>
          <div className="flex items-center gap-1">
            <Wifi className="w-3 h-3 text-emerald-500" />
            <span>LATENCY: 0.8ms</span>
          </div>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
        {/* Brand / Logo */}
        <a 
          href="#dossier" 
          onClick={(e) => { e.preventDefault(); scrollTo('dossier'); }}
          className="group flex items-baseline gap-2.5"
        >
          <span className="font-swiss-sans font-extrabold text-lg sm:text-xl tracking-tighter text-[var(--text-primary)] group-hover:text-[var(--accent-swiss)] transition-colors">
            ORESTIS ZAMBOUNIS
          </span>
          <span className="hidden sm:inline font-swiss-mono text-[10px] uppercase tracking-widest text-[var(--text-tertiary)] border border-[var(--border-hairline)] px-1.5 py-0.5">
            CH-1005
          </span>
        </a>

        {/* Desktop Nav Items */}
        <nav className="hidden lg:flex items-center space-x-1 font-swiss-mono text-xs">
          {navItems.map((item) => {
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                onClick={() => scrollTo(item.id)}
                className={`px-3 py-1.5 transition-all text-left flex items-center gap-1.5 border ${
                  isActive
                    ? 'border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] font-bold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-hairline)]'
                }`}
              >
                <span className={isActive ? 'text-[var(--accent-swiss)]' : 'text-[var(--text-tertiary)]'}>
                  {item.num}
                </span>
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Global Action Tools */}
        <div className="flex items-center gap-2 font-swiss-mono text-xs">
          {/* Quick Search Button */}
          <button
            onClick={onOpenSearch}
            className="flex items-center gap-2 px-2.5 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Search projects and dispatches (Cmd+K or /)"
          >
            <Search className="w-3.5 h-3.5" />
            <span className="hidden md:inline">SEARCH</span>
            <kbd className="hidden md:inline px-1 py-0.2 bg-[var(--bg-subtle)] text-[10px] text-[var(--text-tertiary)]">
              /
            </kbd>
          </button>

          {/* vCard Download */}
          <button
            onClick={() => downloadVCard()}
            className="flex items-center gap-1.5 px-2.5 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--accent-swiss)] hover:text-[var(--accent-swiss)] bg-[var(--bg-surface)] text-[var(--text-primary)] transition-colors font-medium"
            title="Download vCard (.vcf) contact file with photo"
          >
            <Download className="w-3.5 h-3.5 text-[var(--accent-swiss)]" />
            <span className="hidden sm:inline">vCARD</span>
          </button>

          {/* Theme Toggle */}
          <button
            onClick={onToggleTheme}
            className="p-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Mobile Nav strip */}
      <div className="lg:hidden hairline-t px-4 py-2 flex items-center justify-between overflow-x-auto gap-2 text-xs font-swiss-mono">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => scrollTo(item.id)}
            className="whitespace-nowrap px-2 py-1 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          >
            <span className="text-[var(--accent-swiss)] mr-1">{item.num}</span>
            {item.label}
          </button>
        ))}
      </div>
    </header>
  );
};
