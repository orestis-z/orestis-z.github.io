import React, { useState } from 'react';
import { 
  Download, 
  FileText, 
  Calendar, 
  Mail, 
  MapPin, 
  Check, 
  Copy, 
  ExternalLink,
  Cpu,
  Layers,
  ArrowDown
} from 'lucide-react';
import { siteData } from '../data/siteData';
import { downloadVCard } from '../utils/vcard';

export const HeroDossier: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const { profile } = siteData;

  const copyEmail = () => {
    navigator.clipboard.writeText(profile.email);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="dossier" className="hairline-b bg-[var(--bg-canvas)]">
      {/* Section Identifier Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-8 pb-3 flex items-center justify-between border-b border-[var(--border-hairline)] text-xs font-swiss-mono text-[var(--text-tertiary)]">
        <div className="flex items-center gap-2">
          <span className="text-[var(--accent-swiss)] font-bold">SECTION // 01</span>
          <span>SYSTEM DOSSIER & BIOGRAPHY</span>
        </div>
        <div className="hidden sm:flex items-center gap-3">
          <span>SPEC: ETH-ZURICH-RSC</span>
          <span>·</span>
          <span>STATUS: DEPLOYED</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 lg:py-16">
        {/* Main Grid: Asymmetric Swiss Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start">
          
          {/* Left Column: Portrait & Hardware Specs (4 cols) */}
          <div className="lg:col-span-4 flex flex-col space-y-6">
            {/* Technical Framed Portrait */}
            <div className="relative border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
              {/* Corner crosshairs */}
              <span className="absolute -top-1.5 -left-1.5 font-swiss-mono text-xs text-[var(--text-tertiary)]">+</span>
              <span className="absolute -top-1.5 -right-1.5 font-swiss-mono text-xs text-[var(--text-tertiary)]">+</span>
              <span className="absolute -bottom-1.5 -left-1.5 font-swiss-mono text-xs text-[var(--text-tertiary)]">+</span>
              <span className="absolute -bottom-1.5 -right-1.5 font-swiss-mono text-xs text-[var(--text-tertiary)]">+</span>

              <div className="overflow-hidden aspect-square bg-[var(--bg-subtle)] relative group">
                <img 
                  src={profile.picture} 
                  alt={profile.name} 
                  className="w-full h-full object-cover grayscale contrast-110 hover:grayscale-0 transition-all duration-300"
                />
                <div className="absolute bottom-2 left-2 right-2 bg-black/80 backdrop-blur-sm text-white px-2 py-1 flex items-center justify-between text-[10px] font-swiss-mono">
                  <span>FIG 01.0 // ARCHITECT</span>
                  <span className="text-emerald-400">480×480PX</span>
                </div>
              </div>

              {/* Hardware / Location Specs */}
              <div className="mt-3 p-3 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-xs font-swiss-mono space-y-1.5 text-[var(--text-secondary)]">
                <div className="flex justify-between">
                  <span className="text-[var(--text-tertiary)]">LOCATION:</span>
                  <span className="font-semibold text-[var(--text-primary)]">{profile.location}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-tertiary)]">COORDINATES:</span>
                  <span>{profile.coordinates}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-tertiary)]">ALMA MATER:</span>
                  <span className="text-[var(--text-primary)]">ETH Zurich (MSc RSC)</span>
                </div>
                <div className="flex justify-between items-start gap-2">
                  <span className="text-[var(--text-tertiary)]">CURRENT:</span>
                  <div className="text-right">
                    <span className="text-[var(--accent-swiss)] font-semibold block">Senior ML Engineer @ Red Hat</span>
                    <span className="text-[10px] text-[var(--text-secondary)] font-mono block">Maintainer vllm-project/speculators</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Direct Channel Actions */}
            <div className="space-y-2 font-swiss-mono text-xs">
              <a
                href="https://github.com/vllm-project/speculators"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--accent-swiss)] bg-[var(--bg-surface)] flex items-center justify-between transition-colors group"
                title="View vllm-project/speculators on GitHub"
              >
                <div className="flex items-center gap-2 text-[var(--text-primary)]">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className="font-bold">vllm-project/speculators</span>
                </div>
                <span className="text-[10px] text-[var(--accent-swiss)] font-bold">[ MAINTAINER ↗ ]</span>
              </a>

              <button 
                onClick={copyEmail}
                className="w-full py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] flex items-center justify-between transition-colors group"
              >
                <div className="flex items-center gap-2">
                  <Mail className="w-3.5 h-3.5 text-[var(--text-tertiary)] group-hover:text-[var(--text-primary)]" />
                  <span>{profile.email}</span>
                </div>
                <span className="text-[10px] text-[var(--accent-swiss)] font-bold">
                  {copied ? 'COPIED ✓' : '[ COPY ]'}
                </span>
              </button>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => downloadVCard()}
                  className="py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--accent-swiss)] hover:text-[var(--accent-swiss)] bg-[var(--bg-surface)] flex items-center justify-center gap-1.5 font-medium transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>vCARD 3.0</span>
                </button>
                <a
                  href={profile.social.resume}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] flex items-center justify-center gap-1.5 text-[var(--text-primary)] transition-colors"
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>RESUME PDF</span>
                </a>
              </div>

              <a
                href={profile.social.calendly}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-2.5 px-3 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] flex items-center justify-center gap-2 font-semibold transition-colors"
              >
                <Calendar className="w-3.5 h-3.5" />
                <span>BOOK 20-MIN CONSULTATION</span>
              </a>
            </div>
          </div>

          {/* Right Column: Typographic Header, Bio & Telemetry (8 cols) */}
          <div className="lg:col-span-8 flex flex-col space-y-8">
            {/* Massive Display Title */}
            <div>
              <div className="inline-block px-2 py-0.5 mb-3 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-swiss-mono text-[11px] font-semibold tracking-wider uppercase border border-[var(--accent-swiss)]/20">
                AI & FULL-STACK ENGINEER // SYSTEMS ARCHITECT
              </div>
              <h1 className="font-swiss-sans font-black text-4xl sm:text-6xl tracking-tight text-[var(--text-primary)] leading-[1.05]">
                ORESTIS<br className="hidden sm:inline" /> ZAMBOUNIS
              </h1>
              <p className="mt-3 font-swiss-mono text-sm sm:text-base text-[var(--text-secondary)] tracking-tight">
                Machine Learning Systems // Computer Vision // Robotics & Control // Unattended Physical Automation
              </p>
            </div>

            {/* Executive Bio Paragraphs */}
            <div className="space-y-3 text-[var(--text-secondary)] text-sm sm:text-base leading-relaxed border-l-2 border-[var(--border-hairline)] pl-4">
              {profile.aboutParagraphs.map((para, i) => (
                <p key={i}>
                  {para}
                </p>
              ))}
            </div>

            {/* Performance Telemetry Grid (6 Metrics) */}
            <div>
              <div className="text-[11px] font-swiss-mono text-[var(--text-tertiary)] uppercase tracking-wider mb-2 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-[var(--accent-swiss)] inline-block"></span>
                <span>PRODUCTION SYSTEM BENCHMARKS // EMPIRICAL TELEMETRY</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {profile.keyMetrics.map((metric, idx) => (
                  <div 
                    key={idx}
                    className="p-3 bg-[var(--bg-surface)] border border-[var(--border-hairline)] hover:border-[var(--border-strong)] transition-colors relative"
                  >
                    <div className="text-[9px] font-swiss-mono text-[var(--text-tertiary)] tracking-widest uppercase">
                      METRIC // 0{idx + 1}
                    </div>
                    <div className="font-swiss-sans font-black text-2xl sm:text-3xl text-[var(--text-primary)] mt-1 tracking-tight">
                      {metric.value}
                    </div>
                    <div className="font-swiss-mono font-bold text-[11px] text-[var(--accent-swiss)] uppercase mt-0.5">
                      {metric.label}
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-1 leading-snug">
                      {metric.sub}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Technical Disciplines & Stacks */}
            <div>
              <div className="text-[11px] font-swiss-mono text-[var(--text-tertiary)] uppercase tracking-wider mb-2 flex items-center gap-2">
                <Layers className="w-3.5 h-3.5" />
                <span>CORE ENGINEERING DISCIPLINES & STACK</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-swiss-mono">
                {profile.engineeringStacks.map((stack, idx) => (
                  <div key={idx} className="p-3 bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="text-[10px] font-bold text-[var(--text-primary)] mb-1.5 tracking-wider">
                      [{stack.group}]
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {stack.skills.map((skill, sIdx) => (
                        <span 
                          key={sIdx}
                          className="px-1.5 py-0.5 bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[10px] text-[var(--text-secondary)]"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Jump Down Trigger */}
            <div className="pt-2 flex items-center justify-between text-xs font-swiss-mono text-[var(--text-tertiary)]">
              <a 
                href="#projects" 
                className="inline-flex items-center gap-2 hover:text-[var(--accent-swiss)] transition-colors"
              >
                <span>EXPLORE PROJECT MATRIX (15 DEPLOYED SYSTEMS)</span>
                <ArrowDown className="w-3.5 h-3.5 animate-bounce" />
              </a>
              <span className="hidden sm:inline">REV: 2026.09</span>
            </div>

          </div>

        </div>
      </div>
    </section>
  );
};
