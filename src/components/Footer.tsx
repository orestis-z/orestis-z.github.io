import React from 'react';
import { ArrowUp, Terminal, Shield, FileText, Download } from 'lucide-react';
import { siteData } from '../data/siteData';
import { downloadVCard } from '../utils/vcard';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="hairline-t bg-[var(--bg-canvas)] font-swiss-mono text-xs text-[var(--text-secondary)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 space-y-8">
        
        {/* Top Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 border-b border-[var(--border-hairline)] pb-8">
          {/* Identity */}
          <div className="space-y-2">
            <div className="font-swiss-sans font-bold text-sm text-[var(--text-primary)]">
              ORESTIS ZAMBOUNIS
            </div>
            <p className="text-[11px] text-[var(--text-tertiary)] leading-relaxed">
              Senior ML Engineer at Red Hat · Maintainer of vllm-project/speculators · ex Q-SYS / Seervision · ETH Zurich Alumnus.<br />
              Specializing in speculative decoding, LLM inference acceleration, computer vision, and autonomous retail systems.
            </p>
          </div>

          {/* Direct Systems */}
          <div className="space-y-2">
            <div className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider font-bold">
              NAVIGATION
            </div>
            <ul className="space-y-0.5 text-[11px]">
              <li><a href="#dossier" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">// 01 DOSSIER & BIO</a></li>
              <li><a href="#projects" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">// 02 SELECTED WORK [15]</a></li>
              <li><a href="#systems" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">// 03 SHOP AUTOMATION</a></li>
              <li><a href="#dispatches" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">// 04 TECHNICAL PAPERS [6]</a></li>
              <li><a href="#contact" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">// 05 TRANSMISSION & CONTACT</a></li>
            </ul>
          </div>

          {/* Network & Artifacts */}
          <div className="space-y-2">
            <div className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider font-bold">
              ARTIFACTS & OPEN SOURCE
            </div>
            <ul className="space-y-0.5 text-[11px]">
              <li>
                <a href="https://github.com/vllm-project/speculators" target="_blank" rel="noopener noreferrer" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] text-[var(--accent-swiss)] font-bold transition-colors">
                  vllm-project/speculators [MAINTAINER]
                </a>
              </li>
              <li>
                <a href={siteData.profile.social.github} target="_blank" rel="noopener noreferrer" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">
                  GITHUB [ORESTIS-Z]
                </a>
              </li>
              <li>
                <a href={siteData.profile.social.linkedin} target="_blank" rel="noopener noreferrer" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">
                  LINKEDIN [ORESTIS-Z]
                </a>
              </li>
              <li>
                <a href={siteData.profile.social.resume} target="_blank" rel="noopener noreferrer" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">
                  CURRICULUM VITAE (PDF)
                </a>
              </li>
              <li>
                <button onClick={() => downloadVCard()} className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] text-left cursor-pointer transition-colors">
                  DOWNLOAD vCARD 3.0
                </button>
              </li>
              <li>
                <a href="/ai.txt" target="_blank" className="inline-flex items-center min-h-[26px] py-1 hover:text-[var(--accent-swiss)] transition-colors">
                  AI.TXT DIRECTIVES
                </a>
              </li>
            </ul>
          </div>

          {/* Swiss Entity & Back to Top */}
          <div className="space-y-3">
            <div className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider font-bold">
              LEGAL ENTITY
            </div>
            <div className="text-[11px] text-[var(--text-tertiary)] leading-relaxed">
              Zambounis Technology<br />
              Av. Eugène-Rambert 30, 1005 Lausanne<br />
              Switzerland<br />
              <a href="mailto:info@orestis.ch" className="inline-flex items-center min-h-[24px] py-0.5 text-[var(--text-primary)] hover:underline">
                info@orestis.ch
              </a>
            </div>

            <button
              onClick={scrollToTop}
              className="mt-2 px-3 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] flex items-center gap-1.5 transition-colors text-[10px] font-bold"
            >
              <ArrowUp className="w-3 h-3" />
              <span>RETURN TO TOP</span>
            </button>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-[10px] text-[var(--text-tertiary)]">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-[var(--accent-swiss)] inline-block"></span>
            <span>DESIGN SYSTEM: SWISS INTERNATIONAL TYPOGRAPHIC STYLE · RIGOROUS MODULAR GRID</span>
          </div>

          <div>
            © {new Date().getFullYear()} ZAMBOUNIS TECHNOLOGY · LAUSANNE, SWITZERLAND
          </div>
        </div>

      </div>
    </footer>
  );
};
