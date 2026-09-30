import React, { useState } from 'react';
import { 
  Cpu, 
  Server, 
  Smartphone, 
  Unlock, 
  ArrowRight, 
  CheckCircle, 
  FileText, 
  ExternalLink,
  Layers,
  ShieldCheck,
  Zap,
  Quote
} from 'lucide-react';
import { siteData, ShopSolution } from '../data/siteData';

interface ShopAutomationArchitectureProps {
  onOpenArticleBySlug: (slug: string) => void;
}

export const ShopAutomationArchitecture: React.FC<ShopAutomationArchitectureProps> = ({ 
  onOpenArticleBySlug 
}) => {
  const [selectedSolution, setSelectedSolution] = useState<string>('Beachin\'');
  const { shopSolutions } = siteData;

  const beachinItem = shopSolutions.find(s => s.key === "Beachin'");

  return (
    <section id="systems" className="hairline-b bg-[var(--bg-canvas)] relative">
      <div id="shop-automation" className="absolute -top-16 left-0" />
      {/* Section Header Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-3 flex flex-wrap items-center justify-between border-b border-[var(--border-hairline)] text-xs font-swiss-mono text-[var(--text-tertiary)] gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[var(--accent-swiss)] font-bold">SECTION // 03</span>
          <span>HARDWARE & UNATTENDED RETAIL ARCHITECTURE</span>
        </div>
        <div className="flex items-center gap-3">
          <span>PIPELINE: EDGE-TO-CLOUD</span>
          <span>·</span>
          <span>EFFICIENCY: 100% UNATTENDED (0 STAFF)</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 lg:py-16 space-y-12">
        {/* Header & Problem/Solution Statement */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 space-y-4">
            <div className="inline-block px-2 py-0.5 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-swiss-mono text-[11px] font-semibold tracking-wider uppercase border border-[var(--accent-swiss)]/20">
              DEPLOYED CASE STUDY: BEACHIN' RENTALS BARCELONA
            </div>
            <h2 className="font-swiss-sans font-black text-3xl sm:text-5xl text-[var(--text-primary)] tracking-tight leading-tight">
              AUTONOMOUS RETAIL: ZERO-STAFF OPERATION
            </h2>
            <p className="font-swiss-mono text-sm text-[var(--text-secondary)]">
              Transforming physical storefronts into 24/7 automated lockers and kiosk systems.
            </p>
            <p className="text-sm sm:text-base text-[var(--text-secondary)] leading-relaxed font-swiss-sans">
              In retail equipment rentals, on-site staffing is the single largest margin drain. By designing custom "Smart Racks", IoT edge controllers with ESP32 microcontrollers, and headless Shopify POS terminals, we eliminated recurring labor overhead entirely while expanding operation hours to around-the-clock 24/7.
            </p>

            <div className="pt-2 flex flex-wrap gap-3 font-swiss-mono text-xs">
              <button
                onClick={() => onOpenArticleBySlug('automating-beach-rental-store')}
                className="px-4 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] font-bold flex items-center gap-2 transition-colors"
              >
                <FileText className="w-4 h-4" />
                <span>READ IN-DEPTH CASE STUDY DISPATCH →</span>
              </button>
              <a
                href="https://beachinrentalsbcn.es/"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] flex items-center gap-2 transition-colors"
              >
                <ExternalLink className="w-4 h-4 text-[var(--text-tertiary)]" />
                <span>LIVE STOREFRONT</span>
              </a>
            </div>
          </div>

          {/* Testimonial & Metrics Panel */}
          <div className="lg:col-span-5 bg-[var(--bg-surface)] border border-[var(--border-hairline)] p-6 space-y-6">
            {beachinItem?.quote && (
              <div className="space-y-3">
                <Quote className="w-6 h-6 text-[var(--accent-swiss)] opacity-60" />
                <p className="text-sm italic font-swiss-sans text-[var(--text-primary)] leading-relaxed">
                  "{beachinItem.quote.content}"
                </p>
                <div className="font-swiss-mono text-xs text-[var(--text-tertiary)]">
                  — {beachinItem.quote.author}
                </div>
              </div>
            )}

            <div className="hairline-t pt-4 grid grid-cols-2 gap-3 font-swiss-mono text-xs">
              <div className="p-2.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                <div className="text-[10px] text-[var(--text-tertiary)]">OPERATIONAL COST</div>
                <div className="text-lg font-black text-emerald-500 mt-0.5">0.00 CHF / HR</div>
                <div className="text-[10px] text-[var(--text-secondary)]">Zero staffing needed</div>
              </div>
              <div className="p-2.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                <div className="text-[10px] text-[var(--text-tertiary)]">STORE UPTIME</div>
                <div className="text-lg font-black text-[var(--text-primary)] mt-0.5">24/7/365</div>
                <div className="text-[10px] text-[var(--text-secondary)]">Uninterrupted revenue</div>
              </div>
            </div>
          </div>
        </div>

        {/* System Architecture Blueprint / Flow Schematic */}
        <div className="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-4 sm:p-6 space-y-4">
          <div className="flex items-center justify-between text-xs font-swiss-mono text-[var(--text-tertiary)] border-b border-[var(--border-hairline)] pb-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[var(--accent-swiss)]" />
              <span className="font-bold text-[var(--text-primary)]">
                PHYSICAL COMPUTING // HARDWARE-TO-CLOUD SCHEMATIC
              </span>
            </div>
            <span>SCHEMATIC REV 3.2</span>
          </div>

          {/* Interactive Topology Diagram */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-swiss-mono text-xs pt-2">
            {/* Step 1: Customer Touch Terminal */}
            <div className="p-4 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] space-y-2">
              <div className="flex items-center justify-between text-[10px] text-[var(--accent-swiss)] font-bold">
                <span>STAGE 01</span>
                <Smartphone className="w-3.5 h-3.5" />
              </div>
              <div className="font-bold text-[var(--text-primary)]">TOUCH KIOSK / POS</div>
              <div className="text-[11px] text-[var(--text-secondary)] leading-normal">
                High-brightness capacitive screen with contactless Stripe & credit card payment gateway.
              </div>
              <div className="text-[10px] text-[var(--text-tertiary)] pt-1 border-t border-[var(--border-hairline)]">
                PROTO: HTTPS / REST
              </div>
            </div>

            {/* Step 2: Cloud Orchestration */}
            <div className="p-4 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] space-y-2">
              <div className="flex items-center justify-between text-[10px] text-[var(--accent-swiss)] font-bold">
                <span>STAGE 02</span>
                <Server className="w-3.5 h-3.5" />
              </div>
              <div className="font-bold text-[var(--text-primary)]">SHOPIFY & CLOUD API</div>
              <div className="text-[11px] text-[var(--text-secondary)] leading-normal">
                Real-time inventory ledger, order fulfillment webhook dispatcher, and automated customer SMS keys.
              </div>
              <div className="text-[10px] text-[var(--text-tertiary)] pt-1 border-t border-[var(--border-hairline)]">
                PROTO: WEBHOOKS / MQTT
              </div>
            </div>

            {/* Step 3: Edge Controller */}
            <div className="p-4 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] space-y-2">
              <div className="flex items-center justify-between text-[10px] text-[var(--accent-swiss)] font-bold">
                <span>STAGE 03</span>
                <Cpu className="w-3.5 h-3.5" />
              </div>
              <div className="font-bold text-[var(--text-primary)]">ESP32 EDGE GATEWAY</div>
              <div className="text-[11px] text-[var(--text-secondary)] leading-normal">
                Custom firmware with TLS client, I2C port expanders (MCP23017), optical feedback sensors.
              </div>
              <div className="text-[10px] text-[var(--text-tertiary)] pt-1 border-t border-[var(--border-hairline)]">
                PROTO: I2C / GPIO PULSE
              </div>
            </div>

            {/* Step 4: Physical Lock Actuation */}
            <div className="p-4 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] space-y-2">
              <div className="flex items-center justify-between text-[10px] text-[var(--accent-swiss)] font-bold">
                <span>STAGE 04</span>
                <Unlock className="w-3.5 h-3.5" />
              </div>
              <div className="font-bold text-[var(--text-primary)]">SMART RACK ACTUATION</div>
              <div className="text-[11px] text-[var(--text-secondary)] leading-normal">
                12V high-torque solenoid releases and chained buckle mechanisms securing oversized equipment.
              </div>
              <div className="text-[10px] text-[var(--text-tertiary)] pt-1 border-t border-[var(--border-hairline)]">
                MECH: 12V 2A SOLENOIDS
              </div>
            </div>
          </div>
        </div>

        {/* Product System Solutions Grid (4 Modules) */}
        <div>
          <div className="text-xs font-swiss-mono text-[var(--text-tertiary)] uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-[var(--accent-swiss)] inline-block"></span>
            <span>SYSTEM HARDWARE MODULES & INTEGRATIONS</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {shopSolutions.map((solution) => (
              <div 
                key={solution.key}
                className="border border-[var(--border-hairline)] bg-[var(--bg-surface)] flex flex-col justify-between"
              >
                {solution.portfolioImage && (
                  <div className="h-44 overflow-hidden border-b border-[var(--border-hairline)] bg-black/5 dark:bg-white/5 relative">
                    <img 
                      src={solution.portfolioImage} 
                      alt={solution.name}
                      className={`w-full h-full ${solution.portfolioImageContain ? 'object-contain p-4' : 'object-cover'} grayscale contrast-110 hover:grayscale-0 transition-all duration-200`}
                    />
                  </div>
                )}

                <div className="p-4 flex-1 flex flex-col justify-between space-y-3">
                  <div>
                    <h3 className="font-swiss-sans font-bold text-base text-[var(--text-primary)]">
                      {solution.name}
                    </h3>
                    <div className="mt-2 text-xs text-[var(--text-secondary)] leading-relaxed space-y-1.5 font-swiss-sans">
                      {solution.description.split('\n').filter(Boolean).map((line, lIdx) => (
                        <p key={lIdx}>{line}</p>
                      ))}
                    </div>
                  </div>

                  {solution.url && (
                    <div className="pt-2 border-t border-[var(--border-hairline)]">
                      {solution.url.startsWith('/blog/') ? (
                        <button
                          onClick={() => onOpenArticleBySlug('automating-beach-rental-store')}
                          className="font-swiss-mono text-xs text-[var(--accent-swiss)] hover:underline flex items-center gap-1"
                        >
                          <span>{solution.buttonLabel || 'VIEW CASE STUDY'}</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      ) : (
                        <a
                          href={solution.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-swiss-mono text-xs text-[var(--text-primary)] hover:text-[var(--accent-swiss)] flex items-center gap-1"
                        >
                          <span>{solution.buttonLabel || 'VIEW SOLUTION'}</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
};
