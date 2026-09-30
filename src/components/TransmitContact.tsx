import React, { useState } from 'react';
import { 
  Mail, 
  Phone, 
  MapPin, 
  Calendar, 
  Download, 
  Send, 
  Copy, 
  Check, 
  ExternalLink,
  Shield,
  FileText
} from 'lucide-react';
import { siteData } from '../data/siteData';
import { downloadVCard } from '../utils/vcard';

export const TransmitContact: React.FC = () => {
  const { profile } = siteData;
  const [copiedEmail, setCopiedEmail] = useState(false);
  const [copiedPhone, setCopiedPhone] = useState(false);
  const [showImpressumModal, setShowImpressumModal] = useState(false);

  // Form states
  const [name, setName] = useState('');
  const [senderEmail, setSenderEmail] = useState('');
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');
  const [transmissionStatus, setTransmissionStatus] = useState<string | null>(null);

  const copyEmail = () => {
    navigator.clipboard.writeText(profile.email);
    setCopiedEmail(true);
    setTimeout(() => setCopiedEmail(false), 2000);
  };

  const copyPhone = () => {
    navigator.clipboard.writeText(profile.phone);
    setCopiedPhone(true);
    setTimeout(() => setCopiedPhone(false), 2000);
  };

  const handleSendMail = (e: React.FormEvent) => {
    e.preventDefault();
    const mailtoUrl = `mailto:${profile.email}?subject=${encodeURIComponent(
      subject || `Transmission from ${name || 'Contact'}`
    )}&body=${encodeURIComponent(
      `Name: ${name}\nEmail: ${senderEmail}\n\nMessage:\n${message}`
    )}`;
    window.location.href = mailtoUrl;
    setTransmissionStatus('DISPATCHED VIA CLIENT');
    setTimeout(() => setTransmissionStatus(null), 3000);
  };

  return (
    <section id="contact" className="hairline-b bg-[var(--bg-canvas)]">
      {/* Section Header Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-3 flex flex-wrap items-center justify-between border-b border-[var(--border-hairline)] text-xs font-swiss-mono text-[var(--text-tertiary)] gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[var(--accent-swiss)] font-bold">SECTION // 05</span>
          <span>TRANSMISSION, CHANNELS & SWISS IMPRESSUM</span>
        </div>
        <div className="flex items-center gap-3">
          <span>HOST: CH-LAUSANNE</span>
          <span>·</span>
          <span>ENCRYPTION: TLS 1.3</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 lg:py-16 space-y-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Direct Transmission Form (7 cols) */}
          <div className="lg:col-span-7 border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 space-y-6">
            <div>
              <div className="text-[10px] font-swiss-mono text-[var(--text-tertiary)] uppercase tracking-wider mb-1">
                DISPATCH INTERFACE // DIRECT TRANSMISSION
              </div>
              <h2 className="font-swiss-sans font-black text-2xl sm:text-3xl text-[var(--text-primary)] tracking-tight">
                INITIATE COMMUNICATION
              </h2>
              <p className="mt-1 font-swiss-mono text-xs text-[var(--text-secondary)]">
                Direct channel to Orestis Zambounis for systems architecture, ML inference consultation, or retail automation inquiries.
              </p>
            </div>

            <form onSubmit={handleSendMail} className="space-y-4 font-swiss-mono text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-[var(--text-tertiary)]">
                    NAME // SENDER IDENTIFIER
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Dr. H. Müller"
                    className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] focus:border-[var(--text-primary)] text-[var(--text-primary)] focus:outline-hidden"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-[var(--text-tertiary)]">
                    EMAIL ADDRESS
                  </label>
                  <input
                    type="email"
                    required
                    value={senderEmail}
                    onChange={(e) => setSenderEmail(e.target.value)}
                    placeholder="e.g. h.mueller@domain.ch"
                    className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] focus:border-[var(--text-primary)] text-[var(--text-primary)] focus:outline-hidden"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] uppercase text-[var(--text-tertiary)]">
                  SUBJECT LINE
                </label>
                <input
                  type="text"
                  required
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="e.g. ML Inference Optimization Project / Kiosk Automation"
                  className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] focus:border-[var(--text-primary)] text-[var(--text-primary)] focus:outline-hidden"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[10px] uppercase text-[var(--text-tertiary)]">
                  MESSAGE SPECIFICATION
                </label>
                <textarea
                  rows={5}
                  required
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Outline the technical requirements, architecture constraints, or timeline..."
                  className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] focus:border-[var(--text-primary)] text-[var(--text-primary)] focus:outline-hidden resize-y font-swiss-mono"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                <button
                  type="submit"
                  className="px-6 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] font-bold flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>TRANSMIT DISPATCH →</span>
                </button>

                {transmissionStatus && (
                  <span className="text-[10px] text-emerald-500 font-bold">
                    ✓ {transmissionStatus}
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* Right Column: Physical Coordinates, Legal Address, vCard (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Primary Coordinates Box */}
            <div className="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 space-y-4">
              <div className="text-[10px] font-swiss-mono text-[var(--text-tertiary)] uppercase tracking-wider">
                CHANNELS // COORDINATES
              </div>

              <div className="space-y-3 font-swiss-mono text-xs">
                {/* Email row */}
                <div className="p-3 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <Mail className="w-4 h-4 text-[var(--accent-swiss)]" />
                    <div>
                      <div className="text-[9px] text-[var(--text-tertiary)]">PRIMARY TRANSMISSION</div>
                      <a href={`mailto:${profile.email}`} className="font-semibold text-[var(--text-primary)] hover:text-[var(--accent-swiss)]">
                        {profile.email}
                      </a>
                    </div>
                  </div>
                  <button 
                    onClick={copyEmail}
                    className="p-1 hover:text-[var(--accent-swiss)] transition-colors" 
                    title="Copy Email"
                  >
                    {copiedEmail ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>

                {/* Phone row */}
                <div className="p-3 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <Phone className="w-4 h-4 text-[var(--accent-swiss)]" />
                    <div>
                      <div className="text-[9px] text-[var(--text-tertiary)]">VOICE / TELEPHONY</div>
                      <a href={`tel:${profile.phone.replace(/\s+/g, '')}`} className="font-semibold text-[var(--text-primary)] hover:text-[var(--accent-swiss)]">
                        {profile.phone}
                      </a>
                    </div>
                  </div>
                  <button 
                    onClick={copyPhone}
                    className="p-1 hover:text-[var(--accent-swiss)] transition-colors" 
                    title="Copy Phone"
                  >
                    {copiedPhone ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>

                {/* Swiss Physical Address */}
                <div className="p-3 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-start gap-2.5">
                  <MapPin className="w-4 h-4 text-[var(--accent-swiss)] mt-0.5 shrink-0" />
                  <div className="leading-snug">
                    <div className="text-[9px] text-[var(--text-tertiary)]">PHYSICAL HEADQUARTERS</div>
                    <div className="font-bold text-[var(--text-primary)]">{profile.address.entity}</div>
                    <div className="text-[var(--text-secondary)]">{profile.address.street}</div>
                    <div className="text-[var(--text-secondary)]">{profile.address.postalCode} {profile.address.city}, {profile.address.country}</div>
                  </div>
                </div>
              </div>

              {/* Fast Action Buttons */}
              <div className="grid grid-cols-2 gap-2 font-swiss-mono text-xs pt-2">
                <button
                  onClick={() => downloadVCard()}
                  className="py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--accent-swiss)] hover:text-[var(--accent-swiss)] bg-[var(--bg-surface)] flex items-center justify-center gap-1.5 font-medium transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>EXPORT vCARD</span>
                </button>
                <a
                  href={profile.social.calendly}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="py-2.5 px-3 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] flex items-center justify-center gap-1.5 font-medium transition-colors"
                >
                  <Calendar className="w-3.5 h-3.5" />
                  <span>SCHEDULE CALL</span>
                </a>
              </div>
            </div>

            {/* Swiss Impressum Accordion / Modal trigger */}
            <div className="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-5 space-y-3 font-swiss-mono text-xs">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[var(--text-primary)] font-bold">
                  <Shield className="w-3.5 h-3.5 text-[var(--accent-swiss)]" />
                  <span>SWISS IMPRESSUM & LEGAL DISCLOSURES</span>
                </div>
                <button
                  onClick={() => setShowImpressumModal(!showImpressumModal)}
                  className="text-[10px] text-[var(--accent-swiss)] hover:underline"
                >
                  {showImpressumModal ? '[ HIDE ]' : '[ VIEW SPECS ]'}
                </button>
              </div>

              {showImpressumModal && (
                <div className="space-y-3 pt-3 border-t border-[var(--border-hairline)] text-[11px] text-[var(--text-secondary)] font-swiss-sans leading-relaxed animate-in fade-in duration-150">
                  <div>
                    <strong className="font-swiss-mono text-[var(--text-primary)]">Service Provider:</strong><br />
                    Zambounis Technology · Orestis Zambounis<br />
                    Av. Eugène-Rambert 30, 1005 Lausanne, Switzerland<br />
                    Email: info@orestis.ch
                  </div>
                  <div>
                    <strong className="font-swiss-mono text-[var(--text-primary)]">Qualifications:</strong><br />
                    ETH Zurich alumnus (Robotics, Systems & Control). Currently Senior ML Engineer at Red Hat (Model Optimization).
                  </div>
                  <div>
                    <strong className="font-swiss-mono text-[var(--text-primary)]">Dispute Resolution:</strong><br />
                    For EU consumers: <a href="https://ec.europa.eu/consumers/odr/" target="_blank" rel="noopener noreferrer" className="underline">ec.europa.eu/consumers/odr/</a>.<br />
                    As a Swiss-based entity, proceedings are governed under Swiss jurisdiction.
                  </div>
                  <div>
                    <strong className="font-swiss-mono text-[var(--text-primary)]">Liability for Contents:</strong><br />
                    Liable for own content under Swiss law. External links vetted at compilation time.
                  </div>
                </div>
              )}
            </div>

          </div>

        </div>
      </div>
    </section>
  );
};
