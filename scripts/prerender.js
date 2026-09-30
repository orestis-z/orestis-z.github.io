import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { marked } from 'marked';
import katex from 'katex';

// Configure custom link renderer
const renderer = {
  link({ href, title, text }) {
    let cleanHref = href || '#';
    let target = '';
    let rel = '';

    if (cleanHref.startsWith('http://') || cleanHref.startsWith('https://')) {
      target = ' target="_blank"';
      rel = ' rel="noopener noreferrer"';
    } else if (cleanHref.startsWith('/blog/')) {
      cleanHref = '#' + cleanHref.replace('/blog/', '').replace('.html', '');
    } else if (cleanHref === '/shop-automation/' || cleanHref === '/shop-automation') {
      cleanHref = '#systems';
    }

    const titleAttr = title ? ` title="${title}"` : '';
    return `<a href="${cleanHref}"${titleAttr}${target}${rel}>${text}</a>`;
  }
};

marked.use({ renderer });
marked.setOptions({ gfm: true, breaks: true });

function renderMarkdownWithMath(content) {
  if (!content) return '';

  const codeBlocks = [];
  let processed = content.replace(/(```[\s\S]*?```|`[^`]+`)/g, (match) => {
    codeBlocks.push(match);
    return `@@CODE_BLOCK_${codeBlocks.length - 1}@@`;
  });

  const eqMap = {
    'eq:multi-neurons': '(2)',
    'eq:momentum': '(3)',
    'eq:momentum-update': '(4)',
    'eq:adam-bias-correction': '(16)',
    'eq:large-f': '(7)',
    'eq:small-k1': '(8)',
    'eq:large-k1': '(9)',
    'eq:small-document-penalty': '(10)',
    'eq:large-document-penalty': '(11)',
    'eq:large-k1-and-avg-length': '(12)',
  };

  processed = processed.replace(/(?:\$\s*)?\\eqref\{([^}]+)\}(?:\s*\$)?/g, (_, eqId) => {
    const num = eqMap[eqId] || (eqId.startsWith('eq:') ? `(${eqId.slice(3)})` : `(${eqId})`);
    return `<span class="font-swiss-mono font-semibold text-[var(--accent-swiss)]">${num}</span>`;
  });

  processed = processed.replace(/\\label\{[^}]*\}/g, '');

  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, rawFormula) => {
    try {
      const formula = rawFormula.replace(/\\label\{[^}]*\}/g, '').trim();
      const rendered = katex.renderToString(formula, { displayMode: true, throwOnError: false });
      return `<div class="katex-block my-6 overflow-x-auto py-2">${rendered}</div>`;
    } catch {
      return `<code>${rawFormula}</code>`;
    }
  });

  processed = processed.replace(/(?<!\\)\$([^\s\$](?:[^\$\n]*?[^\s\$])?)\$/g, (_, rawFormula) => {
    try {
      const formula = rawFormula.replace(/\\label\{[^}]*\}/g, '').trim();
      const rendered = katex.renderToString(formula, { displayMode: false, throwOnError: false });
      return `<span class="katex-inline inline-block px-1">${rendered}</span>`;
    } catch {
      return `<code>${rawFormula}</code>`;
    }
  });

  processed = processed.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, index) => {
    return codeBlocks[parseInt(index, 10)] || '';
  });

  return marked.parse(processed);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const distDir = path.resolve(rootDir, 'dist');

// Read site data and dist template
const siteData = JSON.parse(fs.readFileSync(path.resolve(rootDir, 'src/data/siteData.json'), 'utf8'));
const baseIndexHtml = fs.readFileSync(path.resolve(distDir, 'index.html'), 'utf8');

// Find built CSS and JS assets from baseIndexHtml
const cssMatch = baseIndexHtml.match(/<link[^>]+href="([^"]+\.css)"[^>]*>/);
const jsMatch = baseIndexHtml.match(/<script[^>]+src="([^"]+\.js)"[^>]*><\/script>/);

const cssHref = cssMatch ? cssMatch[1] : '';
const jsSrc = jsMatch ? jsMatch[1] : '';

console.log('Detected built assets:');
console.log('  CSS:', cssHref);
console.log('  JS: ', jsSrc);

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

// Generate Blog Pages
siteData.posts.forEach((post) => {
  console.log(`Prerendering blog article: ${post.slug}...`);
  const renderedContent = renderMarkdownWithMath(post.content);

  const title = `${post.title} — Orestis Zambounis`;
  const description = post.description || 'Technical publication by Orestis Zambounis, Senior ML Engineer and Systems Architect.';
  const canonicalUrl = `https://orestis.ch/blog/${post.slug}`;
  const imageUrl = post.image ? `https://orestis.ch${post.image}` : 'https://orestis.ch/assets/images/portrait.jpg';

  const schemaJson = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: post.title,
    description: description,
    author: {
      '@type': 'Person',
      name: 'Orestis Zambounis',
      url: 'https://orestis.ch',
      jobTitle: 'Senior ML Engineer',
      worksFor: {
        '@type': 'Organization',
        name: 'Red Hat'
      }
    },
    datePublished: post.date,
    image: imageUrl,
    url: canonicalUrl
  });

  const pageHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <meta name="description" content="${description.replace(/"/g, '&quot;')}" />
    <meta name="keywords" content="${post.tags.join(', ')}" />
    <meta name="author" content="Orestis Zambounis" />
    <link rel="canonical" href="${canonicalUrl}" />

    <!-- OpenGraph Metadata -->
    <meta property="og:type" content="article" />
    <meta property="og:title" content="${post.title.replace(/"/g, '&quot;')}" />
    <meta property="og:description" content="${description.replace(/"/g, '&quot;')}" />
    <meta property="og:url" content="${canonicalUrl}" />
    <meta property="og:image" content="${imageUrl}" />
    <meta property="article:published_time" content="${post.date}" />
    <meta property="article:author" content="Orestis Zambounis" />

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${post.title.replace(/"/g, '&quot;')}" />
    <meta name="twitter:description" content="${description.replace(/"/g, '&quot;')}" />
    <meta name="twitter:image" content="${imageUrl}" />

    <!-- Structured Data -->
    <script type="application/ld+json">
      ${schemaJson}
    </script>

    <!-- Preload Fonts & CSS -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    ${cssHref ? `<link rel="stylesheet" crossorigin href="${cssHref}">` : ''}
  </head>
  <body class="bg-[#fafafa] dark:bg-[#090a0d] text-[#0d0f12] dark:text-[#f2f4f7]">
    <div id="root">
      <div class="min-h-screen bg-[var(--bg-canvas)] text-[var(--text-primary)]">
        <!-- Reader Navigation Bar -->
        <header class="sticky top-0 z-50 hairline-b bg-[var(--bg-canvas)]/95 backdrop-blur-md px-4 sm:px-6 py-3 border-b border-[var(--border-hairline)]">
          <div class="max-w-4xl mx-auto flex items-center justify-between font-swiss-mono text-xs">
            <a href="/" class="flex items-center gap-2 px-3 py-1.5 border border-[var(--border-hairline)] hover:border-[var(--border-strong)] bg-[var(--bg-surface)] text-[var(--text-primary)] transition-colors">
              <span>← RETURN TO DOSSIER</span>
            </a>
            <div class="text-[var(--accent-swiss)] font-bold">
              <span>DOC // ${post.slug.toUpperCase()}</span>
            </div>
            <a href="https://github.com/vllm-project/speculators" target="_blank" rel="noopener noreferrer" class="text-[var(--accent-swiss)] hover:underline hidden sm:inline font-bold">
              [ SPECULATORS ↗ ]
            </a>
          </div>
        </header>

        <!-- Main Article Container -->
        <main class="max-w-4xl mx-auto px-4 sm:px-6 py-10 sm:py-16">
          <article class="space-y-8">
            <header class="space-y-4 border-b border-[var(--border-hairline)] pb-8">
              <div class="flex flex-wrap items-center gap-2 font-swiss-mono text-xs">
                <span class="px-2 py-0.5 bg-[var(--accent-swiss-subtle)] text-[var(--accent-swiss)] font-bold border border-[var(--accent-swiss)]/20">
                  TECHNICAL PUBLICATION
                </span>
                ${post.categories.map((c) => `<span class="px-2 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]">${c}</span>`).join('')}
              </div>

              <h1 class="font-swiss-sans font-black text-3xl sm:text-5xl tracking-tight text-[var(--text-primary)] leading-[1.15]">
                ${post.title}
              </h1>

              ${post.description ? `
              <p class="font-swiss-mono text-sm sm:text-base text-[var(--text-secondary)] leading-relaxed pt-2">
                ${post.description}
              </p>` : ''}

              <div class="pt-4 flex flex-wrap items-center gap-4 sm:gap-6 font-swiss-mono text-xs text-[var(--text-tertiary)] border-t border-[var(--border-hairline)]">
                <span>PUBLISHED: ${post.date}</span>
                <span>·</span>
                <span>LENGTH: ${post.readTime} (${post.wordCount} WORDS)</span>
                <span>·</span>
                <span>AUTHOR: ${post.author.toUpperCase()}</span>
              </div>
            </header>

            ${post.image ? `
            <div class="border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 my-6">
              <img src="${post.image}" alt="${post.title}" class="w-full max-h-[460px] object-cover" />
              <div class="mt-2 px-2 py-1 bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[10px] font-swiss-mono text-[var(--text-tertiary)] flex justify-between">
                <span>FIGURE 01 // OVERVIEW GRAPHIC</span>
                <span>STATUS: DOCUMENTED</span>
              </div>
            </div>` : ''}

            <!-- Full Prerendered Prose -->
            <div class="prose-swiss max-w-none text-base sm:text-lg font-swiss-sans pt-4">
              ${renderedContent}
            </div>

            <!-- Footer Tags & Return -->
            <footer class="pt-10 border-t border-[var(--border-hairline)] space-y-6">
              <div class="space-y-2">
                <div class="text-xs font-swiss-mono text-[var(--text-tertiary)] uppercase">TAXONOMY / TAGS</div>
                <div class="flex flex-wrap gap-1.5 font-swiss-mono text-xs">
                  ${post.tags.map((t) => `<span class="px-2 py-1 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]">#${t}</span>`).join('')}
                </div>
              </div>
              <div class="hairline-t pt-6 flex justify-between items-center font-swiss-mono text-xs border-t border-[var(--border-hairline)]">
                <a href="/" class="px-4 py-2.5 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] font-bold transition-colors">
                  ← RETURN TO ARCHIVE
                </a>
              </div>
            </footer>
          </article>
        </main>
      </div>
    </div>
    ${jsSrc ? `<script type="module" crossorigin src="${jsSrc}"></script>` : ''}
  </body>
</html>`;

  // Write both /blog/slug/index.html and /blog/slug.html for complete compatibility
  const blogSubDir = path.resolve(distDir, 'blog', post.slug);
  ensureDir(blogSubDir);
  fs.writeFileSync(path.resolve(blogSubDir, 'index.html'), pageHtml, 'utf8');
  fs.writeFileSync(path.resolve(distDir, 'blog', `${post.slug}.html`), pageHtml, 'utf8');
});

// Generate Portfolio Page (/portfolio/ and /portfolio.html)
console.log('Prerendering /portfolio/...');
const portfolioDir = path.resolve(distDir, 'portfolio');
ensureDir(portfolioDir);

const portfolioHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Selected Engineering Works & Architecture Portfolio — Orestis Zambounis</title>
    <meta name="description" content="Portfolio of 15 engineered systems spanning LLM speculative decoding, distributed inference, edge automation, computer vision, and cryptographic infrastructure." />
    <link rel="canonical" href="https://orestis.ch/portfolio/" />
    <meta property="og:title" content="Engineering Works Portfolio — Orestis Zambounis" />
    <meta property="og:description" content="Portfolio of 15 engineered systems spanning LLM speculative decoding, distributed inference, edge automation, and robotics." />
    <meta property="og:url" content="https://orestis.ch/portfolio/" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    ${cssHref ? `<link rel="stylesheet" crossorigin href="${cssHref}">` : ''}
  </head>
  <body class="bg-[#fafafa] dark:bg-[#090a0d] text-[#0d0f12] dark:text-[#f2f4f7]">
    <div id="root">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 py-12">
        <header class="mb-10 pb-6 border-b border-[var(--border-hairline)]">
          <a href="/" class="font-swiss-mono text-xs text-[var(--accent-swiss)] font-bold hover:underline">← ORESTIS ZAMBOUNIS // MAIN DOSSIER</a>
          <h1 class="text-3xl sm:text-5xl font-black font-swiss-sans tracking-tight mt-4">SELECTED SYSTEMS MATRIX [15]</h1>
          <p class="font-swiss-mono text-sm text-[var(--text-secondary)] mt-2">INDEX OF ARCHITECTED PRODUCTION & RESEARCH SYSTEMS</p>
        </header>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 font-swiss-sans">
          ${siteData.projects.map((p) => `
            <div class="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 space-y-4">
              <div class="font-swiss-mono text-xs text-[var(--accent-swiss)] font-bold">[ ${p.id} // ${p.year} ]</div>
              <h2 class="text-xl font-bold text-[var(--text-primary)]">${p.name}</h2>
              <p class="text-xs font-swiss-mono text-[var(--text-tertiary)] uppercase">${p.categoryName || ''}</p>
              <p class="text-sm text-[var(--text-secondary)]">${p.description}</p>
              <div class="flex flex-wrap gap-1.5 pt-2">
                ${p.tags.map((t) => `<span class="px-1.5 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-swiss-mono text-[10px] text-[var(--text-secondary)]">${t}</span>`).join('')}
              </div>
              <div class="pt-4 border-t border-[var(--border-hairline)] flex justify-between font-swiss-mono text-xs">
                ${p.url ? `<a href="${p.url}" class="text-[var(--accent-swiss)] hover:underline font-bold">${p.buttonLabel || 'VIEW'} ↗</a>` : '<span></span>'}
                <a href="/#${p.id.toLowerCase()}" class="text-[var(--text-primary)] hover:underline">SPECIFICATION →</a>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
    ${jsSrc ? `<script type="module" crossorigin src="${jsSrc}"></script>` : ''}
  </body>
</html>`;

fs.writeFileSync(path.resolve(portfolioDir, 'index.html'), portfolioHtml, 'utf8');
fs.writeFileSync(path.resolve(distDir, 'portfolio.html'), portfolioHtml, 'utf8');

// Generate Shop Automation Page (/shop-automation/ and /shop-automation.html)
console.log('Prerendering /shop-automation/...');
const shopDir = path.resolve(distDir, 'shop-automation');
ensureDir(shopDir);

const shopHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Autonomous Retail & Edge Architecture — Beachin' Rentals — Orestis Zambounis</title>
    <meta name="description" content="Complete edge-to-cloud hardware schematic and autonomous retail system deployed in Barcelona. Raspberry Pi, RS485 lock controllers, kiosk touchscreen, and Shopify integration." />
    <link rel="canonical" href="https://orestis.ch/shop-automation/" />
    <meta property="og:title" content="Autonomous Retail Architecture — Beachin' Rentals" />
    <meta property="og:description" content="Raspberry Pi, RS485 lock controllers, self-service kiosk UI, and real-time Shopify synchronization." />
    <meta property="og:url" content="https://orestis.ch/shop-automation/" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    ${cssHref ? `<link rel="stylesheet" crossorigin href="${cssHref}">` : ''}
  </head>
  <body class="bg-[#fafafa] dark:bg-[#090a0d] text-[#0d0f12] dark:text-[#f2f4f7]">
    <div id="root">
      <div class="max-w-5xl mx-auto px-4 sm:px-6 py-12 space-y-8 font-swiss-sans">
        <header class="border-b border-[var(--border-hairline)] pb-6">
          <a href="/" class="font-swiss-mono text-xs text-[var(--accent-swiss)] font-bold hover:underline">← ORESTIS ZAMBOUNIS // RETURN TO DOSSIER</a>
          <h1 class="text-3xl sm:text-5xl font-black tracking-tight mt-4">AUTONOMOUS STORE HARDWARE & EDGE ARCHITECTURE</h1>
          <p class="font-swiss-mono text-sm text-[var(--text-secondary)] mt-2">COMMISSIONED FOR BEACHIN' RENTALS // BARCELONA, SPAIN</p>
        </header>
        <div class="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 space-y-6">
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 font-swiss-mono text-xs border-b border-[var(--border-hairline)] pb-4">
            <div><span class="text-[var(--text-tertiary)] block">HARDWARE NODES</span> 32× RS485 LOCKS + PI 4</div>
            <div><span class="text-[var(--text-tertiary)] block">UPTIME</span> 99.98% CONTINUOUS</div>
            <div><span class="text-[var(--text-tertiary)] block">CLOUD SYNC</span> SHOPIFY WEBHOOK PIPELINE</div>
          </div>
          <div class="space-y-4 text-base text-[var(--text-secondary)] leading-relaxed">
            <p>Designed and built an end-to-end autonomous beachfront rental retail facility requiring zero on-site staff. Customers select equipment (umbrellas, beach loungers, paddleboards) through a custom touchscreen kiosk, pay via integrated Nayax POS, and receive automated locker/bay release triggered via RS485 relays.</p>
            <p>The system features offline resilience, automated safety interlocks, and bidirectional inventory synchronization with Shopify.</p>
          </div>
          <div class="pt-4 border-t border-[var(--border-hairline)] flex gap-4 font-swiss-mono text-xs">
            <a href="/blog/automating-beach-rental-store" class="px-4 py-2 border border-[var(--accent-swiss)] bg-[var(--accent-swiss)] text-white font-bold hover:opacity-90">READ CASE STUDY & DEEP DIVE →</a>
            <a href="/#systems" class="px-4 py-2 border border-[var(--border-hairline)] hover:border-[var(--text-primary)]">VIEW INTERACTIVE SCHEMATIC</a>
          </div>
        </div>
      </div>
    </div>
    ${jsSrc ? `<script type="module" crossorigin src="${jsSrc}"></script>` : ''}
  </body>
</html>`;

fs.writeFileSync(path.resolve(shopDir, 'index.html'), shopHtml, 'utf8');
fs.writeFileSync(path.resolve(distDir, 'shop-automation.html'), shopHtml, 'utf8');

// Generate Impressum Page (/impressum/ and /impressum.html)
console.log('Prerendering /impressum/...');
const impressumDir = path.resolve(distDir, 'impressum');
ensureDir(impressumDir);

const impressumHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Impressum & Legal Telemetry — Orestis Zambounis</title>
    <meta name="description" content="Legal telemetry, contact headquarters, and impressum for Orestis Zambounis, Lausanne, Switzerland." />
    <link rel="canonical" href="https://orestis.ch/impressum/" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    ${cssHref ? `<link rel="stylesheet" crossorigin href="${cssHref}">` : ''}
  </head>
  <body class="bg-[#fafafa] dark:bg-[#090a0d] text-[#0d0f12] dark:text-[#f2f4f7]">
    <div id="root">
      <div class="max-w-3xl mx-auto px-4 sm:px-6 py-16 space-y-8 font-swiss-sans">
        <header class="border-b border-[var(--border-hairline)] pb-6">
          <a href="/" class="font-swiss-mono text-xs text-[var(--accent-swiss)] font-bold hover:underline">← RETURN TO DOSSIER</a>
          <h1 class="text-3xl sm:text-4xl font-black tracking-tight mt-4">IMPRESSUM // LEGAL NOTICE</h1>
          <p class="font-swiss-mono text-xs text-[var(--text-tertiary)] uppercase mt-2">SWISS JURISDICTION TELEMETRY & CONTACT</p>
        </header>
        <div class="border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-8 space-y-6 font-swiss-mono text-xs leading-relaxed">
          <div>
            <div class="text-[var(--text-tertiary)] uppercase text-[10px] mb-1">OPERATOR & RESPONSIBLE ENTITY</div>
            <div class="text-base font-bold text-[var(--text-primary)]">Orestis Zambounis</div>
            <div class="text-[var(--text-secondary)]">Senior Machine Learning Engineer & Systems Architect</div>
          </div>
          <div>
            <div class="text-[var(--text-tertiary)] uppercase text-[10px] mb-1">PHYSICAL HEADQUARTERS</div>
            <div class="text-[var(--text-secondary)]">Av. Eugène-Rambert 30</div>
            <div class="text-[var(--text-secondary)]">1005 Lausanne, Switzerland</div>
            <div class="text-[var(--text-tertiary)]">COORDINATES: 46°31'N 6°38'E // ELEVATION: 495M</div>
          </div>
          <div>
            <div class="text-[var(--text-tertiary)] uppercase text-[10px] mb-1">TRANSMISSION CHANNELS</div>
            <div>Email: <a href="mailto:info@orestis.ch" class="text-[var(--accent-swiss)] hover:underline">info@orestis.ch</a></div>
            <div>PGP Key ID: <span class="font-bold">0x9D4C2F8A1B7E3E50</span></div>
          </div>
          <div class="pt-4 border-t border-[var(--border-hairline)] text-[var(--text-tertiary)]">
            This portfolio serves as a technical dossier and knowledge repository. All code references, publications, and hardware designs are authored or maintained by Orestis Zambounis unless otherwise cited.
          </div>
        </div>
      </div>
    </div>
    ${jsSrc ? `<script type="module" crossorigin src="${jsSrc}"></script>` : ''}
  </body>
</html>`;

fs.writeFileSync(path.resolve(impressumDir, 'index.html'), impressumHtml, 'utf8');
fs.writeFileSync(path.resolve(distDir, 'impressum.html'), impressumHtml, 'utf8');

// Generate Smart 404 Page (dist/404.html)
console.log('Generating smart 404.html with intelligent routing...');
const notFoundHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Page Not Found // 404 — Orestis Zambounis</title>
    <script>
      (function() {
        var path = window.location.pathname.toLowerCase();
        // Intelligent route mapping for legacy indexed pages
        if (path.indexOf('adam-w') !== -1) {
          window.location.replace('/blog/adam-w');
        } else if (path.indexOf('rental') !== -1 || path.indexOf('automating') !== -1) {
          window.location.replace('/blog/automating-beach-rental-store');
        } else if (path.indexOf('backprop') !== -1) {
          window.location.replace('/blog/backpropagation');
        } else if (path.indexOf('quantiz') !== -1 || path.indexOf('pruning') !== -1) {
          window.location.replace('/blog/quantization-vs-pruning');
        } else if (path.indexOf('izymaps') !== -1) {
          window.location.replace('/blog/izymaps-shopify-store-locator');
        } else if (path.indexOf('ranking') !== -1 || path.indexOf('retrieval') !== -1) {
          window.location.replace('/blog/information-retrieval-and-ranking');
        } else if (path.indexOf('portfolio') !== -1) {
          window.location.replace('/portfolio/');
        } else if (path.indexOf('shop-automation') !== -1) {
          window.location.replace('/shop-automation/');
        } else if (path.indexOf('impressum') !== -1) {
          window.location.replace('/impressum/');
        }
      })();
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    ${cssHref ? `<link rel="stylesheet" crossorigin href="${cssHref}">` : ''}
  </head>
  <body class="bg-[#fafafa] dark:bg-[#090a0d] text-[#0d0f12] dark:text-[#f2f4f7] min-h-screen flex items-center justify-center p-6">
    <div class="max-w-lg w-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-8 space-y-6 font-swiss-mono">
      <div class="text-[var(--accent-swiss)] font-bold text-xs">[ ERROR // 404 NOT FOUND ]</div>
      <h1 class="text-4xl font-black font-swiss-sans tracking-tight text-[var(--text-primary)]">SIGNAL LOST</h1>
      <p class="text-xs text-[var(--text-secondary)] leading-relaxed">
        The requested transmission endpoint does not exist or has been restructured under the Swiss engineering architecture.
      </p>
      <div class="space-y-2 pt-2 border-t border-[var(--border-hairline)] text-xs">
        <div class="text-[var(--text-tertiary)] uppercase text-[10px]">RECOVERY DIRECTORY:</div>
        <div>• <a href="/" class="text-[var(--text-primary)] hover:text-[var(--accent-swiss)] underline font-bold">Main Dossier & Bio</a></div>
        <div>• <a href="/portfolio/" class="text-[var(--text-primary)] hover:text-[var(--accent-swiss)] underline">Selected Systems Portfolio</a></div>
        <div>• <a href="/blog/adam-w" class="text-[var(--text-primary)] hover:text-[var(--accent-swiss)] underline">The Evolution of AdamW</a></div>
        <div>• <a href="/shop-automation/" class="text-[var(--text-primary)] hover:text-[var(--accent-swiss)] underline">Shop Automation Architecture</a></div>
      </div>
      <div class="pt-4 border-t border-[var(--border-hairline)]">
        <a href="/" class="inline-block px-4 py-2 border border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--bg-canvas)] hover:bg-[var(--accent-swiss)] hover:border-[var(--accent-swiss)] font-bold text-xs transition-colors">
          RETURN TO DOSSIER →
        </a>
      </div>
    </div>
  </body>
</html>`;

fs.writeFileSync(path.resolve(distDir, '404.html'), notFoundHtml, 'utf8');

// Generate XML Sitemap (dist/sitemap.xml and public/sitemap.xml)
console.log('Generating sitemap.xml...');
const today = new Date().toISOString().split('T')[0];

const sitemapXml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://orestis.ch/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/adam-w</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/automating-beach-rental-store</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/backpropagation</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/quantization-vs-pruning</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/izymaps-shopify-store-locator</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/blog/information-retrieval-and-ranking</loc>
    <lastmod>2026-03-20</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://orestis.ch/portfolio/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://orestis.ch/shop-automation/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://orestis.ch/impressum/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>yearly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>`;

fs.writeFileSync(path.resolve(distDir, 'sitemap.xml'), sitemapXml, 'utf8');
fs.writeFileSync(path.resolve(rootDir, 'public/sitemap.xml'), sitemapXml, 'utf8');

console.log('✓ Prerendering and SEO generation completed successfully!');
