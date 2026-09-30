import { marked } from 'marked';
import katex from 'katex';

// Configure custom link renderer
const renderer = {
  link({ href, title, text }: { href: string; title?: string | null; text: string }) {
    let cleanHref = href || '#';
    let target = '';
    let rel = '';

    // Handle external links
    if (cleanHref.startsWith('http://') || cleanHref.startsWith('https://')) {
      target = ' target="_blank"';
      rel = ' rel="noopener noreferrer"';
    } 
    // Normalize internal legacy Jekyll blog links to SPA hash routes
    else if (cleanHref.startsWith('/blog/')) {
      cleanHref = '#' + cleanHref.replace('/blog/', '').replace('.html', '');
    } 
    // Normalize shop-automation link to section hash
    else if (cleanHref === '/shop-automation/' || cleanHref === '/shop-automation') {
      cleanHref = '#systems';
    }

    const titleAttr = title ? ` title="${title}"` : '';
    return `<a href="${cleanHref}"${titleAttr}${target}${rel}>${text}</a>`;
  }
};

marked.use({ renderer });

// Configure marked options
marked.setOptions({
  gfm: true,
  breaks: true,
});

export function renderMarkdownWithMath(content: string): string {
  if (!content) return '';

  // 1. Protect code blocks from math replacement
  const codeBlocks: string[] = [];
  let processed = content.replace(/(```[\s\S]*?```|`[^`]+`)/g, (match) => {
    codeBlocks.push(match);
    return `@@CODE_BLOCK_${codeBlocks.length - 1}@@`;
  });

  // 1.5. Map and resolve equation references (\eqref{...})
  const eqMap: Record<string, string> = {
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

  // Replace any \eqref{...} (with or without enclosing $) with clean equation badges
  processed = processed.replace(/(?:\$\s*)?\\eqref\{([^}]+)\}(?:\s*\$)?/g, (_, eqId) => {
    const num = eqMap[eqId] || (eqId.startsWith('eq:') ? `(${eqId.slice(3)})` : `(${eqId})`);
    return `<span class="font-swiss-mono font-semibold text-[var(--accent-swiss)]">${num}</span>`;
  });

  // Strip any remaining stray \label{...} in text or math
  processed = processed.replace(/\\label\{[^}]*\}/g, '');

  // 2. Parse display math: $$...$$
  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, rawFormula) => {
    try {
      // Clean LaTeX \label{...} which causes red error text in KaTeX and overlaps
      const formula = rawFormula
        .replace(/\\label\{[^}]*\}/g, '')
        .trim();

      const rendered = katex.renderToString(formula, {
        displayMode: true,
        throwOnError: false,
      });
      return `<div class="katex-block my-6 overflow-x-auto py-2">${rendered}</div>`;
    } catch {
      return `<code>${rawFormula}</code>`;
    }
  });

  // 3. Parse inline math: $...$ (requiring non-whitespace borders to prevent greedy captures)
  processed = processed.replace(/(?<!\\)\$([^\s\$](?:[^\$\n]*?[^\s\$])?)\$/g, (_, rawFormula) => {
    try {
      const formula = rawFormula
        .replace(/\\label\{[^}]*\}/g, '')
        .trim();

      const rendered = katex.renderToString(formula, {
        displayMode: false,
        throwOnError: false,
      });
      return `<span class="katex-inline inline-block px-1">${rendered}</span>`;
    } catch {
      return `<code>${rawFormula}</code>`;
    }
  });

  // 4. Restore code blocks
  processed = processed.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, index) => {
    return codeBlocks[parseInt(index, 10)] || '';
  });

  // 5. Parse markdown with marked
  const rawHtml = marked.parse(processed) as string;

  return rawHtml;
}
