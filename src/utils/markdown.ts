import { marked } from 'marked';
import katex from 'katex';

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

  // 3. Parse inline math: $...$ (avoiding dollar signs next to numbers)
  processed = processed.replace(/(?<!\\)\$([^\$\n]+?)\$/g, (_, rawFormula) => {
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
