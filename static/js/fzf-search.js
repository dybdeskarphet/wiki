// Minimal, keyboard-driven full-text search modal (Telescope / fzf style)
// Supports multi-word matching across titles, tags, and full note contents with highlighted snippets.

(function () {
  let searchIndex = [];
  let selectedIndex = 0;
  let filteredResults = [];
  let currentTerms = [];

  const backdrop = document.getElementById('fzf-backdrop');
  const input = document.getElementById('fzf-input');
  const resultsList = document.getElementById('fzf-results');
  const triggerBtn = document.getElementById('search-trigger-btn');
  const countBadge = document.getElementById('fzf-count');

  async function loadIndex() {
    if (searchIndex.length > 0) return;
    try {
      const rootPrefix = document.body.dataset.root || '.';
      const response = await fetch(`${rootPrefix}/search-index.json`);
      if (response.ok) {
        searchIndex = await response.json();
      }
    } catch (e) {
      console.error('Failed to load search index:', e);
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function normalizeText(str) {
    if (!str) return '';
    return String(str)
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase();
  }

  function highlightTerms(text, terms) {
    if (!text) return '';
    const escaped = escapeHtml(text);
    if (!terms || terms.length === 0) return escaped;

    const validTerms = terms.filter(t => t && t.trim().length > 0);
    if (validTerms.length === 0) return escaped;

    const sortedTerms = [...new Set(validTerms)].sort((a, b) => b.length - a.length);
    const patterns = sortedTerms.map(t => {
      return escapeHtml(t).split('').map(c => {
        const lower = c.toLowerCase();
        if (lower === 'a') return '[aàáâãäå]';
        if (lower === 'e') return '[eèéêë]';
        if (lower === 'i') return '[iìíîï]';
        if (lower === 'o') return '[oòóôõö]';
        if (lower === 'u') return '[uùúûü]';
        if (lower === 'c') return '[cç]';
        if (lower === 'n') return '[nñ]';
        return c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      }).join('');
    });

    const regex = new RegExp(`(${patterns.join('|')})`, 'gi');
    return escaped.replace(regex, '<mark class="fzf-hl">$1</mark>');
  }

  function fuzzyMatch(query, text) {
    if (!query) return true;
    query = normalizeText(query);
    text = normalizeText(text);
    let qIdx = 0;
    for (let char of text) {
      if (char === query[qIdx]) {
        qIdx++;
        if (qIdx === query.length) return true;
      }
    }
    return false;
  }

  function extractSnippet(item, terms) {
    const content = item.content || item.description || '';
    if (!content) return '';

    const lowerNormContent = normalizeText(content);
    let bestPos = -1;

    for (const term of terms) {
      const termNorm = normalizeText(term);
      const pos = lowerNormContent.indexOf(termNorm);
      if (pos !== -1 && (bestPos === -1 || pos < bestPos)) {
        bestPos = pos;
      }
    }

    if (bestPos === -1) {
      const snippetRaw = content.slice(0, 110).trim();
      return highlightTerms(snippetRaw, terms) + (content.length > 110 ? '...' : '');
    }

    const start = Math.max(0, bestPos - 35);
    const end = Math.min(content.length, bestPos + 90);
    let snippet = content.slice(start, end).trim();

    if (start > 0) snippet = '...' + snippet;
    if (end < content.length) snippet = snippet + '...';

    return highlightTerms(snippet, terms);
  }

  function matchAndScore(item, terms, rawQuery) {
    const titleNorm = normalizeText(item.title);
    const tagsNorm = normalizeText((item.tags || []).join(' '));
    const contentNorm = normalizeText(item.content);
    const descNorm = normalizeText(item.description);

    // Check that ALL search terms match somewhere in the note
    for (const term of terms) {
      const termNorm = normalizeText(term);
      const inTitle = titleNorm.includes(termNorm);
      const inTags = tagsNorm.includes(termNorm);
      const inContent = contentNorm.includes(termNorm);
      const inDesc = descNorm.includes(termNorm);
      const inFuzzy = fuzzyMatch(termNorm, titleNorm);

      if (!inTitle && !inTags && !inContent && !inDesc && !inFuzzy) {
        return 0; // Term not found anywhere in this note
      }
    }

    let score = 10;
    const queryNorm = normalizeText(rawQuery);

    // Boost exact title match
    if (titleNorm === queryNorm) score += 300;
    else if (titleNorm.startsWith(queryNorm)) score += 200;
    else if (titleNorm.includes(queryNorm)) score += 150;

    // Boost exact content / tags match
    if (tagsNorm.includes(queryNorm)) score += 100;
    if (descNorm.includes(queryNorm)) score += 60;
    if (contentNorm.includes(queryNorm)) score += 40;

    // Term-by-term scoring
    for (const term of terms) {
      const termNorm = normalizeText(term);
      if (titleNorm.includes(termNorm)) score += 50;
      else if (fuzzyMatch(termNorm, titleNorm)) score += 20;

      if (tagsNorm.includes(termNorm)) score += 30;
      if (descNorm.includes(termNorm)) score += 20;

      if (contentNorm.includes(termNorm)) {
        score += 15;
        const count = contentNorm.split(termNorm).length - 1;
        score += Math.min(count, 5) * 2;
      }
    }

    return score;
  }

  function renderResults() {
    resultsList.innerHTML = '';
    if (filteredResults.length === 0) {
      resultsList.innerHTML = '<li style="padding: 1.5rem; color: #928374; text-align: center; font-size: 0.9em;">No matching notes found</li>';
      if (countBadge) countBadge.textContent = `0/${searchIndex.length}`;
      return;
    }

    if (countBadge) {
      countBadge.textContent = `${filteredResults.length}/${searchIndex.length}`;
    }

    const hasQuery = currentTerms.length > 0;

    filteredResults.forEach((result, index) => {
      const item = result.item;
      const li = document.createElement('li');
      li.className = `fzf-item ${index === selectedIndex ? 'selected' : ''}`;
      li.onclick = () => {
        const rootPrefix = document.body.dataset.root || '.';
        window.location.href = `${rootPrefix}/${item.url}`;
      };

      const titleHtml = hasQuery ? highlightTerms(item.title, currentTerms) : escapeHtml(item.title);
      
      const primaryTag = (item.tags && item.tags.length > 0) ? `#${item.tags[0]}` : '';
      const tagHtml = primaryTag ? `<span class="fzf-item-tag">${escapeHtml(primaryTag)}</span>` : '';

      let snippetHtml = '';
      if (hasQuery && result.snippet) {
        snippetHtml = `<div class="fzf-item-snippet">${result.snippet}</div>`;
      }

      li.innerHTML = `
        <div class="fzf-item-main">
          <div class="fzf-item-header">
            <span class="fzf-item-title">${titleHtml}</span>
            <div class="fzf-item-tags">${tagHtml}</div>
          </div>
          ${snippetHtml}
        </div>
      `;
      resultsList.appendChild(li);
    });

    const selectedEl = resultsList.children[selectedIndex];
    if (selectedEl) {
      selectedEl.scrollIntoView({ block: 'nearest' });
    }
  }

  function doSearch(query) {
    const rawQuery = query.trim();
    if (!rawQuery) {
      currentTerms = [];
      filteredResults = searchIndex.map(item => ({ item, snippet: '' }));
    } else {
      currentTerms = rawQuery.toLowerCase().split(/\s+/).filter(Boolean);
      const scored = [];

      for (const item of searchIndex) {
        const score = matchAndScore(item, currentTerms, rawQuery);
        if (score > 0) {
          const snippet = extractSnippet(item, currentTerms);
          scored.push({ item, score, snippet });
        }
      }

      scored.sort((a, b) => b.score - a.score);
      filteredResults = scored;
    }

    selectedIndex = 0;
    renderResults();
  }

  function openSearch() {
    loadIndex().then(() => {
      backdrop.classList.add('active');
      input.value = '';
      doSearch('');
      input.focus();
    });
  }

  function closeSearch() {
    backdrop.classList.remove('active');
  }

  document.addEventListener('keydown', (e) => {
    const isSearchOpen = backdrop.classList.contains('active');

    if (!isSearchOpen && (e.key === '/' || ((e.ctrlKey || e.metaKey) && e.key === 'k'))) {
      const target = e.target;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
      if (!isInput || e.key !== '/') {
        e.preventDefault();
        openSearch();
        return;
      }
    }

    if (isSearchOpen) {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeSearch();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (filteredResults.length > 0) {
          selectedIndex = (selectedIndex + 1) % filteredResults.length;
          renderResults();
        }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (filteredResults.length > 0) {
          selectedIndex = (selectedIndex - 1 + filteredResults.length) % filteredResults.length;
          renderResults();
        }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredResults[selectedIndex]) {
          const rootPrefix = document.body.dataset.root || '.';
          window.location.href = `${rootPrefix}/${filteredResults[selectedIndex].item.url}`;
        }
      }
    }
  });

  if (input) {
    input.addEventListener('input', (e) => {
      doSearch(e.target.value);
    });
  }

  if (backdrop) {
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) {
        closeSearch();
      }
    });
  }

  if (triggerBtn) {
    triggerBtn.addEventListener('click', openSearch);
  }
})();
