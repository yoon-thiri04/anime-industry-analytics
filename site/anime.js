/* Anime Industry Analytics — shared JS. Load AFTER anime_data.js. */
const AD = (typeof ANIME_DATA !== 'undefined') ? ANIME_DATA : undefined;

const C = {ink:'#111114', accent:'#E4384C', muted:'#86868F', line:'#E5E5E1', wash:'#F6F6F4',
           blue:'#3B6FE0', green:'#12A05C', amber:'#EE9B00', violet:'#7C4DEB',
           cyan:'#0E9BC4', pink:'#D6336C', orange:'#E8590C', slate:'#3B6FE0', ochre:'#EE9B00'};
/* categorical ramp — distinguishable in sequence and legible on white */
const SERIES = ['#3B6FE0','#E4384C','#12A05C','#EE9B00','#7C4DEB',
                '#0E9BC4','#E8590C','#D6336C','#0B7285','#6B7280'];
const PLOTCFG = {displayModeBar:false, responsive:true};
const LAYOUT_BASE = {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'Inter,sans-serif', color:'#111114', size:11},
  margin:{l:48,r:12,t:10,b:38},
  xaxis:{gridcolor:'#F0F0EC', zerolinecolor:'#E5E5E1', linecolor:'#E5E5E1'},
  yaxis:{gridcolor:'#F0F0EC', zerolinecolor:'#E5E5E1', linecolor:'#E5E5E1'},
  hoverlabel:{bgcolor:'#111114', bordercolor:'#111114', font:{color:'#fff', family:'Inter,sans-serif', size:11}}};
/* Plotly writes computed ranges back into the layout object it is given, and
   `{...LAYOUT}` only copies the top level — so a single shared xaxis object was
   being overwritten by whichever chart drew last, blanking the earlier ones.
   Every read of LAYOUT now yields its own deep copy. */
Object.defineProperty(window, 'LAYOUT', {
  get: () => (window.structuredClone ? structuredClone(LAYOUT_BASE) : JSON.parse(JSON.stringify(LAYOUT_BASE)))
});

/* Plotly's responsive:true only reacts to WINDOW resize, but these charts sit in
   grid cards that reflow when a neighbour changes. Keep each chart the size of its box. */
(function () {
  if (!('ResizeObserver' in window)) return;
  const seen = new WeakMap();
  const ro = new ResizeObserver(es => { for (const e of es) {
    const d = e.target;
    if (!d._fullLayout || !window.Plotly) continue;
    const w = Math.round(e.contentRect.width), h = Math.round(e.contentRect.height), p = seen.get(d);
    if (p && p.w === w && p.h === h) continue;
    seen.set(d, {w, h});
    requestAnimationFrame(() => { try { Plotly.Plots.resize(d); } catch (_) {} });
  }});
  const attach = () => document.querySelectorAll('.js-plotly-plot').forEach(d => {
    if (!seen.has(d)) { seen.set(d, {w:0, h:0}); ro.observe(d); }
  });
  addEventListener('load', () => { attach(); setTimeout(attach, 500); setTimeout(attach, 1600); });
  setTimeout(attach, 900);
})();

const AICON = {
 search:'<circle cx="11" cy="11" r="7"/><line x1="20" y1="20" x2="16" y2="16"/>',
 reset:'<polyline points="1 4 1 10 7 10"/><path d="M3.5 15a9 9 0 1 0 2.1-9.4L1 10"/>',
 x:'<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
 arrow:'<line x1="4" y1="12" x2="20" y2="12"/><polyline points="14 6 20 12 14 18"/>',
 flask:'<path d="M9 3h6"/><path d="M10 3v6.5L4.6 18a2 2 0 0 0 1.7 3h11.4a2 2 0 0 0 1.7-3L14 9.5V3"/>',
 layers:'<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/>'};
const icn = (n, s = 14) => `<svg class="ic" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${AICON[n]}</svg>`;

const fmtInt = n => (n == null ? '—' : Math.round(n).toLocaleString());
const fmtK = n => {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
  return String(Math.round(n));
};
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const FALLBACK_ART = 'data:image/svg+xml;utf8,' + encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 300">
   <rect width="200" height="300" fill="#F6F6F4"/>
   <rect x="78" y="128" width="44" height="44" fill="none" stroke="#D2D2CC" stroke-width="2"/></svg>`);
const artOf = url => url || FALLBACK_ART;
const isHigh = b => b >= ((AD && AD.BSCORE_BINS) ? AD.BSCORE_BINS.hi_cut : .75);

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({error:'Bad response from the server'}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}
const apiGet = (path, params) => {
  const qs = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (Array.isArray(v)) v.forEach(x => qs.append(k, x));
    else if (v !== '' && v != null) qs.set(k, v);
  });
  return api(path + (qs.toString() ? '?' + qs : ''));
};
const apiPost = (path, body) => api(path, {method:'POST',
  headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});

function posterCard(a, opts = {}) {
  return `<button class="pcard" data-id="${a.mal_id}" aria-label="${esc(a.title)}">
    <div class="shot">
      <img loading="lazy" src="${esc(artOf(a.image))}" alt="" onerror="this.src='${FALLBACK_ART}'">
      ${opts.rank ? `<span class="rk">${String(opts.rank).padStart(2, '0')}</span>` : ''}
      <span class="bs ${isHigh(a.balanced) ? 'hi' : ''}">${a.balanced != null ? a.balanced.toFixed(3) : '—'}</span>
    </div>
    <h4>${esc(a.title)}</h4>
    <div class="mt">${a.year || '—'} · ${esc(a.type || '')}${a.score ? ' · ' + a.score.toFixed(2) : ''}</div>
    ${opts.extra || ''}
  </button>`;
}

/* ── title detail modal ── */
function ensureModal() {
  if (document.getElementById('animeModal')) return;
  const el = document.createElement('div');
  el.className = 'modal'; el.id = 'animeModal';
  el.innerHTML = `<div class="scrim" data-close></div>
    <div class="sheet" role="dialog" aria-modal="true">
      <button class="close" data-close aria-label="Close">${icn('x', 15)}</button>
      <div id="modalBody"></div></div>`;
  document.body.appendChild(el);
  el.addEventListener('click', e => { if (e.target.closest('[data-close]')) closeModal(); });
  addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}
function closeModal() {
  const m = document.getElementById('animeModal');
  if (m) { m.classList.remove('on'); document.body.style.overflow = ''; }
}
async function openAnime(malId) {
  ensureModal();
  const m = document.getElementById('animeModal'), body = document.getElementById('modalBody');
  body.innerHTML = `<div style="padding:1.6rem"><div class="skel" style="height:220px"></div></div>`;
  m.classList.add('on'); document.body.style.overflow = 'hidden';
  let a;
  try { a = await apiGet(`/api/anime/${malId}`); }
  catch (err) { body.innerHTML = `<div class="empty">${esc(err.message)}</div>`; return; }
  const tags = l => (l || []).map(x => `<span class="tag">${esc(x)}</span>`).join(' ');
  const stat = (l, v) => `<div><b>${v}</b><span>${l}</span></div>`;
  /* only 6,798 of the catalogue carry a long enough synopsis to be in the similarity
     index — offering the button on the rest just lands on an error page */
  const simBtn = a.recommendable
    ? `<a class="btn" href="recommend.html?mal_id=${a.mal_id}">Find similar titles ${icn('arrow', 14)}</a>`
    : `<button class="btn" disabled title="Not in the similarity index">Not in similarity index</button>`;
  body.innerHTML = `
    <div class="detail">
      <div class="poster">
        <img src="${esc(artOf(a.image))}" alt="" onerror="this.src='${FALLBACK_ART}'"
             style="width:100%;border-radius:3px;display:block">
        <div class="acts">
          ${simBtn}
          ${a.url ? `<a class="btn ghost" href="${esc(a.url)}" target="_blank" rel="noopener">MyAnimeList ↗</a>` : ''}
        </div>
      </div>
      <div style="min-width:0">
        <div class="lab">${esc(a.type)} · ${a.year || '—'}${a.season ? ' · ' + esc(a.season) : ''}</div>
        <h2 style="font-size:1.35rem;margin:.4rem 0 .2rem">${esc(a.title)}</h2>
        ${a.english && a.english !== a.title
          ? `<div class="sub">${esc(a.english)}</div>` : ''}
        <div class="stats">
          ${stat('balanced', a.balanced.toFixed(3))}
          ${stat('MAL score', a.score != null ? a.score.toFixed(2) : '—')}
          ${stat('members', fmtK(a.members))}
          ${stat('favorites', fmtK(a.favorites))}
          ${stat('episodes', a.episodes != null ? fmtInt(a.episodes) : '—')}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:.25rem">${tags(a.genres)}${tags(a.themes)}</div>
        <div class="sub" style="margin-top:.8rem;font-family:var(--mono);font-size:.68rem">
          ${esc((a.studios || []).join(', ') || 'Unknown studio')} · ${esc(a.source || '—')}
          ${a.nlp_cluster != null ? ` · <a href="storylines.html">cluster #${a.nlp_cluster}</a>` : ''}</div>
        ${a.synopsis ? `<p class="sub" style="margin-top:.8rem">${esc(a.synopsis)}</p>` : ''}
        ${a.trailer ? `<div style="margin-top:1rem;aspect-ratio:16/9;border-radius:3px;overflow:hidden">
           <iframe src="${esc(a.trailer.replace('autoplay=1', 'autoplay=0'))}" style="width:100%;height:100%;border:0"
             allowfullscreen loading="lazy" title="Trailer"></iframe></div>` : ''}
      </div>
    </div>`;
}
addEventListener('click', e => {
  const card = e.target.closest('.pcard');
  if (card && card.dataset.id) openAnime(+card.dataset.id);
});

/* ── nav + footer ── */
function aNav(active) {
  const items = [['index.html','Home'],['catalogue.html','Catalogue'],['descriptive.html','Data'],
    ['patterns.html','Patterns'],['storylines.html','Storylines'],['recommend.html','Similar'],
    ['predict.html','Predict'],['models.html','Evaluation'],['forecast.html','Forecast']];
  document.write(`<div id="progress"></div><nav><div class="bar">
    <a class="logo" href="index.html"><i></i> Anime Industry Analytics</a>
    <button class="navtoggle" id="navToggle" aria-label="Menu"><span></span><span></span><span></span></button>
    <ul>` + items.map(([h, t]) =>
      `<li><a href="${h}" class="${h === active ? 'active' : ''}">${t}</a></li>`).join('') +
    `</ul></div></nav>`);
}
function aFooter() {
  document.write(`<footer><div class="wrap"><div class="row">
   <div class="fine">Yoon Thiri Aung · YKPT-22425 · University of Computer Studies, Yangon</div>
   <div class="links">
     <a href="catalogue.html">Catalogue</a><a href="recommend.html">Similar</a>
     <a href="predict.html">Predict</a><a href="models.html">Evaluation</a>
     <a href="https://docs.api.jikan.moe/" target="_blank" rel="noopener">Jikan API ↗</a></div>
   </div></div></footer>`);
}

addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('nav'), bar = document.getElementById('progress');
  const toggle = document.getElementById('navToggle');
  if (toggle) toggle.addEventListener('click', () => nav.classList.toggle('open'));
  nav?.querySelectorAll('ul a').forEach(a => a.addEventListener('click', () => nav.classList.remove('open')));
  const scrolled = () => (document.scrollingElement || document.documentElement).scrollTop;
  const onScroll = () => { if (bar) bar.style.width =
    (scrolled() / Math.max(1, document.body.scrollHeight - innerHeight) * 100) + '%'; };
  addEventListener('scroll', onScroll, {passive:true}); onScroll();

  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in'); io.unobserve(e.target);
  }), {threshold:.1});
  /* most .reveal nodes are injected by each page's script after DOMContentLoaded,
     so watch the tree rather than taking a single snapshot */
  const seen = new WeakSet();
  const scan = (root = document) => root.querySelectorAll?.('.reveal').forEach(el => {
    if (seen.has(el) || el.classList.contains('in')) return;
    seen.add(el); io.observe(el);
  });
  scan();
  new MutationObserver(ms => { for (const m of ms) for (const n of m.addedNodes) {
    if (n.nodeType !== 1) continue;
    if (n.matches?.('.reveal') && !seen.has(n)) { seen.add(n); io.observe(n); }
    scan(n);
  }}).observe(document.body, {childList:true, subtree:true});
});
