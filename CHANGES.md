# UI Changes Applied to Production

The production deployment at `test.hensler.work/branding-pipeline` diverged from this
example repo through four rounds of UI work. The class names and JS function names are
identical between the two codebases, so all changes below can be dropped in verbatim
(adjust CSS values to match the Inter/violet design tokens if desired).

---

## 1. Demo Mode — locked dropdowns

**Purpose:** Force all runs to use cheap models (nano-banana / gpt-4o-mini), grey out the
dropdowns, and show a notice so users understand why.

### JS — add at the top of `<script>`, before the State block

```js
const DEMO_MODE = true;
```

### JS — add at the end of `loadModels()`, just before the `} catch` line

```js
// Demo mode — lock dropdowns to cheap defaults
if (DEMO_MODE) {
  const imgSel  = document.getElementById('imageModel');
  const editSel = document.getElementById('editModel');
  const provSel = document.getElementById('textProvider');
  const txtSel  = document.getElementById('textModel');

  // Override image/edit model to google/nano-banana
  for (const opt of imgSel.options)  opt.selected = opt.value === 'google/nano-banana';
  for (const opt of editSel.options) opt.selected = opt.value === 'google/nano-banana';

  // Override text provider to openai; text model stays at default (gpt-4o-mini)
  for (const opt of provSel.options) opt.selected = opt.value === 'openai';
  onProviderChange(); // re-populate text model list for openai

  // Disable all four dropdowns
  imgSel.disabled  = true;
  editSel.disabled = true;
  provSel.disabled = true;
  txtSel.disabled  = true;

  document.getElementById('demoBanner').classList.add('visible');
}
```

### CSS — add anywhere in `<style>`

```css
.demo-banner {
  display: none;
  padding: 9px 14px;
  border-radius: var(--radius);
  background: rgba(245,158,11,.08);     /* amber tint — adjust to taste */
  border: 1px solid rgba(245,158,11,.2);
  color: #f59e0b;
  font-size: 11px;
  letter-spacing: .03em;
  line-height: 1.5;
  margin-bottom: 14px;
}
.demo-banner.visible { display: block; }
```

### HTML — add just above the Advanced collapsible toggle

```html
<div class="demo-banner" id="demoBanner">
  ⚑ Demo mode — using cost-optimised models (nano-banana / gpt-4o-mini).
  Other models produce higher-quality results but cost more per run.
</div>
```

---

## 2. Image Lightbox with circular navigation

**Purpose:** Clicking a gallery image opens it full-screen with ‹/› arrows that cycle
through all generated images. Keyboard left/right/Esc also works.

### JS — add to the State block (after existing `let` declarations)

```js
let lightboxImages = {};
let lightboxCurrentStage = null;
const LIGHTBOX_STAGES = [
  'hero_image','logo_image','product_angle','product_topdown',
  'product_macro','product_in_use','merch_tshirt','merch_hat'
];
```

### JS — replace the existing `updateGalleryItem()` function

Key change: store the image URL and use `onclick="openLightbox(...)"` instead of `<a href target="_blank">`.

```js
function updateGalleryItem(stage, src) {
  lightboxImages[stage] = src;  // ← store for lightbox

  const grid = document.getElementById('galleryGrid');
  let item = document.getElementById(`gallery-${stage}`);

  if (!item) {
    item = document.createElement('div');
    item.className = 'gallery-item';
    item.id = `gallery-${stage}`;
    grid.appendChild(item);
  }

  const label = STAGE_LABELS[stage] || stage;
  item.innerHTML = `
    <div onclick="openLightbox('${stage}')">
      <img src="${src}" alt="${label}" loading="lazy" />
    </div>
    <div class="gallery-footer">
      <span class="gallery-label">${label}</span>
      <button class="regen-btn" title="Regenerate ${label}"
        onclick="openRegenModal('${stage}')">↻</button>
    </div>
  `;
}
```

### JS — add before the `// ── Boot` comment

```js
// ── Lightbox ───────────────────────────────────────────────────────────
function openLightbox(stage) {
  lightboxCurrentStage = stage;
  renderLightbox();
  document.getElementById('lightbox').classList.add('open');
}

function renderLightbox() {
  const src   = lightboxImages[lightboxCurrentStage] || '';
  const label = STAGE_LABELS[lightboxCurrentStage] || lightboxCurrentStage;
  const available = LIGHTBOX_STAGES.filter(s => lightboxImages[s]);
  const idx   = available.indexOf(lightboxCurrentStage);

  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightboxImg').alt = label;
  document.getElementById('lightboxLabel').textContent = label;
  document.getElementById('lightboxCount').textContent =
    available.length > 1 ? `${idx + 1} / ${available.length}` : '';
}

function lightboxPrev() {
  const available = LIGHTBOX_STAGES.filter(s => lightboxImages[s]);
  if (!available.length) return;
  const idx = available.indexOf(lightboxCurrentStage);
  lightboxCurrentStage = available[(idx - 1 + available.length) % available.length];
  renderLightbox();
}

function lightboxNext() {
  const available = LIGHTBOX_STAGES.filter(s => lightboxImages[s]);
  if (!available.length) return;
  const idx = available.indexOf(lightboxCurrentStage);
  lightboxCurrentStage = available[(idx + 1) % available.length];
  renderLightbox();
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}

function lightboxClickOutside(event) {
  if (event.target === document.getElementById('lightbox')) closeLightbox();
}

document.addEventListener('keydown', e => {
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'ArrowLeft')  lightboxPrev();
  if (e.key === 'ArrowRight') lightboxNext();
  if (e.key === 'Escape')     closeLightbox();
});
```

### CSS — add anywhere in `<style>`

```css
.gallery-item img { cursor: zoom-in; }

.lightbox-overlay {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,.88);
  z-index: 2000;
  align-items: center; justify-content: center;
  backdrop-filter: blur(6px);
}
.lightbox-overlay.open { display: flex; }

.lb-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 90vw;
  max-height: 85vh;
}
.lb-content img {
  max-width: 100%;
  max-height: calc(85vh - 48px);
  object-fit: contain;
  border-radius: var(--radius);
  border: 1px solid var(--border2);
}
.lb-meta {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 12px;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
}
.lb-close {
  position: fixed;
  top: 20px; right: 24px;
  background: none;
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  color: var(--muted);
  cursor: pointer;
  font-size: 16px;
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  transition: all .12s;
  z-index: 2001;
}
.lb-close:hover { color: var(--text); border-color: var(--text); }
.lb-arrow {
  background: none;
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  color: var(--muted);
  cursor: pointer;
  font-size: 32px;
  width: 44px; height: 64px;
  display: flex; align-items: center; justify-content: center;
  transition: all .12s;
  flex-shrink: 0;
  margin: 0 16px;
}
.lb-arrow:hover { color: var(--accent2); border-color: var(--accent); }
```

### HTML — add before the existing `<!-- Regen modal -->` comment

```html
<!-- Lightbox -->
<div class="lightbox-overlay" id="lightbox" onclick="lightboxClickOutside(event)">
  <button class="lb-close" onclick="closeLightbox()">✕</button>
  <button class="lb-arrow lb-prev" onclick="lightboxPrev(); event.stopPropagation()">‹</button>
  <div class="lb-content" onclick="event.stopPropagation()">
    <img id="lightboxImg" src="" alt="" />
    <div class="lb-meta">
      <span id="lightboxLabel"></span>
      <span id="lightboxCount"></span>
    </div>
  </div>
  <button class="lb-arrow lb-next" onclick="lightboxNext(); event.stopPropagation()">›</button>
</div>
```

---

## 3. UI Tooltips (palette consistency explanation)

All are `title` attribute additions — no CSS or JS required.

### Tone buttons (in the HTML form)

```html
<button class="tone-btn active" data-tone="silly"
  title="Playful and absurd — exaggerated claims, fun personality">😄 Silly</button>
<button class="tone-btn" data-tone="serious"
  title="Professional and premium — authoritative, minimal, trustworthy">🎩 Serious</button>
<button class="tone-btn" data-tone="scam"
  title="Deliberately cheesy — infomercial energy, over-the-top promises">💀 Scam</button>
```

### Seed label

```html
<label title="Set a seed to reproduce the same result. Leave blank for random.">
  Seed <span class="text-muted">(optional)</span>
</label>
```

### Output Quality and Safety Tolerance labels (in adv-grid)

```html
<label title="JPEG compression quality (60–100). Higher = larger files, sharper detail.">
  Output Quality …
</label>

<label title="1 = strictest content filtering, 5 = most permissive.">
  Safety Tolerance …
</label>
```

### Image Gallery card title

Add `title="All images generated with the brand color palette and voice as prompt context"`
to the `.card-title` div that contains "Image Gallery".

### Palette row

Add `title="Used as style guidance in all image prompts"` to `<div … id="resPalette">`.

---

## 4. Mobile responsiveness

**Summary of problems fixed:**
- Two-column layout breaks on small screens — sidebar needs to be a slide-in overlay
- `.main` padding (20–36px) is too generous on phones
- `.brand-layout` (1fr + hero image column) needs to stack, image first
- `.adv-grid`, `.two-col`, `.form-row` all need to collapse to single column
- Brand name font (large px value) needs `clamp()` scaling
- Pipeline stages overflow horizontally — needs `overflow-x: auto`
- Gallery grid `minmax` value too wide for one column on narrow phones
- Lightbox arrows consume too much width; need to shrink on mobile

### HTML — add a hamburger button inside `<header>` (first child)

```html
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Archive">☰</button>
```

### HTML — add a scrim overlay just before `<div class="layout">`

```html
<div class="sidebar-scrim" id="sidebarScrim" onclick="closeSidebar()"></div>
```

### JS — add before the Advanced toggle function

```js
// ── Sidebar (mobile) ───────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const scrim   = document.getElementById('sidebarScrim');
  const open    = sidebar.classList.toggle('open');
  scrim.classList.toggle('visible', open);
}

function closeSidebar() {
  document.querySelector('.sidebar').classList.remove('open');
  document.getElementById('sidebarScrim').classList.remove('visible');
}
```

### JS — in `openHistoryRun()`, add `closeSidebar();` as the first line after the `classList.add('active')` call

This auto-closes the sidebar on mobile when the user taps a history run.

### CSS — add at the end of `<style>`, before the closing `</style>` tag

Adjust pixel values and font sizes to match your design tokens. The class names used
(`.sidebar`, `.main`, `.brand-layout`, `.two-col`, `.adv-grid`, `.form-row`,
`.pipeline-stages`, `.gallery-grid`, `.modal-overlay`, `.modal`) are identical in both
repos.

```css
/* ── Sidebar toggle ─────────────────────────────────────── */
.sidebar-toggle {
  display: none;
  background: none;
  border: none;
  color: var(--muted);
  cursor: pointer;
  font-size: 20px;
  padding: 4px 6px;
  line-height: 1;
  flex-shrink: 0;
}
.sidebar-toggle:hover { color: var(--text); }

.sidebar-scrim {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,.55);
  z-index: 890;
}
.sidebar-scrim.visible { display: block; }

/* ── Mobile breakpoint ──────────────────────────────────── */
@media (max-width: 720px) {
  /* Header: hide decorative subtitle, show hamburger */
  header { padding: 0 14px; gap: 10px; }
  .hdr-pipe, .hdr-sub { display: none; }   /* remove if your header has no .hdr-sub */
  .sidebar-toggle { display: block; }

  /* Layout: drop to single column; sidebar becomes fixed overlay */
  .layout { grid-template-columns: 1fr; }

  .sidebar {
    position: fixed;
    left: -100%;
    top: 57px;                /* match your header height */
    width: min(280px, 85vw);
    height: calc(100vh - 57px);
    z-index: 900;
    transition: left .2s ease;
  }
  .sidebar.open { left: 0; }

  /* Reduce whitespace */
  .main  { padding: 14px; gap: 14px; }
  .card  { padding: 16px; }          /* if your .card has padding */

  /* Stack all two-column grids */
  .brand-layout { grid-template-columns: 1fr; }
  .brand-layout > div:last-child { order: -1; margin-bottom: 16px; } /* hero image first */
  .form-row { grid-template-columns: 1fr; }
  .adv-grid  { grid-template-columns: 1fr; }
  .two-col   { grid-template-columns: 1fr; }

  /* Fluid brand name */
  .brand-name { font-size: clamp(28px, 9vw, 52px); }

  /* Pipeline: horizontal scroll rather than wrapping */
  .pipeline-stages { overflow-x: auto; padding-bottom: 6px; }

  /* Gallery: allow two columns on narrow phones */
  .gallery-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }

  /* Modal: full-width on mobile */
  .modal-overlay .modal { max-width: 95%; padding: 18px; }

  /* Lightbox arrows: shrink so the image has room */
  .lb-arrow { width: 34px; height: 48px; font-size: 26px; margin: 0 5px; }
  .lb-close { top: 14px; right: 14px; }
}
```
