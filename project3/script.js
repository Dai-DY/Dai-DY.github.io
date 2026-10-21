(() => {
  const nav = document.querySelector('.bookmark-nav');
  const toggle = nav.querySelector('.bookmark-toggle');
  const mobile = window.matchMedia('(max-width: 1100px)');
  const entries = Array.from(nav.querySelectorAll('a[href^="#"]'), link => ({link, target: document.getElementById(link.hash.slice(1))})).filter(entry => entry.target);
  nav.classList.add('is-enhanced');
  const setMenu = open => {
    nav.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  };
  toggle.addEventListener('click', () => setMenu(!nav.classList.contains('is-open')));
  nav.addEventListener('click', event => { if (mobile.matches && event.target.closest('a')) setMenu(false); });
  nav.addEventListener('keydown', event => { if (event.key === 'Escape') { setMenu(false); toggle.focus(); } });
  mobile.addEventListener('change', () => setMenu(false));
  let pending = false;
  function updateCurrent() {
    pending = false;
    const offset = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--anchor-offset')) + 12;
    let current = entries[0];
    for (const entry of entries) if (entry.target.getBoundingClientRect().top <= offset) current = entry;
    for (const {link} of entries) link.removeAttribute('aria-current');
    if (current) current.link.setAttribute('aria-current', 'location');
  }
  const scheduleUpdate = () => { if (!pending) { pending = true; requestAnimationFrame(updateCurrent); } };
  document.addEventListener('scroll', scheduleUpdate, {passive: true});
  document.addEventListener('load', scheduleUpdate, true);
  window.addEventListener('resize', scheduleUpdate);
  updateCurrent();

  const dialog = document.querySelector('.lightbox');
  const enlarged = dialog.querySelector('img');
  const label = dialog.querySelector('p');
  if (typeof dialog.showModal === 'function') {
    document.querySelectorAll('a[data-enlarge]').forEach(link => link.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const original = link.querySelector('img');
      enlarged.src = link.href;
      enlarged.alt = original.alt;
      label.textContent = original.alt;
      dialog.showModal();
    }));
    dialog.querySelector('button').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
  }
  document.querySelectorAll('[data-rotate]').forEach(button => button.addEventListener('click', () => {
    const img = document.getElementById(button.dataset.rotate);
    const rotated = img.classList.toggle('rotate-view');
    button.setAttribute('aria-pressed', String(rotated));
    button.closest('figure').querySelector('.orientation').textContent = rotated ? 'Rotated 180°' : 'Upright';
  }));
  const printDetails = [];
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details').forEach(detail => { if (!detail.open) { printDetails.push(detail); detail.open = true; } });
    // Ensure the exported report includes figures below the current viewport.
    document.querySelectorAll('img[loading="lazy"]').forEach(img => img.loading = 'eager');
  });
  window.addEventListener('afterprint', () => { printDetails.splice(0).forEach(detail => detail.open = false); });
})();
