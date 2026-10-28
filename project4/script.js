(() => {
  const nav = document.querySelector('.bookmark-nav');
  const toggle = nav.querySelector('.bookmark-toggle');
  const mobile = window.matchMedia('(max-width: 1100px)');
  const entries = Array.from(nav.querySelectorAll('a[href^="#"]'), link => ({
    link, target: document.getElementById(link.hash.slice(1))
  })).filter(entry => entry.target);

  nav.classList.add('is-enhanced');
  function setMenu(open) {
    nav.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  }
  toggle.addEventListener('click', () => setMenu(!nav.classList.contains('is-open')));
  nav.addEventListener('click', event => {
    if (mobile.matches && event.target.closest('a')) setMenu(false);
  });
  nav.addEventListener('keydown', event => {
    if (event.key === 'Escape') { setMenu(false); toggle.focus(); }
  });
  mobile.addEventListener('change', () => setMenu(false));

  let pending = false;
  function updateCurrent() {
    pending = false;
    const offset = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--anchor-offset')) + 12;
    let current = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= offset) current = entry;
    }
    for (const {link} of entries) link.removeAttribute('aria-current');
    if (current) current.link.setAttribute('aria-current', 'location');
  }
  function scheduleUpdate() {
    if (!pending) { pending = true; requestAnimationFrame(updateCurrent); }
  }
  document.addEventListener('scroll', scheduleUpdate, {passive: true});
  document.addEventListener('load', scheduleUpdate, true);
  window.addEventListener('resize', scheduleUpdate);
  updateCurrent();

  const dialog = document.querySelector('.lightbox');
  if (typeof dialog.showModal === 'function') {
    const enlarged = dialog.querySelector('img');
    const caption = dialog.querySelector('p');
    document.querySelectorAll('a[data-enlarge]').forEach(link => {
      link.addEventListener('click', event => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        const original = link.querySelector('img') || link.closest('figure').querySelector('img');
        enlarged.src = link.href;
        enlarged.alt = original.alt;
        caption.textContent = original.alt;
        dialog.showModal();
      });
    });
    dialog.querySelector('button').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      const box = dialog.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
    });
  }

  document.querySelectorAll('.blend-study').forEach(study => {
    const reveal = study.querySelector('input[type="range"]');
    const comparison = study.querySelector('[data-compare]');
    if (!reveal || !comparison) return;
    reveal.addEventListener('input', () => comparison.style.setProperty('--reveal', `${reveal.value}%`));
  });

  const printDetails = [];
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details').forEach(detail => {
      if (!detail.open) { printDetails.push(detail); detail.open = true; }
    });
    document.querySelectorAll('img[loading="lazy"]').forEach(img => { img.loading = 'eager'; });
  });
  window.addEventListener('afterprint', () => printDetails.splice(0).forEach(detail => { detail.open = false; }));
})();
