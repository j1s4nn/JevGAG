// Theme toggle (persisted) + copy-BibTeX button.
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('jevgag-theme'); } catch (e) {}
  if (saved === 'light') root.setAttribute('data-theme', 'light');

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var isLight = root.getAttribute('data-theme') === 'light';
      if (isLight) root.removeAttribute('data-theme');
      else root.setAttribute('data-theme', 'light');
      try { localStorage.setItem('jevgag-theme', isLight ? 'dark' : 'light'); } catch (e) {}
    });
  }

  var copyBtn = document.getElementById('copy-bib');
  var bib = document.getElementById('bibtex');
  if (copyBtn && bib) {
    copyBtn.addEventListener('click', function () {
      var text = bib.innerText;
      var done = function () {
        copyBtn.textContent = 'Copied';
        setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var r = document.createRange();
        r.selectNode(bib);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(r);
        try { document.execCommand('copy'); } catch (e) {}
        done();
      }
    });
  }
})();
