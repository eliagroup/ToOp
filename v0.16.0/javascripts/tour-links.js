/* Open the interactive tour in a new tab: the buttons on the docs pages and the
   "Interactive tour" entry in the navigation. The tour is a standalone page
   without the docs navigation, so a new tab keeps the docs one click away. */
function openTourInNewTab() {
  document.querySelectorAll('a[href]').forEach((a) => {
    if (/(^|\/)tour\/(index\.html)?$/.test(a.getAttribute('href'))) {
      a.target = '_blank';
      a.rel = 'noopener';
    }
  });
}

// With navigation.instant, document$ emits on every page change; without it, run once.
if (typeof document$ !== 'undefined') {
  document$.subscribe(openTourInNewTab);
} else {
  document.addEventListener('DOMContentLoaded', openTourInNewTab);
}
