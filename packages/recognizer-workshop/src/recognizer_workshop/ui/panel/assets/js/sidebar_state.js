/* Persist the help sidebar open/closed state across page reloads via localStorage,
   and inject a ? button to the left of the theme toggle. */
(function () {
    var KEY = 'rw_sidebar_open';

    // Safety reveal: if something goes wrong, always show the sidebar after 3 s.
    var safetyTimer = setTimeout(reveal, 3000);

    function reveal() {
        clearTimeout(safetyTimer);
        document.documentElement.classList.add('rw-sidebar-ready');
    }

    function currentState() {
        var btn = document.getElementById('sidebar-button');
        if (!btn) return null;
        var oc = btn.getAttribute('onclick') || '';
        if (oc.includes('closeNav')) return 'open';
        if (oc.includes('openNav')) return 'closed';
        return null;
    }

    function injectHelpButton() {
        if (document.getElementById('rw-help-btn')) return;
        var themeToggle = document.querySelector('.pn-toggle-theme');
        if (!themeToggle) { setTimeout(injectHelpButton, 200); return; }

        var btn = document.createElement('span');
        btn.id = 'rw-help-btn';
        btn.textContent = 'Help';
        btn.onclick = function () {
            // Delegate to the original (hidden) sidebar button — it manages its own state.
            var sidebarBtn = document.getElementById('sidebar-button');
            if (sidebarBtn) sidebarBtn.click();
        };
        themeToggle.parentNode.insertBefore(btn, themeToggle);
    }

    function restoreState() {
        var saved = localStorage.getItem(KEY);
        var state = currentState();
        if (state === null) { setTimeout(restoreState, 200); return; }
        if (saved === null) {
            if (state === 'open') document.getElementById('sidebar-button').click();
            reveal();
            return;
        }
        if (saved === '1' && state === 'closed') document.getElementById('sidebar-button').click();
        if (saved === '0' && state === 'open')   document.getElementById('sidebar-button').click();
        reveal();
    }

    function hookFunctions() {
        if (typeof window.openNav !== 'function') { setTimeout(hookFunctions, 200); return; }
        var origOpen = window.openNav, origClose = window.closeNav;
        window.openNav  = function () { origOpen.apply(this, arguments);  localStorage.setItem(KEY, '1'); };
        window.closeNav = function () { origClose.apply(this, arguments); localStorage.setItem(KEY, '0'); };
        injectHelpButton();
        restoreState();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hookFunctions);
    } else {
        setTimeout(hookFunctions, 300);
    }
}());
