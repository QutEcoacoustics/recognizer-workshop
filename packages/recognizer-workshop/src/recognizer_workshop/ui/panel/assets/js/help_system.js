export function render({ model }) {
    console.log('HelpSystem JSComponent initialized');
    model.on('sidebar_bump', () => {
        // Always open the sidebar when bumped.
        openNav();
    });

    model.on('scroll_to', () => {
        console.log('scroll_to changed to:', model.scroll_to);
        if (model.scroll_to) {
            // Wait for the sidebar to finish opening.
            setTimeout(() => {
                let element = document.getElementById(model.scroll_to);
                console.log('Found in main document:', element);

                if (!element) {
                    console.log('Looking in shadow roots...');
                    const htmlElements = document.querySelectorAll('.bk-panel-models-markup-HTML');

                    for (const htmlEl of htmlElements) {
                        if (htmlEl.shadowRoot) {
                            console.log('Checking shadow root:', htmlEl.shadowRoot);
                            element = htmlEl.shadowRoot.getElementById(model.scroll_to);
                            if (element) {
                                console.log('Found in shadow root:', element);
                                break;
                            }
                        }
                    }
                }

                if (element) {
                    element.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                        inline: 'nearest'
                    });

                    element.style.transition = 'background-color 0.5s';
                    element.style.backgroundColor = '#fff3cd';
                    setTimeout(() => {
                        element.style.backgroundColor = '';
                    }, 2000);
                } else {
                    console.log('Element not found anywhere with ID:', model.scroll_to);
                }
            }, 300);
        }
    });
}
