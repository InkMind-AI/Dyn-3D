// ================= BibTeX Copy =================
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (!bibtexElement || !button || !copyText) return;

    const showCopied = () => {
        button.classList.add('copied');
        copyText.textContent = 'Cop';
        setTimeout(() => {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    };

    navigator.clipboard.writeText(bibtexElement.textContent).then(showCopied).catch(function() {
        const textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showCopied();
    });
}

// ================= Scroll to Top =================
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// ================= Demo Video Autoplay =================
function setupDemoVideoAutoplay() {
    const demoVideos = document.querySelectorAll('.demo-video');
    if (demoVideos.length === 0) return;

    demoVideos.forEach(video => {
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    }, {
        threshold: 0.35
    });

    demoVideos.forEach(video => {
        observer.observe(video);
        if (video.readyState >= 2) {
            video.play().catch(() => {});
        } else {
            video.addEventListener('loadeddata', () => {
                video.play().catch(() => {});
            }, { once: true });
        }
    });
}

document.addEventListener('DOMContentLoaded', setupDemoVideoAutoplay);
