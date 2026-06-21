window.HELP_IMPROVE_VIDEOJS = false;

// ================= BibTeX Copy =================
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// ================= Scroll to Top =================
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
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

// ================= Video Carousel Autoplay =================
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// ================= Spatial Quiz Module =================
// 展开/收起 “Ground Truth vs Gemini” 答案
function toggleQuizAnswer(answerId, el) {
    const ans = document.getElementById(answerId);
    if (!ans) return;
    const visible = ans.style.display === 'block';
    if (visible) {
        ans.style.display = 'none';
        el.innerHTML = '👉 Click to view Ground Truth and Gemini\'s answer!';
    } else {
        ans.style.display = 'block';
        el.innerHTML = '🙈 Hide answer';
    }
}

// 初始化预览缩略图点击逻辑
function setupSpatialQuiz() {
    const previews = document.querySelectorAll('.quiz-preview');
    const blocks = document.querySelectorAll('.quiz-video-block');
    if (!previews.length || !blocks.length) return; // 页面上没有这个模块就直接跳过

    previews.forEach(p => {
        p.addEventListener('click', () => {
            const targetId = p.getAttribute('data-target');
            // 隐藏所有视频块
            blocks.forEach(b => b.classList.add('is-hidden'));
            // 显示目标视频块
            const target = document.getElementById(targetId);
            if (target) {
                target.classList.remove('is-hidden');
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ================= DOM Ready =================
$(document).ready(function() {
    // Bulma carousel & slider
    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel and demo section
    setupVideoCarouselAutoplay();
    setupDemoVideoAutoplay();

    // Setup spatial quiz (Compare your spatial intelligence with Gemini)
    setupSpatialQuiz();
});
