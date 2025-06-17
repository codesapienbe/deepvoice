// DeepVoice web application JavaScript

// Utility functions
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!');
    }).catch(() => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showNotification('Copied to clipboard!');
    });
}

function copyCode(elementId) {
    const codeElement = document.getElementById(elementId);
    if (codeElement) {
        const code = codeElement.textContent;
        copyToClipboard(code);
    }
}

function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--color-success);
        color: var(--color-btn-primary-text);
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        z-index: 1000;
        opacity: 0;
        transform: translateY(-20px);
        transition: all 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

function simulateProcessing(callback, duration = 2000) {
    // Add loading state
    const button = event.target;
    const originalText = button.textContent;
    button.textContent = 'Processing...';
    button.classList.add('loading');
    button.disabled = true;
    
    setTimeout(() => {
        button.textContent = originalText;
        button.classList.remove('loading');
        button.disabled = false;
        callback();
    }, duration);
}

// File upload handlers
function triggerFileUpload(inputId) {
    const input = document.getElementById(inputId + '-file');
    if (input) {
        input.click();
    }
}

// Speaker Diarization Demo
function processDiarization() {
    const fileInput = document.getElementById('diarization-file');
    if (!fileInput.files.length) return;
    
    simulateProcessing(() => {
        showDiarizationResults();
    });
}

function loadSampleDiarization() {
    simulateProcessing(() => {
        showDiarizationResults();
    });
}

function showDiarizationResults() {
    const resultsPanel = document.getElementById('diarization-results');
    resultsPanel.classList.remove('hidden');
    
    // Animate segments
    const segments = resultsPanel.querySelectorAll('.segment');
    segments.forEach((segment, index) => {
        segment.style.opacity = '0';
        segment.style.transform = 'translateY(20px)';
        setTimeout(() => {
            segment.style.transition = 'all 0.3s ease';
            segment.style.opacity = '1';
            segment.style.transform = 'translateY(0)';
        }, index * 200);
    });
    
    showNotification('Speaker diarization completed!');
}

// Voice Verification Demo
function processVerification() {
    const voice1Input = document.getElementById('voice1-file');
    const voice2Input = document.getElementById('voice2-file');
    
    if (!voice1Input.files.length || !voice2Input.files.length) {
        showNotification('Please upload both voice samples');
        return;
    }
    
    simulateProcessing(() => {
        showVerificationResults();
    });
}

function loadSampleVerification() {
    simulateProcessing(() => {
        showVerificationResults();
    });
}

function showVerificationResults() {
    const resultsPanel = document.getElementById('verification-results');
    resultsPanel.classList.remove('hidden');
    
    // Animate similarity meter
    const meterFill = resultsPanel.querySelector('.meter-fill');
    meterFill.style.width = '0%';
    setTimeout(() => {
        meterFill.style.width = '87%';
    }, 500);
    
    showNotification('Voice verification completed!');
}

// Voice Embeddings Demo
function processEmbeddings() {
    const fileInput = document.getElementById('embeddings-file');
    if (!fileInput.files.length) return;
    
    simulateProcessing(() => {
        showEmbeddingsResults();
    });
}

function loadSampleEmbeddings() {
    simulateProcessing(() => {
        showEmbeddingsResults();
    });
}

function showEmbeddingsResults() {
    const resultsPanel = document.getElementById('embeddings-results');
    resultsPanel.classList.remove('hidden');
    
    // Draw embedding chart
    drawEmbeddingChart();
    
    showNotification('Voice embeddings generated!');
}

function drawEmbeddingChart() {
    const canvas = document.getElementById('embedding-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Generate sample embedding visualization
    const numBars = 50;
    const barWidth = width / numBars;
    const maxHeight = height - 20;
    
    // Set colors
    const primaryColor = getComputedStyle(document.documentElement)
        .getPropertyValue('--color-primary').trim();
    const secondaryColor = getComputedStyle(document.documentElement)
        .getPropertyValue('--color-secondary').trim();
    
    ctx.fillStyle = primaryColor;
    
    // Draw bars representing embedding values
    for (let i = 0; i < numBars; i++) {
        const value = Math.sin(i * 0.2) * 0.5 + Math.random() * 0.5;
        const barHeight = Math.abs(value) * maxHeight;
        const x = i * barWidth;
        const y = height - barHeight - 10;
        
        ctx.fillRect(x, y, barWidth - 1, barHeight);
    }
    
    // Add axis lines
    ctx.strokeStyle = getComputedStyle(document.documentElement)
        .getPropertyValue('--color-border').trim();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 10);
    ctx.lineTo(width, height - 10);
    ctx.stroke();
}

// Voice Search Demo
function processSearch() {
    const fileInput = document.getElementById('search-file');
    if (!fileInput.files.length) return;
    
    simulateProcessing(() => {
        showSearchResults();
    }, 3000); // Longer processing time for search
}

function loadSampleSearch() {
    simulateProcessing(() => {
        showSearchResults();
    }, 3000);
}

function showSearchResults() {
    const resultsPanel = document.getElementById('search-results');
    resultsPanel.classList.remove('hidden');
    
    // Animate search results
    const matches = resultsPanel.querySelectorAll('.match-item');
    matches.forEach((match, index) => {
        match.style.opacity = '0';
        match.style.transform = 'translateX(-20px)';
        setTimeout(() => {
            match.style.transition = 'all 0.3s ease';
            match.style.opacity = '1';
            match.style.transform = 'translateX(0)';
        }, index * 150);
    });
    
    showNotification('Voice search completed!');
}

// Enhanced code syntax highlighting
function highlightCode() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        let html = block.innerHTML;
        
        // Highlight Python keywords
        html = html.replace(/\b(from|import|def|class|if|else|elif|for|while|try|except|finally|with|as|return|yield|lambda|and|or|not|in|is|None|True|False|print)\b/g, 
            '<span style="color: var(--color-primary); font-weight: 600;">$1</span>');
        
        // Highlight strings
        html = html.replace(/(['"])((?:(?!\1)[^\\]|\\.)*)(\1)/g, 
            '<span style="color: var(--color-success);">$1$2$3</span>');
        
        // Highlight comments
        html = html.replace(/(#.*$)/gm, 
            '<span style="color: var(--color-text-secondary); font-style: italic;">$1</span>');
        
        // Highlight function calls
        html = html.replace(/(\w+)(\()/g, 
            '<span style="color: var(--color-warning);">$1</span>$2');
        
        block.innerHTML = html;
    });
}

// Smooth scroll for navigation
function setupSmoothScroll() {
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
}

// Intersection Observer for scroll animations
function setupScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe feature demos and cards
    const animatedElements = document.querySelectorAll('.feature-demo, .install-step, .perf-card');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

// Add CSS for scroll animations
function addScrollAnimationStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .feature-demo,
        .install-step,
        .perf-card {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s ease;
        }
        
        .feature-demo.animate-in,
        .install-step.animate-in,
        .perf-card.animate-in {
            opacity: 1;
            transform: translateY(0);
        }
        
        .feature-demo {
            transition-delay: 0.1s;
        }
        
        .install-step:nth-child(2) {
            transition-delay: 0.2s;
        }
        
        .install-step:nth-child(3) {
            transition-delay: 0.4s;
        }
        
        .perf-card:nth-child(2) {
            transition-delay: 0.1s;
        }
        
        .perf-card:nth-child(3) {
            transition-delay: 0.2s;
        }
        
        .perf-card:nth-child(4) {
            transition-delay: 0.3s;
        }
    `;
    document.head.appendChild(style);
}

// Theme switching (bonus feature)
function setupThemeToggle() {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-color-scheme', savedTheme);
    
    // Create theme toggle button (optional)
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = savedTheme === 'dark' ? '☀️' : '🌙';
    themeToggle.className = 'theme-toggle';
    themeToggle.style.cssText = `
        position: fixed;
        top: 20px;
        left: 20px;
        width: 40px;
        height: 40px;
        border: none;
        border-radius: 50%;
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        cursor: pointer;
        font-size: 18px;
        z-index: 1000;
        transition: all 0.3s ease;
    `;
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-color-scheme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-color-scheme', newTheme);
        localStorage.setItem('theme', newTheme);
        themeToggle.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
        
        // Redraw embedding chart with new colors
        if (!document.getElementById('embeddings-results').classList.contains('hidden')) {
            setTimeout(drawEmbeddingChart, 100);
        }
    });
    
    document.body.appendChild(themeToggle);
}

// Enhanced file drag and drop
function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--color-primary)';
            area.style.backgroundColor = 'var(--color-secondary)';
        });
        
        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--color-border)';
            area.style.backgroundColor = 'var(--color-background)';
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--color-border)';
            area.style.backgroundColor = 'var(--color-background)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('audio/')) {
                    // Find the associated input and trigger processing
                    const input = area.querySelector('input[type="file"]');
                    if (input) {
                        // Create a new FileList-like object
                        const dt = new DataTransfer();
                        dt.items.add(file);
                        input.files = dt.files;
                        
                        // Trigger the appropriate processing function
                        if (input.id.includes('diarization')) {
                            processDiarization();
                        } else if (input.id.includes('voice1') || input.id.includes('voice2')) {
                            // Check if both files are uploaded for verification
                            const voice1 = document.getElementById('voice1-file');
                            const voice2 = document.getElementById('voice2-file');
                            if (voice1.files.length && voice2.files.length) {
                                processVerification();
                            }
                        } else if (input.id.includes('embeddings')) {
                            processEmbeddings();
                        } else if (input.id.includes('search')) {
                            processSearch();
                        }
                    }
                } else {
                    showNotification('Please upload an audio file');
                }
            }
        });
    });
}

// Add loading states for buttons
function enhanceButtons() {
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                this.style.transform = 'scale(0.98)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 150);
            }
        });
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎤 DeepVoice Demo Application Loaded');
    
    // Setup all functionality
    highlightCode();
    setupSmoothScroll();
    addScrollAnimationStyles();
    setupScrollAnimations();
    setupDragAndDrop();
    enhanceButtons();
    
    // Optional theme toggle (can be enabled)
    // setupThemeToggle();
    
    // Add some initial animations
    setTimeout(() => {
        const heroElements = document.querySelectorAll('.hero-title, .hero-subtitle, .hero-features, .hero-actions');
        heroElements.forEach((el, index) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            setTimeout(() => {
                el.style.transition = 'all 0.6s ease';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }, 100);
    
    // Show welcome message
    setTimeout(() => {
        showNotification('Welcome to DeepVoice! Try the interactive demos below.');
    }, 2000);
});

// Additional utility functions for demo enhancement
function getRandomEmbeddingValues(count = 512) {
    return Array.from({length: count}, () => (Math.random() - 0.5) * 2);
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function generateMockWaveform() {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 80;
    const ctx = canvas.getContext('2d');
    
    // Generate waveform pattern
    const numBars = 100;
    const barWidth = canvas.width / numBars;
    
    ctx.fillStyle = getComputedStyle(document.documentElement)
        .getPropertyValue('--color-primary').trim();
    
    for (let i = 0; i < numBars; i++) {
        const height = Math.random() * canvas.height * 0.8;
        const x = i * barWidth;
        const y = (canvas.height - height) / 2;
        
        ctx.fillRect(x, y, barWidth - 1, height);
    }
    
    return canvas.toDataURL();
}

// Export functions for potential external use
window.DeepVoiceDemo = {
    scrollToSection,
    copyToClipboard,
    copyCode,
    showNotification,
    processDiarization,
    loadSampleDiarization,
    processVerification,
    loadSampleVerification,
    processEmbeddings,
    loadSampleEmbeddings,
    processSearch,
    loadSampleSearch
};