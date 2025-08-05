// Motor Fault Detection App JavaScript - Phase 2: Batch Processing

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Initialize dark mode
    initializeDarkMode();
    
    // Initialize file upload handling (batch support)
    initializeBatchFileUpload();
    
    // Initialize clear functionality
    initializeClearButton();
    
    // Initialize download buttons
    initializeDownloadButtons();
    
    // Initialize batch progress tracking
    initializeBatchProgress();
    
    // Initialize file details toggles
    initializeFileDetailsToggles();
    
    // Initialize drag and drop
    initializeDragAndDrop();
}

// Dark Mode Toggle Functionality
function initializeDarkMode() {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    
    // Add click event listener
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const currentTheme = body.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            setTheme(newTheme);
        });
    }
}

function setTheme(theme) {
    const body = document.body;
    const themeToggle = document.getElementById('theme-toggle');
    
    body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    if (themeToggle) {
        themeToggle.textContent = theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
    }
}

// Batch File Upload Handling
function initializeBatchFileUpload() {
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const uploadBtn = document.getElementById('upload-btn');
    
    if (fileInput && fileInfo) {
        fileInput.addEventListener('change', function(event) {
            const files = event.target.files;
            handleFileSelection(files);
        });
    }
    
    function handleFileSelection(files) {
        if (files.length > 0) {
            let totalSize = 0;
            let fileList = [];
            let validFiles = 0;
            const maxFiles = parseInt(document.body.getAttribute('data-max-files')) || 100;
            
            // Process each file
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const isValid = validateAudioFile(file);
                
                if (isValid) {
                    validFiles++;
                    totalSize += file.size;
                    fileList.push({
                        name: file.name,
                        size: formatFileSize(file.size),
                        valid: true
                    });
                } else {
                    fileList.push({
                        name: file.name,
                        size: formatFileSize(file.size),
                        valid: false
                    });
                }
            }
            
            // Display file information
            displayFileInfo(fileList, totalSize, validFiles, maxFiles);
            
            // Enable/disable upload button
            if (uploadBtn) {
                uploadBtn.disabled = validFiles === 0 || validFiles > maxFiles;
            }
        } else {
            if (fileInfo) {
                fileInfo.style.display = 'none';
            }
            if (uploadBtn) {
                uploadBtn.disabled = true;
            }
        }
    }
    
    function displayFileInfo(fileList, totalSize, validFiles, maxFiles) {
        if (!fileInfo) return;
        
        const validColor = 'var(--success-color)';
        const invalidColor = 'var(--danger-color)';
        const warningColor = 'var(--warning-color)';
        
        let statusMessage = '';
        if (validFiles > maxFiles) {
            statusMessage = `<div style="color: ${invalidColor}; font-weight: bold; margin-bottom: 10px;">
                ⚠️ Too many files! Maximum ${maxFiles} files allowed.
            </div>`;
        } else if (validFiles > 0) {
            statusMessage = `<div style="color: ${validColor}; font-weight: bold; margin-bottom: 10px;">
                ✅ ${validFiles} valid files ready for processing
            </div>`;
        }
        
        const fileListHtml = fileList.map(file => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--border-color);">
                <span style="color: ${file.valid ? validColor : invalidColor};">
                    ${file.valid ? '✅' : '❌'} ${file.name}
                </span>
                <span style="font-family: monospace;">${file.size}</span>
            </div>
        `).join('');
        
        fileInfo.innerHTML = `
            ${statusMessage}
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div><strong>Selected files:</strong> ${fileList.length}</div>
                <div><strong>Valid files:</strong> ${validFiles}</div>
                <div><strong>Total size:</strong> ${formatFileSize(totalSize)}</div>
                <div><strong>Status:</strong> ${validFiles > maxFiles ? 'Too many' : 'Ready'}</div>
            </div>
            <div style="max-height: 200px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 6px; padding: 10px;">
                ${fileListHtml}
            </div>
        `;
        fileInfo.style.display = 'block';
    }
}

// File validation
function validateAudioFile(file) {
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/ogg'];
    const allowedExtensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    // Check file size
    if (file.size > maxSize) {
        return false;
    }
    
    // Check extension
    const extension = file.name.split('.').pop().toLowerCase();
    return allowedExtensions.includes(extension);
}

// Drag and Drop functionality
function initializeDragAndDrop() {
    const fileInputWrapper = document.querySelector('.file-input-wrapper');
    const fileInput = document.getElementById('file-input');
    
    if (fileInputWrapper && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            fileInputWrapper.classList.add('drag-over');
        }
        
        function unhighlight(e) {
            fileInputWrapper.classList.remove('drag-over');
        }
        
        fileInputWrapper.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
}

// Batch Progress Tracking
function initializeBatchProgress() {
    if (window.location.pathname.includes('batch_progress')) {
        startProgressTracking();
    }
}

function startProgressTracking() {
    let progressInterval;
    
    function updateProgress() {
        fetch('/batch_status')
            .then(response => response.json())
            .then(data => {
                updateProgressDisplay(data);
                
                // Stop tracking when complete or error
                if (data.status === 'completed' || data.status === 'error') {
                    clearInterval(progressInterval);
                    
                    if (data.status === 'completed') {
                        setTimeout(() => {
                            window.location.href = '/results';
                        }, 1000);
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching progress:', error);
                showAlert('Error fetching progress. Please refresh the page.', 'danger');
            });
    }
    
    function updateProgressDisplay(data) {
        // Update progress bar
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = data.progress + '%';
        }
        
        // Update status text
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = data.status;
            statusElement.className = `status-${data.status}`;
        }
        
        // Update progress text
        const progressElement = document.getElementById('progress');
        if (progressElement) {
            progressElement.textContent = data.current_file;
        }
        
        // Update current file
        const currentFileElement = document.getElementById('current-file');
        if (currentFileElement) {
            currentFileElement.textContent = data.current_filename || 'Processing...';
        }
        
        // Update completed files list
        const completedList = document.getElementById('completed-list');
        if (completedList) {
            completedList.innerHTML = data.completed_files.map(file => 
                `<li class="fade-in">✅ ${file}</li>`
            ).join('');
        }
        
        // Update errors list
        if (data.errors && data.errors.length > 0) {
            let errorContainer = document.getElementById('error-files');
            if (!errorContainer) {
                errorContainer = document.createElement('div');
                errorContainer.id = 'error-files';
                errorContainer.style.marginTop = '20px';
                document.getElementById('completed-files').after(errorContainer);
            }
            
            errorContainer.innerHTML = `
                <h3 style="color: var(--danger-color);">Errors:</h3>
                <ul id="error-list">
                    ${data.errors.map(error => `<li class="fade-in">❌ ${error}</li>`).join('')}
                </ul>
            `;
        }
    }
    
    // Start tracking
    progressInterval = setInterval(updateProgress, 2000);
    updateProgress(); // Initial call
}

// File Details Toggle
function initializeFileDetailsToggles() {
    // This is handled by the onclick in the template, but we can add enhanced functionality
    window.toggleDetails = function(fileId) {
        const details = document.getElementById('details-' + fileId);
        const button = event.target;
        
        if (details.style.display === 'none' || details.style.display === '') {
            details.style.display = 'block';
            details.classList.add('slide-up');
            button.innerHTML = '🙈 Hide Details';
            button.classList.add('btn-secondary');
            button.classList.remove('btn-info');
        } else {
            details.style.display = 'none';
            button.innerHTML = '👁️ View Details';
            button.classList.add('btn-info');
            button.classList.remove('btn-secondary');
        }
    };
}

// Clear Button Functionality
function initializeClearButton() {
    const clearButton = document.getElementById('clear-btn');
    
    if (clearButton) {
        clearButton.addEventListener('click', function(event) {
            const confirmMessage = 'Are you sure you want to clear all results? This action cannot be undone.';
            if (!confirm(confirmMessage)) {
                event.preventDefault();
                return false;
            }
            
            showLoading('Clearing results...');
        });
    }
}

// Download Button Functionality
function initializeDownloadButtons() {
    const downloadButtons = document.querySelectorAll('.download-btn, a[href*="/download/"]');
    
    downloadButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            let format = 'file';
            
            if (href.includes('csv')) format = 'CSV';
            else if (href.includes('json')) format = 'JSON';
            else if (href.includes('zip')) format = 'ZIP';
            
            showAlert(`Preparing ${format} download...`, 'info');
            
            // Add download animation
            this.classList.add('pulse');
            setTimeout(() => {
                this.classList.remove('pulse');
            }, 2000);
        });
    });
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-dynamic');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dynamic fade-in`;
    alert.innerHTML = message;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alert, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 300);
            }
        }, 5000);
    }
}

function showLoading(message = 'Processing...') {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'block';
        const loadingText = loading.querySelector('p');
        if (loadingText) {
            loadingText.textContent = message;
        }
    }
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

// Feature value copy functionality
function initializeFeatureCopy() {
    const featureValues = document.querySelectorAll('.feature-value');
    featureValues.forEach(value => {
        value.style.cursor = 'pointer';
        value.title = 'Click to copy';
        
        value.addEventListener('click', function() {
            const textToCopy = this.textContent.trim();
            
            if (navigator.clipboard) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showAlert('Value copied to clipboard!', 'success');
                    
                    // Visual feedback
                    this.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        this.style.transform = 'scale(1)';
                    }, 150);
                });
            } else {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = textToCopy;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                showAlert('Value copied to clipboard!', 'success');
            }
        });
    });
}

// Image error handling
function initializeImageErrorHandling() {
    const spectrogramImages = document.querySelectorAll('.spectrogram-item img, .thumbnail img, .spectrogram-full img');
    
    spectrogramImages.forEach(img => {
        img.addEventListener('error', function() {
            this.style.display = 'none';
            
            const errorMsg = document.createElement('div');
            errorMsg.className = 'alert alert-danger';
            errorMsg.textContent = 'Error loading spectrogram image';
            errorMsg.style.margin = '10px';
            
            this.parentNode.appendChild(errorMsg);
        });
        
        img.addEventListener('load', function() {
            this.classList.add('fade-in');
        });
    });
}

// Progressive enhancement for results pages
if (window.location.pathname.includes('results')) {
    document.addEventListener('DOMContentLoaded', function() {
        initializeFeatureCopy();
        initializeImageErrorHandling();
        
        // Add smooth scrolling for navigation
        const links = document.querySelectorAll('a[href^="#"]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // ESC key to close modals or clear forms
    if (event.key === 'Escape') {
        const fileInput = document.getElementById('file-input');
        if (fileInput && fileInput.value) {
            fileInput.value = '';
            const fileInfo = document.getElementById('file-info');
            if (fileInfo) {
                fileInfo.style.display = 'none';
            }
            const uploadBtn = document.getElementById('upload-btn');
            if (uploadBtn) {
                uploadBtn.disabled = true;
            }
        }
        
        // Close any open details
        const openDetails = document.querySelectorAll('.file-details[style*="display: block"]');
        openDetails.forEach(detail => {
            detail.style.display = 'none';
        });
    }
    
    // Ctrl/Cmd + D for download (if on results page)
    if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        const csvDownload = document.querySelector('a[href*="csv"]');
        if (csvDownload) {
            csvDownload.click();
        }
    }
    
    // Ctrl/Cmd + R for refresh progress (if on progress page)
    if ((event.ctrlKey || event.metaKey) && event.key === 'r' && window.location.pathname.includes('progress')) {
        event.preventDefault();
        location.reload();
    }
});

// Performance optimization: Intersection Observer for lazy loading
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                }
            });
        }, {
            rootMargin: '50px 0px',
            threshold: 0.01
        });

        const lazyImages = document.querySelectorAll('img[data-src]');
        lazyImages.forEach(img => imageObserver.observe(img));
    }
}

// Initialize lazy loading on page load
document.addEventListener('DOMContentLoaded', initializeLazyLoading);

// Auto-save form data (for development)
function initializeAutoSave() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input[type="file"]');
        inputs.forEach(input => {
            input.addEventListener('change', function() {
                localStorage.setItem('last_upload_time', new Date().toISOString());
            });
        });
    });
}

// Initialize auto-save
document.addEventListener('DOMContentLoaded', initializeAutoSave);

// Add visual feedback for all interactive elements
function addVisualFeedback() {
    const interactiveElements = document.querySelectorAll('button, .btn, input[type="file"] + label');
    
    interactiveElements.forEach(element => {
        element.addEventListener('mousedown', function() {
            this.style.transform = 'scale(0.98)';
        });
        
        element.addEventListener('mouseup', function() {
            this.style.transform = 'scale(1)';
        });
        
        element.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// Initialize visual feedback
document.addEventListener('DOMContentLoaded', addVisualFeedback);

console.log('🎯 Motor Fault Detection App - Phase 2 Initialized Successfully!');
