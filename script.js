// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set example text
    const exampleText = "I always fail at interviews, so I'm sure I'll fail this one too. I'm just not good at this kind of thing.";
    document.getElementById('inputText').value = exampleText;
    updateCharCounter();
    
    // Add input event for character counter
    document.getElementById('inputText').addEventListener('input', updateCharCounter);
    
    // Add enter key support for textarea
    document.getElementById('inputText').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeText();
        }
    });
});

// Update character counter
function updateCharCounter() {
    const text = document.getElementById('inputText').value;
    document.getElementById('charCounter').textContent = text.length;
}

// Main analysis function
async function analyzeText() {
    const text = document.getElementById('inputText').value.trim();
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Validation
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (text.length < 10) {
        showError('Please enter at least 10 characters for meaningful analysis.');
        return;
    }
    
    // Reset UI
    hideError();
    hideResult();
    showLoading();
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        // Call your backend API
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update UI with results
        updateResults(data);
        
        // Show results
        hideLoading();
        showResult();
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Failed to analyze text: ${error.message}. Make sure the backend server is running at http://127.0.0.1:8000`);
    } finally {
        // Reset button
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Bias';
    }
}

// Update UI with results
function updateResults(data) {
    // Set values from backend response
    document.getElementById('biasResult').textContent = data.bias || 'No bias detected';
    document.getElementById('explanationResult').textContent = data.explanation || 'No explanation available';
    document.getElementById('correctionResult').textContent = data.correction || 'No correction available';
    document.getElementById('severityValue').textContent = (data.severity || 0) + '/5';
    
    // Update severity badge color
    updateSeverityBadge(data.severity || 0);
    
    // Add a helpful suggestion based on bias type
    addSuggestion(data.bias);
}

// Update severity badge color
function updateSeverityBadge(severity) {
    const badge = document.querySelector('.severity-badge');
    const value = document.getElementById('severityValue');
    
    if (severity >= 4) {
        badge.style.background = 'rgba(255, 87, 87, 0.2)';
        badge.style.color = '#ff5757';
    } else if (severity >= 3) {
        badge.style.background = 'rgba(255, 193, 7, 0.2)';
        badge.style.color = '#ffc107';
    } else if (severity >= 1) {
        badge.style.background = 'rgba(0, 242, 96, 0.2)';
        badge.style.color = '#00f260';
    } else {
        badge.style.background = 'rgba(100, 100, 100, 0.2)';
        badge.style.color = '#cccccc';
    }
}

// Add helpful suggestion based on bias type
function addSuggestion(biasType) {
    const suggestionEl = document.getElementById('suggestionResult');
    
    const suggestions = {
        'confirmation': 'Try seeking out information that challenges your viewpoint.',
        'overgeneralization': 'Look for exceptions to your general statement.',
        'black-and-white': 'Consider the middle ground between extremes.',
        'emotional': 'Separate your feelings from objective facts.',
        'anchoring': 'Consider multiple reference points, not just the first one.',
        'availability': 'Check if your impression matches actual statistics.',
        'bandwagon': 'Evaluate whether something is truly right for you, not just popular.',
        'default': 'Try to consider alternative perspectives and challenge absolute statements.'
    };
    
    let suggestion = suggestions.default;
    
    if (biasType) {
        const biasLower = biasType.toLowerCase();
        for (const [key, value] of Object.entries(suggestions)) {
            if (biasLower.includes(key)) {
                suggestion = value;
                break;
            }
        }
    }
    
    suggestionEl.textContent = suggestion;
}

// UI Helper Functions
function showLoading() {
    document.getElementById('loadingIndicator').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loadingIndicator').style.display = 'none';
}

function showResult() {
    document.getElementById('resultBox').style.display = 'block';
    // Smooth scroll to results
    setTimeout(() => {
        document.getElementById('resultBox').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

function hideResult() {
    document.getElementById('resultBox').style.display = 'none';
}

function showError(message) {
    document.getElementById('errorText').textContent = message;
    document.getElementById('errorMessage').style.display = 'flex';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}