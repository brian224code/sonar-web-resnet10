
// Placeholder for your JavaScript code
export function addLog(message, type = 'info') {
	const logOutput = document.getElementById('logOutput');
	const time = new Date().toLocaleTimeString();
	const logEntry = document.createElement('div');
	logEntry.className = `log-entry log-${type}`;
	logEntry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
	logOutput.appendChild(logEntry);
	logOutput.scrollTop = logOutput.scrollHeight;
}

export function getTrainingConfig() {
    return {
        epochs: parseInt(document.getElementById('epochs').value),
        batchSize: parseInt(document.getElementById('batchSize').value),
        validationSplit: parseFloat(document.getElementById('validationSplit').value),
        shuffle: document.getElementById('shuffle').checked,
        verbose: document.getElementById('verbose').checked
    };
}