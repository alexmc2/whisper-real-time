HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Transcription</title>
    <style>
        /* Reset */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        /* Main layout */
        html {
            overflow-x: hidden;
            width: 100vw;
            height: 100%;
        }
        
        body {
            font-family: Arial, sans-serif;
            width: 100vw;
            max-width: 100vw;
            overflow-x: hidden;
            min-height: 100%;
            position: relative;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }
        
        /* Container */
        .main-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0;
            position: relative;
        }
        
        /* Header */
        h1 {
            font-size: 24px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
            width: 100%;
            word-break: break-word;
        }
        
        /* Controls */
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
            width: 100%;
        }
        
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        button.blue {
            background-color: #2196F3;
        }
        
        button.blue:hover {
            background-color: #0b7dda;
        }
        
        button.red {
            background-color: #f44336;
        }
        
        button.red:hover {
            background-color: #da190b;
        }
        
        button.purple {
            background-color: #9c27b0;
        }
        
        button.purple:hover {
            background-color: #7b1fa2;
        }
        
        button.teal {
            background-color: #009688;
        }
        
        button.teal:hover {
            background-color: #00796b;
        }
        
        /* Status message */
        .status {
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
            font-size: 14px;
            width: 100%;
            word-break: break-word;
            min-height: 3em; /* Fixed height to prevent layout shifts */
            display: flex;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            border-radius: 4px;
            padding: 0 10px;
        }
        
        .status-text {
            padding: 10px 0;
        }
        
        /* Auto-save status */
        .auto-save-status {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            margin-bottom: 10px;
            padding: 3px 5px;
            background-color: #f1f8e9;
            border-radius: 4px;
            border-left: 3px solid #8bc34a;
            display: none; /* Hidden by default */
        }
        
        /* Format selection dropdown */
        .format-selection {
            display: flex;
            align-items: center;
            margin-top: 10px;
            margin-bottom: 15px;
        }
        
        .format-selection select {
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
            background-color: #f9f9f9;
            margin-left: 10px;
            font-size: 14px;
        }
        
        .format-selection label {
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }
        
        /* Transcript textarea */
        #transcription {
            width: 100%;
            height: 300px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            font-size: 16px;
            resize: vertical;
            font-family: inherit;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .edit-mode {
            border: 2px solid #2196F3 !important;
            background-color: #F1F5F9 !important;
        }
        
        /* Responsive adjustments */
        @media screen and (max-width: 600px) {
            body {
                padding: 10px;
            }
            
            h1 {
                font-size: 20px;
            }
            
            button {
                padding: 8px 12px;
                font-size: 13px;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>Whisper Real-Time Transcription</h1>
        
        <div class="controls">
            <button id="copyBtn">Copy All Text</button>
            <button id="downloadBtn" class="teal">Download Text</button>
            <button id="editBtn" class="blue">Edit Text</button>
            <button id="saveBtn" class="blue" style="display:none">Save Changes</button>
            <button id="cleanBtn" class="purple">Clean Transcription</button>
            <button id="undoCleanBtn" class="purple" style="display:none">Undo Cleaning</button>
            <button id="resetCleanBtn" class="purple" style="display:none">Reset Cleaning History</button>
            <button id="clearBtn" class="red">Clear All</button>
        </div>
        
        <div id="autoSaveStatus" class="auto-save-status">
            Auto-save: <span id="lastSaveTime">Not saved yet</span>
        </div>
        
        <div class="format-selection">
            <label for="formatMode">Cleaning Format:</label>
            <select id="formatMode">
                <option value="standard">Standard Cleaning</option>
                <option value="formal">Formal Language</option>
                <option value="casual">Casual</option>
                <option value="concise">Concise</option>
                <option value="paragraph">Paragraph Structure</option>
                <option value="instructions">Process as Instructions</option>
            </select>
        </div>
        
        <div id="status" class="status">
            <div id="statusText" class="status-text"></div>
        </div>
        
        <textarea id="transcription" readonly></textarea>
    </div>

    <script>
        (() => {
            console.log('[whisper-ui] script initialized');
            let isEditing = false;
            let autoUpdate = true;
            const transcriptionEl = document.getElementById('transcription');
            const editBtn = document.getElementById('editBtn');
            const saveBtn = document.getElementById('saveBtn');
            const statusEl = document.getElementById('statusText');

            // Initial load
            transcriptionEl.value = 'Waiting for speech...';

            // Copy
            document.getElementById('copyBtn').addEventListener('click', () => {
                const text = transcriptionEl.value;
                navigator.clipboard.writeText(text)
                    .then(() => showStatus('Transcription copied to clipboard!'))
                    .catch(err => {
                        console.error('Failed to copy:', err);
                        showStatus('Failed to copy to clipboard.');
                    });
            });

            // Download
            document.getElementById('downloadBtn').addEventListener('click', () => {
                const text = transcriptionEl.value;
                if (!text || text.trim() === '') {
                    showStatus('No transcription text to download.');
                    return;
                }

                const now = new Date();
                const dateStr = now.toISOString().slice(0, 19).replace(/[-:T]/g, '');
                const filename = `transcription_${dateStr}.txt`;
                const blob = new Blob([text], { type: 'text/plain' });

                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();

                setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showStatus(`Transcription downloaded as "${filename}"`);
                }, 100);
            });

            // Clear
            document.getElementById('clearBtn').addEventListener('click', () => {
                if (!confirm('Are you sure you want to clear all transcription text?')) return;
                fetch('/clear_transcription', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                })
                    .then(() => {
                        transcriptionEl.value = '';
                        document.getElementById('undoCleanBtn').style.display = 'none';
                        document.getElementById('resetCleanBtn').style.display = 'none';
                        showStatus('Transcription cleared.');
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showStatus('Error clearing transcription.');
                    });
            });

            // Clean
            document.getElementById('cleanBtn').addEventListener('click', () => {
                const formatMode = document.getElementById('formatMode').value;
                showStatus(`Cleaning transcription with ${formatMode} formatting...`);

                fetch('/clean_transcription', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ formatting_mode: formatMode })
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            showStatus(`Transcription cleaned successfully with ${formatMode} formatting!`);
                            document.getElementById('undoCleanBtn').style.display = 'inline-block';
                            document.getElementById('resetCleanBtn').style.display = 'inline-block';
                            updateTranscription();
                        } else {
                            showStatus(data.message || 'Error cleaning transcription.');
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showStatus('Error cleaning transcription. Please check if API key is set.');
                    });
            });

            // Undo Clean
            document.getElementById('undoCleanBtn').addEventListener('click', () => {
                showStatus('Undoing cleaning...');
                fetch('/undo_cleaning', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            showStatus('Reverted to original transcription. You can clean again or continue from here.');
                            document.getElementById('resetCleanBtn').style.display = 'inline-block';
                            updateTranscription();
                        } else {
                            showStatus(data.message || 'Error undoing changes.');
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showStatus('Error undoing changes.');
                    });
            });

            // Reset Cleaning History
            document.getElementById('resetCleanBtn').addEventListener('click', () => {
                showStatus('Resetting cleaning history...');
                fetch('/reset_cleaning_history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            showStatus('Cleaning history reset.');
                            document.getElementById('undoCleanBtn').style.display = 'none';
                            document.getElementById('resetCleanBtn').style.display = 'none';
                        } else {
                            showStatus(data.message || 'Error resetting cleaning history.');
                        }
                    })
                    .catch(error => {
                        console.error('Error resetting cleaning history:', error);
                        showStatus('Error resetting cleaning history.');
                    });
            });

            // Edit
            editBtn.addEventListener('click', () => toggleEditMode());
            // Save
            saveBtn.addEventListener('click', () => saveChanges());

            function toggleEditMode() {
                isEditing = !isEditing;
                if (isEditing) {
                    autoUpdate = false;
                    transcriptionEl.removeAttribute('readonly');
                    transcriptionEl.classList.add('edit-mode');
                    editBtn.style.display = 'none';
                    saveBtn.style.display = 'inline-block';
                    showStatus('Edit mode: Changes will NOT be overwritten by new speech.');
                } else {
                    transcriptionEl.setAttribute('readonly', 'readonly');
                    transcriptionEl.classList.remove('edit-mode');
                    editBtn.style.display = 'inline-block';
                    saveBtn.style.display = 'none';
                    autoUpdate = true;
                    showStatus('Returned to automatic update mode.');
                }
            }

            function saveChanges() {
                const newText = transcriptionEl.value;
                const lines = newText.split('\\n').filter(line => line.trim() !== '');
                fetch('/update_transcription', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: lines }),
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            showStatus('Changes saved successfully!');
                            toggleEditMode();
                        } else {
                            showStatus('Error saving changes.');
                        }
                    })
                    .catch(error => {
                        console.error('Error saving changes:', error);
                        showStatus('Error saving changes. Please try again.');
                    });
            }

            function showStatus(message) {
                statusEl.textContent = message;
                setTimeout(() => { statusEl.textContent = ''; }, 10000);
            }

            function updateTranscription() {
                if (!autoUpdate) return;
                fetch('/get_transcription')
                    .then(response => response.json())
                    .then(data => {
                        const newText = data.text.join('\\n');
                        if (transcriptionEl.value !== newText && !isEditing) {
                            transcriptionEl.value = newText;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching transcription:', error);
                        if (!isEditing) {
                            transcriptionEl.value = 'Error connecting to transcription service.';
                        }
                    });
            }

            setInterval(updateTranscription, 1000);

            function checkAutoSaveStatus() {
                fetch('/get_last_save_info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            document.getElementById('autoSaveStatus').style.display = 'block';
                            document.getElementById('lastSaveTime').textContent = 'Last saved at ' + data.last_save_time + ' (' + data.last_save_file + ')';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching auto-save status:', error);
                    });
            }

            setInterval(checkAutoSaveStatus, 5000);
        })();
    </script>
</body>
</html>
'''
