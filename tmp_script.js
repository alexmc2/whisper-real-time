
        document.addEventListener('DOMContentLoaded', () => {
            console.log('[whisper-ui] script initialized');
            let isEditing = false;
            let autoUpdate = true;
            const transcriptionEl = document.getElementById('transcription');
            const editBtn = document.getElementById('editBtn');
            const saveBtn = document.getElementById('saveBtn');
            const statusEl = document.getElementById('statusText');
            
            // Initial load
            transcriptionEl.value = 'Waiting for speech...';
            updateTranscription();
        
            // Copy
            document.getElementById('copyBtn').addEventListener('click', function() {
                const text = transcriptionEl.value;
                navigator.clipboard.writeText(text)
                    .then(() => {
                        showStatus('Transcription copied to clipboard!');
                    })
                .catch(err => {
                    console.error('Failed to copy:', err);
                    showStatus('Failed to copy to clipboard.');
                });
            });
        
            // Download
            document.getElementById('downloadBtn').addEventListener('click', function() {
            const text = transcriptionEl.value;
            if (!text || text.trim() === '') {
                showStatus('No transcription text to download.');
                return;
            }
            
            // Create a date-time stamp for the filename
            const now = new Date();
            const dateStr = now.toISOString().slice(0, 19).replace(/[-:T]/g, '');
            const filename = `transcription_${dateStr}.txt`;
            
            // Create a Blob with the text content
            const blob = new Blob([text], { type: 'text/plain' });
            
            // Create a download link and trigger it
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showStatus(`Transcription downloaded as "${filename}"`);
            }, 100);
            });
        
            // Clear
            document.getElementById('clearBtn').addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all transcription text?')) {
                fetch('/clear_transcription', { 
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                })
                .then(() => {
                    transcriptionEl.value = '';
                    document.getElementById('undoCleanBtn').style.display = 'none'; // Hide undo button
                    document.getElementById('resetCleanBtn').style.display = 'none'; // Hide reset button
                    showStatus('Transcription cleared.');
                })
                .catch(error => {
                    console.error('Error:', error);
                    showStatus('Error clearing transcription.');
                });
            }
            });
        
            // Clean
            document.getElementById('cleanBtn').addEventListener('click', function() {
            const formatMode = document.getElementById('formatMode').value;
            showStatus(`Cleaning transcription with ${formatMode} formatting...`);
            
            fetch('/clean_transcription', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    formatting_mode: formatMode 
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`Transcription cleaned successfully with ${formatMode} formatting!`);
                    
                    document.getElementById('undoCleanBtn').style.display = 'inline-block'; // Show undo button
                    document.getElementById('resetCleanBtn').style.display = 'inline-block'; // Show reset button
                    updateTranscription(); // Refresh to show cleaned text
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
            document.getElementById('undoCleanBtn').addEventListener('click', function() {
            showStatus('Undoing cleaning...');
            fetch('/undo_cleaning', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({}) 
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('Reverted to original transcription. You can clean again or continue from here.');
                    // Keep the undo button visible since we can undo multiple times now
                    document.getElementById('resetCleanBtn').style.display = 'inline-block'; // Show reset button
                    updateTranscription(); // Refresh to show original text
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
            document.getElementById('resetCleanBtn').addEventListener('click', function() {
            showStatus('Resetting cleaning history...');
            fetch('/reset_cleaning_history', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('Cleaning history reset.');
                    document.getElementById('undoCleanBtn').style.display = 'none'; // Hide undo button
                    document.getElementById('resetCleanBtn').style.display = 'none'; // Hide reset button
                } else {
                    showStatus(data.message || 'Error resetting cleaning history.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showStatus('Error resetting cleaning history.');
            });
            });
        
            // Edit
            editBtn.addEventListener('click', function() {
                toggleEditMode();
            });
            
            // Save
            saveBtn.addEventListener('click', function() {
                saveChanges();
            });
            
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
                const lines = newText.split('\n').filter(line => line.trim() !== '');
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
                setTimeout(() => { statusEl.textContent = ''; }, 10000); // Increased from 3000 to 10000 ms (10 seconds)
            }
            
            function updateTranscription() {
                if (!autoUpdate) return;
                fetch('/get_transcription')
                    .then(response => response.json())
                    .then(data => {
                        const newText = data.text.join('\n');
                        if (transcriptionEl.value !== newText && !isEditing) {
                            transcriptionEl.value = newText;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching transcription:', error);
                        if (!isEditing) {
                            transcriptionEl.value = "Error connecting to transcription service.";
                        }
                    });
            }
            
            // Update every second
            setInterval(updateTranscription, 1000);

            // Add auto-save status check
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
            
            // Check auto-save status every 5 seconds
            setInterval(checkAutoSaveStatus, 5000);
        });
    