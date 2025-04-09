import argparse
import gc
import logging
import os
import threading
import warnings
from datetime import datetime, timedelta, timezone
from queue import Queue
from sys import platform
from time import sleep

import google.generativeai as genai  # Added for Gemini API
import numpy as np
import speech_recognition as sr
import torch
import whisper
from flask import Flask, jsonify, render_template_string, request
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Add filter for specific warnings
warnings.filterwarnings("ignore", message=".*attention_mask.*")


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Removed FileHandler to prevent creating log file
        logging.StreamHandler()  # Print to console only
    ]
)
logger = logging.getLogger("whisper_app")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
transcriptions = [
    "Waiting for speech..."
]
# Store original transcription for undo feature
original_transcriptions = []

# Additional configuration
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['GEMINI_API_KEY'] = ""  # Will be set from command line arguments

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
        let isEditing = false;
        let autoUpdate = true;
        const transcriptionEl = document.getElementById('transcription');
        const editBtn = document.getElementById('editBtn');
        const saveBtn = document.getElementById('saveBtn');
        const statusEl = document.getElementById('statusText');
        
        // Initial load
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
            setTimeout(() => { statusEl.textContent = ''; }, 10000); // Increased from 3000 to 10000 ms (10 seconds)
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
                        transcriptionEl.value = "Error connecting to transcription service.";
                    }
                });
        }
        
        // Update every second
        setInterval(updateTranscription, 1000);
    </script>
</body>
</html>
'''

# Initialize Gemini API


def setup_gemini_api(api_key):
    if not api_key:
        logger.warning(
            "No Gemini API key provided. The Clean function will not work.")
        return

    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully.")

        # Diagnostic: List all available models
        models = genai.list_models()
        gemini_models = [
            model for model in models if "gemini" in model.name.lower()]

        logger.info(f"Available Gemini models ({len(gemini_models)}):")
        for model in gemini_models:
            logger.info(
                f"- {model.name} (supported methods: {model.supported_generation_methods})")

        # Try to check if Gemini 2.5 is available in any form
        gemini_25_models = [
            model for model in gemini_models if "2.5" in model.name]
        if gemini_25_models:
            logger.info("FOUND GEMINI 2.5 MODELS:")
            for model in gemini_25_models:
                logger.info(f"  - {model.name}")
        else:
            logger.info("No Gemini 2.5 models found in API list.")

    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}")
        logger.exception("Full exception details:")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/get_transcription')
def get_transcription():
    global transcriptions
    return jsonify({'text': transcriptions if transcriptions else ["Waiting for speech..."]})


@app.route('/clear_transcription', methods=['POST'])
def clear_transcription():
    global transcriptions, original_transcriptions
    # Silently handle JSON or non-JSON requests
    _ = request.get_json(silent=True)
    transcriptions = []
    original_transcriptions = []  # Also clear original transcriptions
    return jsonify({'status': 'success'})


@app.route('/update_transcription', methods=['POST'])
def update_transcription():
    global transcriptions
    try:
        data = request.get_json()
        if 'text' in data and isinstance(data['text'], list):
            transcriptions = data['text']
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data format'})
    except Exception as e:
        print(f"Error updating transcription: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/clean_transcription', methods=['POST'])
def clean_transcription():
    global transcriptions, original_transcriptions
    if not app.config['GEMINI_API_KEY']:
        return jsonify({'status': 'error', 'message': 'Gemini API key not set.'})

    try:
        # Get formatting mode if provided
        data = request.get_json(silent=True) or {}
        formatting_mode = data.get('formatting_mode', 'standard')

        # If the request did not have JSON content type, fall back to standard mode
        if data == {}:
            logger.warning(
                "Request to clean_transcription did not have JSON content type, using standard mode")
            formatting_mode = 'standard'

        if not transcriptions or len(transcriptions) == 0:
            return jsonify({'status': 'error', 'message': 'No transcription to clean.'})

        # Store original transcription for undo feature ONLY if we don't already have an original saved
        if not original_transcriptions:
            original_transcriptions = transcriptions.copy()

        # Join all transcription lines
        text_to_clean = "\n".join(transcriptions)

        # Log what we're trying to do
        logger.info(
            f"Attempting to clean transcription with Gemini API (mode: {formatting_mode})")

        # Try different model names
        model_names_to_try = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-pro-exp-03-25",
            "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/gemini-1.5-pro"                  # Then 1.5
        ]

        cleaned_text = None
        success_model = None

        for model_name in model_names_to_try:
            try:
                logger.info(f"Attempting to use model: {model_name}")

                # Set up safety settings (to avoid content filtering issues)
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                # Set up the model with safety settings and reduced temperature for more consistent outputs
                model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=safety_settings,
                    generation_config={
                        "temperature": 0.1,  # Lower temperature for more precise formatting
                        "top_p": 0.95,       # Slightly reduce top_p for more predictable outputs
                        "top_k": 40,         # Standard top_k
                        "max_output_tokens": 8192,  # Allow longer outputs for detailed cleaning
                    }
                )

                # Create the prompt for aggressive cleaning
                formatting_instructions = ""
                if formatting_mode == 'formal':
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please make the text more formal:
                    - Use proper grammar and complete sentences
                    - Remove casual language and slang
                    - Maintain a professional tone throughout
                    - Structure the content with clear paragraph breaks
                    - Use more sophisticated vocabulary where appropriate
                    """
                elif formatting_mode == 'casual':
                    # Set moderate temperature for casual mode
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        safety_settings=safety_settings,
                        generation_config={
                            "temperature": 0.3,  # Moderate temperature for natural but not overly casual rewrites
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 8192,
                        }
                    )

                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Gently humanize the text to make it more natural:
                    - Use some contractions where it sounds natural (e.g., "don't", "can't", "we're")
                    - Simplify overly complex or formal language
                    - Use a warm, approachable tone while maintaining professionalism
                    - Convert extremely formal phrasings to more natural alternatives
                    - Use active voice instead of passive voice where appropriate
                    - Keep complex ideas intact but express them in more straightforward language
                    - Maintain the original meaning and most of the original structure
                    - Only make modest adjustments to make the text sound more natural
                    - DO NOT add casual expressions like "kinda", "y'know", or rhetorical questions
                    - DO NOT change the level of detail or information presented
                    - Preserve the professional nature of the content, just make it warmer
                    """
                elif formatting_mode == 'concise':
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please make the text more concise:
                    - Remove all unnecessary words and phrases
                    - Condense multiple sentences into shorter, clearer statements
                    - Focus on key information only
                    - Aim to reduce the overall length by at least 30% while preserving meaning
                    - Use shorter sentences and simpler structures
                    """
                elif formatting_mode == 'paragraph':
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please format the text into proper paragraphs:
                    - Group related sentences into coherent paragraphs
                    - Add paragraph breaks where topics change
                    - Each paragraph should represent a complete thought or topic
                    - Aim for 3-5 sentences per paragraph where appropriate
                    - Remove sentence fragments that break paragraph flow
                    """
                elif formatting_mode == 'instructions':
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Format this as spoken instructions or directions:
                    - Identify key steps and number them if appropriate
                    - Organize information in a sequential, logical order
                    - Make instructions clear and actionable
                    - Use imperative language ("Do this" rather than "You should do this")
                    - Separate different sections with clear breaks
                    - Focus on preserving the step-by-step nature of the content
                    - Break complex instructions into simpler steps
                    """

                prompt = f"""
                    Clean up the following speech transcription by:
                    1. AGGRESSIVELY removing repeated phrases or words that appear consecutively or across line breaks
                    2. Removing sequences of identical phrases like "Thank you. Thank you. Thank you."
                    3. Removing isolated single words that are likely speech fragments (e.g., standalone "you")
                    4. Removing filler words like "um", "uh", "like", etc.
                    5. Consolidating multiple consecutive identical phrases into a single instance
                    6. Preserving one instance of meaningful phrases even if repeated
                    7. IMPROVING PUNCTUATION: Add proper periods, commas, question marks where appropriate
                    8. FIXING CAPITALIZATION: Ensure sentences begin with capital letters and proper nouns are capitalized
                    9. Ensuring proper spacing after punctuation marks (one space after a period, comma, etc.)
                    10. CONVERTING TO BRITISH SPELLING: Convert American English spellings to British English (e.g., "color" to "colour", "center" to "centre")
                    
                    {formatting_instructions}
                    
                    IMPORTANT RULES FOR PUNCTUATION:
                    - Add periods at the end of complete thoughts
                    - Add question marks for questions
                    - Add commas for natural pauses or to separate clauses
                    - Ensure proper capitalization at the start of sentences
                    - DO NOT over-punctuate with excessive commas or periods
                    - Make sure each sentence has proper subject-verb structure when possible
                    - Maintain the original speaker's style but with better punctuation
                    
                    IMPORTANT BRITISH SPELLING RULES:
                    - Replace "-ize" endings with "-ise" (e.g., "organize" to "organise")
                    - Replace "-yze" endings with "-yse" (e.g., "analyze" to "analyse") 
                    - Add 'u' to words like "color/colour", "favor/favour", "humor/humour"
                    - Replace "-er" endings with "-re" in words like "center/centre", "meter/metre"
                    - Replace "-og" endings with "-ogue" in words like "dialog/dialogue", "catalog/catalogue"
                    - Double the final 'l' in certain verbs (e.g., "traveled/travelled", "canceled/cancelled")
                    - Use British spelling for these common words:
                      * "defense" → "defence"
                      * "offense" → "offence"
                      * "gray" → "grey"
                      * "program" → "programme" (except for computer programs)
                      * "check" → "cheque" (for banking)
                    
                    IMPORTANT GENERAL RULES:
                    - ONLY return text in the SAME LANGUAGE as the original transcription
                    - NEVER translate or add foreign language text
                    {"- For CASUAL mode, make modest adjustments to sound more natural while preserving original meaning" if formatting_mode == "casual" else "- DO NOT add new content or change the meaning of what was said"}
                    {"- For CASUAL mode, simplify overly formal language while keeping the core message intact" if formatting_mode == "casual" else "- DO NOT rewrite or rephrase the content beyond correcting punctuation and removing repetition"}
                    - Treat each line break as a potential silence marker in speech
                    - When the same phrase repeats across multiple lines, keep only ONE instance
                    - Remove any single-word lines that are likely fragments like just "you" or "the"
                    - If a phrase is repeated more than twice consecutively, keep only ONE instance
                    - Remove any fragments or incomplete thoughts that don't contribute to meaning
                    - If you see the word "Gemini" followed by instructions (like "Gemini, make this formal"), FOLLOW those instructions and apply them to the nearby content, then remove the instruction itself
                    
                    Here is the transcription to clean:
                    {text_to_clean}
                    """

                # Get response from Gemini
                logger.info(f"Sending cleaning request to {model_name}")
                response = model.generate_content(prompt)

                # If we get here, the model worked!
                logger.info(
                    f"Successfully cleaned transcription using {model_name}")
                cleaned_text = response.text.strip()
                success_model = model_name
                break

            except Exception as e:
                logger.error(f"Error with model {model_name}: {str(e)}")
                logger.exception("Full exception details:")
                continue

        if cleaned_text is None:
            logger.error("All model attempts failed for cleaning")
            return jsonify({'status': 'error', 'message': 'All model attempts failed. Check logs for details.'})

        # Split by line but filter out empty lines
        cleaned_lines = [
            line for line in cleaned_text.split('\n') if line.strip()]

        # Dictionary of common American to British spelling conversions
        # This is used as a fallback in case the Gemini model doesn't fully convert all spellings
        us_to_uk_spelling = {
            # -or to -our
            'color': 'colour',
            'favor': 'favour',
            'flavor': 'flavour',
            'humor': 'humour',
            'labor': 'labour',
            'neighbor': 'neighbour',
            'rumor': 'rumour',
            'valor': 'valour',

            # -ize to -ise
            'organize': 'organise',
            'recognize': 'recognise',
            'apologize': 'apologise',
            'criticize': 'criticise',
            'realize': 'realise',
            'emphasize': 'emphasise',
            'summarize': 'summarise',
            'prioritize': 'prioritise',

            # -yze to -yse
            'analyze': 'analyse',
            'paralyze': 'paralyse',
            'catalyze': 'catalyse',

            # -er to -re
            'center': 'centre',
            'meter': 'metre',
            'theater': 'theatre',
            'liter': 'litre',
            'caliber': 'calibre',
            'fiber': 'fibre',

            # -og to -ogue
            'dialog': 'dialogue',
            'catalog': 'catalogue',
            'analog': 'analogue',
            'monolog': 'monologue',

            # Various other differences
            'gray': 'grey',
            'defense': 'defence',
            'offense': 'offence',
            'plow': 'plough',
            'check': 'cheque',
            'program': 'programme',
            'draft': 'draught',
            'tire': 'tyre',
            'pajamas': 'pyjamas',
            'judgment': 'judgement',
            'aluminum': 'aluminium',
            'estrogen': 'oestrogen',
            'maneuver': 'manoeuvre',
            'pediatric': 'paediatric',
            'encyclopedia': 'encyclopaedia',
            'artifact': 'artefact',
            'pretense': 'pretence',
            'skeptic': 'sceptic',
            'specialty': 'speciality',
            'traveled': 'travelled',
            'traveling': 'travelling',
            'canceled': 'cancelled',
            'canceling': 'cancelling',
        }

        # Additional post-processing for better punctuation and text cleaning
        improved_lines = []
        for line in cleaned_lines:
            # Skip single-word lines that are likely fragments
            if len(line.split()) == 1 and len(line) < 5 and line.lower() in ["you", "the", "a", "and", "but", "or", "so", "ah", "oh", "uh", "um"]:
                continue

            # Fix common transcription issues
            processed_line = line

            # Remove specific problematic phrases that often appear incorrectly in whisper outputs
            # These are typically not intentional foreign language but transcription artifacts
            problematic_phrases = [
                "продолжение следует",  # Russian "to be continued"
                "подписывайтесь на канал",  # Russian "subscribe to the channel"
                "спасибо за просмотр",  # Russian "thanks for watching"
                "продолжение в следующей",  # Russian "continued in the next"
                "конец фильма",  # Russian "end of film"
                "перерыв",  # Russian "break/intermission"
                "Chào mừng quý vị đến với bộ phim",  # Vietnamese "welcome to the movie"
                "Hãy subscribe cho kênh",  # Vietnamese "subscribe to the channel"
                "Để không bỏ lỡ những video hấp dẫn",  # Vietnamese "don't miss exciting videos"
                "Thanks for watching!",  # English outro
                "Please subscribe",  # English outro
                "Please like and subscribe",  # English outro
                "Don't forget to subscribe",  # English outro
                "Thank you for watching"  # English outro
            ]

            # Common end markers that often appear incorrectly
            end_markers = [
                "...",  # Ellipsis (can appear as trailing dots)
                "…",    # Unicode ellipsis
                "..!",  # Strange combinations
                "!..",
                "!...",
                "...!",
                ".....",
                "......",
            ]

            # Handle excessive ellipsis and trailing dots
            for marker in end_markers:
                # Only remove if at the end of line or standalone
                if processed_line.endswith(marker):
                    # Replace with a single period if it's at the end of a sentence
                    processed_line = processed_line[:-len(marker)] + "."
                    logger.debug(f"Replaced end marker {marker} with period")
                elif processed_line == marker:
                    # Skip entirely if it's just an ellipsis
                    processed_line = ""
                    logger.debug(f"Removed standalone marker {marker}")
                    break

            # Process problematic phrases without using the unused variable
            for phrase in problematic_phrases:
                if phrase.lower() in processed_line.lower():
                    # Only flag as problematic if:
                    # 1. This phrase is the majority of the line, or
                    # 2. It appears at the start or end of the line, or
                    # 3. It appears as a standalone item

                    # Calculate what percentage of the line this phrase represents
                    phrase_words = len(phrase.split())
                    line_words = len(processed_line.split())

                    phrase_position_start = processed_line.lower().find(phrase.lower())
                    phrase_position_end = phrase_position_start + len(phrase)

                    # Check if phrase is at start or end of line (with some tolerance)
                    at_start = phrase_position_start <= 5  # Within first 5 chars
                    at_end = phrase_position_end >= (
                        len(processed_line) - 5)  # Within last 5 chars

                    # Check if phrase is majority of line content
                    is_majority = phrase_words / max(1, line_words) > 0.5

                    # Skip deletion if this appears to be intentional foreign language content
                    # (e.g., part of a longer paragraph or surrounded by similar characters)
                    is_intentional = False

                    # Check if surrounded by similar script (Cyrillic in this case)
                    if phrase_position_start > 0 and phrase_position_end < len(processed_line):
                        # Check characters before and after
                        char_before = processed_line[phrase_position_start - 1]
                        char_after = processed_line[phrase_position_end] if phrase_position_end < len(
                            processed_line) else ""

                        # Simple detection of Cyrillic script before and after
                        def is_cyrillic(
                            c): return 'а' <= c.lower() <= 'я' if c else False

                        # If surrounded by similar script, likely intentional
                        if is_cyrillic(char_before) and is_cyrillic(char_after):
                            is_intentional = True

                    # Only remove if meets our criteria and doesn't look intentional
                    if (at_start or at_end or is_majority) and not is_intentional:
                        logger.info(
                            f"Removing problematic artifact phrase: '{phrase}' from line")
                        # Remove the phrase
                        processed_line = processed_line.lower().replace(phrase.lower(), "").strip()

            # Skip empty lines after processing
            if not processed_line.strip():
                continue

            # Ensure first letter is capitalized if line isn't already starting with a capital
            if processed_line and not processed_line[0].isupper() and processed_line[0].isalpha():
                processed_line = processed_line[0].upper() + processed_line[1:]

            # Ensure line ends with proper punctuation if it doesn't already
            if processed_line and processed_line[-1] not in ['.', '!', '?', ':', ';']:
                # Check if the line seems like a question (starting with who, what, where, when, why, how)
                question_starters = ['who', 'what', 'where', 'when', 'why', 'how',
                                     'is', 'are', 'was', 'were', 'will', 'can', 'could', 'should', 'would']
                first_word = processed_line.split()[0].lower(
                ) if processed_line.split() else ""

                if first_word in question_starters and '?' not in processed_line:
                    processed_line += '?'
                else:
                    processed_line += '.'

            # Fix spacing after punctuation
            for punct in ['.', ',', '!', '?', ':', ';']:
                processed_line = processed_line.replace(
                    f'{punct}', f'{punct} ')
                # Fix double spaces
                while '  ' in processed_line:
                    processed_line = processed_line.replace('  ', ' ')

            # Fix spaces before punctuation
            for punct in ['.', ',', '!', '?', ':', ';']:
                processed_line = processed_line.replace(
                    f' {punct}', f'{punct}')

            # Trim any extra spaces
            processed_line = processed_line.strip()

            # Apply British spelling conversion manually as a fallback
            # This helps catch any words the Gemini model might have missed
            for us_word, uk_word in us_to_uk_spelling.items():
                # Case-sensitive replacement
                if us_word in processed_line:
                    processed_line = processed_line.replace(us_word, uk_word)

                # Capitalized version
                us_word_cap = us_word.capitalize()
                uk_word_cap = uk_word.capitalize()
                if us_word_cap in processed_line:
                    processed_line = processed_line.replace(
                        us_word_cap, uk_word_cap)

            improved_lines.append(processed_line)

        transcriptions = improved_lines if improved_lines else [
            "[Cleaned transcription - no significant content detected]"]

        # Log before and after samples for comparison
        if logger.isEnabledFor(logging.DEBUG) and cleaned_text:
            original_sample = '\n'.join(original_transcriptions[:3] if len(
                original_transcriptions) > 3 else original_transcriptions)
            cleaned_sample = '\n'.join(transcriptions[:3] if len(
                transcriptions) > 3 else transcriptions)

            logger.debug("==== CLEANING COMPARISON (SAMPLE) ====")
            logger.debug(f"BEFORE CLEANING:\n{original_sample}\n")
            logger.debug(f"AFTER CLEANING:\n{cleaned_sample}")
            logger.debug("====================================")

        logger.info(
            f"Successfully cleaned transcription using model: {success_model}")
        return jsonify({'status': 'success', 'model_used': success_model})
    except Exception as e:
        logger.error(f"Error cleaning transcription: {e}")
        logger.exception("Full exception details:")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/undo_cleaning', methods=['POST'])
def undo_cleaning():
    global transcriptions, original_transcriptions
    # Silently handle JSON or non-JSON requests
    _ = request.get_json(silent=True)
    try:
        if not original_transcriptions or len(original_transcriptions) == 0:
            return jsonify({'status': 'error', 'message': 'No original transcription to restore.'})

        # Restore the original transcription
        transcriptions = original_transcriptions.copy()
        # We're no longer clearing original_transcriptions, allowing multiple undos
        # original_transcriptions = []  # Clear the backup after restoring

        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error undoing cleaning: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/reset_cleaning_history', methods=['POST'])
def reset_cleaning_history():
    global original_transcriptions
    # Silently handle JSON or non-JSON requests
    _ = request.get_json(silent=True)
    try:
        original_transcriptions = []  # Clear the original transcriptions
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error resetting cleaning history: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


def run_flask():
    global app
    # Try to run the server, handle port-in-use errors gracefully
    port = app.config.get('PORT', 5000)
    max_attempts = 10

    for attempt in range(max_attempts):
        try:
            app.run(debug=False, host='0.0.0.0', port=port, threaded=True,
                    processes=1, use_reloader=False)
            break  # If successful, exit the loop
        except OSError as e:
            if "Address already in use" in str(e) and attempt < max_attempts - 1:
                print(f"Port {port} is busy, trying {port + 1}...")
                port += 1
                app.config['PORT'] = port  # Update the port
            else:
                print(f"Failed to start web server: {e}")
                print(
                    "Try manually killing the process using the port with 'sudo fuser -k PORT/tcp'")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, type=float,
                        help="Approx chunk length for each background capture.")
    parser.add_argument("--phrase_timeout", default=3, type=float,
                        help="Silence (in seconds) before we consider it a new line.")
    parser.add_argument("--debug", action='store_true',
                        help="Enable debug output (prints to console).")
    parser.add_argument("--cpu_threads", default=4, type=int,
                        help="Number of CPU threads to use for inference.")
    parser.add_argument("--sleep_duration", default=0.25, type=float,
                        help="Sleep time in main loop when idle (seconds).")
    parser.add_argument("--port", default=5000, type=int,
                        help="Port to run the web server on.")
    parser.add_argument("--gemini_api_key", default="", type=str,
                        help="Google Gemini API key for the clean transcription feature.")
    parser.add_argument("--gemini_debug", action='store_true',
                        help="Run Gemini API diagnostics at startup.")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='default',
                            help="Default microphone name. Use 'list' to show all.", type=str)
    args = parser.parse_args()

    # Set initial port
    app.config['PORT'] = args.port
    app.config['GEMINI_API_KEY'] = args.gemini_api_key

    logger.info("Starting Whisper Transcription App")
    logger.info(f"Using Whisper model: {args.model}")

    # Setup Gemini API if key provided
    if args.gemini_api_key:
        logger.info("Gemini API key provided, setting up...")
        setup_gemini_api(args.gemini_api_key)

        # Run comprehensive Gemini diagnostics if requested
        if args.gemini_debug:
            logger.info("Running comprehensive Gemini API diagnostics...")
            try:
                # Try all possible model name formats
                model_names_to_try = [
                    "models/gemini-2.5-pro-exp-03-25",       # First try 2.5
                    "models/gemini-2.0-flash",               # Then 2.0 models
                    "models/gemini-2.0-flash-thinking-exp-01-21",
                    "models/gemini-1.5-pro"                  # Then 1.5
                ]

                logger.info(
                    "Testing model access with different name formats:")

                for model_name in model_names_to_try:
                    try:
                        logger.info(f"Trying to access: {model_name}")
                        model = genai.GenerativeModel(model_name)

                        # Test with minimal prompt and use the response
                        response = model.generate_content("Hello")
                        logger.info(
                            f"✓ SUCCESS: Model {model_name} is accessible! Response: '{response.text[:20]}...'")
                    except Exception as e:
                        logger.error(
                            f"✗ FAILED: Model {model_name} error: {str(e)}")
            except Exception as e:
                logger.error(f"Error in Gemini diagnostics: {e}")
                logger.exception("Full exception details:")
    else:
        logger.warning(
            "No Gemini API key provided. Clean transcription will not be available.")

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print("=" * 80)
    print(f"Web interface running at: http://localhost:{app.config['PORT']}")
    print("Open that URL in your browser to see and copy transcriptions.")
    print("=" * 80)

    debug_mode = args.debug
    torch.set_num_threads(args.cpu_threads)
    if debug_mode:
        print(f"Using {args.cpu_threads} CPU threads for inference")

    phrase_time = None
    data_queue = Queue()

    global transcriptions
    if not transcriptions or len(transcriptions) == 0:
        transcriptions = [
            "Waiting for speech..."
        ]
    elif len(transcriptions) == 1 and not transcriptions[0]:
        transcriptions = [
            "Waiting for speech..."
        ]

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Set up microphone
    source = None
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            print(f"Using microphone '{mic_name}'")
            try:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        print(f"Found matching microphone: {name}")
                        source = sr.Microphone(
                            sample_rate=16000, device_index=index)
                        break
                if source is None:
                    print(
                        f"Microphone '{mic_name}' not found. Using default/first microphone...")
                    source = sr.Microphone(sample_rate=16000)
            except Exception as e:
                print(f"Error selecting microphone: {e}")
                print("Falling back to default microphone...")
                source = sr.Microphone(sample_rate=16000)
    else:
        source = sr.Microphone(sample_rate=16000)

    if source is None:
        print("No microphone found or selected.")
        return

    # Determine device and dtype
    device = "cuda:0" if torch.cuda.is_available(
    ) else "cpu"  # Use cuda:0 for HF compatibility
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Initialize model/pipeline variables
    audio_model = None
    hf_pipe = None
    # loaded_model_name = args.model # Removed unused variable

    if args.model == "large-v3-turbo":
        print(f"Loading Hugging Face model openai/whisper-{args.model}...")
        model_id = f"openai/whisper-{args.model}"
        try:
            # Filter warnings
            warnings.filterwarnings(
                "ignore", message="The input name `inputs` is deprecated")
            warnings.filterwarnings(
                "ignore", message="You have passed task=transcribe")

            # Add Flash Attention 2 or SDPA acceleration
            from transformers.utils import is_flash_attn_2_available

            # Modify model loading with advanced attention implementations
            if is_flash_attn_2_available():
                print("Flash Attention 2 is available - using for faster inference")
                hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="flash_attention_2"
                )
            else:
                # Fall back to SDPA if available
                print("Flash Attention 2 not available - using SDPA instead")
                hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="sdpa"  # Use PyTorch's scaled dot-product attention
                )

            # Enable static cache for better memory efficiency
            hf_model.generation_config.cache_implementation = "static"

            hf_model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)

            # Enhanced pipeline with chunking and batching
            hf_pipe = pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
                chunk_length_s=30,  # Optimal for large-v3
                batch_size=8,  # Adjust based on your GPU memory
            )
            print(
                f"Optimized Hugging Face model {model_id} loaded on {device}")

        except Exception as e:
            print(f"Error loading Hugging Face model {model_id}: {e}")
            print("Please ensure 'transformers', 'torch', 'accelerate', and 'safetensors' are installed correctly.")
            return  # Exit main if HF model fails to load

    # Only try to load standard Whisper model if we haven't already loaded a HF model
    elif not hf_pipe:  # Add this "elif" instead of "else" to prevent double loading
        # Load standard Whisper model
        print("Loading standard Whisper model, please wait...")
        model_name = args.model
        # Apply .en suffix logic only for standard models, exclude large models like large-v2, large-v3
        if model_name not in ["large", "large-v2", "large-v3"] and not args.non_english:
            model_name += ".en"
        try:
            audio_model = whisper.load_model(model_name)
            # Use the whisper lib device format ("cuda" or "cpu")
            whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
            audio_model.to(whisper_device)
            print(
                f"Standard Whisper model {model_name} loaded on {whisper_device}")
            # loaded_model_name = model_name # Removed unused variable
        except Exception as e:
            print(f"Error loading standard Whisper model {model_name}: {e}")
            print("Please ensure 'openai-whisper' is installed correctly.")
            return  # Exit main if standard model fails to load

    # Check if any model loaded successfully
    if audio_model is None and hf_pipe is None:
        print("Failed to load any transcription model. Exiting.")
        return

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    print("Adjusting for ambient noise; please stand by...")
    try:
        with source:
            recorder.adjust_for_ambient_noise(source)
            print("Done. Listening...")

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        # Start the microphone listener
        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=record_timeout
        )
    except Exception as e:
        print("\n=== Microphone Setup Note ===")
        print("Unable to initialize the microphone. This is usually a temporary issue.")
        print("Please try running the program again - no changes needed.")
        print("If the problem persists, try unplugging and reconnecting your microphone.")
        print(f"Technical details: {e}")
        print("===========================\n")
        return  # Exit if microphone initialization fails

    idle_count = 0

    while True:
        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                idle_count = 0
                phrase_complete = False
                # If enough time has passed between recordings, consider a new line
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # Skip very small chunks (likely silence)
                if len(audio_data) < 8000:
                    if debug_mode:
                        print("Skipping small audio chunk (likely silence).")
                    continue

                # Convert raw data to float32
                audio_np = np.frombuffer(
                    audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                text = ""  # Initialize text
                if hf_pipe:
                    # Use Hugging Face pipeline
                    # The pipeline handles preprocessing (feature extraction, attention mask) internally
                    # Simply pass the raw audio numpy array
                    result = hf_pipe(audio_np.copy())  # <-- CORRECTED LINE
                    text = result['text'].strip()
                elif audio_model:
                    # Use standard Whisper model
                    # Use the correct device name for standard whisper check ("cuda" or "cpu")
                    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
                    result = audio_model.transcribe(
                        audio_np,
                        fp16=(whisper_device == "cuda"),
                        language=None if args.non_english else "en"
                    )
                    text = result['text'].strip()
                else:
                    # Should not happen if loading checks passed, but include for safety
                    print("Error: No transcription model available.")
                    continue  # Skip this loop iteration

                # Filter out hallucinations with high non-ASCII character counts (likely Vietnamese, Russian, etc.)
                if len(text) > 0:
                    # Count how many non-ascii characters are in the text
                    non_ascii_count = sum(1 for c in text if ord(c) > 127)
                    non_ascii_ratio = non_ascii_count / len(text)

                    # If more than 30% of the characters are non-ASCII, it's likely a hallucination
                    if non_ascii_ratio > 0.3:
                        if debug_mode:
                            print(
                                f"Filtering out likely hallucination with {non_ascii_ratio:.1%} non-ASCII characters: '{text}'")
                        continue  # Skip this text completely

                if len(text) < 2:
                    if debug_mode:
                        print(f"Ignoring short result: '{text}'")
                    continue

                # Append instead of overwrite
                if phrase_complete:
                    transcriptions.append(text)
                else:
                    transcriptions[-1] = transcriptions[-1].strip() + \
                        " " + text

                if debug_mode:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("\nTranscription so far:")
                    for line in transcriptions:
                        print(line)
                    print(
                        f"\nVisit http://localhost:{app.config['PORT']} for the web interface.")

                gc.collect()

            else:
                # No new data
                sleep(args.sleep_duration)
                idle_count += 1
                # Periodic GC
                if idle_count > 20:
                    gc.collect()
                    idle_count = 0

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()

    print("\nFinal Transcription:")
    for line in transcriptions:
        print(line)

    print("Cleaning up resources...")
    del audio_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        print("\nExiting...")
