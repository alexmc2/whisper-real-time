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
        
        <div id="status" class="status"></div>
        
        <textarea id="transcription" readonly></textarea>
    </div>

    <script>
        let isEditing = false;
        let autoUpdate = true;
        const transcriptionEl = document.getElementById('transcription');
        const editBtn = document.getElementById('editBtn');
        const saveBtn = document.getElementById('saveBtn');
        const statusEl = document.getElementById('status');
        
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
                fetch('/clear_transcription', { method: 'POST' })
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
            showStatus('Cleaning transcription...');
            fetch('/clean_transcription', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showStatus('Transcription cleaned successfully!');
                        
                        document.getElementById('undoCleanBtn').style.display = 'inline-block'; // Show undo button
                        document.getElementById('resetCleanBtn').style.display = 'inline-block'; // Show reset button
                        // No need to disable the clean button - we want to allow multiple cleanings
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
            fetch('/undo_cleaning', { method: 'POST' })
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
            fetch('/reset_cleaning_history', { method: 'POST' })
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
        if not transcriptions or len(transcriptions) == 0:
            return jsonify({'status': 'error', 'message': 'No transcription to clean.'})

        # Store original transcription for undo feature ONLY if we don't already have an original saved
        if not original_transcriptions:
            original_transcriptions = transcriptions.copy()

        # Join all transcription lines
        text_to_clean = "\n".join(transcriptions)

        # Log what we're trying to do
        logger.info("Attempting to clean transcription with Gemini API")

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

                # Set up the model with safety settings
                model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=safety_settings,
                    generation_config={"temperature": 0.2}
                )

                # Create the prompt for aggressive cleaning
                prompt = f"""
                    Clean up the following transcription by:
                    1. AGGRESSIVELY removing repeated phrases or words that appear consecutively or across line breaks
                    2. Removing sequences of identical phrases like "Thank you. Thank you. Thank you."
                    3. Removing isolated single words that are likely speech fragments (e.g., standalone "you")
                    4. Removing filler words like "um", "uh", "like", etc.
                    5. Consolidating multiple consecutive identical phrases into a single instance
                    6. Preserving one instance of meaningful phrases even if repeated
                    7. Join sentences that end unnaturally with a period with the next sentence
                    8. Replace uppercase letters with lowercase letters if they occur unnaturally in the middle of a sentence
                    
                    IMPORTANT RULES:
                    - Treat each line break as a potential silence marker in speech
                    - When the same phrase repeats across multiple lines, keep only ONE instance
                    - DO NOT add new content or change the meaning of what was said
                    - DO NOT rewrite or rephrase the content beyond removing repetitions
                    - If a phrase is repeated more than twice consecutively, keep only ONE instance
                    - Remove any single-word lines that are fragments like just "you" or "the"
                    
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

        # Additional post-processing to remove any remaining single-word fragments
        filtered_lines = []
        for line in cleaned_lines:
            # Skip single-word lines that are likely fragments
            if len(line.split()) == 1 and len(line) < 5 and line.lower() in ["you", "the", "a", "and", "but", "or", "so", "ah", "oh", "uh", "um"]:
                continue
            filtered_lines.append(line)

        transcriptions = filtered_lines if filtered_lines else [
            "[Cleaned transcription - no significant content detected]"]

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

    # Find this section in your code (around line 584):
    if args.model == "large-v3-turbo":
        print(f"Loading Hugging Face model openai/whisper-{args.model}...")
        model_id = f"openai/whisper-{args.model}"
        try:
            # Filter warnings
            warnings.filterwarnings(
                "ignore", message="The input name `inputs` is deprecated")
            warnings.filterwarnings(
                "ignore", message="You have passed task=transcribe")

            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            hf_model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            hf_pipe = pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            print(f"Hugging Face model {model_id} loaded on {device}")
            # loaded_model_name = model_id # Removed unused variable
        except Exception as e:
            print(f"Error loading Hugging Face model {model_id}: {e}")
            print("Please ensure 'transformers', 'torch', 'accelerate', and 'safetensors' are installed correctly.")
            return  # Exit main if HF model fails to load

    else:
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
    with source:
        recorder.adjust_for_ambient_noise(source)
    print("Done. Listening...")

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    # We do NOT store the returned function -> no Ruff unused-variable issue
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )

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
                    # The pipeline handles device and dtype automatically based on setup
                    # Language is also typically handled, or can be set via generate_kwargs if needed
                    # Pass a copy to avoid potential issues
                    result = hf_pipe(audio_np.copy())
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
