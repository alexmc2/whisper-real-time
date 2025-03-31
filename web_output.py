import argparse
import gc
import os
import threading
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep

import numpy as np
import speech_recognition as sr
import torch
import whisper
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
transcriptions = [
    "Welcome! Start speaking to see the transcription appear here."
]

# Additional configuration
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['JSON_SORT_KEYS'] = False

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Transcription</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        #transcription {
            display: block;
            width: 100%;
            height: 300px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            font-size: 1em;
            line-height: 1.6;
            resize: vertical;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
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
        .status {
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }
        .edit-mode {
            border: 2px solid #2196F3 !important;
        }
    </style>
</head>
<body>
    <h1>Whisper Real-Time Transcription</h1>
    <div class="controls">
        <button id="copyBtn">Copy All Text</button>
        <button id="editBtn" class="blue">Edit Text</button>
        <button id="saveBtn" class="blue" style="display:none">Save Changes</button>
        <button id="clearBtn" class="red">Clear All</button>
    </div>
    <div id="status" class="status"></div>
    <textarea id="transcription" readonly></textarea>
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
        
        // Clear
        document.getElementById('clearBtn').addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all transcription text?')) {
                fetch('/clear_transcription', { method: 'POST' })
                    .then(() => {
                        transcriptionEl.value = '';
                        showStatus('Transcription cleared.');
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showStatus('Error clearing transcription.');
                    });
            }
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
            setTimeout(() => { statusEl.textContent = ''; }, 3000);
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


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/get_transcription')
def get_transcription():
    global transcriptions
    return jsonify({'text': transcriptions if transcriptions else ["Waiting for speech..."]})


@app.route('/clear_transcription', methods=['POST'])
def clear_transcription():
    global transcriptions
    transcriptions = []
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
                        choices=["tiny", "base", "small", "medium", "large"])
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
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='default',
                            help="Default microphone name. Use 'list' to show all.", type=str)
    args = parser.parse_args()

    # Set initial port
    app.config['PORT'] = args.port

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
            "Welcome! Start speaking to see the transcription appear here."
        ]
    elif len(transcriptions) == 1 and not transcriptions[0]:
        transcriptions = [
            "Welcome! Start speaking to see the transcription appear here."
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

    print("Loading Whisper model, please wait...")
    model_name = args.model
    if model_name != "large" and not args.non_english:
        model_name += ".en"
    audio_model = whisper.load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model.to(device)
    print(f"Model {model_name} loaded on {device}")

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
            now = datetime.utcnow()
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

                result = audio_model.transcribe(
                    audio_np,
                    fp16=(device == "cuda"),
                    language=None if args.non_english else "en"
                )
                text = result['text'].strip()

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
