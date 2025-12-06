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

import google.generativeai as genai
import numpy as np
import speech_recognition as sr
import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from autosave import install_crash_handler, save_transcription_to_log, start_auto_save
from gemini_cleaning import setup_gemini_api
from transcription_state import DEFAULT_WAITING_MESSAGE, TranscriptionState
from web_routes import create_app

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "whisper_app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("whisper_app")


def run_flask(app) -> None:
    port = app.config.get("PORT", 5000)
    max_attempts = 10

    for attempt in range(max_attempts):
        try:
            app.run(
                debug=False,
                host="0.0.0.0",
                port=port,
                threaded=True,
                processes=1,
                use_reloader=False,
            )
            break
        except OSError as exc:  # noqa: PERF203
            if "Address already in use" in str(exc) and attempt < max_attempts - 1:
                print(f"Port {port} is busy, trying {port + 1}...")
                port += 1
                app.config["PORT"] = port
            else:
                print(f"Failed to start web server: {exc}")
                print(
                    "Try manually killing the process using the port with 'sudo fuser -k PORT/tcp'"
                )
                break


def run_gemini_diagnostics() -> None:
    try:
        model_names_to_try = [
            "models/gemini-2.5-pro-exp-03-25",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/gemini-1.5-pro",
        ]

        logger.info("Testing model access with different name formats:")

        for model_name in model_names_to_try:
            try:
                logger.info("Trying to access: %s", model_name)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                preview = response.text[:20] if response and response.text else ""
                logger.info("SUCCESS: Model %s is accessible. Response: '%s...'", model_name, preview)
            except Exception as exc:  # noqa: BLE001
                logger.error("FAILED: Model %s error: %s", model_name, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in Gemini diagnostics: %s", exc)
        logger.exception("Full exception details:")


def configure_app_from_args(app, args) -> None:
    app.config["PORT"] = args.port
    app.config["GEMINI_API_KEY"] = args.gemini_api_key
    app.config["AUTO_SAVE_INTERVAL"] = args.auto_save_interval

    if args.log_dir:
        app.config["TRANSCRIPTION_LOG_DIR"] = os.path.expanduser(args.log_dir)
    os.makedirs(app.config["TRANSCRIPTION_LOG_DIR"], exist_ok=True)


def prepare_microphone(args):
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are:")
            for _, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return None
        print(f"Using microphone '{mic_name}'")
        try:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    print(f"Found matching microphone: {name}")
                    return sr.Microphone(sample_rate=16000, device_index=index)
            print(f"Microphone '{mic_name}' not found. Using default/first microphone...")
            return sr.Microphone(sample_rate=16000)
        except Exception as exc:  # noqa: BLE001
            print(f"Error selecting microphone: {exc}")
            print("Falling back to default microphone...")
            return sr.Microphone(sample_rate=16000)

    return sr.Microphone(sample_rate=16000)


def load_transcription_model(args, torch_dtype, device):
    audio_model = None
    hf_pipe = None

    if args.model == "large-v3-turbo":
        print(f"Loading Hugging Face model openai/whisper-{args.model}...")
        model_id = f"openai/whisper-{args.model}"
        try:
            from transformers.utils import is_flash_attn_2_available

            if is_flash_attn_2_available():
                print("Flash Attention 2 is available - using for faster inference")
                hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                print("Flash Attention 2 not available - using SDPA instead")
                hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="sdpa",
                )

            hf_model.generation_config.cache_implementation = "static"

            hf_model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            # Ensure HF pipeline provides attention_mask so generation does not warn when pad_token_id == eos_token_id.
            processor.feature_extractor.return_attention_mask = True

            hf_pipe = pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
                chunk_length_s=30,
                batch_size=8,
            )
            print(f"Optimized Hugging Face model {model_id} loaded on {device}")
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading Hugging Face model {model_id}: {exc}")
            print(
                "Please ensure 'transformers', 'torch', 'accelerate', and 'safetensors' are installed correctly."
            )
            return None, None

    if not hf_pipe:
        print("Loading standard Whisper model, please wait...")
        model_name = args.model
        if model_name not in ["large", "large-v2", "large-v3"] and not args.non_english:
            model_name += ".en"
        try:
            audio_model = whisper.load_model(model_name)
            whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
            audio_model.to(whisper_device)
            print(f"Standard Whisper model {model_name} loaded on {whisper_device}")
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading standard Whisper model {model_name}: {exc}")
            print("Please ensure 'openai-whisper' is installed correctly.")
            return None, None

    return audio_model, hf_pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="small", help="Model to use", choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"]
    )
    parser.add_argument("--non_english", action="store_true", help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument(
        "--record_timeout", default=2, type=float, help="Approx chunk length for each background capture."
    )
    parser.add_argument(
        "--phrase_timeout", default=3, type=float, help="Silence (in seconds) before we consider it a new line."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output (prints to console).")
    parser.add_argument("--cpu_threads", default=4, type=int, help="Number of CPU threads to use for inference.")
    parser.add_argument("--sleep_duration", default=0.25, type=float, help="Sleep time in main loop when idle (seconds).")
    parser.add_argument("--port", default=5000, type=int, help="Port to run the web server on.")
    parser.add_argument(
        "--gemini_api_key", default="", type=str, help="Google Gemini API key for the clean transcription feature."
    )
    parser.add_argument("--gemini_debug", action="store_true", help="Run Gemini API diagnostics at startup.")
    parser.add_argument(
        "--auto_save_interval", default=600, type=int, help="Auto-save transcription interval in seconds. Use 0 to disable."
    )
    parser.add_argument(
        "--log_dir",
        default="",
        type=str,
        help="Custom directory for saving transcription logs. Default is ~/transcription_logs.",
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone", default="default", help="Default microphone name. Use 'list' to show all.", type=str
        )
    args = parser.parse_args()

    state = TranscriptionState()
    app = create_app(state, logger)
    configure_app_from_args(app, args)

    install_crash_handler(state, app.config["TRANSCRIPTION_LOG_DIR"], app.config, logger)

    if app.config["AUTO_SAVE_INTERVAL"] > 0:
        start_auto_save(
            app.config["AUTO_SAVE_INTERVAL"],
            state,
            app.config["TRANSCRIPTION_LOG_DIR"],
            app.config,
            logger,
        )
        logger.info(
            "Auto-save enabled: Saving transcriptions every %s seconds",
            app.config["AUTO_SAVE_INTERVAL"],
        )
        logger.info("Transcription logs will be saved to: %s", app.config["TRANSCRIPTION_LOG_DIR"])
    else:
        logger.info("Auto-save disabled. Transcriptions will only be saved on exit or error.")

    logger.info("Starting Whisper Transcription App")
    logger.info("Using Whisper model: %s", args.model)

    if args.gemini_api_key:
        logger.info("Gemini API key provided, setting up...")
        setup_gemini_api(args.gemini_api_key, logger)
        if args.gemini_debug:
            logger.info("Running comprehensive Gemini API diagnostics...")
            run_gemini_diagnostics()
    else:
        logger.warning("No Gemini API key provided. Clean transcription will not be available.")

    flask_thread = threading.Thread(target=run_flask, args=(app,), daemon=True)
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

    state.ensure_placeholder()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = prepare_microphone(args)
    if source is None:
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    audio_model, hf_pipe = load_transcription_model(args, torch_dtype, device)

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

        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=record_timeout
        )
    except Exception as exc:  # noqa: BLE001
        print("\n=== Microphone Setup Note ===")
        print("Unable to initialize the microphone. This is usually a temporary issue.")
        print("Please try running the program again - no changes needed.")
        print("If the problem persists, try unplugging and reconnecting your microphone.")
        print(f"Technical details: {exc}")
        print("===========================\n")
        return

    idle_count = 0
    last_logged_text = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                idle_count = 0
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b"".join(data_queue.queue)
                data_queue.queue.clear()

                if len(audio_data) < 8000:
                    if debug_mode:
                        print("Skipping small audio chunk (likely silence).")
                    continue

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                text = ""
                if hf_pipe:
                    result = hf_pipe(audio_np.copy())
                    text = result["text"].strip()
                elif audio_model:
                    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
                    result = audio_model.transcribe(
                        audio_np,
                        fp16=(whisper_device == "cuda"),
                        language=None if args.non_english else "en",
                    )
                    text = result["text"].strip()
                else:
                    print("Error: No transcription model available.")
                    continue

                if len(text) > 0:
                    non_ascii_count = sum(1 for c in text if ord(c) > 127)
                    non_ascii_ratio = non_ascii_count / len(text)

                    if non_ascii_ratio > 0.3:
                        if debug_mode:
                            print(
                                f"Filtering out likely hallucination with {non_ascii_ratio:.1%} non-ASCII characters: '{text}'"
                            )
                        continue

                if len(text) < 2:
                    if debug_mode:
                        print(f"Ignoring short result: '{text}'")
                    continue

                state.ensure_placeholder()

                if phrase_complete:
                    state.transcriptions.append(text)
                else:
                    if state.transcriptions and state.transcriptions[-1] == DEFAULT_WAITING_MESSAGE:
                        state.transcriptions[-1] = text
                    else:
                        state.transcriptions[-1] = state.transcriptions[-1].strip() + " " + text

                if debug_mode:
                    os.system("cls" if os.name == "nt" else "clear")
                    print("\nTranscription so far:")
                    for line in state.transcriptions:
                        print(line)
                    print(f"\nVisit http://localhost:{app.config['PORT']} for the web interface.")

                current_line = state.transcriptions[-1].strip()
                if (
                    current_line
                    and current_line != DEFAULT_WAITING_MESSAGE
                    and current_line != last_logged_text
                ):
                    logger.info("Transcript update: %s", current_line)
                    last_logged_text = current_line

                gc.collect()

            else:
                sleep(args.sleep_duration)
                idle_count += 1
                if idle_count > 20:
                    gc.collect()
                    idle_count = 0

        except KeyboardInterrupt:
            logger.info("Manual exit detected. Saving transcription before exiting.")
            save_transcription_to_log(
                state, app.config["TRANSCRIPTION_LOG_DIR"], app.config, logger
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            logger.error("Error occurred during processing: %s", exc)
            save_transcription_to_log(
                state, app.config["TRANSCRIPTION_LOG_DIR"], app.config, logger
            )
            if debug_mode:
                import traceback

                traceback.print_exc()

    print("\nFinal Transcription:")
    for line in state.transcriptions:
        print(line)

    save_transcription_to_log(state, app.config["TRANSCRIPTION_LOG_DIR"], app.config, logger)

    print("Transcription saved to log file in:", app.config["TRANSCRIPTION_LOG_DIR"])
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
