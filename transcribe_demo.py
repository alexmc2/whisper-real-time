# # transcribe_demo.py

# import argparse
# import importlib.util  # Added for module availability check
# import logging  # Added for better output
# import os
# from datetime import datetime, timedelta
# from queue import Queue
# from sys import platform
# from time import sleep

# # --- Dependency Imports ---
# try:
#     import google.generativeai as genai
#     from google.generativeai.types import HarmBlockThreshold, HarmCategory
# except ImportError:
#     print("WARNING: google.generativeai library not found. AI Cleaning (--clean) will not work. pip install google-generativeai")
#     genai = None
#     HarmCategory = None
#     HarmBlockThreshold = None

# try:
#     import numpy as np
# except ImportError:
#     print("ERROR: numpy library not found. pip install numpy")
#     exit(1)

# try:
#     import speech_recognition as sr
# except ImportError:
#     print("ERROR: SpeechRecognition library not found. pip install SpeechRecognition")
#     exit(1)

# try:
#     import torch
# except ImportError:
#     print("ERROR: PyTorch library not found. Install from https://pytorch.org/")
#     exit(1)

# try:
#     import whisper
# except ImportError:
#     print("ERROR: openai-whisper library not found. pip install -U openai-whisper")
#     exit(1)

# try:
#     from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#     if torch.cuda.is_available():
#         if not importlib.util.find_spec("accelerate"):
#             print("WARNING: 'accelerate' not found, may impact HF GPU performance.")
# except ImportError:
#     print("WARNING: 'transformers' library not found. HF models ('large-v3-turbo') unavailable.")

#     class AutoModelForSpeechSeq2Seq:
#         pass

#     class AutoProcessor:
#         pass

#     def pipeline(*args, **kwargs): return None
#     hf_pipe_available = False
# else:
#     hf_pipe_available = True


# # --- Logging Setup (Simple for CLI) ---
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("TranscribeDemo")


# # --- Updated Gemini Cleaning Function ---
# def clean_transcription_with_gemini(text_lines, api_key):
#     """Clean transcription using Google Gemini API (CLI version)."""
#     if not genai or not HarmCategory or not HarmBlockThreshold:
#         logger.error("Gemini library not available. Cleaning skipped.")
#         return text_lines
#     if not api_key:
#         logger.error(
#             "No Gemini API key provided via --gemini_api_key. Cleaning skipped.")
#         return text_lines

#     # Join lines using literal \n for the prompt
#     text_to_clean = "\\n".join(text_lines)
#     logger.info(f"Attempting AI cleaning on {len(text_lines)} lines...")
#     logger.debug(f"Text for cleaning:\n---\n{text_to_clean}\n---")

#     try:
#         genai.configure(api_key=api_key)
#         # Test connection by listing available models
#         models = genai.list_models()
#         gemini_models = [
#             model for model in models if "gemini" in model.name.lower()]

#         # Log available Gemini models
#         logger.info(f"Found {len(gemini_models)} Gemini models")
#         gemini_25_models = [
#             model for model in gemini_models if "2.5" in model.name]
#         if gemini_25_models:
#             logger.info("Found Gemini 2.5 models:")
#             for model in gemini_25_models:
#                 logger.info(f"  - {model.name}")

#     except Exception as e:
#         logger.error(f"Failed to configure or connect to Gemini API: {e}")
#         logger.error("Cleaning aborted. Please check API key and connection.")
#         return text_lines  # Return original on setup failure

#     # Updated to try Gemini 2.5 first, then fall back to other models
#     model_names_to_try = [
#         # Primary - Gemini 2.5 Pro (correct format)
#         "models/gemini-2.5-pro-exp-03-25",
#         "models/gemini-1.5-pro-latest",     # Fallback 1
#         "gemini-1.5-pro"                    # Fallback 2
#     ]
#     cleaned_text_from_gemini = None
#     success_model = None
#     last_error = "No models attempted."

#     for model_name in model_names_to_try:
#         try:
#             logger.info(f"Trying cleaning model: {model_name}")
#             safety_settings = {
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             }
#             model = genai.GenerativeModel(
#                 model_name=model_name,
#                 safety_settings=safety_settings,
#                 generation_config={"temperature": 0.3, "top_p": 0.95}
#             )

#             # Same refined prompt as web_output.py
#             prompt = f"""
#             You are an expert transcription cleaner. Refine the following raw speech transcription by applying these rules STRICTLY:

#             1.  **Remove Redundancy & Fillers:**
#                 *   Delete filler words/sounds (e.g., "um", "uh", "ah", "like", "you know", "so", "well", "okay").
#                 *   Delete unintentional immediate word repetitions (e.g., "the the cat" -> "the cat").
#                 *   If a short phrase repeats many times consecutively (e.g., "Thank you. Thank you. Thank you."), keep only ONE instance. If the phrase seems like a verbal tic with no semantic value in context, remove it entirely.

#             2.  **Handle Line Breaks & Fragments:**
#                 *   Preserve meaningful line breaks that separate distinct thoughts or sentences.
#                 *   Delete entire lines containing only a single, common, isolated word likely to be a speech fragment or error (e.g., delete lines that are just "you", "the", "a", "is", "it", "i", "and", "but", "so", "okay", "yeah"). Do NOT delete meaningful single-word lines (e.g., "Yes.", "No.", "Right.", "Okay?" used as a question).

#             3.  **Minor Corrections (Use Sparingly):**
#                 *   Correct obvious, high-confidence transcription errors ONLY if the correction is clear and doesn't alter meaning (e.g., "recognise speech" -> "recognize speech" if context is US English, "undue" -> "undo" if context implies an action). Be very conservative. Avoid changing proper nouns unless clearly misspelled.
#                 *   Ensure basic sentence capitalization (start lines/sentences with caps) and add terminal punctuation (.?!) ONLY where clearly appropriate and missing based on conversational flow. Do not over-punctuate or force formal sentence structure onto natural speech.

#             4.  **Constraints:**
#                 *   **DO NOT** add new information or invent content.
#                 *   **DO NOT** significantly rephrase sentences or change the speaker's original meaning. Focus on removal and very minor fixes.
#                 *   **DO NOT** merge separate thoughts/sentences across line breaks unless fixing an obvious run-on error where the break occurred mid-thought.

#             Apply these rules meticulously to the following transcription, which uses "\\n" to denote line breaks:

#             --- TRANSCRIPTION START ---
#             {text_to_clean}
#             --- TRANSCRIPTION END ---

#             Return ONLY the cleaned transcription text, using "\\n" for line breaks. Do not add any introductory or concluding remarks, apologies, or explanations. Just the cleaned text.
#             """

#             logger.info(f"Sending cleaning request to {model_name}...")
#             response = model.generate_content(prompt)

#             if hasattr(response, 'text') and response.text and response.text.strip():
#                 cleaned_text_from_gemini = response.text.strip()
#                 logger.info(
#                     f"Successfully received response from {model_name}: '{response.text[:30]}...'")
#                 success_model = model_name
#                 break  # Success
#             else:
#                 reason = "Unknown reason (empty response)"
#                 if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
#                     reason = f"Blocked: {response.prompt_feedback.block_reason}"
#                 logger.warning(
#                     f"Model {model_name} returned empty/blocked response. Reason: {reason}")
#                 last_error = f"Empty/blocked from {model_name} ({reason})"

#         except Exception as e:
#             logger.error(f"Exception using model {model_name}: {str(e)}")
#             last_error = f"Error with {model_name}: {str(e)}"
#             if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e):
#                 logger.critical(
#                     "Gemini API key invalid/permission denied. Aborting cleaning.")
#                 return text_lines  # Return original
#             continue  # Try next model

#     # --- Post-Gemini Processing ---
#     if cleaned_text_from_gemini is None:
#         logger.error(
#             f"All Gemini cleaning attempts failed. Last error: {last_error}")
#         return text_lines  # Return original

#     logger.info(
#         f"Raw cleaned text received ({success_model}):\n---\n{cleaned_text_from_gemini}\n---")

#     # Process the cleaned text (split by literal \n, strip, filter empty)
#     lines_from_gemini = [line.strip()
#                          for line in cleaned_text_from_gemini.split('\\n')]
#     lines_after_empty_removal = [line for line in lines_from_gemini if line]

#     if not lines_after_empty_removal:
#         logger.warning("AI cleaning resulted in empty text after processing.")
#         return ["[AI cleaning resulted in empty text]"]

#     # Remove consecutive identical lines
#     final_cleaned_lines = []
#     if lines_after_empty_removal:
#         final_cleaned_lines.append(lines_after_empty_removal[0])
#         for i in range(1, len(lines_after_empty_removal)):
#             if lines_after_empty_removal[i] != final_cleaned_lines[-1]:
#                 final_cleaned_lines.append(lines_after_empty_removal[i])
#             else:
#                 logger.debug(
#                     f"Post-proc removed consecutive duplicate: '{lines_after_empty_removal[i]}'")

#     # Remove single-word fragment lines
#     fragment_words = {"you", "the", "a", "an", "and", "but", "or", "so", "is", "it", "i", "he",
#                       "she", "they", "we", "um", "uh", "ah", "oh", "hmm", "yeah", "ok", "okay", "like", "well"}
#     lines_after_fragment_removal = []
#     for line in final_cleaned_lines:
#         if line.lower() in fragment_words:
#             logger.debug(f"Post-proc removed fragment line: '{line}'")
#             continue
#         lines_after_fragment_removal.append(line)

#     logger.info(
#         f"AI cleaning successful using {success_model} + post-processing.")
#     return lines_after_fragment_removal if lines_after_fragment_removal else ["[AI cleaning removed all content]"]


# # --- Main Function ---
# def main():
#     parser = argparse.ArgumentParser(
#         description="Real-time Whisper transcription (CLI version) with optional AI cleaning.")
#     parser.add_argument("--model", default="small", help="Whisper model size", choices=[
#                         "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"])
#     parser.add_argument("--non_english", action='store_true',
#                         help="Use multilingual model")
#     parser.add_argument("--energy_threshold", default=1000,
#                         help="Mic energy threshold", type=int)
#     parser.add_argument("--record_timeout", default=2.0,
#                         help="Audio chunk duration (sec)", type=float)
#     parser.add_argument("--phrase_timeout", default=3.0,
#                         help="Silence duration for new line (sec)", type=float)
#     parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_API_KEY",
#                         ""), help="Google Gemini API key for cleaning (or use env var)")
#     parser.add_argument("--clean", action='store_true',
#                         help="Auto-clean transcription after completion using Gemini (requires API key)")
#     parser.add_argument("--debug", action='store_true',
#                         help="Enable verbose debug logging")
#     parser.add_argument("--gemini_debug", action='store_true',
#                         help="Print detailed information about available Gemini models")
#     if 'linux' in platform:
#         parser.add_argument("--default_microphone", default='default',
#                             help="Substring of mic name (Linux), or 'list'")
#     args = parser.parse_args()

#     if args.debug:
#         logger.setLevel(logging.DEBUG)
#         logging.getLogger().setLevel(logging.DEBUG)
#         logger.info("Debug logging enabled.")

#     # Gemini Debug Info if requested
#     if args.gemini_api_key and args.gemini_debug and genai:
#         try:
#             logger.info("Running Gemini API diagnostics...")
#             genai.configure(api_key=args.gemini_api_key)
#             models = genai.list_models()
#             gemini_models = [
#                 model for model in models if "gemini" in model.name.lower()]

#             logger.info(f"Available Gemini models ({len(gemini_models)}):")
#             for model in gemini_models:
#                 logger.info(
#                     f"- {model.name} (supported: {model.supported_generation_methods})")

#             # Check specifically for Gemini 2.5 models
#             gemini_25_models = [
#                 model for model in gemini_models if "2.5" in model.name]
#             if gemini_25_models:
#                 logger.info("FOUND GEMINI 2.5 MODELS:")
#                 for model in gemini_25_models:
#                     logger.info(f"  - {model.name}")
#             else:
#                 logger.info("No Gemini 2.5 models found in API list.")

#             # Test if we can use the model
#             test_model_names = [
#                 "models/gemini-2.5-pro-exp-03-25",
#                 "models/gemini-1.5-pro",
#                 "gemini-1.5-pro"
#             ]

#             for model_name in test_model_names:
#                 try:
#                     logger.info(f"Testing access to: {model_name}")
#                     model = genai.GenerativeModel(model_name)
#                     response = model.generate_content("Hello")
#                     logger.info(
#                         f"✓ Success! Model {model_name} response: '{response.text[:30]}...'")
#                     break
#                 except Exception as e:
#                     logger.error(f"× Failed to access {model_name}: {e}")
#         except Exception as e:
#             logger.error(f"Error during Gemini diagnostics: {e}")

#     # --- Audio / Whisper Initialization ---
#     phrase_time = None
#     last_sample = bytes()
#     data_queue = Queue()
#     transcription = ['']  # Start with an empty string in the list

#     try:
#         recorder = sr.Recognizer()
#         recorder.energy_threshold = args.energy_threshold
#         recorder.dynamic_energy_threshold = False

#         # Mic Setup
#         source = None
#         mic_name = "Default"
#         if 'linux' in platform and args.default_microphone != 'default':
#             mic_name = args.default_microphone
#             if mic_name == 'list':
#                 logger.info("Available microphones:")
#                 for i, name in enumerate(sr.Microphone.list_microphone_names()):
#                     logger.info(f" {i}: \"{name}\"")
#                 return
#             else:
#                 found_mic_index = next((i for i, name in enumerate(
#                     sr.Microphone.list_microphone_names()) if mic_name.lower() in name.lower()), None)
#                 if found_mic_index is not None:
#                     mic_name = sr.Microphone.list_microphone_names()[
#                         found_mic_index]
#                     logger.info(
#                         f"Using mic: Index {found_mic_index} - \"{mic_name}\"")
#                     source = sr.Microphone(
#                         sample_rate=16000, device_index=found_mic_index)
#                 else:
#                     logger.warning(
#                         f"Mic '{args.default_microphone}' not found. Using default.")
#                     mic_name = "Default (Fallback)"
#         if source is None:
#             source = sr.Microphone(sample_rate=16000)
#             logger.info(f"Using microphone: {mic_name}")

#         logger.info("Adjusting for ambient noise...")
#         with source:
#             recorder.adjust_for_ambient_noise(source, duration=1.0)
#         logger.info(
#             f"Noise adjustment done. Threshold: {recorder.energy_threshold:.2f}")

#     except Exception as e:
#         logger.error(f"Audio setup failed: {e}")
#         return

#     # Load Model
#     device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     logger.info(
#         f"Loading Whisper model ({args.model}) on {device_type} ({torch_dtype})...")

#     audio_model = None
#     hf_pipe = None
#     loaded_model_name = args.model

#     try:
#         if args.model == "large-v3-turbo" and hf_pipe_available:
#             model_id = "openai/whisper-large-v3"
#             logger.warning(
#                 f"Attempting HF model '{model_id}' for 'large-v3-turbo'.")
#             hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
#                 model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(device_type)
#             processor = AutoProcessor.from_pretrained(model_id)
#             hf_pipe = pipeline("automatic-speech-recognition", model=hf_model, tokenizer=processor.tokenizer,
#                                feature_extractor=processor.feature_extractor, chunk_length_s=30, stride_length_s=5, torch_dtype=torch_dtype, device=device_type)
#             loaded_model_name = model_id
#         elif args.model == "large-v3-turbo" and not hf_pipe_available:
#             logger.error(
#                 "Transformers library not found. Cannot load 'large-v3-turbo'. Falling back to 'large-v3'.")
#             args.model = "large-v3"

#         if audio_model is None and hf_pipe is None:
#             model_name = args.model
#             if model_name not in ["large", "large-v2", "large-v3"] and not args.non_english:
#                 model_name += ".en"
#             whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
#             audio_model = whisper.load_model(model_name, device=whisper_device)
#             loaded_model_name = model_name
#         logger.info(f"Model '{loaded_model_name}' loaded.")

#     except Exception as e:
#         logger.error(f"Failed to load model '{loaded_model_name}': {e}")
#         return

#     # Background Recording Callback
#     def record_callback(_, audio: sr.AudioData) -> None:
#         data = audio.get_raw_data()
#         data_queue.put(data)

#     stop_listening = recorder.listen_in_background(
#         source, record_callback, phrase_time_limit=args.record_timeout)
#     logger.info("\nModel loaded. Listening... Press Ctrl+C to stop.\n")

#     # Main Loop
#     try:
#         while True:
#             try:
#                 now = datetime.utcnow()
#                 if not data_queue.empty():
#                     phrase_complete = phrase_time and now - \
#                         phrase_time > timedelta(seconds=args.phrase_timeout)
#                     if phrase_complete:
#                         last_sample = bytes()
#                     phrase_time = now

#                     while not data_queue.empty():
#                         last_sample += data_queue.get()
#                     audio_np = np.frombuffer(
#                         last_sample, dtype=np.int16).astype(np.float32) / 32768.0

#                     if len(audio_np) < 1000:
#                         continue  # Skip fragments

#                     result_text = ""
#                     if hf_pipe:
#                         output = hf_pipe(audio_np.copy(), generate_kwargs={
#                                          "task": "transcribe", "language": "en" if not args.non_english else None}, return_timestamps=False)
#                         result_text = output['text'].strip()
#                     elif audio_model:
#                         result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(
#                         ), language=None if args.non_english else "en")
#                         result_text = result['text'].strip()

#                     if result_text:
#                         if phrase_complete or transcription == ['']:
#                             if transcription == ['']:
#                                 # Overwrite initial empty string
#                                 transcription[0] = result_text
#                             else:
#                                 transcription.append(result_text)
#                         else:
#                             transcription[-1] = transcription[-1].strip() + \
#                                 " " + result_text

#                         # Dynamic output to console
#                         os.system('cls' if os.name == 'nt' else 'clear')
#                         print("\n--- Current Transcription ---\n")
#                         for line in transcription:
#                             print(line)
#                         print(
#                             "\n-----------------------------\nListening... Press Ctrl+C to stop.", end='', flush=True)

#                         last_sample = bytes()  # Clear buffer after processing

#                 else:
#                     sleep(0.25)  # Wait if queue empty

#             except Exception as loop_e:
#                 logger.error(f"Error in loop: {loop_e}")
#                 sleep(1)  # Avoid fast error loops

#     except KeyboardInterrupt:
#         logger.info("\nKeyboard interrupt received.")
#     finally:
#         logger.info("Stopping listener...")
#         if 'stop_listening' in locals():
#             stop_listening(wait_for_stop=False)

#         # Final Output
#         print("\n\n" + "="*50 + "\nFinal Raw Transcription:\n" + "="*50)
#         # Filter out the initial empty string if it's still there and nothing else was added
#         final_raw = [line for line in transcription if line]
#         if not final_raw:
#             print("[No transcription recorded]")
#         else:
#             for line in final_raw:
#                 print(line)
#         print("="*50)

#         # Clean if requested and possible
#         if args.clean and final_raw:
#             logger.info("\nAttempting AI cleaning...")
#             cleaned_transcription = clean_transcription_with_gemini(
#                 final_raw, args.gemini_api_key)

#             print("\n" + "="*50 + "\nCleaned Transcription:\n" + "="*50)
#             if not cleaned_transcription or cleaned_transcription == ["[AI cleaning resulted in empty text]"] or cleaned_transcription == ["[AI cleaning removed all content]"]:
#                 print("[Cleaning resulted in no content or failed]")
#             else:
#                 for line in cleaned_transcription:
#                     print(line)
#             print("="*50)
#         elif args.clean and not final_raw:
#             logger.info("Skipping cleaning as there is no raw transcription.")

#         logger.info("Exiting.")


# # --- Entry Point ---
# if __name__ == "__main__":
#     if not torch or not np or not sr or not whisper:
#         logger.critical("Core dependencies missing. Exiting.")
#         exit(1)
#     try:
#         main()
#     except KeyboardInterrupt:
#         logger.info("Application terminated.")
#     except Exception as e:
#         logger.critical(f"Unhandled critical error in main: {e}")
#         logger.exception("See details:")
