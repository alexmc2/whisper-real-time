#! python3.7

import argparse
import os
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep

import numpy as np
import speech_recognition as sr
import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline # Added for HF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"]) # Added large-v3-turbo
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real-time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="Silence (in seconds) before we consider it a new phrase/line.")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    phrase_time = None
    data_queue = Queue()
    transcription = ['']

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Select microphone (Linux)
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Determine device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use cuda:0 for HF compatibility
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Initialize model/pipeline variables
    audio_model = None
    hf_pipe = None
    # loaded_model_name = args.model # Removed unused variable

    if args.model == "large-v3-turbo":
        print(f"Loading Hugging Face model openai/whisper-{args.model}...")
        model_id = f"openai/whisper-{args.model}"
        try:
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
            return # Exit main if HF model fails to load

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
            print(f"Standard Whisper model {model_name} loaded on {whisper_device}")
            # loaded_model_name = model_name # Removed unused variable
        except Exception as e:
            print(f"Error loading standard Whisper model {model_name}: {e}")
            print("Please ensure 'openai-whisper' is installed correctly.")
            return # Exit main if standard model fails to load

    # Check if any model loaded successfully
    if audio_model is None and hf_pipe is None:
        print("Failed to load any transcription model. Exiting.")
        return

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start background listening (no unused variable)
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )

    print("Model loaded and listening! Speak into your microphone.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed, we consider it a new phrase.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # Convert to float32
                audio_np = np.frombuffer(
                    audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                text = "" # Initialize text
                if hf_pipe:
                    # Use Hugging Face pipeline
                    # Pipeline handles device, dtype, language automatically
                    result = hf_pipe(audio_np.copy()) # Pass a copy
                    text = result['text'].strip()
                elif audio_model:
                    # Use standard Whisper model
                    # Use the correct device name for standard whisper check ("cuda" or "cpu")
                    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
                    result = audio_model.transcribe(
                        audio_np,
                        fp16=(whisper_device == "cuda"),
                        language=None if args.non_english else "en" # Match web_output logic
                    )
                    text = result['text'].strip()
                else:
                    # Should not happen if loading checks passed
                    print("Error: No transcription model available.")
                    continue # Skip this loop iteration

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = transcription[-1].strip() + " " + text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
