import os
import sys
import threading
from datetime import datetime
from time import sleep
from typing import MutableMapping, Optional

from transcription_state import DEFAULT_WAITING_MESSAGE, TranscriptionState


def save_transcription_to_log(
    state: TranscriptionState,
    log_dir: str,
    app_config: MutableMapping,
    logger,
) -> bool:
    """Persist the current transcription to disk and record metadata."""
    if not state.transcriptions or (
        len(state.transcriptions) == 1 and state.transcriptions[0] == DEFAULT_WAITING_MESSAGE
    ):
        return False

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"transcription_log_{timestamp}.txt")

    try:
        with open(log_file, "w") as handle:
            handle.write("\n".join(state.transcriptions))
        logger.info("Transcription auto-saved to: %s", log_file)
        app_config["LAST_SAVE_TIME"] = datetime.now().strftime("%H:%M:%S")
        app_config["LAST_SAVE_FILE"] = log_file
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Error saving transcription log: %s", exc)
        return False


def auto_save_timer(
    state: TranscriptionState,
    interval_seconds: int,
    log_dir: str,
    app_config: MutableMapping,
    logger,
) -> None:
    while True:
        sleep(interval_seconds)
        save_transcription_to_log(state, log_dir, app_config, logger)


def start_auto_save(
    interval_seconds: int,
    state: TranscriptionState,
    log_dir: str,
    app_config: MutableMapping,
    logger,
) -> Optional[threading.Thread]:
    """Kick off the auto-save thread if enabled."""
    if interval_seconds <= 0:
        return None

    thread = threading.Thread(
        target=auto_save_timer,
        args=(state, interval_seconds, log_dir, app_config, logger),
        daemon=True,
    )
    thread.start()
    return thread


def install_crash_handler(
    state: TranscriptionState, log_dir: str, app_config: MutableMapping, logger
) -> None:
    def save_on_crash(exctype, value, traceback):
        logger.error("Program is crashing. Attempting to save transcription.")
        save_transcription_to_log(state, log_dir, app_config, logger)
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = save_on_crash
