import os
from flask import Flask, jsonify, render_template_string, request

from gemini_cleaning import clean_transcription
from html_template import HTML_TEMPLATE
from transcription_state import DEFAULT_WAITING_MESSAGE, TranscriptionState


def create_app(state: TranscriptionState, logger) -> Flask:
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = False
    app.config["JSON_SORT_KEYS"] = False
    app.config["GEMINI_API_KEY"] = ""
    app.config["AUTO_SAVE_INTERVAL"] = 600
    app.config["TRANSCRIPTION_LOG_DIR"] = os.path.join(
        os.path.expanduser("~"), "transcription_logs"
    )
    register_routes(app, state, logger)
    os.makedirs(app.config["TRANSCRIPTION_LOG_DIR"], exist_ok=True)
    return app


def register_routes(app: Flask, state: TranscriptionState, logger) -> None:
    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/get_transcription")
    def get_transcription():
        state.ensure_placeholder()
        if not state.transcriptions:
            return jsonify({"text": [DEFAULT_WAITING_MESSAGE]})
        return jsonify({"text": state.transcriptions})

    @app.route("/clear_transcription", methods=["POST"])
    def clear_transcription():
        _ = request.get_json(silent=True)
        state.clear()
        return jsonify({"status": "success"})

    @app.route("/update_transcription", methods=["POST"])
    def update_transcription():
        try:
            data = request.get_json()
            if "text" in data and isinstance(data["text"], list):
                state.transcriptions = data["text"]
                return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "Invalid data format"})
        except Exception as exc:  # noqa: BLE001
            logger.error("Error updating transcription: %s", exc)
            return jsonify({"status": "error", "message": str(exc)})

    @app.route("/clean_transcription", methods=["POST"])
    def clean_transcription_route():
        data = request.get_json(silent=True) or {}
        formatting_mode = data.get("formatting_mode", "standard")
        if not request.is_json:
            logger.warning(
                "Request to clean_transcription did not have JSON content type, using standard mode"
            )
            formatting_mode = "standard"

        result = clean_transcription(
            state, formatting_mode, app.config.get("GEMINI_API_KEY", ""), logger
        )
        return jsonify(result)

    @app.route("/undo_cleaning", methods=["POST"])
    def undo_cleaning():
        _ = request.get_json(silent=True)
        try:
            if not state.original_transcriptions:
                return jsonify(
                    {"status": "error", "message": "No original transcription to restore."}
                )
            state.transcriptions = state.original_transcriptions.copy()
            return jsonify({"status": "success"})
        except Exception as exc:  # noqa: BLE001
            logger.error("Error undoing cleaning: %s", exc)
            return jsonify({"status": "error", "message": str(exc)})

    @app.route("/reset_cleaning_history", methods=["POST"])
    def reset_cleaning_history():
        _ = request.get_json(silent=True)
        try:
            state.reset_original()
            return jsonify({"status": "success"})
        except Exception as exc:  # noqa: BLE001
            logger.error("Error resetting cleaning history: %s", exc)
            return jsonify({"status": "error", "message": str(exc)})

    @app.route("/get_last_save_info")
    def get_last_save_info():
        last_save_time = app.config.get("LAST_SAVE_TIME", None)
        last_save_file = app.config.get("LAST_SAVE_FILE", None)

        if last_save_time and last_save_file:
            return jsonify(
                {
                    "status": "success",
                    "last_save_time": last_save_time,
                    "last_save_file": os.path.basename(last_save_file),
                }
            )
        return jsonify({"status": "info", "message": "No auto-save has occurred yet"})
