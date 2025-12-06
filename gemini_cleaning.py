import logging

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from transcription_state import DEFAULT_WAITING_MESSAGE, TranscriptionState


def setup_gemini_api(api_key: str, logger) -> None:
    if not api_key:
        logger.warning("No Gemini API key provided. The Clean function will not work.")
        return

    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully.")

        models = genai.list_models()
        gemini_models = [model for model in models if "gemini" in model.name.lower()]

        logger.info("Available Gemini models (%s):", len(gemini_models))
        for model in gemini_models:
            logger.info(
                "- %s (supported methods: %s)",
                model.name,
                model.supported_generation_methods,
            )

        gemini_25_models = [model for model in gemini_models if "2.5" in model.name]
        if gemini_25_models:
            logger.info("FOUND GEMINI 2.5 MODELS:")
            for model in gemini_25_models:
                logger.info("  - %s", model.name)
        else:
            logger.info("No Gemini 2.5 models found in API list.")

    except Exception as exc:  # noqa: BLE001
        logger.error("Error configuring Gemini API: %s", exc)
        logger.exception("Full exception details:")


def clean_transcription(
    state: TranscriptionState, formatting_mode: str, api_key: str, logger
):
    if not api_key:
        return {"status": "error", "message": "Gemini API key not set."}

    if not state.transcriptions or state.transcriptions == [DEFAULT_WAITING_MESSAGE]:
        return {"status": "error", "message": "No transcription to clean."}

    try:
        state.backup_original_if_needed()

        text_to_clean = "\n".join(state.transcriptions)
        logger.info(
            "Attempting to clean transcription with Gemini API (mode: %s)",
            formatting_mode,
        )

        model_names_to_try = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-pro-exp-03-25",
            "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/gemini-1.5-pro",
        ]

        cleaned_text = None
        success_model = None

        for model_name in model_names_to_try:
            try:
                logger.info("Attempting to use model: %s", model_name)

                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }

                model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )

                if formatting_mode == "casual":
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        safety_settings=safety_settings,
                        generation_config={
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 8192,
                        },
                    )

                formatting_instructions = ""
                if formatting_mode == "formal":
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please make the text more formal:
                    - Use proper grammar and complete sentences
                    - Remove casual language and slang
                    - Maintain a professional tone throughout
                    - Structure the content with clear paragraph breaks
                    - Use more sophisticated vocabulary where appropriate
                    """
                elif formatting_mode == "casual":
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
                elif formatting_mode == "concise":
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please make the text more concise:
                    - Remove all unnecessary words and phrases
                    - Condense multiple sentences into shorter, clearer statements
                    - Focus on key information only
                    - Aim to reduce the overall length by at least 30% while preserving meaning
                    - Use shorter sentences and simpler structures
                    """
                elif formatting_mode == "paragraph":
                    formatting_instructions = """
                    ADDITIONAL FORMATTING: Please format the text into proper paragraphs:
                    - Group related sentences into coherent paragraphs
                    - Add paragraph breaks where topics change
                    - Each paragraph should represent a complete thought or topic
                    - Aim for 3-5 sentences per paragraph where appropriate
                    - Remove sentence fragments that break paragraph flow
                    """
                elif formatting_mode == "instructions":
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
                    - When a phrase repeats more than twice consecutively, keep only ONE instance
                    - Remove any fragments or incomplete thoughts that don't contribute to meaning
                    - If you see the word "Gemini" followed by instructions (like "Gemini, make this formal"), FOLLOW those instructions and apply them to the nearby content, then remove the instruction itself
                    
                    Here is the transcription to clean:
                    {text_to_clean}
                    """

                logger.info("Sending cleaning request to %s", model_name)
                response = model.generate_content(prompt)
                logger.info("Successfully cleaned transcription using %s", model_name)
                cleaned_text = response.text.strip()
                success_model = model_name
                break

            except Exception as exc:  # noqa: BLE001
                logger.error("Error with model %s: %s", model_name, str(exc))
                logger.exception("Full exception details:")
                continue

        if cleaned_text is None:
            logger.error("All model attempts failed for cleaning")
            return {
                "status": "error",
                "message": "All model attempts failed. Check logs for details.",
            }

        cleaned_lines = [line for line in cleaned_text.split("\n") if line.strip()]

        us_to_uk_spelling = {
            "color": "colour",
            "favor": "favour",
            "flavor": "flavour",
            "humor": "humour",
            "labor": "labour",
            "neighbor": "neighbour",
            "rumor": "rumour",
            "valor": "valour",
            "organize": "organise",
            "recognize": "recognise",
            "apologize": "apologise",
            "criticize": "criticise",
            "realize": "realise",
            "emphasize": "emphasise",
            "summarize": "summarise",
            "prioritize": "prioritise",
            "analyze": "analyse",
            "paralyze": "paralyse",
            "catalyze": "catalyse",
            "center": "centre",
            "meter": "metre",
            "theater": "theatre",
            "liter": "litre",
            "caliber": "calibre",
            "fiber": "fibre",
            "dialog": "dialogue",
            "catalog": "catalogue",
            "analog": "analogue",
            "monolog": "monologue",
            "gray": "grey",
            "defense": "defence",
            "offense": "offence",
            "plow": "plough",
            "check": "cheque",
            "program": "programme",
            "draft": "draught",
            "tire": "tyre",
            "pajamas": "pyjamas",
            "judgment": "judgement",
            "aluminum": "aluminium",
            "estrogen": "oestrogen",
            "maneuver": "manoeuvre",
            "pediatric": "paediatric",
            "encyclopedia": "encyclopaedia",
            "artifact": "artefact",
            "pretense": "pretence",
            "skeptic": "sceptic",
            "specialty": "speciality",
            "traveled": "travelled",
            "traveling": "travelling",
            "canceled": "cancelled",
            "canceling": "cancelling",
        }

        improved_lines = []
        for line in cleaned_lines:
            if len(line.split()) == 1 and len(line) < 5 and line.lower() in [
                "you",
                "the",
                "a",
                "and",
                "but",
                "or",
                "so",
                "ah",
                "oh",
                "uh",
                "um",
            ]:
                continue

            processed_line = line

            problematic_phrases = [
                "продолжение следует",
                "подписывайтесь на канал",
                "спасибо за просмотр",
                "продолжение в следующей",
                "конец фильма",
                "перерыв",
                "Chào mừng quý vị đến với bộ phim",
                "Hãy subscribe cho kênh",
                "Để không bỏ lỡ những video hấp dẫn",
                "Thanks for watching!",
                "Please subscribe",
                "Please like and subscribe",
                "Don't forget to subscribe",
                "Thank you for watching",
            ]

            end_markers = [
                "...",
                "…",
                "..!",
                "!..",
                "!...",
                "...!",
                ".....",
                "......",
            ]

            for marker in end_markers:
                if processed_line.endswith(marker):
                    processed_line = processed_line[:-len(marker)] + "."
                    logger.debug("Replaced end marker %s with period", marker)
                elif processed_line == marker:
                    processed_line = ""
                    logger.debug("Removed standalone marker %s", marker)
                    break

            for phrase in problematic_phrases:
                if phrase.lower() in processed_line.lower():
                    phrase_words = len(phrase.split())
                    line_words = len(processed_line.split())

                    phrase_position_start = processed_line.lower().find(phrase.lower())
                    phrase_position_end = phrase_position_start + len(phrase)

                    at_start = phrase_position_start <= 5
                    at_end = phrase_position_end >= (len(processed_line) - 5)

                    is_majority = phrase_words / max(1, line_words) > 0.5

                    is_intentional = False

                    if phrase_position_start > 0 and phrase_position_end < len(processed_line):
                        char_before = processed_line[phrase_position_start - 1]
                        char_after = (
                            processed_line[phrase_position_end]
                            if phrase_position_end < len(processed_line)
                            else ""
                        )

                        def is_cyrillic(c):
                            return "а" <= c.lower() <= "я" if c else False

                        if is_cyrillic(char_before) and is_cyrillic(char_after):
                            is_intentional = True

                    if (at_start or at_end or is_majority) and not is_intentional:
                        logger.info(
                            "Removing problematic artifact phrase: '%s' from line",
                            phrase,
                        )
                        processed_line = processed_line.lower().replace(
                            phrase.lower(), ""
                        ).strip()

            if not processed_line.strip():
                continue

            if processed_line and not processed_line[0].isupper() and processed_line[0].isalpha():
                processed_line = processed_line[0].upper() + processed_line[1:]

            if processed_line and processed_line[-1] not in [".", "!", "?", ":", ";"]:
                question_starters = [
                    "who",
                    "what",
                    "where",
                    "when",
                    "why",
                    "how",
                    "is",
                    "are",
                    "was",
                    "were",
                    "will",
                    "can",
                    "could",
                    "should",
                    "would",
                ]
                first_word = processed_line.split()[0].lower() if processed_line.split() else ""

                if first_word in question_starters and "?" not in processed_line:
                    processed_line += "?"
                else:
                    processed_line += "."

            for punct in [".", ",", "!", "?", ":", ";"]:
                processed_line = processed_line.replace(f"{punct}", f"{punct} ")
                while "  " in processed_line:
                    processed_line = processed_line.replace("  ", " ")

            for punct in [".", ",", "!", "?", ":", ";"]:
                processed_line = processed_line.replace(f" {punct}", f"{punct}")

            processed_line = processed_line.strip()

            for us_word, uk_word in us_to_uk_spelling.items():
                if us_word in processed_line:
                    processed_line = processed_line.replace(us_word, uk_word)

                us_word_cap = us_word.capitalize()
                uk_word_cap = uk_word.capitalize()
                if us_word_cap in processed_line:
                    processed_line = processed_line.replace(us_word_cap, uk_word_cap)

            improved_lines.append(processed_line)

        state.transcriptions = (
            improved_lines
            if improved_lines
            else ["[Cleaned transcription - no significant content detected]"]
        )

        if logger.isEnabledFor(logging.DEBUG):
            original_sample = "\n".join(
                state.original_transcriptions[:3]
                if len(state.original_transcriptions) > 3
                else state.original_transcriptions
            )
            cleaned_sample = "\n".join(
                state.transcriptions[:3]
                if len(state.transcriptions) > 3
                else state.transcriptions
            )

            logger.debug("==== CLEANING COMPARISON (SAMPLE) ====")
            logger.debug("BEFORE CLEANING:\n%s\n", original_sample)
            logger.debug("AFTER CLEANING:\n%s", cleaned_sample)
            logger.debug("====================================")

        logger.info("Successfully cleaned transcription using model: %s", success_model)
        return {"status": "success", "model_used": success_model}
    except Exception as exc:  # noqa: BLE001
        logger.error("Error cleaning transcription: %s", exc)
        logger.exception("Full exception details:")
        return {"status": "error", "message": str(exc)}
