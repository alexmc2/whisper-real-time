from dataclasses import dataclass, field
from typing import List

DEFAULT_WAITING_MESSAGE = "Waiting for speech..."


@dataclass
class TranscriptionState:
    transcriptions: List[str] = field(
        default_factory=lambda: [DEFAULT_WAITING_MESSAGE]
    )
    original_transcriptions: List[str] = field(default_factory=list)

    def clear(self) -> None:
        self.transcriptions = []
        self.original_transcriptions = []

    def ensure_placeholder(self) -> None:
        if not self.transcriptions:
            self.transcriptions = [DEFAULT_WAITING_MESSAGE]
        elif len(self.transcriptions) == 1 and not self.transcriptions[0]:
            self.transcriptions = [DEFAULT_WAITING_MESSAGE]

    def reset_original(self) -> None:
        self.original_transcriptions = []

    def backup_original_if_needed(self) -> None:
        if not self.original_transcriptions:
            self.original_transcriptions = self.transcriptions.copy()
