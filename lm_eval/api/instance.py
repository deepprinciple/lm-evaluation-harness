from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class GenerationResult:
    """Generated text plus JSON-compatible execution metadata for sample logs."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: Tuple[Optional[str], Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None
    response_metadata: list[Optional[dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    def add_response(self, response: Any) -> None:
        """Keep execution metadata out of the responses passed to task filters."""
        # Request caches written before this field was added lack the attribute.
        if not hasattr(self, "response_metadata"):
            self.response_metadata = [None] * len(self.resps)
        if isinstance(response, GenerationResult):
            if self.request_type != "generate_until":
                raise ValueError("GenerationResult requires a generate_until request")
            self.resps.append(response.text)
            self.response_metadata.append(response.metadata)
        else:
            self.resps.append(response)
            self.response_metadata.append(None)

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
