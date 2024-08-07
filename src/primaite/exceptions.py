# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
class PrimaiteError(Exception):
    """The root PrimAITe Error."""

    pass


class RLlibAgentError(PrimaiteError):
    """Raised when there is a generic error with a RLlib agent that is specific to PRimAITE."""

    pass


class LLMGrammarError(Exception):
    """Custom exception for when grammar generations cannot be parsed"""

    def __init__(self, message: str):
        super().__init__(message)
