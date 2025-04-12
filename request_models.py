from pydantic import BaseModel, Field


class PromptArgs(BaseModel):
    """Prompt endpoint customizable arguments."""
    mss: float = Field(default=0.5, gt=0, le=1.0)
    top_k: int = Field(default=3, gt=0)
