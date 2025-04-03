from pydantic import BaseModel, Field


class Mss(BaseModel):
    mss: float = Field(0.5, gt=0, le=1.0)
