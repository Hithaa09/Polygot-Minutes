from typing import Literal, Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float
    end: float
    text: str


class ActionItem(BaseModel):
    item: str
    priority: Literal["High", "Medium", "Low"]
    assignee: Optional[str] = None


class TranscriptionResult(BaseModel):
    transcript: str
    segments: list[Segment]
    language: str
    duration: float


class MeetingAnalysis(BaseModel):
    title: str
    key_points: list[str]
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


class ProcessingResult(BaseModel):
    transcript: str
    segments: list[Segment]
    language: str
    duration: float
    title: str
    key_points: list[str]
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


class SummarizeRequest(BaseModel):
    transcript: str = Field(..., min_length=10, description="Raw meeting transcript text")


class HealthResponse(BaseModel):
    status: str
    whisper_model: str
    ollama_model: str
    version: str
