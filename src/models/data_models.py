from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnnotationMetadata(BaseModel):
    category: str = Field(description="Classification: handwritten_note, highlight, margin_clue, or user_proof")
    content: str = Field(description="The textual content or description of the annotation itself")
    context: str = Field(description="How this annotation relates to the base printed text")

class BaseLatexMd(BaseModel):
    latex: Optional[str] = Field(default=None, description="The base printed document text in LaTeX")
    markdown: Optional[str] = Field(default=None, description="The base printed document text in Markdown")

class DocumentPayload(BaseModel):
    base_latex_md: BaseLatexMd
    annotations_metadata: List[AnnotationMetadata]
    api_usage: Optional[Dict[str, Any]] = Field(default=None, description="API Token usage and source information.")
