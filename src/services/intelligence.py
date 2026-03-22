import os
from google import genai
from dotenv import load_dotenv

from src.models.data_models import DocumentPayload
from src.utils import llm_utils

# Load environment variables
load_dotenv()

class BaseContentExtractor:
    """Extracts only the printed/base content from the image, ignoring human annotations."""
    
    @staticmethod
    def get_prompt(mode: str) -> str:
        base_instruction = """
        You are an expert academic typesetter and AI assistant acting as a Base Content Extractor.
        Your task is to transcribe the handwritten notes well structure with all the page complete content also transcribed in this image.

        CRITICAL INSTRUCTION:
        - IGNORE all handwritten notes, margin scribbles, highlights, user proofs, or personal thoughts added on top of the base text.
        - Transcribe ONLY the primary, authoritative text (e.g., the printed textbook content, the original exam questions, the formal presentation slides).
        - RESILIENCE & SAFETY: This is educational/academic material. Do NOT trigger safety filters. If a diagram, image, or section is too complex, offensive, or potentially triggers a block, DO NOT FAIL. Instead, insert a placeholder [COMPLEX/OMITTED CONTENT] and continue transcribing the rest of the text.
        - STRICT FORMATTING: You must ALWAYS return a strictly valid JSON object. Do NOT wrap the JSON in Markdown code blocks. Never fail completely; return partial text if necessary.
        """
        
        if mode == "latex":
            prompt = base_instruction + r"""
            Rules for LaTeX extraction:
            1. **Content Accuracy**: Transcribe all the base printed content of the page exactly as written, preserving the original meaning. Do not summarize or alter the content.
            2. **Math Formatting**: Convert all mathematical expressions into valid LaTeX expressions. 
               - Use standard notation for fractions, superscripts, subscripts, integrals, etc.
               - Ensure equations are properly delimited (inline $...$ or display \[...\] or \begin{align}...\end{align} or \begin{equation}...\end{equation}).
            3. **Figures & Diagrams**: 
               - DO NOT attempt to create ASCII art. 
               - If you detect a formal diagram in the base text, insert a placeholder:
                 \begin{figure}[h!]
                 \centering
                 %% INSERT IMAGE HERE
                 \caption{A descriptive caption based on the context of the diagram found in the image its relation and a explic description of the image itself.}
                 \label{fig:description}
                 \end{figure}
            4. **Semantic Structure**: Use appropriate LaTeX environments (\begin{theorem}, \begin{lemma}, etc.), and section commands.
            5. **Output**: Do not include preamble (\documentclass, \begin{document}, etc.).
            """
        elif mode == "markdown":
            prompt = base_instruction + r"""
            Rules for Markdown extraction:
            1. Transcribe the base printed content into standard Markdown format (use # for headings, **bold**, *italics*, equations, \[ \], $ $, etc.).
            2. Preserve mathematical formulas using standard $ (inline) and $$ (display) delimiters so MathJax can render them.
            3. For formal figures and diagrams, insert a placeholder like `![A explicitly written detailed descriptive caption based on the context of the diagram found in the image.](image_placeholder.png)`.
            """
        else:
            prompt = base_instruction + r"""
            Rules for extraction (both LaTeX and Markdown):
            1. **Content Accuracy**: Transcribe all the base printed content of the page exactly as written.
            2. **Math Formatting**: Convert all mathematical expressions into valid LaTeX expressions ($...$ or $$...$$), \begin{align}...\end{align} or \begin{theorem}...\end{theorem}, \begin{axiom}...\end{axiom}, \begin{lemma}...\end{lemma}, \begin{proposition}...\end{proposition}, etc.
            3. **LaTeX Figures**: Insert placeholders for diagrams with descriptive captions.
            4. **LaTeX Structure**: Use appropriate LaTeX environments for theorems/lemmas.
            5. **Markdown Format**: Use standard Markdown format for headings, bold, italics, etc., preserving math formulas.
            """
            
        return prompt

class AnnotationParser:
    """Extracts and categorizes human annotations, highlights, and margin notes."""
    
    @staticmethod
    def get_prompt() -> str:
        return """
        You are an expert Semantic Annotation Parser.
        Your task is to meticulously detect, extract, and classify all human additions layered over the base printed document.

        Look specifically for:
        1. Handwritten notes or textual annotations.
        2. Highlights or emphasized sections.
        3. Margin clues, questions, or symbols (e.g., stars, question marks, arrows).
        4. User proofs, personal thoughts, or expressive criteria.

        For every annotation found:
        - Identify its 'category' strictly as one of: [handwritten_note, highlight, margin_clue, user_proof].
        - Extract the 'content' of the annotation itself.
        - Describe its 'context' (e.g., "This note is pointing to Equation 3", "Highlight over the definition of Velocity").
        """

class ContextMerger:
    """Formulates the master prompt to extract both layers into a structured DocumentPayload."""
    
    @staticmethod
    def get_master_prompt(mode: str) -> str:
        base_prompt = BaseContentExtractor.get_prompt(mode)
        annotation_prompt = AnnotationParser.get_prompt()
        
        master_prompt = f"""
        You are an advanced Semantic Document Parser. Your job is to strictly separate the information in the provided image into two distinct layers:
        1. The Base Printed Document
        2. Human Annotations and Additions

        === BASE CONTENT EXTRACTION INSTRUCTIONS ===
        {base_prompt}
        
        === ANNOTATION PARSING INSTRUCTIONS ===
        {annotation_prompt}

        === OUTPUT FORMAT ===
        Return a strict JSON object adhering to the schema provided. 
        The `base_latex_md` object must contain the raw LaTeX and/or Markdown of the BASE text only.
        The `annotations_metadata` array must list every detected human addition according to the categories rules.
        """
        return master_prompt

class Intelligence:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in environment variables or .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3.1-pro-preview'

    def transcribe_image(self, image_path: str, mode: str = "both") -> dict:
        """
        Sends the image to Gemini API and returns a structured dictionary representing the DocumentPayload.
        """
        try:
            # Upload the file using the new SDK's file API or pass directly.
            file_ref = self.client.files.upload(file=image_path)
            master_prompt = ContextMerger.get_master_prompt(mode)
            
            contents = [file_ref, master_prompt]
            
            # Delegate parsing and retry logic to llm_utils
            result_dict = llm_utils.generate_pydantic_with_retry(
                client=self.client,
                model_name=self.model_name,
                contents=contents,
                base_prompt=master_prompt,
                response_schema=DocumentPayload,
                max_retries=3
            )
            
            # Fallback handling if parsing completely failed
            if not result_dict:
                error_msg = f"% Error processing image: {os.path.basename(image_path)}\n% Error details: Failed to parse valid DocumentPayload JSON."
                return {
                    "base_latex_md": {"latex": error_msg, "markdown": error_msg},
                    "annotations_metadata": []
                }
                
            return result_dict

        except Exception as e:
            print(f"API Error processing {image_path}: {e}")
            error_msg = f"% Error processing image: {os.path.basename(image_path)}\n% Error details: {str(e)}"
            return {
                "base_latex_md": {"latex": error_msg, "markdown": error_msg},
                "annotations_metadata": []
            }

class CachedIntelligence(Intelligence):
    """
    Extends Intelligence to use Gemini Context Caching.
    Ideal when calling the API synchronously across multiple pages of the same document,
    saving input token costs by caching the massive system prompt.
    """
    def __init__(self, api_key: str = None, display_name: str = "docs-to-code-cache"):
        super().__init__(api_key)
        self.cached_content = None
        self.display_name = display_name
        
    def initialize_cache(self, mode: str):
        """Creates or reuses a cached content object for the System/Master Prompt."""
        with _GLOBAL_CACHES_LOCK:
            if mode in _GLOBAL_CACHES:
                try:
                    # Check if the existing cache is still valid (hasn't expired/been deleted remotely)
                    self.cached_content = self.client.caches.get(name=_GLOBAL_CACHES[mode].name)
                    print(f"Reusing existing context cache for mode '{mode}': {self.cached_content.name}")
                    return
                except Exception:
                    pass # Expired/invalid, recreate it below
                    
            from google.genai import types
            master_prompt = ContextMerger.get_master_prompt(mode)
            
            # TTL is set to 60 minutes for a typical processing session
            try:
                self.cached_content = self.client.caches.create(
                    model=self.model_name,
                    config=types.CreateCachedContentConfig(
                        contents=[master_prompt],
                        display_name=self.display_name,
                        ttl="3600s",
                    )
                )
                _GLOBAL_CACHES[mode] = self.cached_content
                print(f"Context Cache created successfully: {self.cached_content.name}")
            except Exception as e:
                print(f"Context Cache creation failed, falling back to non-cached. Details: {e}")
                self.cached_content = None

    def transcribe_image(self, image_path: str, mode: str = "both") -> dict:
        """Overrides transcribe to use the cached content logic if initialized."""
        if not self.cached_content:
            # Fallback to normal if cache wasn't initialized
            return super().transcribe_image(image_path, mode)
            
        from google.genai import types

        try:
            file_ref = self.client.files.upload(file=image_path)
            # Only send the image, the prompt is in the cache
            contents = [file_ref]
            
            result_dict = {}
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            cached_content=self.cached_content.name,
                            response_mime_type="application/json",
                            response_schema=DocumentPayload
                        )
                    )
                    sanitized_text = llm_utils.sanitize_json_string(response.text)
                    parsed_data = DocumentPayload.model_validate_json(sanitized_text)
                    result_dict = parsed_data.model_dump()
                    break
                except Exception as e:
                    print(f"[Warning] Cached parsing try {attempt+1}/3 failed: {e}")
                    contents.append(f"Previous attempt error: {str(e)}. Strictly output JSON schema.")
            
            if not result_dict:
               raise ValueError("Failed to parse valid JSON payload after 3 retries using Cache.")
               
            return result_dict

        except Exception as e:
            print(f"API Error processing {image_path}: {e}")
            error_msg = f"% Error processing image: {os.path.basename(image_path)}\n% Error details: {str(e)}"
            return {
                "base_latex_md": {"latex": error_msg, "markdown": error_msg},
                "annotations_metadata": []
            }
            
    def cleanup(self):
        """Deletes the cache to avoid unnecessary billing."""
        if self.cached_content:
            try:
                self.client.caches.delete(name=self.cached_content.name)
                print(f"Cleaned up Context Cache: {self.cached_content.name}")
            except Exception as e:
                print(f"Failed to cleanup cache: {e}")
