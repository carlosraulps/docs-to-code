import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, Callable
from google.genai import types

from src.services.intelligence import ContextMerger

logger = logging.getLogger("progressive_writer")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.expanduser("~/.cache/docs-to-code/gemini_api.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - PROGRESSIVE - %(message)s')
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)

class OrderedBufferWriter:
    """Consumes pages asynchronously but writes them synchronously in strict increasing order."""
    def __init__(self, output_file: str, mode: str, start_page: int = 1):
        self.output_file = output_file
        self.mode = mode
        self.next_page = start_page
        self.buffer = {}
        self.lock = asyncio.Lock()
        
        # Ensure fresh file or append if resuming
        if not os.path.exists(output_file):
            with open(self.output_file, 'w') as f:
                ext_str = "LaTeX" if mode == "latex" else "Markdown"
                f.write(f"% Auto-Generated {ext_str} Output\n" if mode == "latex" else f"<!-- Auto-Generated {ext_str} Output -->\n")

    async def add_and_flush(self, page_num: int, content: str):
        """Adds a completed page to the buffer and flushes sequentially to disk."""
        async with self.lock:
            self.buffer[page_num] = content
            
            # Flush continuously in strictly increasing order
            while self.next_page in self.buffer:
                page_content = self.buffer.pop(self.next_page)
                
                # Sanitize content wrapper
                page_content = page_content.strip()
                if page_content.startswith('```json'): page_content = page_content[7:]
                if page_content.startswith('```'): page_content = page_content[3:]
                if page_content.endswith('```'): page_content = page_content[:-3]
                
                try:
                    content_json = json.loads(page_content)
                    base = content_json.get('base_latex_md')
                    if isinstance(base, dict):
                        extracted_text = base.get(self.mode, '')
                    elif isinstance(base, str):
                        extracted_text = base
                    else:
                        extracted_text = page_content
                except:
                    extracted_text = page_content
                
                with open(self.output_file, 'a') as f:
                    if self.mode == "latex":
                        f.write(f"\n% --- Page {self.next_page} ---\n{extracted_text}\n")
                    else:
                        f.write(f"\n<!-- --- Page {self.next_page} --- -->\n{extracted_text}\n")
                
                logger.info(f"Wrote page {self.next_page} to disk.")
                self.next_page += 1

class ProgressiveGenerator:
    """Manages the asynchronous processing of images against the Gemini API."""
    def __init__(self, client, job_id, doc_path, mode, update_state_cb: Callable, generated_pages: list):
        self.client = client
        self.job_id = job_id
        self.mode = mode
        self.update_state_cb = update_state_cb
        
        base_name = os.path.splitext(os.path.basename(doc_path))[0]
        output_dir = os.path.dirname(doc_path) if doc_path.endswith('.pdf') else doc_path
        ext = ".tex" if mode == "latex" else ".md"
        self.output_file = os.path.join(output_dir, f"{base_name}_output{ext}")
        
        # Calculate highest consecutive generated page to know where to start buffer
        start_page = 1
        if generated_pages:
            start_page = max(generated_pages) + 1
        
        self.writer = OrderedBufferWriter(self.output_file, mode, start_page=start_page)

    def extract_page_num(self, local_path: str) -> int:
        """Parses the page number from the image format 'TitleXImage12.png'."""
        match = re.search(r'(?:Image|page|file)[_-]?(\d+)', os.path.basename(local_path), re.IGNORECASE)
        # If we can't find one via regex, we default to 0, which might break order but acts as fallback
        return int(match.group(1)) if match else 0

    async def generate_page(self, local_path: str, file_uri: str, prompt: str, sem: asyncio.Semaphore):
        page_num = self.extract_page_num(local_path)
        async with sem:
            logger.info(f"Generating page {page_num}...")
            # Wrap dynamic inference with generic Retry fallback
            for attempt in range(3):
                try:
                    # Async thread offload for native generic calls
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model='gemini-3-flash-preview',
                        contents=[prompt, types.Part.from_uri(file_uri, mime_type="image/jpeg")]
                    )
                    
                    text = response.text
                    await self.writer.add_and_flush(page_num, text)
                    
                    # Update checkpoint via callback securely
                    self.update_state_cb(local_path, "generated", page_num)
                    break
                    
                except Exception as e:
                    logger.error(f"Failed generating page {page_num} (Attempt {attempt+1}/3): {e}")
                    if attempt == 2:
                        self.update_state_cb(local_path, "failed", page_num)
                    await asyncio.sleep(2 ** attempt) # Exponential backoff locally

    async def run(self, uploaded_files_map: Dict[str, str], prompt: str, generated_pages: list):
        """Asynchronously dispatches all pending paths."""
        sem = asyncio.Semaphore(5)
        
        tasks = []
        for path, uri in uploaded_files_map.items():
            page_num = self.extract_page_num(path)
            if page_num not in generated_pages:
                tasks.append(self.generate_page(path, uri, prompt, sem))
        
        if tasks:
            logger.info(f"Starting async batched generation for {len(tasks)} items...")
            await asyncio.gather(*tasks)
            logger.info("Progressive Generation fully completed.")
        else:
            logger.info("No tasks to generate found. Everything already assembled.")
