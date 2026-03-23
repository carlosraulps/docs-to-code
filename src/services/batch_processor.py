import os
import json
import time
import logging
import concurrent.futures
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import List, Dict, Any, Callable, Optional, Union
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models.data_models import DocumentPayload
from src.services.intelligence import ContextMerger

load_dotenv()

# Setup robust logging
log_dir = os.path.expanduser("~/.cache/docs-to-code")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("batch_processor")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(log_dir, "gemini_api.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)

class BatchUploadWorker:
    """Decoupled worker responsible purely for resilient HTTP interactions with Gemini API."""
    def __init__(self, client):
        self.client = client

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), 
        stop=stop_after_attempt(5),
        reraise=True
    )
    def upload_with_retry(self, path: str) -> Any:
        """Uploads a file with resilient retry logic."""
        logger.info(f"Attempting API upload for: {os.path.basename(path)}")
        try:
            f_ref = self.client.files.upload(file=path)
            return f_ref
        except Exception as e:
            logger.warning(f"Upload attempt failed for {os.path.basename(path)}: {e}")
            raise e

    def upload_with_timeout(self, path: str, timeout_sec: int = 120) -> Any:
        """Wraps the retry worker with a hard thread-enforced timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.upload_with_retry, path)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                logger.error(f"Upload stalled indefinitely for {path}. Execution timed out after {timeout_sec}s.")
                # We raise a custom exception that tenacity can potentially catch if desired, 
                # but here it's caught at the orchestrator level.
                raise TimeoutError(f"Upload timed out for {path}")

class BatchProcessor:
    """
    Manages the creation and submission of Gemini Batch API jobs for massive document directories.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3-flash-preview'

    def process_directory_batch(self, image_paths: List[str], mode: str, update_state_cb: Callable = None, existing_state: Dict = None) -> Dict[str, Any]:
        """
        Takes a list of images, uploads them to Gemini Files with resumability and concurrency,
        creates a JSONL buffer, and submits the batch job.
        """
        if not image_paths:
            return {"status": "error", "message": "No images provided for batching."}

        # 1. Initialize resumable checkpoint mapping: local path -> Remote File URI
        uploaded_files_map: Dict[str, str] = {}
        if existing_state and "uploaded_files_map" in existing_state:
            uploaded_files_map = existing_state["uploaded_files_map"]
            logger.info(f"Found existing state checkpoint. Resuming from {len(uploaded_files_map)} uploaded files.")

        master_prompt = ContextMerger.get_master_prompt(mode)
        total_files = len(image_paths)
        
        # 2. Filter out already processed artifacts
        pending_paths = [p for p in image_paths if p not in uploaded_files_map]
        
        if pending_paths:
            print(f"Preparing batch job: {len(uploaded_files_map)} already uploaded, uploading {len(pending_paths)} pending files...")
            worker = BatchUploadWorker(self.client)
            
            # 3. Controlled Async Concurrency (Pool of 5)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Submit mapping: { Future: path }
                future_to_path = {
                    executor.submit(worker.upload_with_timeout, path, 120): path 
                    for path in pending_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        f_ref = future.result()
                        # Store the URI or identifier returned by the API
                        uploaded_files_map[path] = getattr(f_ref, 'uri', str(f_ref))
                        
                        # Dynamic Checkpoint save
                        if update_state_cb:
                            update_state_cb({
                                "status": "uploading_images",
                                "uploaded": len(uploaded_files_map),
                                "total": total_files,
                                "uploaded_files_map": uploaded_files_map,
                                "current_file": os.path.basename(path)
                            })
                        print(f"Uploaded {os.path.basename(path)} ({len(uploaded_files_map)}/{total_files})")
                            
                    except Exception as exc:
                        logger.error(f"Critical failure uploading {path}: {exc}", exc_info=True)
                        return {
                            "status": "error", 
                            "message": f"Pipeline failed on {os.path.basename(path)}: {str(exc)}",
                            "uploaded_files_map": uploaded_files_map # Return current progress for future retry
                        }
        else:
            print("All files already uploaded. Proceeding to batch submission.")

        # 4. Construct JSONL payload via the checkpoint map
        jsonl_lines = []
        for local_path in image_paths:
            file_uri = uploaded_files_map.get(local_path)
            if not file_uri:
                logger.error(f"Missing file URI for {local_path} despite successful upload loop.")
                continue

            request_id = os.path.basename(local_path)
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": master_prompt},
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": file_uri} 
                                }
                            ]
                        }
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            jsonl_lines.append(json.dumps(request))

        # 3. Create the JSONL file locally temporarily
        import tempfile
        fd, batch_input_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(jsonl_lines))

        # 4. Upload the JSONL definition to Gemini
        try:
            batch_input_file = self.client.files.upload(
                file=batch_input_path,
                config={"mime_type": "application/jsonl"}
            )
            print(f"Uploaded JSONL definition. Triggering Job...")
            
            # 5. Execute Job
            batch_job = self.client.batches.create(
                model=self.model_name,
                src=batch_input_file.name
            )
            
            # Cleanup temp file locally
            if os.path.exists(batch_input_path):
                os.remove(batch_input_path)
            
            result = {
                "status": "processing_background",
                "job_id": batch_job.name,
                "job_state": batch_job.state,
                "uploaded": len(uploaded_files_map),
                "total": total_files,
                "uploaded_files_map": uploaded_files_map,
                "message": f"Successfully queued {len(uploaded_files_map)} pages. Job ID: {batch_job.name}. Please inform user and check status later."
            }
            if update_state_cb:
                update_state_cb(result)
                
            return result
            
        except Exception as e:
            if os.path.exists(batch_input_path):
                os.remove(batch_input_path)
            err_res = {"status": "error", "message": f"Batch API staging failed: {str(e)}"}
            if update_state_cb:
                update_state_cb(err_res)
            return err_res

    def check_job_status(self, job_name: str) -> Dict[str, Any]:
        """Polls the API for the batch status."""
        try:
            job = self.client.batches.get(name=job_name)
            state_str = str(job.state)
            
            if "SUCCEEDED" in state_str:
                return {
                    "status": "completed",
                    "output_uri": getattr(job, "output_uri", ""),
                    "message": "Job finished. Ready to parse results."
                }
            elif "FAILED" in state_str:
                return {"status": "failed", "message": "The Background Batch Job Failed."}
            else:
                return {
                    "status": "processing", 
                    "state": state_str,
                    "message": f"Job is currently: {state_str}"
                }
                
        except Exception as e:
             return {"status": "error", "message": str(e)}

    def download_and_extract_results(self, job_name: str, output_format: str, output_dir: str) -> str:
        """Downloads the batch results and extracts latex or markdown in sorted order."""
        try:
            job = self.client.batches.get(name=job_name)
            state_str = str(job.state)
            if "SUCCEEDED" not in state_str:
                return f"Job is not completed yet. Current state: {state_str}"
            
            file_name = job.dest.file_name
            print(f"Downloading {file_name}...")
            content = self.client.files.download(file=file_name)
            
            raw_path = os.path.join(output_dir, "raw_batch_results.jsonl")
            with open(raw_path, "wb") as f:
                f.write(content)
                
            import json
            import re
            final_files = []
            formats_to_extract = ["latex", "markdown"] if output_format == "both" else [output_format]
            
            for fmt in formats_to_extract:
                ext = ".tex" if fmt == "latex" else ".md"
                out_path = os.path.join(output_dir, f"extracted_document{ext}")
                
                pages_data = {}
                
                with open(raw_path, 'r', encoding='utf-8') as f_in:
                    for line_num, line in enumerate(f_in, 1):
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            custom_id = data.get('custom_id', '')
                            
                            # Parse page number using basic regex
                            match = re.search(r'(?:Image|page|file)[_-]?(\d+)', custom_id, re.IGNORECASE)
                            if match:
                                page_num = int(match.group(1))
                            else:
                                page_num = line_num
                                
                            content_str = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                            if not content_str: continue
                            
                            content_str = content_str.strip()
                            if content_str.startswith('```json'): content_str = content_str[7:]
                            if content_str.startswith('```'): content_str = content_str[3:]
                            if content_str.endswith('```'): content_str = content_str[:-3]
                            
                            try:
                                content_json = json.loads(content_str)
                                base = content_json.get('base_latex_md')
                                
                                extracted_text = ""
                                if isinstance(base, dict):
                                    extracted_text = base.get(fmt, '')
                                elif isinstance(base, str):
                                    extracted_text = base
                            except:
                                extracted_text = content_str
                                
                            if extracted_text:
                                pages_data[page_num] = extracted_text
                                
                        except Exception as e:
                            pass
                            
                if pages_data:
                    min_page = min(pages_data.keys())
                    max_page = max(pages_data.keys())
                    expected = set(range(min_page, max_page + 1))
                    actual = set(pages_data.keys())
                    missing = sorted(list(expected - actual))
                else:
                    missing = []
                    
                with open(out_path, 'w', encoding='utf-8') as f_out:
                    for p_num in sorted(pages_data.keys()):
                        f_out.write(f"\n% --- Page {p_num} --- \n" if fmt=="latex" else f"\n<!-- Page {p_num} -->\n")
                        f_out.write(pages_data[p_num] + "\n\n")
                        
                msg = f"Extracted and sorted {len(pages_data)} pages to {out_path}."
                if missing:
                    msg += f" Missing pages: {missing}"
                final_files.append(msg)
            
            return "Successfully downloaded and sorted results:\n" + "\n".join(final_files)
            
        except Exception as e:
            return f"Error downloading/extracting results: {str(e)}"