import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models.data_models import DocumentPayload
from src.services.intelligence import ContextMerger

load_dotenv()

class BatchProcessor:
    """
    Manages the creation and submission of Gemini Batch API jobs for massive document directories.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3.1-pro-preview'

    def process_directory_batch(self, image_paths: List[str], mode: str, update_state_cb: Callable = None) -> Dict[str, Any]:
        """
        Takes a list of images, uploads them to Gemini Files, creates a JSONL buffer,
        and submits the batch job. Returns the Batch Job Metadata.
        """
        if not image_paths:
            return {"status": "error", "message": "No images provided for batching."}

        master_prompt = ContextMerger.get_master_prompt(mode)
        jsonl_lines = []
        
        print(f"Preparing batch job for {len(image_paths)} files...")
        
        # 1. Upload files securely for the batch
        uploaded_files = []
        total_files = len(image_paths)
        import threading
        # Ensure we only have 3 concurrent staging uploads globally at any given time
        _upload_semaphore = getattr(BatchProcessor, '_upload_sem', threading.Semaphore(3))
        BatchProcessor._upload_sem = _upload_semaphore
        
        for i, path in enumerate(image_paths):
            try:
                if update_state_cb:
                    update_state_cb({
                        "status": "uploading_images",
                        "uploaded": i,
                        "total": total_files,
                        "current_file": os.path.basename(path)
                    })
                # In production, check if file exists on Gemini first, but for now upload
                with _upload_semaphore:
                    f_ref = self.client.files.upload(file=path)
                    uploaded_files.append((path, f_ref))
                print(f"Uploaded {os.path.basename(path)} to staging. ({i+1}/{total_files})")
                time.sleep(1) # Be nice to the API rate limits during staging
            except Exception as e:
                print(f"Failed to stage {path}: {e}")

        if not uploaded_files:
            return {"status": "error", "message": "Failed to upload any files to staging."}

        if update_state_cb:
            update_state_cb({
                "status": "submitting_batch_job",
                "uploaded": len(uploaded_files),
                "total": total_files
            })

        # 2. Construct JSONL payload
        for local_path, file_ref in uploaded_files:
            # The custom ID allows us to map the async result back to the specific image
            request_id = os.path.basename(local_path)
            
            # The structure dictated by the Gemini Batch API
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
                                    "image_url": {"url": file_ref.uri} 
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
                "uploaded": len(uploaded_files),
                "total": total_files,
                "message": f"Successfully queued {len(uploaded_files)} pages. Job ID: {batch_job.name}. Please inform user and check status later."
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