import os
import json
import uuid
import threading
from pydantic import BaseModel, Field
from typing import Literal, Optional

from src.services import vision
from src.services.intelligence import CachedIntelligence
from src.services.batch_processor import BatchProcessor
from src.utils import job_manager

class ProcessDocumentInput(BaseModel):
    document_path: str = Field(..., description="The absolute file path to the PDF document or a folder of images.")
    mode: Literal["latex", "markdown", "both"] = Field(default="both", description="The desired textual output format.")
    threshold_pages: Optional[int] = Field(default=50, description="Optional page threshold to automatically switch to the async Batch API.")
    start_page: Optional[int] = Field(default=None, description="Start page for PDF extraction.")
    end_page: Optional[int] = Field(default=None, description="End page for PDF extraction.")
    verbose: bool = Field(default=False, description="Enable verbose logging to gemini_api.log")


def log_progress(msg, verbose):
    if verbose:
        log_dir = os.path.expanduser("~/.cache/docs-to-code")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "gemini_api.log"), "a") as lf:
            lf.write(msg + "\n")

def background_task(doc_path, work_dir, mode, is_pdf, local_job_id, start_page, end_page, verbose):
    job_manager.add_job(local_job_id, doc_path, mode)
    state_file = job_manager.get_job_state_file(local_job_id)
    try:
        def update_state(state_data):
            with open(state_file, "w") as f:
                json.dump(state_data, f)
            status = state_data.get("status")
            if status:
                job_manager.update_job(
                    local_job_id,
                    status=status,
                    gemini_job_id=state_data.get("job_id"),
                    state_data=state_data
                )
                
        update_state({"status": "extracting_images"})
        log_progress(f"Job {local_job_id}: extracting images...", verbose)
            
        if is_pdf:
            image_paths = vision.process_pdf(doc_path, work_dir, start_page=start_page, end_page=end_page)
            if not image_paths:
                raise Exception("vision.process_pdf returned empty list. PDF extraction failed or found no pages.")
        else:
            image_paths = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            image_paths.sort()
            
        update_state({"status": "uploading_images"})
        log_progress(f"Job {local_job_id}: uploading images to Gemini API...", verbose)
            
        processor = BatchProcessor()
        log_progress(f"Job {local_job_id}: submitting batch job...", verbose)
        
        result = processor.process_directory_batch(image_paths, mode, update_state_cb=update_state)
        
        log_progress(f"Job {local_job_id}: batch job submitted. Results: {result}", verbose)
        
        # result was already written by update_state inside process_directory_batch if it succeeded
        # but just to be sure:
        update_state(result)
            
    except Exception as e:
        error_state = {"status": "failed", "error_type": type(e).__name__, "message": str(e)}
        with open(state_file, "w") as f:
            json.dump(error_state, f)
        job_manager.update_job(local_job_id, status="failed", error_msg=str(e), state_data=error_state)
        log_progress(f"Job {local_job_id} failed: {e}", verbose)

def smart_process_document(input_data: ProcessDocumentInput) -> str:
    doc_path = input_data.document_path
    mode = input_data.mode
    threshold = input_data.threshold_pages
    start_page = input_data.start_page
    end_page = input_data.end_page
    verbose = input_data.verbose

    if not os.path.exists(doc_path):
        return json.dumps({"error": "FileNotFound", "details": f"Cannot find {doc_path}"})

    try:
        is_pdf = doc_path.lower().endswith(".pdf")
        
        # If it's a folder, output_dir should be parallel or inside it, etc.
        if is_pdf:
            output_dir = os.path.dirname(doc_path)
            base_name = os.path.splitext(os.path.basename(doc_path))[0]
            work_dir = os.path.join(output_dir, base_name, "figures")
        else:
            work_dir = doc_path # For directory, work_dir is the dir itself
            
        os.makedirs(work_dir, exist_ok=True)
        
        num_images = 0
        if is_pdf:
            try:
                import pdf2image
                info = pdf2image.pdfinfo_from_path(doc_path)
                num_images = int(info.get("Pages", 0))
                if end_page is not None:
                    num_images = min(num_images, end_page)
                if start_page is not None:
                    num_images = max(1, num_images - start_page + 1)
            except Exception:
                num_images = 100 
        else:
            if os.path.isdir(doc_path):
                num_images = len([f for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            elif doc_path.lower().endswith((".png", ".jpg", ".jpeg")):
                num_images = 1
            else:
                num_images = 0

        if num_images == 0:
            return json.dumps({"error": "NoImages", "details": "Found no valid images to parse."})

        if threshold is not None and num_images > threshold:
            local_job_id = f"local-{uuid.uuid4().hex[:8]}"
            t = threading.Thread(target=background_task, args=(doc_path, work_dir, mode, is_pdf, local_job_id, start_page, end_page, verbose))
            t.daemon = True
            t.start()
            
            return json.dumps({
                "status": "processing_background",
                "job_id": local_job_id,
                "message": f"Document has {num_images} pages (> threshold {threshold}). Processing started in the background. Please don't worry. Check status later using this job_id."
            }, indent=2)

        else:
            intel = CachedIntelligence()
            try:
                if is_pdf:
                    image_paths = vision.process_pdf(doc_path, work_dir, start_page=start_page, end_page=end_page)
                else:
                    if os.path.isdir(doc_path):
                        image_paths = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    else:
                        image_paths = [doc_path]
                    image_paths.sort()

                intel.initialize_cache(mode)
                results_log = []
                for p in image_paths:
                    enhanced = vision.enhance_image(p)
                    content = intel.transcribe_image(enhanced, mode)
                    if enhanced != p:
                        try: os.remove(enhanced)
                        except: pass
                    results_log.append({"file": os.path.basename(p), "content": content})
                intel.cleanup()
                return json.dumps({"status": "success", "results": results_log}, indent=2)
            except Exception as e:
                intel.cleanup()
                raise e

    except Exception as e:
        return json.dumps({"error": "TrafficControllerException", "details": str(e)})