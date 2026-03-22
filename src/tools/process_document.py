import os
import json
import uuid
import threading
from pydantic import BaseModel, Field
from typing import Literal, Optional

from src.services import vision
from src.services.intelligence import CachedIntelligence
from src.services.batch_processor import BatchProcessor

class ProcessDocumentInput(BaseModel):
    document_path: str = Field(..., description="The absolute file path to the PDF document or a folder of images.")
    mode: Literal["latex", "markdown", "both"] = Field(default="both", description="The desired textual output format.")
    threshold_pages: Optional[int] = Field(default=50, description="Optional page threshold to automatically switch to the async Batch API.")
    start_page: Optional[int] = Field(default=None, description="Start page for PDF extraction.")
    end_page: Optional[int] = Field(default=None, description="End page for PDF extraction.")
    verbose: bool = Field(default=False, description="Enable verbose logging to gemini_api.log")


def log_progress(msg, verbose):
    if verbose:
        with open("/Users/apple/Research/docs-to-code/gemini_api.log", "a") as lf:
            lf.write(msg + "\n")

def background_task(doc_path, work_dir, mode, is_pdf, local_job_id, start_page, end_page, verbose):
    state_file = f"/Users/apple/Research/docs-to-code/{local_job_id}.json"
    try:
        with open(state_file, "w") as f:
            json.dump({"status": "extracting_images"}, f)
        log_progress(f"Job {local_job_id}: extracting images...", verbose)
            
        if is_pdf:
            image_paths = vision.process_pdf(doc_path, work_dir, start_page=start_page, end_page=end_page)
        else:
            image_paths = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            
        with open(state_file, "w") as f:
            json.dump({"status": "uploading_images"}, f)
        log_progress(f"Job {local_job_id}: uploading images to Gemini API...", verbose)
            
        processor = BatchProcessor()
        log_progress(f"Job {local_job_id}: submitting batch job...", verbose)
        result = processor.process_directory_batch(image_paths, mode)
        log_progress(f"Job {local_job_id}: batch job submitted. Results: {result}", verbose)
        
        with open(state_file, "w") as f:
            json.dump(result, f)
            
    except Exception as e:
        with open(state_file, "w") as f:
            json.dump({"status": "failed", "message": str(e)}, f)

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
        output_dir = os.path.dirname(doc_path)
        base_name = os.path.splitext(os.path.basename(doc_path))[0]
        work_dir = os.path.join(output_dir, base_name, "figures")
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
            num_images = len([f for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        if num_images == 0:
            return json.dumps({"error": "NoImages", "details": "Found no valid images to parse."})

        if num_images > threshold:
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
                    image_paths = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

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
