import json
import os
from pydantic import BaseModel, Field
from src.services.batch_processor import BatchProcessor

class CheckBatchStatusInput(BaseModel):
    job_id: str = Field(..., description="The Batch Job ID returned by the process_document tool.")
    output_format: str = Field(default="both", description="The desired extracted format: 'latex', 'markdown', or 'both'.")
    output_dir: str = Field(default="./output", description="Where to save the extracted files.")

def check_batch_job(input_data: CheckBatchStatusInput) -> str:
    try:
        job_id = input_data.job_id
        
        if job_id.startswith("local-"):
            from src.utils import job_manager
            state_file = job_manager.get_job_state_file(job_id)
            if not os.path.exists(state_file):
                return json.dumps({"status": "processing_background", "message": "Background task is starting or not found..."}, indent=2)
                
            with open(state_file, "r") as f:
                state = json.load(f)
                
            if state.get("status") == "extracting_images":
                return json.dumps({"status": "processing_background", "message": "Background task is currently: extracting_images... (Converting PDF to images)"}, indent=2)

            if state.get("status") == "uploading_images":
                uploaded = state.get("uploaded", 0)
                total = state.get("total", "?")
                current = state.get("current_file", "")
                progress_msg = f"Uploading images to Gemini API: {uploaded}/{total} complete."
                if current:
                    progress_msg += f" Currently processing: {current}"
                return json.dumps({"status": "processing_background", "message": progress_msg, "details": state}, indent=2)

            if state.get("status") == "submitting_batch_job":
                return json.dumps({"status": "processing_background", "message": "All images uploaded successfully. Submitting the batch job to Gemini now..."}, indent=2)
                
            if state.get("status") in ["failed", "error"]:
                return json.dumps({
                    "status": "error",
                    "message": "The background process failed.",
                    "details": state
                }, indent=2)
                
            real_job_id = state.get("job_id")
            if not real_job_id:
                return json.dumps({"status": "processing_background", "message": "Waiting for Gemini Batch API Job ID...", "state_data": state}, indent=2)
                
            job_id = real_job_id
            
        processor = BatchProcessor()
        result_meta = processor.check_job_status(job_id)
        status = result_meta.get("status")
        
        if status == "completed":
            os.makedirs(input_data.output_dir, exist_ok=True)
            extraction_result = processor.download_and_extract_results(
                job_id, 
                input_data.output_format, 
                input_data.output_dir
            )
            return json.dumps({
                "status": "success",
                "message": extraction_result
            }, indent=2)
            
        elif status == "processing":
            return json.dumps({
                "status": "processing",
                "message": result_meta.get("message"),
                "details": result_meta
            }, indent=2)
            
        else:
             return json.dumps(result_meta, indent=2)

    except Exception as e:
        return json.dumps({
            "error": "BatchCheckException",
            "details": str(e)
        }, indent=2)