import os
import json
from typing import Optional
from mcp.server.fastmcp import FastMCP

from src.tools.process_document import smart_process_document
from src.tools.check_batch_status import check_batch_job
from src.tools.process_document import ProcessDocumentInput
from src.tools.check_batch_status import CheckBatchStatusInput

mcp = FastMCP("docs-to-code")

@mcp.tool()
def process_document(document_path: str, mode: str = "latex", threshold_pages: int = 50, start_page: Optional[int] = None, end_page: Optional[int] = None, verbose: bool = False) -> str:
    """
    Converts a PDF or image of handwritten notes/equations into LaTeX or Markdown code.
    """
    try:
        input_data = ProcessDocumentInput(
            document_path=document_path, 
            mode=mode,
            threshold_pages=threshold_pages,
            start_page=start_page,
            end_page=end_page,
            verbose=verbose
        )
        return smart_process_document(input_data)
        
    except ValueError as val_err:
        import json
        return json.dumps({
            "error": "InputValidationError",
            "details": str(val_err)
        })

@mcp.tool()
def check_document_status(job_id: str, output_format: str = "both") -> str:
    """
    Use this if `process_document` returned a status of 'processing_background'.
    Downloads and extracts the code when finished.
    
    Args:
        job_id: The job ID returned by the server.
        output_format: 'latex', 'markdown', or 'both'. Defaults to 'both'.
    """
    try:
        input_data = CheckBatchStatusInput(job_id=job_id, output_format=output_format)
        return check_batch_job(input_data)
        
    except ValueError as val_err:
        import json
        return json.dumps({
            "error": "InputValidationError",
            "details": str(val_err)
        })

@mcp.tool()
def process_multiple_documents(document_paths: list[str], mode: str = "latex", threshold_pages: int = 50, verbose: bool = False) -> str:
    """
    Converts multiple PDFs or images into LaTeX or Markdown code efficiently using the secure pipeline.
    Use this when you need to process an array of files simultaneously.
    """
    import json
    from src.tools.process_document import ProcessDocumentInput, smart_process_document
    results = {}
    for path in document_paths:
        try:
            input_data = ProcessDocumentInput(
                document_path=path,
                mode=mode,
                threshold_pages=threshold_pages,
                verbose=verbose
            )
            result_str = smart_process_document(input_data)
            try:
                results[path] = json.loads(result_str)
            except:
                results[path] = {"raw_output": result_str}
        except Exception as e:
            results[path] = {"error": "ProcessingException", "details": str(e)}
            
    return json.dumps(results, indent=2)

@mcp.tool()
def manage_jobs(action: str = "list", job_id: Optional[str] = None) -> str:
    """
    Manage the local background jobs tracker.
    
    Args:
        action: 'list' to show tracked jobs, 'cleanup' to delete local state files for finished/failed jobs.
        job_id: Optional specific job ID to fetch info.
    """
    from src.utils import job_manager
    import json
    try:
        if action == "list":
            ledger = job_manager.get_ledger()
            if job_id:
                if job_id in ledger:
                    return json.dumps({job_id: ledger[job_id]}, indent=2)
                return json.dumps({"error": "Job not found in ledger"}, indent=2)
            return json.dumps(ledger, indent=2)
        elif action == "cleanup":
            removed = job_manager.cleanup_jobs()
            return json.dumps({"status": "success", "message": f"Cleaned up {removed} local job state files."})
        else:
            return json.dumps({"error": "Invalid action. Use 'list' or 'cleanup'."})
    except Exception as e:
        return json.dumps({"error": "ManageJobsException", "details": str(e)})

if __name__ == "__main__":
    mcp.run(transport='stdio')
