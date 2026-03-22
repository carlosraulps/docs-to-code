import os
from mcp.server.fastmcp import FastMCP

from src.tools.process_document import smart_process_document
from src.tools.check_batch_status import check_batch_job
from src.tools.process_document import ProcessDocumentInput
from src.tools.check_batch_status import CheckBatchStatusInput

mcp = FastMCP("docs-to-code")

@mcp.tool()
def process_document(document_path: str, mode: str = "latex", threshold_pages: int = 50, start_page: int = None, end_page: int = None, verbose: bool = False) -> str:
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

if __name__ == "__main__":
    mcp.run(transport='stdio')
