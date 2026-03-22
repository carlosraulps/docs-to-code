import os
import json
import datetime
import threading
from typing import Dict, Any, Optional

JOBS_DIR = os.path.join(os.getcwd(), ".jobs")
LEDGER_FILE = os.path.join(JOBS_DIR, "jobs_ledger.json")
_ledger_lock = threading.Lock()

def _ensure_dir():
    os.makedirs(JOBS_DIR, exist_ok=True)

def _load_ledger() -> Dict[str, Any]:
    _ensure_dir()
    if not os.path.exists(LEDGER_FILE):
        return {}
    try:
        with open(LEDGER_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def _save_ledger(ledger: Dict[str, Any]):
    _ensure_dir()
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)

def add_job(local_job_id: str, doc_path: str, mode: str):
    with _ledger_lock:
        ledger = _load_ledger()
        ledger[local_job_id] = {
            "local_job_id": local_job_id,
            "doc_path": doc_path,
            "mode": mode,
            "status": "started",
            "gemini_job_id": None,
            "timestamp": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "api_retries": 0
        }
        _save_ledger(ledger)

def update_job(local_job_id: str, status: str, gemini_job_id: Optional[str] = None, error_msg: Optional[str] = None, state_data: Optional[Dict] = None):
    with _ledger_lock:
        ledger = _load_ledger()
        if local_job_id not in ledger:
            return
            
        ledger[local_job_id]["status"] = status
        ledger[local_job_id]["updated_at"] = datetime.datetime.now().isoformat()
        if gemini_job_id:
            ledger[local_job_id]["gemini_job_id"] = gemini_job_id
        if error_msg:
            ledger[local_job_id]["error"] = error_msg
            
        if state_data:
            if "message" in state_data:
                ledger[local_job_id]["last_message"] = state_data["message"]
            if "api_retries" in state_data:
                current_retries = ledger[local_job_id].get("api_retries", 0)
                ledger[local_job_id]["api_retries"] = current_retries + state_data["api_retries"]
            
        _save_ledger(ledger)

def get_job_state_file(local_job_id: str) -> str:
    _ensure_dir()
    return os.path.join(JOBS_DIR, f"{local_job_id}.json")

def get_ledger() -> Dict[str, Any]:
    with _ledger_lock:
        return _load_ledger()

def delete_job(local_job_id: str) -> bool:
    """Forcefully removes a job from the ledger and deletes its local state file."""
    with _ledger_lock:
        ledger = _load_ledger()
        if local_job_id in ledger:
            ledger.pop(local_job_id, None)
            _save_ledger(ledger)
            
            state_file = get_job_state_file(local_job_id)
            if os.path.exists(state_file):
                try:
                    os.remove(state_file)
                except Exception:
                    pass
            return True
        return False

def cleanup_jobs() -> int:
    """Removes local state files for jobs that are completed, failed, or errored."""
    with _ledger_lock:
        ledger = _load_ledger()
        removed_count: int = 0
        for local_job_id, info in list(ledger.items()):
            status = info.get("status")
            # cleanup if it is definitively done or failed
            if status in ["completed", "failed", "error", "success"]:
                state_file = get_job_state_file(local_job_id)
                if os.path.exists(state_file):
                    try:
                        os.remove(state_file)
                        removed_count += 1
                    except Exception:
                        pass
        return removed_count
