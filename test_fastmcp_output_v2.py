import sys
import os
import io

# Backup real stdout
_REAL_STDOUT = sys.stdout
_REAL_BUFFER = sys.stdout.buffer

# Mock stdout entirely
class MockStream:
    def __init__(self, buffer):
        self.buffer = buffer
        self.encoding = "utf-8"
    def write(self, s):
        self.buffer.write(s.encode("utf-8"))
    def flush(self):
        self.buffer.flush()

mock_buffer = io.BytesIO()
sys.stdout = MockStream(mock_buffer)

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("test")
    print("Test message from standard print")
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")

# Revert to original
sys.stdout = _REAL_STDOUT

# Report results
captured_data = mock_buffer.getvalue()
sys.stderr.write(f"CAPTURED FROM STDOUT: {repr(captured_data)}\n")
if captured_data:
    sys.stderr.write("Fix: We should ensure this doesn't happen on real stdout.\n")
else:
    sys.stderr.write("Stdout seems clean for this initialization.\n")
