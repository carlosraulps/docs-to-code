import sys
import io
import os

# Capture everything
stdout_capture = io.BytesIO()
old_stdout_buffer = sys.stdout.buffer

# Mock stdout buffer
class MockBuffer:
    def write(self, b):
        stdout_capture.write(b)
    def flush(self):
        pass

sys.stdout.buffer = MockBuffer()

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("test")
    print("This is a test print", file=sys.stdout)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)

# Check what was captured
captured = stdout_capture.getvalue()
sys.stdout.buffer = old_stdout_buffer

print(f"Captured on buffer: {repr(captured)}", file=sys.stderr)
