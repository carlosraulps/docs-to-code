# OCR-to-LaTeX & Markdown Converter

A modular Python application that transforms handwritten notes (docs, pdfs or Images) into editable, semantic LaTeX code using the **Google Gemini 3.1 Pro Preview** API.

## Features

- **Dual Output**: Generates both `.tex` and `.md` files in a single pass.
- **Automated Workflow**: Splits PDFs, processes images, and generates documents automatically.
- **Pagination Control**: Extract specific page ranges from large PDFs using `start_page` and `end_page` to prevent memory overflow.
- **Live Logging**: Real-time progress tracking of background batch API jobs via `gemini_api.log` (`verbose` mode).
- **Smart Vision**: Enhances images (denoising, deskewing) with OpenCV before processing.
- **Incremental Processing**: Skips already processed images using a smart cache to save time and API tokens.
- **Self-Correction & Resiliency**: Built-in retry loops and prompt engineering handle malformed LLM JSON outputs autonomously.
- **Semantic Understanding**: Uses Gemini models to interpret equations, theorems, and proofs correctly.
- **Figure Handling**: Detected diagrams are captioned and placed in proper `figure` environments (LaTeX) or image placeholders (Markdown).

## Installation

### Prerequisites

- Python 3.9+
- [Poppler](https://github.com/check-repos/poppler) (Required for PDF processing)
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`

### Setup

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd docs-to-code
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Key**:
    Get your API key from [Google AI Studio](https://aistudio.google.com/).

    Create a `.env` file in the root directory:

    ```bash
    GOOGLE_API_KEY=your_api_key_here
    ```

## Usage

Run the `app.py` script pointing to your source directory:

```bash
python3 app.py /path/to/your/notes/folder
```

You will be prompted to choose the output format interactively:

```text
What would you like to generate? (latex/markdown/both) [both]:
```

### Folder Structure & Naming

- The script looks for **PDFs** (which it splits automatically) or **Images**.
- Images are grouped by title using the pattern: `TitleNameXImageNumber.png`.
  - Example: `Calculus_Ch1XImage1.png`, `Calculus_Ch1XImage2.png`.

### Output

- A `.tex` file is generated for each title (e.g., `Calculus_Ch1.tex`).
- A `.md` file is generated for each title (e.g., `Calculus_Ch1.md`).
- A `processed_log.json` file is created to track progress.

## Architecture

- `vision.py`: Image pre-processing and PDF handling.
- `intelligence.py`: Interface with Google Gen AI SDK (Gemini).
- `llm_utils.py`: Utilities for LLM JSON sanitization and self-correction retry loops.
- `memory.py`: State management for incremental builds.
- `latex.py`: LaTeX generation and package management.
- `markdown.py`: Markdown file generation.
- `app.py`: Main entry point and orchestration.

## MCP Server & Agent Skill Setup

This repository can now be run as a standard Model Context Protocol (MCP) server, allowing AI assistants to interact with it directly as a "Skill".

### 1. Install MCP Dependencies

```bash
pip install mcp
```

### 2. Register the Extension

Add the connection configuration to your AI agent's extension config (e.g., `gemini-extension.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "docs-to-code": {
      "command": "python3",
      "args": [
        "/absolute/path/to/docs-to-code/mcp_server.py"
      ],
      "env": {
        "GOOGLE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### 3. Invoke the Skill

Once registered, your AI agent can natively run commands like:
> "Convert the handwritten notes at `/path/to/notes.png` into LaTeX code."

**New in v1.1:** For large PDFs, you can specify `start_page` and `end_page` to process in chunks, and enable `verbose` to tail logs in `gemini_api.log`.

## License

[MIT](LICENSE)
