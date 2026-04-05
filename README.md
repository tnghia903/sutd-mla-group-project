# Whiteboard Digitizer

This repository contains the Machine Learning pipeline (YOLOv8 & PP-OCRv3) to parse and digitize whiteboard photographs into structured Markdown files.

## 🚀 Quick Start for Team Members

To clone and run this project, you just need Python 3 installed. We use a standalone bash script to initialize a robust virtual environment and grab all heavy dependencies without polluting your global system.

### 1. Clone & Setup
```bash
git clone <repository_url>
cd mla-proj

# Run the automated setup script
./setup.sh
```

### 2. Activate Environment
Always run this before executing the Python scripts directly:
```bash
source .venv/bin/activate
```

### 3. Testing the Pipeline
You can test the Layout Detector script manually:
```bash
python src/detect_layout.py
```

## 🤖 Agentic Architecture

The real magic happens when you load this workspace into **Antigravity**. The AI Agent will read the `skills/whiteboard_digitizer/SKILL.md` runbook, load the two LLM rulesets (`GEMINI.md` and `CLAUDE.md`), and orchestrate the two sub-agents (`agents/layout_analyst.md` and `agents/ocr_specialist.md`) to read your whiteboards automatically!
