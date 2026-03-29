---
name: Whiteboard Digitizer
description: End-to-end skill to orchestrate Layout Analyst and OCR Specialist sub-agents for reading whiteboards.
---

# Whiteboard Digitizer Skill

This skill allows the Agent to digitize a whiteboard photo.

## Prerequisites
1. You must have access to the `detect_layout.py` and `transcribe_ocr.py` scripts located in `src/`.
2. A valid input image file.

## Execution Steps

When digitizing a whiteboard:

1. **Layout Analysis Phase**
   Instruct the **Layout Analyst** sub-agent (`agents/layout_analyst.md`) to run the layout detection script:
   ```bash
   python src/detect_layout.py <IMAGE_PATH>
   ```
   *Objective:* Gain a spatial understanding of where all handwriting, diagrams, equations, arrows, and sticky notes exist. The sub-agent will return a JSON structure.

2. **OCR Phase**
   With the regions mapped, instruct the **OCR Specialist** sub-agent (`agents/ocr_specialist.md`) to run the transcription script on the text-bearing regions:
   ```bash
   python src/transcribe_ocr.py <IMAGE_PATH> --regions <JSON_REGIONS>
   ```
   *Objective:* Secure high-fidelity text for the handwritten notes and equations.

3. **Markdown Assembly**
   As the main orchestrator, combine the output of both sub-agents.
   - Use `GEMINI.md` to map out the physical connectivity (e.g. this arrow connects this handwritten block to that diagram).
   - Use `CLAUDE.md` to structure the final text cleanly, using `$..$` for equations and callout blocks for sticky notes.

4. **Return Results**
   Write the final Markdown back to the user or save it.
