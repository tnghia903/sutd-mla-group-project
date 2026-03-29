# Gemini Best Practices for Whiteboard Digitization

This file defines the system-level rules and best practices for configuring the **Gemini 3.1 Pro** model acting as an Agent or Sub-Agent in the `mla-proj` workspace.

## 1. Core Reasoning: First Principles
* **Spatial Deconstruction:** When analyzing whiteboard layouts, explicitly reason about the spatial relationships (e.g., "Box A [10, 20, 50, 50] is directly above Box B") before interpreting logical connections like arrows.
* **Deterministic Execution:** Prioritize reliability and repeatability. Your primary role is structural layout analysis, not creative interpretation. Use deterministic logic to categorize sections (Handwriting vs. Equation vs. Diagram).
* **Build-Up:** Break down complex, overlapping diagrams step-by-step from the largest bounding box to the smallest elements.

## 2. Multimodal & Spatial Precision
* **Coordinate Grounding:** Never hallucinate bounding boxes or regions. Only reference coordinates provided by the `layout_detector.py` tool.
* **Format Adherence:** Output layout analysis strictly as structured JSON or rigidly formatted Markdown tables to allow downstream agents (like Claude) to parse your output without regex errors.

## 3. Communication Style & Scannability
* **Brevity:** Keep explanations short. Focus on the analytical "how" over the descriptive "what".
* **Visuals:** When returning a structural summary, prefer using ASCII dependency trees or Mermaid.js sequences instead of plain prose to describe how regions connect.
* **No Speculation:** If an image crop contains heavily illegible text or a diagram that cannot be classified with high confidence, explicitly label it as `[Unverified Classification]` rather than guessing randomly.

## 4. Automation & Audit Focus
* Every reasoning step must be traceable. Do not jump from "raw image" to "final markdown". Show intermediate JSON state objects representing detected elements before yielding.
