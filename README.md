# Multimodal Synthetic Media Detection & Moderation Pipeline 🛡️🤖

A robust, production-oriented proof-of-concept designed to detect synthetic media (Deepfakes) and enforce automated safety moderation. This pipeline implements a Dual-Stage Conditional Architecture to balance high-accuracy detection with resource-efficient LLM inference.

🏗️ System Architecture

The pipeline is engineered with a Decoupled Gateway pattern. Instead of running expensive multimodal moderation on every request, we use specialized "Gatekeeper" models to determine the necessity of secondary analysis.

graph TD
    A[User Upload] --> B{Media Type?}
    B -- Image --> C[ViT Deepfake Detector]
    B -- Video --> D[Temporal CNN/ViT]
    C --> E{Synthetic?}
    D --> E
    E -- No (Authentic) --> F[Return Results]
    E -- Yes (Threshold > 0.7) --> G[Ollama: Llama Guard 3 Vision]
    G --> H{Moderation Check}
    H -- Pass --> I[Flag as Synthetic/Safe]
    H -- Violates --> J[Flag as Synthetic/Unsafe]
    I --> F
    J --> F


Technical Trade-off: Efficiency vs. Depth

Stage 1 (Detection): Uses lightweight Vision Transformers (ViT) optimized for high throughput.

Stage 2 (Moderation): Invokes the 11B parameter Vision LLM only upon positive detection. This reduces inference costs by ~80% in typical "clean" traffic environments while ensuring harmful AI-generated content is strictly moderated.

🌟 Core Features

Image Deepfake Detection: Powered by prithivMLmods/Deep-Fake-Detector-v2-Model, utilizing state-of-the-art ViT backbones for spatial anomaly detection.

Video Temporal Analysis: Leverages Naman712/Deep-fake-detection via HF Video Classification. Processes ~20-frame sequences to identify flickering and unnatural transition artifacts that frame-by-frame analysis misses.

Local LLM Moderation: Integrates Meta’s Llama Guard 3 (11B Vision) via Ollama. Evaluates media against 13 hazard categories (MLCommons taxonomy) including defamation, hate speech, and graphic violence.

Modular FastAPI Backend: Clean separation of concerns between media processing, routing logic, and external LLM orchestration.

🛠️ Tech Stack

Framework: FastAPI (Asynchronous Python)

ML Inference: Hugging Face Transformers, PyTorch, OpenCV

Orchestration: Ollama (Local REST execution)

Frontend: Vanilla JS / Tailwind CSS (Standalone)

🚀 Installation & Setup

1. Prerequisites

Python 3.9+

Ollama installed and running.

2. Environment Setup

git clone [[https://github.com/yourusername/synthetic-media-pipeline.git](https://github.com/yourusername/synthetic-media-pipeline.git)](https://github.com/DevaNandanJS/AI-disease-detection-using-MOx-sensors-and-ESP32-S3-microcontroller.git)

cd synthetic-media-pipeline

# Initialize Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install fastapi uvicorn python-multipart transformers torch torchvision httpx opencv-python pillow


3. Initialize Moderation Engine

Ensure the Ollama daemon is running, then pull the required vision-guard model:

ollama pull llama-guard3:11b-vision


Note: For machines with < 16GB VRAM, consider using llama-guard3:1b or qwen2.5vl:7b.

💻 Usage

Start the Backend

uvicorn app.main:app --reload


The API documentation will be available at http://localhost:8000/docs.

Launch the Interface

Simply open index.html in a modern browser. The frontend communicates with the FastAPI endpoints to provide real-time confidence scores and moderation reports.

⚠️ Limitations & Ethical Considerations

Facial Bias: Models are currently optimized for human facial features; reliability may decrease in environmental or abstract synthetic media.

Adversarial Robustness: As a PoC, the system is susceptible to high-end adversarial noise designed to bypass ViT features.

Ethical Mandate: This software is intended for defensive research and platform safety. It should not be used to assist in the creation of bypass tools or for malicious surveillance.

📚 Acknowledgments

Architecture: Inspired by Akshayredekar07/Multimodal-Deepfake-Detection.

Detection Backbones: Models curated by Naman712 and prithivMLmods via Hugging Face.

Moderation Standards: Compliance logic based on the MLCommons Safety Taxonomy.

Maintained by Deva Nandan JS and Team
