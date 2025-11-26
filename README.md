# AI Tour Guide Platform

AI Tour Guide is a multi-component system for translating ancient Egyptian hieroglyphs, surfacing RAG-backed history, and enabling persona-driven conversations (text or voice) with Pharaohs such as Ramesses II.

## Features

*   **Hieroglyph Translation**: Upload an image containing hieroglyphs and receive an English translation.
*   **AI Chatbot**: Engage in a conversation with a Retrieval-Augmented Generation (RAG) chatbot knowledgeable about ancient Egypt.
*   **Talk To Pharos**: Persona-driven experience where you can speak (text or voice) with specific pharaohs such as Ramesses II, powered by the same Agentic_RAG stack.
*   **Web Interface**: A user-friendly web application built with Django provides a central point of access to both features.
*   **Microservices Architecture**: The backend logic is split into two independent FastAPI microservices for scalability and maintainability.

## Architecture

The system is built using a microservices architecture:

1.  **Django Frontend**: The main web application that users interact with. It serves the UI and communicates with the backend services.
2.  **Translation API**: A FastAPI service (under `translation_service/`) that uses a deep learning model (InceptionV3) to analyze images and translate hieroglyphic symbols.
3.  **Chatbot API**: A FastAPI service powered by a RAG system (LangChain, Hugging Face, FAISS) that provides context-aware answers to user queries about ancient Egypt.
4.  **Talk To Pharos Service**: A dedicated FastAPI microservice (`talk_to_pharos_service/`) that wraps the Agentic_RAG pipeline with persona prompts and exposes `/pharos` + `/converse` endpoints for the voice-enabled UI.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.10 or higher
*   Git
*   A Hugging Face account and an API token

### 1. Clone the Repository

```bash
git clone <your-new-ai-tour-guide-repo>.git
cd AI-Tour-Guide
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The Chatbot API requires a Hugging Face Hub API token to download models. Create a `.env` file in the root directory of the project:

```
HUGGING_FACE_HUB_TOKEN="your_hugging_face_api_token_here"
```

## Running the Application

You will need to run the four components (three APIs and the Django server) in **four separate terminals**.

### Terminal 1: Start the Translation API

This service runs on port `8000`.

```bash
uvicorn translation_service.api_server:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Start the Chatbot API

This service runs on port `8001`. Make sure your current directory is the project root.

```bash
uvicorn chatbot.chatbot_api:app --host 127.0.0.1 --port 8001
```

### Terminal 3: Start the Talk To Pharos Service

```bash
cd talk_to_pharos_service
uvicorn talk_to_pharos_service.app:app --host 127.0.0.1 --port 8050 --reload
```

Set `OPEN_ROUTER_API_KEY` in `Agentic_RAG/.env` or `Agentic_RAG/src/.env`. Optional env vars:

- `PHAROS_ALLOWED_ORIGINS` – CSV whitelist for CORS (defaults to `*`)
- `PHAROS_SERVICE_URL` – Django uses this to point to the microservice (default `http://localhost:8050`)

### Terminal 4: Run the Django Web Application

Navigate to the `django` directory and run the development server on port `9000` to avoid conflicts.

```bash
cd django
python manage.py runserver 9000
```

### Access the Application

Once all services are running, open your web browser and navigate to:

**http://127.0.0.1:9000/**

You can now use the web interface to translate hieroglyphs and interact with the chatbot.

> **Voice support:** The Talk To Pharos page relies on the browser's Web Speech API. Voice input/output is currently supported in Chromium-based browsers (Chrome, Edge) and gracefully degrades to text-only elsewhere.

## Project Structure

```
translation_service/
├── __init__.py
├── api_server.py                 # FastAPI REST server
├── hieroglyph_pipeline.py        # Main pipeline code
├── batch_processor.py            # Batch processing script
├── batch_translation_processor.py
├── optimized_batch_processor.py
├── translation_judge.py          # RAG-based evaluator
├── model_downloader.py
├── config_loader.py
├── config.yaml                   # Default configuration
├── deploy.sh
├── segmentation_evaluator.py
├── extract_paper_results.py
└── setup.py
```

## Configuration

The system uses several configuration parameters that can be modified in `hieroglyph_pipeline.py`:

- Image processing parameters (size, kernel)
- SAM model parameters (IoU threshold, stability score)
- Story generation settings (prompt template, model selection)

## API Usage

### Start the Server

```bash
python -m translation_service.api_server
```

The server runs on `http://localhost:8000` by default.

### API Endpoints

1. **Health Check**
```http
GET /health
```

2. **Get Configuration**
```http
GET /config
```

3. **Translate Hieroglyph Image**
```http
POST /translate
Content-Type: multipart/form-data
file: <image_file>
```

### Example Response

```json
{
  "processing_time": 2.5,
  "image_path": "path/to/input.jpg",
  "symbols_found": 12,
  "classifications": [
    {
      "Gardiner Code": "D21",
      "confidence": 0.85,
      "Description": "Mouth",
      "Details": "Represents speech or eating"
    }
  ],
  "story": "Translation and cultural context...",
  "session_dir": "output/session_20250529_141311_34c2135f",
  "file_paths": {
    "json_results": "path/to/results.json",
    "translation": "path/to/translation.txt",
    "symbols_dir": "path/to/symbols",
    "input_image": "path/to/input.jpg"
  }
}
```

## Output Directory Structure

```
output/
├── images/           # Original uploaded images
├── symbols/         # Extracted hieroglyph symbols
├── translations/    # Generated stories/translations
└── json/           # Complete processing results
```

## Error Handling

The API provides detailed error messages for:
- Invalid image formats
- Missing model files
- Processing failures
- Story generation issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta's Segment Anything Model (SAM)
- Alan Gardiner's Sign List
- OpenRouter/Qwen for LLM capabilities
- TensorFlow and PyTorch communities
