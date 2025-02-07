import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import whisper
import os
import shutil
import torch
import json
import mimetypes
from openai import OpenAI
import asyncio
import edge_tts

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Path to FFmpeg executable (ensure FFmpeg is installed and in your PATH)
ffmpeg_path = "ffmpeg"

# Chat history file path
CHAT_HISTORY_FILE = "chat_history.json"

# LM Studio setup (update these as needed)
lm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_identifier = "model-identifier"  # Replace with your LM Studio model identifier

# Check if GPU is available; otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
model = whisper.load_model("base", device=device)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (allow all origins for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your specific front-end URL(s) as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Edge TTS Configuration
VOICE = "en-CA-LiamNeural"  # Select voice (verify this voice is supported)
TTS_OUTPUT_FILE = "response.mp3"


@app.get("/")
async def root():
    return {"message": "Hello World"}


def is_audio_file(file_path: str) -> bool:
    """Check if the file is a valid audio file based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    valid = mime_type is not None and mime_type.startswith("audio")
    if not valid:
        logger.error(f"File {file_path} not recognized as a valid audio file (MIME type: {mime_type}).")
    return valid


def load_chat_history():
    """Load chat history from file. Return empty list if file does not exist or is empty."""
    if not os.path.exists(CHAT_HISTORY_FILE) or os.stat(CHAT_HISTORY_FILE).st_size == 0:
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            history = json.load(file)
            return history
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []


def save_chat_history(history):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")


@app.get("/clear")
async def clear_history():
    """Clears the chat history file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)
        logger.info("Chat history cleared successfully.")
        return {"message": "Chat history has been cleared"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to clear chat history"})


async def text_to_speech(text: str):
    """Generates speech from text using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(TTS_OUTPUT_FILE)
        logger.info("TTS generation successful.")
        return TTS_OUTPUT_FILE
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None


@app.post("/talk")
async def post_audio(
    file: UploadFile,
    prompt: str = Form(default=(
        "You are an interviewer you are hiring candidate for a junioer ai ml Engineer position,"
        "Ask one by one structured question to candidate about covering technical skills, problem-solving, real-world applications, and debugging.ask only one question at a time. if user give answer short ask him to explain his answer. "
        "Keep it professional but add occasional humor. Avoid repetitive patterns—donot just ask about technologies or past projects. "
        "Include scenario-based and logic questions. Do not overpraise; analyze responses and challenge weak answers. if usert says that he or she dont know about answer just move forward to next question "
        "Conclude by informing candidate that you'll get results via email."
        "Stay formal and stick to the script.Behave like you are interviwer."
    )),
):
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save the uploaded file to a temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {temp_file_path}")

        # Validate if the file is a proper audio file
        if not is_audio_file(temp_file_path):
            os.remove(temp_file_path)
            return JSONResponse(status_code=400, content={"error": "Invalid audio file format. Please upload a valid MP3 or WAV file."})

        # Transcribe the audio file using Whisper
        result = model.transcribe(temp_file_path)
        user_message = result.get("text", "")
        logger.info(f"Transcribed Text: {user_message}")

        # Load chat history
        chat_history = load_chat_history()

        # Add custom prompt as a system message, if provided
        if prompt:
            chat_history.append({"role": "system", "content": prompt})

        # Add user's message to chat history
        chat_history.append({"role": "user", "content": user_message})

        # Get LM Studio response with context
        lm_response = lm_client.chat.completions.create(
            model=model_identifier,
            messages=chat_history,
            temperature=0.7,
        )
        assistant_message = lm_response.choices[0].message.content
        logger.info(f"Assistant Response: {assistant_message}")

        # Add assistant's message to chat history and save history
        chat_history.append({"role": "assistant", "content": assistant_message})
        save_chat_history(chat_history)

        # Generate TTS Audio
        tts_file = await text_to_speech(assistant_message)

        if tts_file and os.path.exists(tts_file):
            return FileResponse(
                tts_file,
                media_type="audio/mpeg",
                filename="response.mp3",
                headers={"Content-Disposition": "attachment; filename=response.mp3"},
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "transcribed_text": user_message,
                    "response": assistant_message,
                    "tts": "Failed to generate speech",
                },
            )
    except Exception as e:
        logger.error(f"Error during transcription or response generation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Ensure temporary file is cleaned up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file {temp_file_path}")


if __name__ == "__main__":
    import uvicorn

    # Run the app with uvicorn on port 8000 (adjust as needed)
    uvicorn.run("your_script_filename:app", host="0.0.0.0", port=8000, reload=True)
