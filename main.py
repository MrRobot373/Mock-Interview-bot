import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
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
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Path to FFmpeg executable
ffmpeg_path = "ffmpeg"

# Chat history file path
CHAT_HISTORY_FILE = "chat_history.json"

# LM Studio setup
lm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_identifier = "model-identifier"  # Replace with your LM Studio model identifier

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.setLevel(logging.INFO)  # Only show INFO or higher levels
model = whisper.load_model("base", device=device)

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Edge TTS Configuration
VOICE = "en-CA-LiamNeural"  # Select voice
TTS_OUTPUT_FILE = "response.mp3"


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Check if the file is a valid audio file
def is_audio_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith("audio")


# Load chat history from a file
def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE) or os.stat(CHAT_HISTORY_FILE).st_size == 0:
        return []
    with open(CHAT_HISTORY_FILE, "r") as file:
        return json.load(file)


# Save chat history to a file
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)


@app.get("/clear")
async def clear_history():
    """Clears the chat history file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)
        return {"message": "Chat history has been cleared"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to clear chat history"})


async def text_to_speech(text: str):
    """Generates speech from text using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(TTS_OUTPUT_FILE)
        return TTS_OUTPUT_FILE
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None


@app.post("/talk")
async def post_audio(
    file: UploadFile,
    prompt: str = Form(default="You are Alex, an interviewer for a junior AI/ML Engineer position, interviewing Niharika. Ask 15-20 structured questions covering technical skills, problem-solving, real-world applications, and debugging. Keep it professional but add occasional humor. Avoid repetitive patterns—don’t just ask about technologies or past projects. Include scenario-based and logic questions. Don’t overpraise; analyze responses and challenge weak answers. Conclude by informing Yash that you'll get results via email, then provide a final evaluation on their suitability for the role based on their strengths and weaknesses. Stay formal and stick to the script."),
):
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save the uploaded file to a temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate if the file is a proper audio file
        if not is_audio_file(temp_file_path):
            os.remove(temp_file_path)
            return {"error": "Invalid audio file format. Please upload a valid MP3 or WAV file."}

        # Transcribe the audio file using Whisper
        result = model.transcribe(temp_file_path)
        user_message = result["text"]
        logger.info(f"Transcribed Text: {user_message}")

        # Load chat history
        chat_history = load_chat_history()

        # Add custom prompt as a system message, if provided
        if prompt:
            chat_history.append({"role": "system", "content": prompt})

        # Add user's message to history
        chat_history.append({"role": "user", "content": user_message})

        # Get LM Studio response with context
        lm_response = lm_client.chat.completions.create(
            model=model_identifier,
            messages=chat_history,
            temperature=0.7,
        )
        assistant_message = lm_response.choices[0].message.content
        logger.info(f"Assistant Response: {assistant_message}")

        # Add assistant's message to history
        chat_history.append({"role": "assistant", "content": assistant_message})

        # Save updated chat history
        save_chat_history(chat_history)

        # Generate TTS Audio
        tts_file = await text_to_speech(assistant_message)

        # Return the assistant's response and the TTS file
        if tts_file:
            return FileResponse(
                tts_file,
                media_type="audio/mpeg",
                filename="response.mp3",
                headers={"Content-Disposition": "attachment; filename=response.mp3"}
            )
        else:
            return {"transcribed_text": user_message, "response": assistant_message, "tts": "Failed to generate speech"}

    except Exception as e:
        logger.error(f"Error during transcription or response generation: {e}")
        return {"error": str(e)}

    finally:
        # Ensure temp file is cleaned up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
