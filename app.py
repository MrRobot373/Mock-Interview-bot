import os
import logging
import shutil
import torch
import json
import mimetypes
import whisper
import edge_tts
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Directly set the Gemini API Key here
GEMINI_API_KEY = "AIzaSyCIhzKAOCeRUL-GX2q0jbJL6-vgxUMPIeM" 

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Path to FFmpeg executable
ffmpeg_path = "ffmpeg"

# Chat history file path
CHAT_HISTORY_FILE = "chat_history.json"

# Whisper model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.setLevel(logging.INFO)  # Only show INFO or higher levels
whisper_model = whisper.load_model("tiny", device=device)

# Edge TTS Configuration
VOICE = "en-CA-LiamNeural"  # Select voice male1
TTS_OUTPUT_FILE = "response.mp3"

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your specific front-end URL(s) as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini model configuration
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
    system_instruction=(
   "You are Alex, an interviewer for a jonior ai ml engineer \"\n        \"first ask about tell me about yourself and then on based of it Ask 10-15 structured questions covering technical skills, problem-solving, real-world applications, and debugging. \"\n        \"Keep it professional. Avoid repetitive patterns—don’t just ask about technologies or past projects. do not ask coding questions.ask verbal questions only and ask sensible and complete questions\"\n        \"Include scenario-based and logic questions. Don’t overpraise; analyze responses and challenge weak answers. \"\n        \"Conclude by informing Yash that you'll get results via email, then provide a final evaluation on their suitability for the role based on their strengths and weaknesses. \"\n        \"Stay formal and stick to the script.\\n\"\n        \"Here is how you MUST behave:\\n\"\n        \"- **Stay in character**: Always remain a formal interviewer.\\n\"\n        \"- **Ask only ONE question at a time.** \\n\"\n        \"- **Do NOT provide the candidate with the correct answer**; do not reveal inside knowledge or solutions.\\n\"\n        \"- If the candidate’s answer is incomplete, politely ask for more details.if candidate dident give answer properly or ignore your answer just ask him again and ask to be stay serious its interwiew \\n\"\n          \"- If candidate says they don’t know, move on to the next question without giving answers.\\n\\n\"\n        \"DO NOT say things like “How can I help you?” \\n\"\n        \"DO NOT include asterisks `*` in your response.\\n\"\n        \"DO NOT break character. \""),
)

# Start a new chat session
chat_session = gemini_model.start_chat(history=[])

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

# Transcribe speech to text using Whisper
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

# Generate speech from text using Edge TTS
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
    prompt: str = Form(default=(
       "You are Alex, an interviewer for a jonior ai ml engineer position \"\n        \"first ask about tell me about yourself and then on based of itAsk 10-15 structured questions covering technical skills, problem-solving, real-world applications, and debugging. \"\n        \"Keep it professional. Avoid repetitive patterns—don’t just ask about technologies or past projects. do not ask coding questions..ask verbal questions only and ask sensible and complete questions \"\n        \"Include scenario-based and logic questions. Don’t overpraise; analyze responses and challenge weak answers. \"\n        \"Conclude by informing Yash that you'll get results via email, then provide a final evaluation on their suitability for the role based on their strengths and weaknesses. \"\n        \"Stay formal and stick to the script.\\n\"\n        \"Here is how you MUST behave:\\n\"\n        \"- **Stay in character**: Always remain a formal interviewer.\\n\"\n        \"- **Ask only ONE question at a time.** \\n\"\n        \"- **Do NOT provide the candidate with the correct answer**; do not reveal inside knowledge or solutions.\\n\"\n        \"- If the candidate’s answer is incomplete, politely ask for more details.if candidate dident give answer properly or ignore your answer just ask him again and ask to be stay serious its interwiew \\n\"\n          \"- If candidate says they don’t know, move on to the next question without giving answers.\\n\\n\"\n        \"DO NOT say things like “How can I help you?” \\n\"\n        \"DO NOT include asterisks `*` in your response.\\n\"\n        \"DO NOT break character. \""
    )),
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
        user_message = transcribe_audio(temp_file_path)
        logger.info(f"Transcribed Text: {user_message}")

        # Load chat history
        chat_history = load_chat_history()

        # Add custom prompt as a system message, if provided
        if prompt:
            chat_history.append({"role": "system", "content": prompt})

        # Add user's message to history
        chat_history.append({"role": "user", "content": user_message})

        # Get Gemini response with context
        response = chat_session.send_message(user_message)
        assistant_message = response.text
        logger.info(f"Assistant Response: {assistant_message}")

        # Add assistant's message to history
        chat_history.append({"role": "assistant", "content": assistant_message})

        # Save updated chat history
        save_chat_history(chat_history)

        # Generate TTS Audio (optional)
        tts_file = await text_to_speech(assistant_message)

        # Return the assistant's response and the TTS file if available
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

# API root endpoint
@app.get("/")
async def root():
    return {"message": "Hello, I am the Interviewer Bot!"}

# Clear chat history endpoint
@app.get("/clear")
async def clear_history():
    """Clears the chat history file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)
        return {"message": "Chat history has been cleared."}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to clear chat history"})

# Final interview report endpoint returning JSON
@app.get("/final_report")
async def final_report():
    # Load the chat history
    chat_history = load_chat_history()
    if not chat_history:
        return JSONResponse(status_code=404, content={"error": "No interview history found. Please conduct an interview first."})

    # Build a transcript from the chat history for clarity
    transcript = ""
    for message in chat_history:
        transcript += f"{message['role']}: {message['content']}\n"
    
    # Construct the evaluation prompt with clear instructions for analysis
    evaluation_prompt = (
        "You are now evaluating the interview. give normal evaluation  Based on the following transcript, provide a final report analyzing the candidate's performance. in 500 words \"\n        \"Assess the candidate's answers with attention to tone, pitch, fluency, knowledge, and confidence. Determine if the candidate is suitable for role, explaining your reasoning. also give marks out of 10 to given answers as per questions.\"\n        \"Also, specify areas where the candidate could improve. \"\n        \"Interview Transcript:\\n\\n" + transcript
    )
    
    # Start a new chat session for the evaluation
    evaluation_session = gemini_model.start_chat(history=[])
    response = evaluation_session.send_message(evaluation_prompt)
    final_report_text = response.text
    logger.info(f"Final Report: {final_report_text}")

    # Return the final report as JSON
    return JSONResponse(content={"final_report": final_report_text})
