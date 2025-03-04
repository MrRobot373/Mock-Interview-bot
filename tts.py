import asyncio
import edge_tts

VOICES = ['en-AU-NatashaNeural', 'en-AU-WilliamNeural', 'en-CA-ClaraNeural', 
          'en-CA-LiamNeural', 'en-GB-LibbyNeural', 'en-GB-MaisieNeural']
TEXT = "You are an AI assistant named ALEX interviewing the user for an AI/ML engineer position. Ask short questions, and evaluate the candidate with some humor"
VOICE = VOICES[2]  # This might be out of range, adjust accordingly
OUTPUT_FILE = "test.mp3"

async def main() -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

loop = asyncio.get_event_loop_policy().get_event_loop()
try:
    loop.run_until_complete(main())
finally:
    loop.close()
