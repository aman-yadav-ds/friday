import asyncio
import speech_recognition as sr
from faster_whisper import WhisperModel
import google.generativeai as genai
import edge_tts
import pygame
import os

# --- CONFIGURATION ---
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# genai.configure(api_key=GEMINI_API_KEY)

# --- INITIALIZE MODELS (Load once to be fast) ---
print("Loading Whisper Model...")
# 'base' is a good balance. Use 'small' for better accuracy if your PC can handle it.
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

# print("Configuring Gemini...")
# llm_model = genai.GenerativeModel('gemini-1.5-flash')
# # Initialize chat history so it remembers the conversation
# chat_session = llm_model.start_chat(history=[])

print("Initializing Audio...")
pygame.mixer.init()

# --- HELPER FUNCTIONS ---

async def text_to_speech(text, output_file="response.mp3"):
    """Converts text to audio using Edge-TTS"""
    communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
    await communicate.save(output_file)

def play_audio(file_path):
    """Plays audio using Pygame"""
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    # Unload to release the file so we can overwrite it next time
    pygame.mixer.music.unload()

def record_audio(filename="input.wav"):
    """Records audio from the microphone"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening... (Speak now)")
        # Adjust for background noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=10)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            return True
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return False

# --- MAIN LOOP ---
async def main_loop():
    print("✅ Agent is ready! Say 'Exit' to stop.")
    
    while True:
        # 1. RECORD
        if not record_audio("input.wav"):
            continue

        # 2. TRANSCRIBE (STT)
        print("👂 Transcribing...")
        segments, _ = stt_model.transcribe(
                "input.wav", 
                beam_size=5, 
                language="hi",  # Telling it 'Hindi' usually captures Hinglish better than 'English'
                initial_prompt="This is a conversation in Hinglish. The speakers often mix Hindi and English words." 
            )
        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"👤 You said: {user_text}")

        if "exit" in user_text.lower():
            print("Goodbye!")
            break

        # 3. THINK (LLM)
        # print("🧠 Thinking...")
        # response = chat_session.send_message(user_text)
        # ai_text = response.text
        # print(f"🤖 AI: {ai_text}")

        # 4. SPEAK (TTS)
        print("🗣️ Speaking...")
        # We need to await this because edge-tts is async
        # await text_to_speech(ai_text, "response.mp3")
        await text_to_speech(user_text, "response.mp3")

        # 5. PLAY
        play_audio("response.mp3")

        # Optional: Clean up files
        # os.remove("input.wav")
        # os.remove("response.mp3")

if __name__ == "__main__":
    asyncio.run(main_loop())