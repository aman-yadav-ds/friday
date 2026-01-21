
import asyncio
import os
from src.audio_input import AudioInput
from src.llm_engine import LLMEngine
from src.audio_output import AudioOutput

HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.45

async def main_loop():
    """
    The Big Boss. The Conductor. The Main Loop. 🎩
    This function ties the Ear, Brain, and Mouth together into one somewhat cohesive unit.
    It's the puppet master pulling the strings.
    """
    print("✅ Agent is locked and loaded!")
    MODE = "text"
    
    # The "Shut Up" Button
    # If this event is set, the Mouth stops talking immediately.
    stop_speaking_event = asyncio.Event()
    
    # --- The Crew ---
    # 1. The Ear: Handles the mic, VAD, and transcription.
    audio_input = AudioInput(stop_speaking_event)
    
    # 2. The Brain: The smart part (hopefully). Talks to Gemini.
    llm_engine = LLMEngine()
    
    # 3. The Mouth: The loud part. Turns text into noise.
    audio_output = AudioOutput(stop_speaking_event)
    
    # --- Spin up the background workers ---
    # These guys run forever, doing the heavy lifting in the background.
    await audio_input.start()
    await audio_output.start()
    
    while True:
        # 1. LISTEN: Wait for the Ear to hand us a complete thought.
        if MODE == "text":
            user_text = str(input("Boss: "))
            confidence = 0.99
        else: 
            payload = await audio_input.text_queue.get()
            user_text = payload["text"]
            confidence = payload["confidence"]

        print(f"👤 Boss said (Final): {user_text}")

        # Check for exit commands
        # Because sometimes you just need some peace and quiet.
        exit_phrases = ["exit", "shutdown", "shut down"]
        if any(phrase in user_text.lower() for phrase in exit_phrases):
            await audio_output.speak("Catch you on the flip side!", on_start_speaking)
            os._exit(0)

        if user_text.lower() == "voice":
            MODE = "voice"
        elif user_text.lower() == "text":
            MODE = "text"

        # --- CONFIDENCE POLICY ---
        if confidence < LOW_CONFIDENCE:
            print("🤔 Low confidence. Asking to repeat.")
            await audio_output.speak(
                "Sorry, I didn’t catch that. Can you say it again?",
                on_start_speaking
            )
            audio_input.is_speaking = False
            continue

        elif confidence < HIGH_CONFIDENCE:
            print("🤨 Medium confidence. Clarifying.")
            user_text = f"(Unclear speech) {user_text}"


        # 2. THINK: Send it to the Brain and get a stream of thoughts back.
        response_stream = llm_engine.generate_response_stream(
            user_text,
            confidence=confidence
        )
        
        # 3. SPEAK: Pipe those thoughts directly to the Mouth.
        
        # We need this callback so the Ear knows when to cover its ears.
        # Otherwise, the AI hears itself and we get an infinite echo loop of doom.
        def on_start_speaking():
            audio_input.is_speaking = True
            stop_speaking_event.clear() # Reset the kill switch
            
        await audio_output.speak(response_stream, on_start_speaking)
        
        # Done talking? Cool, tell the Ear it's safe to listen again.
        audio_input.is_speaking = False
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass