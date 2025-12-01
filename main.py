
import asyncio
import os
from src.audio_input import AudioInput
from src.llm_engine import LLMEngine
from src.audio_output import AudioOutput

async def main_loop():
    """
    The Big Boss. The Conductor. The Main Loop.
    This ties the Ear, Brain, and Mouth together into one somewhat cohesive unit.
    """
    print("✅ Agent is locked and loaded!")
    
    # The "Shut Up" Button
    # If this gets set, we kill the audio output immediately.
    stop_speaking_event = asyncio.Event()
    
    # --- The Crew ---
    # 1. The Ear: Handles the mic and figures out if you're actually talking.
    audio_input = AudioInput(stop_speaking_event)
    
    # 2. The Brain: The smart part (hopefully). Talks to Gemini.
    llm_engine = LLMEngine()
    
    # 3. The Mouth: The loud part. Turns text into noise.
    audio_output = AudioOutput(stop_speaking_event)
    
    # --- Spin up the background workers ---
    # These guys run forever, doing the heavy lifting.
    await audio_input.start()
    await audio_output.start()
    
    while True:
        # 1. LISTEN: Wait for the Ear to hand us a complete thought.
        user_text = await audio_input.text_queue.get()
        print(f"👤 You said (Final): {user_text}")

        if "exit" in user_text.lower():
            print("Catch you on the flip side!")
            os._exit(0)

        # 2. THINK: Send it to the Brain and get a stream of thoughts back.
        response_stream = llm_engine.generate_response_stream(user_text)
        
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
        pass