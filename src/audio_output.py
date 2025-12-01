import asyncio
import edge_tts
import pygame
import io
import os

class AudioOutput:
    """
    The 'Mouth' of the AI.
    It takes text and turns it into sound waves. Magic.
    """
    def __init__(self, stop_event):
        self.stop_event = stop_event
        # Queue to hold audio streams waiting to be played
        self.audio_queue = asyncio.Queue()
        # Initialize Pygame mixer for audio playback
        pygame.mixer.init()
        
    async def start(self):
        """Starts the background playback loop."""
        asyncio.create_task(self.playback_loop())

    async def generate_audio_stream(self, text):
        """
        Uses Edge TTS to convert text -> audio bytes.
        Returns an in-memory stream (BytesIO) because saving files to disk is so 2010.
        """
        communicate = edge_tts.Communicate(text, "en-US-AndrewMultilingualNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return io.BytesIO(audio_data)

    async def play_audio_stream(self, audio_stream):
        """
        Plays a single audio stream using Pygame.
        Checks 'stop_event' constantly so we can shut it up instantly if needed.
        """
        audio_stream.seek(0)
        try:
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            # Wait while audio is playing...
            while pygame.mixer.music.get_busy():
                # If user interrupted, STOP immediately!
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    return False
                await asyncio.sleep(0.05)
            
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return True

    async def playback_loop(self):
        """
        The DJ Booth.
        Endlessly pulls audio tracks from the queue and plays them.
        This allows us to 'pipeline' audio: generating sentence 2 while playing sentence 1.
        """
        while True:
            audio_stream = await self.audio_queue.get()
            
            # If we were interrupted, discard this audio chunk
            if self.stop_event.is_set():
                self.audio_queue.task_done()
                continue
                
            await self.play_audio_stream(audio_stream)
            self.audio_queue.task_done()

    async def speak(self, text_stream, on_start_speaking=None):
        """
        Takes a stream of text from the LLM, breaks it into sentences,
        and queues them up for the DJ.
        """
        text_buffer = ""
        import re
        
        for chunk in text_stream:
            # If interrupted, stop processing the LLM stream
            if self.stop_event.is_set():
                break
            
            # Handle different chunk types (LangChain vs Google SDK vs String)
            text_chunk = ""
            if hasattr(chunk, "content"):
                text_chunk = chunk.content # LangChain
            elif hasattr(chunk, "text"):
                text_chunk = chunk.text # Google GenAI SDK
            elif isinstance(chunk, str):
                text_chunk = chunk # Raw string
            
            if text_chunk:
                text_buffer += text_chunk
                
                # Check for sentence endings (. ? !)
                if any(punct in text_buffer for punct in [".", "?", "!"]):
                    sentences = re.split(r'(?<=[.?!])\s+', text_buffer)
                    
                    # Process all complete sentences
                    for sentence in sentences[:-1]:
                        if self.stop_event.is_set():
                            break
                        
                        # Notify that we are about to speak (so VAD knows to expect noise)
                        # "Hey Ear, cover your ears, I'm about to yell!"
                        if on_start_speaking:
                            on_start_speaking()
                            
                        print(f"🗣️ AI Speaking: {sentence}")
                        
                        # Generate audio in background
                        audio_stream = await self.generate_audio_stream(sentence)
                        # Add to playback queue
                        await self.audio_queue.put(audio_stream)
                        
                    text_buffer = sentences[-1]
        
        # Process any remaining text (the last sentence)
        if text_buffer and not self.stop_event.is_set():
            if on_start_speaking:
                on_start_speaking()
            print(f"🗣️ AI Speaking: {text_buffer}")
            audio_stream = await self.generate_audio_stream(text_buffer)
            await self.audio_queue.put(audio_stream)
            
        # Wait for all audio to finish playing
        await self.audio_queue.join()
