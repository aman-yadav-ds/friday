import asyncio
from piper import PiperVoice
import pygame
import io
import wave
import re
from utils.helpers import read_yaml_config

class AudioOutput:
    """
    The 'Mouth' of the operation. 🗣️
    Converts text to speech using Piper TTS and plays it back via Pygame.
    """
    def __init__(self, stop_event, config_path="config/config.yaml"):
        self.config = read_yaml_config(config_path)
        self.voice_settings = self.config.get("voice_settings", {})
        self.stop_event = stop_event
        # Queue for holding audio streams. FIFO (First In, First Out).
        self.audio_queue = asyncio.Queue()
        # Load the voice (assuming the model file is in the root directory)
        # Verify the path if using a different model
        self.voice = PiperVoice.load(self.voice_settings.get("model", "./piper_voices/en_GB-cori-high.onnx"))
        # Initialize the DJ (Pygame Mixer)
        pygame.mixer.init()
        
    async def start(self):
        """Kicks off the background playback loop."""
        asyncio.create_task(self.playback_loop())

    async def generate_audio_stream(self, text):
        """
        The Voice Box. 🎙️
        Uses Piper TTS to generate audio bytes from text.
        Returns an in-memory BytesIO stream (formatted as WAV).
        """
        if not text or not text.strip():
            return None
            
        try:
            # Piper generates raw PCM samples. We need to collect them.
            audio_data = b""
            # synthesize yields PCM bytes
            for chunk in self.voice.synthesize(text):
                # FIX: Check if chunk is bytes, otherwise extract the data
                try:
                    audio_data += chunk.audio_int16_bytes
                except AttributeError:
                    # Debugging helper if the above fail
                    print(f"DEBUG: Chunk type is {type(chunk)}, attributes: {dir(chunk)}")
            
            if not audio_data:
                return None
            
            # Create a WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.voice.config.sample_rate)
                wav_file.writeframes(audio_data)
                
            # Rewind buffer for reading
            wav_buffer.seek(0)
            return wav_buffer

        except Exception as e:
            print(f"⚠️ TTS Generation Error: {e}")
            return None

    async def play_audio_stream(self, audio_stream):
        """
        The Speaker. 🔊
        Plays a single audio stream. Monitors the 'stop_event' like a hawk
        to cut off speech instantly if interrupted.
        """
        if not audio_stream:
            return True
            
        try:
            # pygame needs a file-like object or path
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            # Busy wait loop for playback
            while pygame.mixer.music.get_busy():
                # The "Shut Up" check
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    return False
                # Low latency sleep for quick reaction time
                await asyncio.sleep(0.01)
            
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            print(f"⚠️ Audio playback error: {e}")
            return True

    async def playback_loop(self):
        """
        The DJ Booth. 🎧
        Continuously pulls tracks from the queue and spins them.
        """
        while True:
            audio_stream = await self.audio_queue.get()
            
            # Check if we should skip this track (interruption)
            if self.stop_event.is_set():
                self.audio_queue.task_done()
                continue
            
            if audio_stream:
                await self.play_audio_stream(audio_stream)
            
            self.audio_queue.task_done()

    async def speak(self, text_stream, on_start_speaking=None):
        """
        The Orator. 📜
        Consumes a stream of text tokens and queues them for playback.
        """
        text_buffer = ""
        
        for chunk in text_stream:
            # Check for interruption
            if self.stop_event.is_set():
                break
            
            # Extract text from various chunk types
            text_chunk = ""
            if hasattr(chunk, "content"):
                text_chunk = chunk.content # LangChain
            elif hasattr(chunk, "text"):
                text_chunk = chunk.text # Google GenAI SDK
            elif isinstance(chunk, str):
                text_chunk = chunk # Raw string
            
            if text_chunk:
                text_buffer += text_chunk
                
                # Sentence boundary detection
                if any(punct in text_buffer for punct in [".", "?", "!"]):
                    # Split by sentence endings, keeping the delimiter
                    sentences = re.split(r'(?<=[.?!])\s+', text_buffer)
                    
                    # Process complete sentences
                    for sentence in sentences[:-1]:
                        if self.stop_event.is_set():
                            break
                        
                        if not sentence.strip():
                            continue

                        # Signal start of speech
                        if on_start_speaking:
                            on_start_speaking()
                            
                        print(f"🗣️ AI Speaking: {sentence}")
                        
                        # Generate and queue audio
                        # Note: This is now a blocking call (Piper is fast but blocking)
                        # To make it truly async we might need run_in_executor if it hitches
                        audio_stream = await self.generate_audio_stream(sentence)
                        if audio_stream:
                            await self.audio_queue.put(audio_stream)
                        
                    text_buffer = sentences[-1]
        
        # Flush remaining text
        if text_buffer and text_buffer.strip() and not self.stop_event.is_set():
            if on_start_speaking:
                on_start_speaking()
            print(f"🗣️ AI Speaking: {text_buffer}")
            audio_stream = await self.generate_audio_stream(text_buffer)
            if audio_stream:
                await self.audio_queue.put(audio_stream)
            
        # Wait for the queue to drain
        await self.audio_queue.join()
