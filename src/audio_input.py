import asyncio
import queue
import numpy as np
import pyaudio
import torch
import wave
import time
from faster_whisper import WhisperModel

class AudioInput:
    def __init__(self, stop_event):
        # This event is the "Shut Up" button. If it's set, we stop talking.
        self.stop_event = stop_event
        
        # The "Inbox" for the main loop. Final text goes here.
        self.text_queue = asyncio.Queue()
        
        # The "Raw Feed". Audio bytes from the mic get dumped here.
        self.audio_queue = queue.Queue()
        
        # --- Audio Settings ---
        self.CHUNKS = 512       # Chunk size. Too small = CPU burn, too big = lag.
        self.FORMAT = pyaudio.paInt16 # 16-bit audio. Good enough for voice.
        self.CHANNELS = 1       # Mono. We have one mouth, so one ear is fine.
        self.RATE = 16000       # 16kHz. The gold standard for speech models.
        
        # --- "Can you hear me now?" Thresholds ---
        # RMS = Volume. How loud are you screaming?
        self.RMS_THRESHOLD = 200      # Silence threshold. Don't transcribe breathing.
        self.BARGE_IN_RMS = 3000      # Interruption threshold. You gotta yell to stop the bot.
        self.BARGE_IN_CONFIDENCE = 0.7 # VAD confidence. Are we SURE it's a human?
        
        # Are we currently blabbering?
        self.is_speaking = False
        
        # --- Brain Transplants ---
        print("Loading Whisper... (The ears)")
        # Using 'small' because we like speed. Int8 because we like RAM.
        self.stt_model = WhisperModel("small", device="cpu", compute_type="int8")
        
        print("Loading Silero VAD... (The reflex)")
        # This thing is crazy fast at telling speech from a car horn.
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )

    def mic_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio calls this when it has data. 
        We treat it like a hot potato: grab it and throw it in the queue.
        Don't do heavy math here or the audio will glitch.
        """
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """Starts the background listening and transcription tasks."""
        asyncio.create_task(self.listen_loop())
        asyncio.create_task(self.transcription_loop())

    async def listen_loop(self):
        """
        The 'Ear' Loop.
        It sits there, listening to the mic, filtering out the noise,
        and waiting for you to say something profound.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNKS,
                        stream_callback=self.mic_callback)
        
        stream.start_stream()
        print("\n🎤 Listening... (Say 'Exit' to stop)")

        audio_buffer = []
        started_talking = False
        silence_counter = 0
        silence_limit = 60 # ~2 seconds of silence = "I'm done talking."
        phrase_limit = 15  # ~0.5 seconds = "I'm taking a breath."
        phrase_processed = False
        
        # The conveyor belt to the Brain (transcription_loop)
        self.transcription_queue = asyncio.Queue()

        while True:
            try:
                # Get the latest audio chunk from the mic
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                # If no data yet, take a tiny nap to let other tasks run
                await asyncio.sleep(0.01)
                continue

            # Convert raw bytes to numbers so we can do math on them
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Calculate volume (RMS)
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
            
            # Ask the VAD model: "Is this speech?" (returns 0.0 to 1.0)
            voice_confidence = self.vad_model(torch.from_numpy(audio_float32), self.RATE).item()
            
            # --- The "Shut Up" Logic ---
            # If the AI is talking, we need to be LOUDER to interrupt it.
            # Otherwise, it hears its own voice and thinks "Wow, I'm interrupting myself!"
            # Solving this issue gave me the run for my money.
            if self.is_speaking:
                is_speech = (voice_confidence > self.BARGE_IN_CONFIDENCE) and (rms > self.BARGE_IN_RMS)
            else:
                is_speech = (voice_confidence > 0.5) and (rms > self.RMS_THRESHOLD)

            if is_speech:
                # If we detect speech while the AI is talking, SHUT IT UP!
                if self.is_speaking:
                    print("\n🛑 Interruption detected! Stopping AI...")
                    self.stop_event.set() # Signal the output module to stop
                    self.is_speaking = False
                
                if not started_talking:
                    print(f"\n🗣️ User started speaking...")
                    started_talking = True
                    audio_buffer = []
                    phrase_processed = False
                
                silence_counter = 0
                audio_buffer.append(data)
                phrase_processed = False
                
            elif started_talking:
                # User was talking, but now it's silent.
                audio_buffer.append(data)
                silence_counter += 1
                
                # If silent for a bit, try to transcribe what we have (Incremental STT)
                # This makes it feel faster because we don't wait for the full sentence.
                if silence_counter > phrase_limit and not phrase_processed:
                    await self.process_audio_input(audio_buffer, is_final=False)
                    phrase_processed = True
                
                # If silent for a long time, assume the user is done.
                if silence_counter > silence_limit:
                    print("🔇 End of speech detected.")
                    await self.process_audio_input(audio_buffer, is_final=True)
                    
                    # Reset everything for the next turn
                    started_talking = False
                    audio_buffer = []
                    silence_counter = 0
                    phrase_processed = False

    async def process_audio_input(self, audio_buffer, is_final=False):
        """Helper to prepare audio data for the transcriber."""
        if not audio_buffer:
            return
        # Combine all chunks into one byte stream
        audio_data = b''.join(audio_buffer)
        audio_int16 = np.frombuffer(audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Send it to the transcription loop
        await self.transcription_queue.put((audio_float32, is_final))

    async def transcription_loop(self):
        """
        The 'Translator' Loop.
        Takes raw audio bytes and turns them into words using Whisper.
        It's like a court stenographer, but digital and less likely to get carpal tunnel.
        """
        accumulated_text = ""
        while True:
            audio_data, is_final = await self.transcription_queue.get()
            
            print(f"👂 Transcribing segment...")
            segments, _ = self.stt_model.transcribe(
                    audio_data, 
                    beam_size=1, 
                    language="en", 
                    task="transcribe",
                    vad_filter=True, # Double check silence with Whisper's internal VAD (belt and suspenders)
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            
            text_parts = []
            for segment in segments:
                # Filter out "hallucinations" (when the model guesses but isn't sure)
                if segment.avg_logprob > -1.0:
                    text_parts.append(segment.text)
                else:
                    print(f"⚠️ Ignored low confidence segment: '{segment.text}' ({segment.avg_logprob:.2f})")
            
            text = "".join(text_parts).strip()
            
            # Filter out common "ghost" phrases Whisper hears in silence
            # Just don't thank the Agent, It'll ignore it.
            if text.lower() in ["thank you.", "thank you", "you", "thanks."]:
                 print(f"⚠️ Ignored hallucination: '{text}'")
                 text = ""

            if text:
                accumulated_text = text
                print(f"📝 Partial: {accumulated_text}")
            
            # Only send the text to the Main Brain if it's the FINAL part of the sentence
            if is_final:
                if accumulated_text:
                    await self.text_queue.put(accumulated_text)
                    accumulated_text = ""
            
            self.transcription_queue.task_done()
