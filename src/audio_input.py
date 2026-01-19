import asyncio
import queue
import numpy as np
import pyaudio
import torch
import sys
import time
import io
import scipy.io.wavfile as wavfile
from groq import Groq
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from utils.helpers import read_yaml_config

load_dotenv()

class AudioInput:
    """
    The 'Ear' of the operation. 👂
    Responsible for listening to the microphone, detecting voice activity (VAD),
    and transcribing speech to text using Faster-Whisper.
    """
    def __init__(self, stop_event, wake_word_enabled=True, config_path="config/config.yaml"):
        self.config = read_yaml_config(config_path)
        self.audio_settings = self.config.get("audio_settings", {})
        # The "Kill Switch" event. If set, we stop any active output.
        self.stop_event = stop_event
        
        # The "Inbox" for the main loop. Final, polished text lands here.
        self.text_queue = asyncio.Queue()
        
        # The "Raw Feed". Raw audio bytes from the mic are dumped here.
        self.audio_queue = queue.Queue()
        
        # --- Groq Model Setup (for future use) ---
        self.groq_client = Groq()
        self.GROQ_MODEL = "whisper-large-v3-turbo"
        print(f"Connected to Groq Cloud {self.GROQ_MODEL}")

        # --- Wake Word Configuration ---
        self.wake_word = self.audio_settings.get("wake_word", "emma").lower()
        self.wake_word_enabled = wake_word_enabled
        self.is_awake = not wake_word_enabled  # If disabled, we're always listening.
        self.wake_word_timeout = self.audio_settings.get("wake_word_timeout", 30)  # Seconds of silence before we snooze.
        self.last_wake_time = 0
        self.wake_word_buffer_size = self.audio_settings.get("wake_word_buffer_size", 3.0)  # Rolling buffer size (in seconds) for wake detection.
        
        # Throttling for wake word checks (prevent CPU spam)
        self.last_wake_check_time = 0
        self.WAKE_CHECK_INTERVAL = self.audio_settings.get("wake_check_interval", 0.5) # Check at most every 0.5 seconds
        
        # --- Audio Stream Settings ---
        self.CHUNKS = self.audio_settings.get("chunks", 512)       # Buffer size. Lower = less latency, Higher = less CPU load.
        self.FORMAT = pyaudio.paInt16 # 16-bit audio. Standard for speech recognition.
        self.CHANNELS = self.audio_settings.get("channels", 1)       # Mono audio. We only need one ear.
        self.RATE = self.audio_settings.get("sample_rate", 16000)       # 16kHz sample rate. The native tongue of Whisper.
        
        # --- Voice Activity Detection (VAD) Thresholds ---
        # RMS (Root Mean Square) = Volume.
        self.RMS_THRESHOLD = self.audio_settings.get("threshold", 400)      # Noise floor. Ignore anything quieter than this.
        self.BARGE_IN_RMS = self.audio_settings.get("barge_in_rms", 3000)      # Interruption threshold. Yell louder than this to stop the bot.
        self.BARGE_IN_CONFIDENCE = self.audio_settings.get("barge_in_confidence", 0.8) # VAD confidence required to interrupt.
        
        # State flag: Are we currently outputting audio?
        self.is_speaking = False
        self.STREAMING_INTERVAL = 1.0
        
        # --- Model Initialization ---
        print("🎧 Initializing Audio Subsystems...")
        
        # Initialize Faster-Whisper
        # We use a tiny model for quick wake-word detection and a small model for accurate transcription.
        
        print("  - Loading 'base.en' model for wake word detection...")
        self.wake_word_model = WhisperModel(
            "base.en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8",  # Quantized for CPU efficiency
            num_workers=2,
            cpu_threads=4
        )
        
        print("  - Loading Silero VAD for voice detection...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        
        # --- Wake Word Buffer Setup ---
        if self.wake_word_enabled:
            print(f"✅ Wake word ready! Say '{self.wake_word.capitalize()}' to activate.")
            self.wake_word_buffer = []
            # Calculate max chunks for the rolling buffer
            self.max_wake_buffer_chunks = int(self.wake_word_buffer_size * (self.RATE / self.CHUNKS))

    def _create_wave_file(self, audio_float32):
        """
        Converts raw numpy float32 audio data into a WAV file object in memory.
        Groq API requires a file-like object with a filename.
        """
        # Convert back to int16 for WAV Standard
        audio_int16 = (audio_float32 * 32768).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, self.RATE, audio_int16)
        buffer.seek(0)

        return ("audio.wav", buffer.read(), "audio/wav")
    
    # --- Helper: Groq Transcription ---
    async def _transcribe_with_groq(self, audio_float32, prompt=""):
        """Sends audio to Groq and returns text."""
        try: 
            wav_file = self._create_wave_file(audio_float32)

            # Run blocking API call in a thread to keep the loop async
            transcription_result = await asyncio.to_thread(
                self.groq_client.audio.transcriptions.create,
                file=wav_file,
                model=self.GROQ_MODEL,
                prompt=prompt,
                language="en",
                temperature=0.0
            )

            return transcription_result.text.strip()
        except Exception as e:
            print(f"⚠️ Groq transcription failed: {e}")
            return ""

    def mic_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback.
        This needs to be lightning fast. Grab the data, shove it in the queue, and get out.
        """
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """Ignites the background listening and transcription engines."""
        asyncio.create_task(self.listen_loop())
        asyncio.create_task(self.transcription_loop())
        if self.wake_word_enabled:
            asyncio.create_task(self.wake_word_timeout_loop())

    def _transcribe_wake_word(self, audio_float32):
        """
        Blocking transcription call to be run in a thread.
        """
        try:
            segments, _ = self.wake_word_model.transcribe(
                audio_float32,
                beam_size=1,
                language="en",
                vad_filter=False,
                without_timestamps=True
            )
            return " ".join([segment.text.lower().strip() for segment in segments])
        except Exception as e:
            print(f"⚠️ Wake word transcription failed: {e}")
            return ""

    async def check_for_wake_word(self, audio_buffer):
        """
        Checks the audio buffer for the wake word using the tiny model.
        Runs in a separate thread to avoid blocking the main loop.
        Returns True if detected, False otherwise.
        """
        try:
            # Flatten buffer to a single byte string
            audio_data = b''.join(audio_buffer)
            # Convert to normalized float32 array
            audio_int16 = np.frombuffer(audio_data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Run transcription in a thread
            text = await asyncio.to_thread(self._transcribe_wake_word, audio_float32)
            
            if self.wake_word in text:
                print(f"\n🎯 Wake word '{self.wake_word.capitalize()}' detected!")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Wake word check failed: {e}")
            return False

    async def wake_word_timeout_loop(self):
        """
        The Sandman. 😴
        Puts the agent back to sleep if no one talks to it for a while.
        """
        while True:
            await asyncio.sleep(1)
            
            if self.is_awake and self.last_wake_time > 0:
                elapsed = time.time() - self.last_wake_time
                if elapsed > self.wake_word_timeout:
                    self.is_awake = False
                    print(f"\n😴 Timeout reached. Going to sleep.")
                    print(f"   Say '{self.wake_word.capitalize()}' to wake me up.")

    def deserves_transcription(self, audio_buffer: list[bytes]) -> bool:
        audio_data = b"".join(audio_buffer)
        audio_int16 = np.frombuffer(audio_data, np.int16)

        duration = len(audio_int16) / self.RATE
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))

        # Hard rules
        if duration < 0.5:
            return False

        if rms < self.RMS_THRESHOLD * 1.2:
            return False

        return True


    async def listen_loop(self):
        """
        The Main Listening Loop.
        Captures audio, checks for voice activity, and manages recording state.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNKS,
                        stream_callback=self.mic_callback)
        
        stream.start_stream()
        
        if self.wake_word_enabled:
            print(f"\n🎤 Listening for wake word '{self.wake_word.capitalize()}'...")
        else:
            print("\n🎤 Listening... (Say 'Exit' to stop)")

        audio_buffer = []
        is_recording = False
        silence_counter = 0
        # Calculate silence limit (0.8 seconds) - Reduced for lower latency
        silence_limit = int(0.8 * (self.RATE / self.CHUNKS)) 
        
        # Track when recording started for grace period
        recording_start_time = 0
        
        self.transcription_queue = asyncio.Queue()

        while True:
            try:
                # Non-blocking get from queue
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01) # Yield to other tasks
                continue

            # Convert for processing
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Compute metrics
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
            voice_confidence = self.vad_model(torch.from_numpy(audio_float32), self.RATE).item()
            
            # --- Wake Word Mode ---
            if self.wake_word_enabled and not self.is_awake:
                self.wake_word_buffer.append(data)
                if len(self.wake_word_buffer) > self.max_wake_buffer_chunks:
                    self.wake_word_buffer.pop(0)
                
                # Check for wake word only if we hear something resembling speech
                if voice_confidence > 0.6 and rms > self.RMS_THRESHOLD:
                    # Throttle checks
                    current_time = time.time()
                    if current_time - self.last_wake_check_time > self.WAKE_CHECK_INTERVAL:
                        self.last_wake_check_time = current_time
                        
                        if await self.check_for_wake_word(self.wake_word_buffer):
                            self.is_awake = True
                            self.last_wake_time = time.time()
                            print("✅ Awake and listening!")
                            
                            # Transition directly to recording state to capture the rest of the sentence
                            # We keep the wake buffer as the start of the recording
                            is_recording = True
                            recording_start_time = time.time()

                            # Keep only last 0.4–0.6s AFTER wake word
                            max_chunks = int(0.5 * self.RATE / self.CHUNKS)
                            audio_buffer = self.wake_word_buffer[-max_chunks:]

                            self.wake_word_buffer = []
                            silence_counter = 0
                continue
            
            # --- Active Mode ---
            
            # Update activity timer
            if self.wake_word_enabled and voice_confidence > 0.5 and rms > self.RMS_THRESHOLD:
                self.last_wake_time = time.time()
            
            # 1. Interruption Logic (Barge-In)
            if self.is_speaking:
                if (voice_confidence > self.BARGE_IN_CONFIDENCE) and (rms > self.BARGE_IN_RMS):
                    print("\n🛑 Interruption detected! Cutting speech.")
                    self.stop_event.set()
                    self.is_speaking = False
                    
                    if not is_recording:
                        print(f"\n🗣️ User interrupted...")
                        is_recording = True
                        recording_start_time = time.time()
                        audio_buffer = []
                        silence_counter = 0
            
            # 2. Start Recording
            elif not is_recording:
                if (voice_confidence > 0.65) and (rms > self.RMS_THRESHOLD):
                    print(f"\n🗣️ Speech detected...")
                    is_recording = True
                    recording_start_time = time.time()
                    audio_buffer = []
                    silence_counter = 0

            # 3. Continue Recording
            if is_recording:
                audio_buffer.append(data)
                
                # Silence Detection
                if (voice_confidence < 0.5) or (rms < self.RMS_THRESHOLD):
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # End of Turn
                # Grace Period: Don't cut off due to silence if we just started recording (e.g. < 2.0s)
                # This allows for a pause between "Emma" and the command.
                if silence_counter > silence_limit:
                    if (time.time() - recording_start_time > 2.0):
                        is_recording = False

                        if self.deserves_transcription(audio_buffer):
                            print("🔇 End of turn. Valid speech.")
                            await self.process_audio_input(audio_buffer, is_final=True)
                        else:
                            print("🗑️ Discarded low-quality turn.")

                        audio_buffer = []
                        silence_counter = 0
                    # else: inside grace period, ignore silence

    async def process_audio_input(self, audio_buffer, is_final=False):
        """Prepares audio data for the transcription loop."""
        if not audio_buffer:
            return
        audio_data = b''.join(audio_buffer)
        audio_int16 = np.frombuffer(audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        await self.transcription_queue.put((audio_float32, is_final))

    # --- HALLUCINATION DETECTOR ---
    def _estimate_confidence(self, text: str, audio_seconds: float) -> float:
        """
        Heuristic confidence score [0.0 – 1.0]
        """
        if not text.strip():
            return 0.0

        tokens = len(text.split())

        # Unrealistic speech rate
        tokens_per_sec = tokens / max(audio_seconds, 0.1)
        if tokens_per_sec > 6:
            return 0.2

        # Very short audio, long sentence
        if audio_seconds < 1.0 and tokens > 10:
            return 0.3

        # Whisper politeness hallucinations
        if text.lower().startswith(("thank", "thanks")):
            return 0.2

        return 0.9 if tokens >= 2 else 0.5


    def _is_garbage(self, text: str, audio_seconds: float) -> bool:
        """
        Final hallucination / noise rejection.
        """
        if not text.strip():
            return True

        tokens = len(text.split())

        if tokens == 1 and audio_seconds > 2.0:
            return True

        if tokens > 20 and audio_seconds < 2.0:
            return True

        tokens_per_sec = tokens / max(audio_seconds, 0.1)
        if tokens_per_sec > 7:
            return True

        return False


    async def transcription_loop(self):
        """
        The Scribe. ✍️
        - Preview STT: local, disposable
        - Final STT: Groq, validated
        - Emits (text, confidence)
        """
        while True:
            audio_float32, is_final = await self.transcription_queue.get()

            try:
                audio_seconds = len(audio_float32) / self.RATE

                # ---------------------------
                # PREVIEW (UI ONLY)
                # ---------------------------
                if not is_final:
                    segments, _ = self.wake_word_model.transcribe(
                        audio_float32,
                        beam_size=1,
                        language="en",
                        vad_filter=True,
                        temperature=0.0
                    )

                    preview = " ".join(s.text.strip() for s in segments)
                    if preview:
                        sys.stdout.write(f"\r👂 Hearing: {preview}...")
                        sys.stdout.flush()

                    continue

                # ---------------------------
                # FINAL TRANSCRIPTION
                # ---------------------------
                sys.stdout.write("\r")
                sys.stdout.flush()

                groq_text = await self._transcribe_with_groq(audio_float32)

                if self._is_garbage(groq_text, audio_seconds):
                    print("🗑️ Discarded low-quality speech")
                    continue

                confidence = self._estimate_confidence(groq_text, audio_seconds)

                print(f"📝 Final: {groq_text} (conf={confidence:.2f})")

                # IMPORTANT: queue text + confidence
                await self.text_queue.put({
                    "text": groq_text,
                    "confidence": confidence,
                    "duration": round(audio_seconds, 2)
                })

            except Exception as e:
                print(f"⚠️ Transcription failed: {e}")

            finally:
                self.transcription_queue.task_done()
