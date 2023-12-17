import os
import tempfile
import torch
import soundfile as sf
import whisperx
import noisereduce
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class SentenceTTS:
    def __init__(self, model_dir, sample_path):
        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_dir, "config.json"))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=model_dir,
            use_deepspeed=False,
        )
        _ = self.model.cuda()
        self.sample = sample_path
        (
            self.gpt_cond_latent,
            self.speaker_embedding,
        ) = self.model.get_conditioning_latents(audio_path=[self.sample])
        self.whisper_model = whisperx.load_model(
            "large-v3", "cuda", compute_type="float16"
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device="cuda"
        )

    def inference(self, text):
        text = text.trim() + " pause"
        out = self.model.inference(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, out["wav"], 24000)
            temp_file_path = temp_file.name
            audio = whisperx.load_audio(temp_file_path)
        result = self.whisper_model.transcribe(audio, batch_size=16)
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            "cuda",
            return_char_alignments=False,
        )

        pause_word = result["word_segments"][-1]
        end_sample = int(pause_word["start"] * 24000)

        # Define the duration of the fade in seconds
        fade_duration = 0.2  # for example, a 50ms fade

        # Calculate the number of samples for the fade duration
        fade_samples = int(fade_duration * 24000)  # assuming a sample rate of 24000 Hz

        # Create a fade-out envelope (linear fade)
        fade_out_envelope = torch.linspace(1, 0, fade_samples)

        # Apply the fade to the last `fade_samples` of the audio
        audio_tensor = torch.tensor(out["wav"])
        fade_start = end_sample - fade_samples
        audio_tensor[fade_start : int(end_sample)] *= fade_out_envelope

        # Ensure the audio ends with zero amplitude to avoid the click
        output = audio_tensor[: end_sample + 1]
        output[end_sample] = 0

        reduced_noise = noisereduce.reduce_noise(
            y=output, sr=24000, stationary=True, prop_decrease=0.75
        )
        return reduced_noise
