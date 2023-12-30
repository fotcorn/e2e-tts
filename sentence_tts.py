import os
import math
import tempfile
import torch
import soundfile as sf
import noisereduce
import json
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import whisper_timestamped


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

        self.whisper_model = whisper_timestamped.load_model("small", "cuda")

    def inference(self, text):
        for i in range(5):
            try:
                return self._inference(text)
            except Exception as ex:
                print(f"Try {i} failed: {ex}")
                continue
        raise Exception("Inference failed after 5 tries")

    def _inference(self, text):
        text = text.strip()
        padded_text = text + " pause"

        out = self.model.inference(
            padded_text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, out["wav"], 24000)
            temp_file_path = temp_file.name
            audio = whisper_timestamped.load_audio(temp_file_path)

        result = whisper_timestamped.transcribe(
            self.whisper_model, audio, language="en", seed=None
        )

        last_word = result["segments"][-1]["words"][-1]
        last_word_text = last_word["text"].lower()

        def normalize_word(word):
            word = word.lower()
            word = word.replace(".", "").replace(",", "").replace(";", "")
            word = word.strip()
            return word

        if not "pause" in normalize_word(last_word_text):
            if normalize_word(text.rsplit(" ", 1)[1]) == normalize_word(last_word_text):
                print("did not find pause, but found last word")
                end_sample = int(last_word["end"] * 24000)
            else:
                import IPython

                IPython.display.display(
                    IPython.display.Audio(
                        torch.tensor(out["wav"]).unsqueeze(0), rate=24000
                    )
                )
                print(last_word)
                print("-------------")
                print(padded_text)
                print("-------------")
                print(
                    json.dumps(
                        result["segments"][-1]["words"], indent=2, ensure_ascii=False
                    )
                )
                raise Exception("'pause' is not the last detected word")
        else:
            end_sample = int(last_word["start"] * 24000)

        fade_duration = 0.2

        # Calculate the number of samples for the fade duration
        fade_samples = int(fade_duration * 24000)

        # Create a fade-out envelope (linear fade)
        fade_out_envelope = torch.linspace(1, 0, fade_samples)

        # Apply the fade to the last `fade_samples` of the audio
        audio_tensor = torch.tensor(out["wav"])
        fade_start = end_sample - fade_samples
        audio_tensor[fade_start : int(end_sample)] *= fade_out_envelope

        output = audio_tensor[: end_sample + 1]
        # Ensure the audio ends with zero amplitude to avoid clicking noise
        output[end_sample] = 0

        reduced_noise = noisereduce.reduce_noise(
            y=output, sr=24000, stationary=True, prop_decrease=0.75
        )
        return reduced_noise
