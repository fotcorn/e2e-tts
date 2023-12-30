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
        end_of_sentence_text = " end of sentence pause"
        end_of_sentence_words = [s for s in end_of_sentence_text.split(" ") if s]
        padded_text = text + end_of_sentence_text

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

        def normalize_word(word):
            word = word.lower()
            word = word.replace(".", "").replace(",", "").replace(";", "")
            word = word.strip()
            return word

        num_words = len(end_of_sentence_words)
        last_generated_words = result["segments"][-1]["words"][-num_words:]
        # Check all words except the last, this one might be garbled.
        for expected_word, actual_word in zip(
            end_of_sentence_words[:-1], last_generated_words[:-1]
        ):
            if expected_word not in normalize_word(actual_word["text"]):
                raise Exception(f"Expected word {expected_word} not in {actual_word}")

        last_word_expected = end_of_sentence_words[-1]
        last_word_generated = normalize_word(last_generated_words[-1]["text"])
        if not last_word_expected in last_word_generated:
            print(
                f"Warning: Last word not in whisper output: {last_word_expected}, {last_word_generated}"
            )

        end_sample = int(last_generated_words[0]["start"] * 24000)

        fade_duration = 0.2

        # Calculate the number of samples for the fade duration
        fade_samples = int(fade_duration * 24000)

        # Create a fade-out envelope (linear fade)
        fade_out_envelope = torch.linspace(1, 0, fade_samples)

        # Apply the fade to the last `fade_samples` of the audio
        audio_tensor = torch.tensor(out["wav"])
        fade_start = end_sample - fade_samples
        audio_tensor[fade_start:end_sample] *= fade_out_envelope

        output = audio_tensor[: end_sample + 1]
        # Ensure the audio ends with zero amplitude to avoid clicking noise
        output[end_sample] = 0

        reduced_noise = noisereduce.reduce_noise(
            y=output, sr=24000, stationary=True, prop_decrease=0.75
        )
        return reduced_noise
