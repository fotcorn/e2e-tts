import argparse
import os
import numpy as np
import medium
import trafilatura
import soundfile


def main():
    parser = argparse.ArgumentParser(description="Process TTS arguments.")
    parser.add_argument(
        "path_to_voice_sample", type=str, help="Path to the voice sample file."
    )
    parser.add_argument("url", type=str, help="URL of the text to synthesize.")
    args = parser.parse_args()

    # lazy-load dependencies for fast fail on argument parsing errors
    from preprocessing import TextPreprocessor
    from sentence_tts import SentenceTTS

    tts = SentenceTTS(
        os.path.expanduser(
            "~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        ),
        args.path_to_voice_sample,
    )

    if "medium" in args.url:
        text = medium.get_text(args.url)
    else:
        page = trafilatura.fetch_url(args.url)
        text = trafilatura.extract(page, include_comments=False, include_tables=False)

    preprocessor = TextPreprocessor(max_sentence_length=240)
    sentences = preprocessor.preprocess(text)

    audio_data = np.array([], dtype=np.float32)
    for sentence in sentences[:4]:
        audio = tts.inference(sentence)
        audio_data = np.concatenate((audio_data, audio))

    soundfile.write("output.wav", audio_data, 24000)


if __name__ == "__main__":
    main()
