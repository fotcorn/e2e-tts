import re
from nemo_text_processing.text_normalization.normalize import Normalizer
from num2words import num2words
from nltk.tokenize import sent_tokenize


def replace_boeing_numbers(text):
    def replacer(match):
        number = int(match.group(1))
        return f"Boeing {num2words(number, to='year')}"

    return re.sub(r"Boeing (\d+)", replacer, text)


def replace_number_ranges(text):
    def replacer(match):
        num1 = int(match.group(1))
        num2 = int(match.group(2))
        return f"{num2words(num1, to='year')} {num2words(num2)}"

    text = re.sub(r"(\d+)-(\d+)", replacer, text)
    return re.sub(r"(\d+)–(\d+)", replacer, text)


def replace_text_hyphens(text):
    pattern = r"([a-zA-Z]+)-([a-zA-Z]+)"

    def replacer(match):
        return f"{match.group(1)} {match.group(2)}"

    return re.sub(pattern, replacer, text)


def preprocess(text):
    text = replace_boeing_numbers(text)
    text = replace_number_ranges(text)
    text = replace_text_hyphens(text)

    normalizer = Normalizer(input_case="cased", lang="en")
    sentences = normalizer.split_text_into_sentences(text)
    sentences = normalizer.normalize_list(sentences)

    return sentences


def _force_splitter(sentence, token, max_length=250):
    out = []
    while True:
        s = sentence.split(token, 1)
        if len(s) == 1:
            out.append(sentence)
            return out
        s1, s2 = s
        out.append(s1)
        if len(s2) > max_length:
            sentence = s2
        else:
            out.append(s2)
            return out


def split_text(text, max_length=250):
    sentences = sent_tokenize(text)
    out = []
    for sentence in sentences:
        if len(sentence) < max_length:
            out.append(sentence)
            continue
        splits = _force_splitter(sentence, ",", max_length)
        for split in splits:
            if len(split) < max_length:
                out.append(split)
            else:
                out.extend(_force_splitter(split, ";", max_length))
    return out
