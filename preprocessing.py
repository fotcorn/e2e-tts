import re
from nemo_text_processing.text_normalization.normalize import Normalizer
from num2words import num2words
from nltk.tokenize import sent_tokenize


class TextPreprocessor:
    def __init__(self, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        self.normalizer = Normalizer(input_case="cased", lang="en")

    def _replace_boeing_numbers(self, text):
        def replacer(match):
            number = int(match.group(1))
            return f"Boeing {num2words(number, to='year')}"

        return re.sub(r"Boeing (\d+)", replacer, text)

    def _replace_number_ranges(self, text):
        def replacer(match):
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            return f"{num2words(num1, to='year')} {num2words(num2)}"

        text = re.sub(r"(\d+)-(\d+)", replacer, text)
        return re.sub(r"(\d+)–(\d+)", replacer, text)

    def _replace_text_hyphens(self, text):
        pattern = r"([a-zA-Z]+)-([a-zA-Z]+)"

        def replacer(match):
            return f"{match.group(1)} {match.group(2)}"

        return re.sub(pattern, replacer, text)

    def _force_splitter(self, sentence, token, max_length):
        out = []
        while True:
            s = sentence.split(token, 1)
            if len(s) == 1:
                out.append(sentence)
                return out
            s1, s2 = s
            out.append(s1.strip() + token)
            if len(s2) > max_length:
                sentence = s2
            else:
                out.append(s2.strip())
                return out

    def _split_text(self, sentence, max_length):
        if len(sentence) < max_length:
            return [sentence]
        out = []
        splits = self._force_splitter(sentence, ",", max_length)
        for split in splits:
            if len(split) < max_length:
                out.append(split)
            else:
                out.extend(self._force_splitter(split, ";", max_length))
        return out

    def preprocess(self, text):
        text = self._replace_boeing_numbers(text)
        text = self._replace_number_ranges(text)
        text = self._replace_text_hyphens(text)

        sentences = sent_tokenize(text)
        sentences = self.normalizer.normalize_list(sentences)

        split_sentences = []
        for sentence in sentences:
            split_sentences.extend(self._split_text(sentence, self.max_sentence_length))

        return split_sentences
