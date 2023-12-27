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

    def _split_sentence(self, sentence, max_length):
        if len(sentence) < max_length:
            return [sentence]
        indices = [i for i, c in enumerate(sentence) if c == "," or c == ";"]
        subsequence_lengths = (
            [indices[0]]
            + [j - i for i, j in zip(indices[:-1], indices[1:])]
            + [len(sentence) - indices[-1]]
        )

        def helper(subseq, idx):
            if idx == len(subsequence_lengths):
                if subseq:
                    yield subseq
                return
            yield from helper(subseq + [[subsequence_lengths[idx]]], idx + 1)
            if subseq:
                yield from helper(
                    subseq[:-1] + [subseq[-1] + [subsequence_lengths[idx]]], idx + 1
                )

        subsequences = list(helper([], 0))

        best_seq = []
        best_seq_score = 10**100

        def calc_score(subseq):
            score = 0
            for s in subseq:
                total = sum(s)
                if total > max_length:
                    return None
                # Cube, so differences close to 200 are better
                score += (max_length - total) ** 2
            return score

        for subseq in subsequences:
            score = calc_score(subseq)
            if score and score < best_seq_score:
                best_seq = subseq
                best_seq_score = score

        # reconstruct the sentence
        start = 0
        end = 0
        splits = []
        for seq in best_seq:
            for index in seq:
                end += index
            splits.append(sentence[start : end + 1].strip())
            start = end + 1
        return splits

    def preprocess(self, text):
        text = self._replace_boeing_numbers(text)
        text = self._replace_number_ranges(text)
        text = self._replace_text_hyphens(text)

        sentences = sent_tokenize(text)
        sentences = self.normalizer.normalize_list(sentences)

        split_sentences = []
        for sentence in sentences:
            split_sentences.extend(
                self._split_sentence(sentence, self.max_sentence_length)
            )

        return split_sentences
