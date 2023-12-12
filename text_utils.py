from nltk.tokenize import sent_tokenize


def force_splitter(sentence, token, max_length=250):
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


def splitter(text, max_length=250):
    sentences = sent_tokenize(text)
    out = []
    for sentence in sentences:
        if len(sentence) < max_length:
            out.append(sentence)
            continue
        splits = force_splitter(sentence, ",", max_length)
        for split in splits:
            if len(split) < max_length:
                out.append(split)
            else:
                out.extend(force_splitter(split, ";", max_length))
    return out
