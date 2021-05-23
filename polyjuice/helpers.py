import itertools
from spacy.tokens import Token, Span, Doc
import spacy
def _normalize_chunk(doc, start_idx, end_idx):
    # end idx is not included: [start, end)
    # the function is for normalizing the chunk so it does not start/end with punctuation
    punctuation = "''\"!, -.:;<=>?\^_|~”’ "
    if not doc:
        return None
    end_idx = min([len(doc), end_idx])
    start_idx = max([0, start_idx])
    while start_idx < end_idx and (doc[start_idx].text in punctuation or doc[start_idx].is_punct or doc[start_idx].is_space):
        start_idx += 1
    while start_idx < end_idx and (doc[end_idx-1].text in punctuation or doc[end_idx-1].is_punct or doc[end_idx-1].is_space):
        end_idx -= 1
    return doc[start_idx:end_idx]


def _normalize_span(span):
    #print(span,type(span))
    if type(span) == Span: return _normalize_chunk(span.doc, span.start, span.end)
    if type(span) == Token: return _normalize_chunk(span.doc, span.i, span.i+1)
    return None

def unify_tags(dep):
    norm_dep = dep
    if dep.endswith("mod") or dep in ["poss", "appos", "compound", "meta", "nn", "neg"]:
        # "nummod", "quantmod"
        norm_dep = "modifier"
    elif dep.endswith("cl") or dep.endswith("comp") or dep in ["mark", "parataxis", "attr", "oprd", "prep", "cop"]:
        norm_dep = "clause"
    elif dep.endswith("obj") or dep in ["dative", "obl"]:
        norm_dep = "obj"
    elif "subj" in dep:
        norm_dep = "subj"
    elif dep in ["cc", "conj", "preconj", "predet"]:
        norm_dep = "conj" 
    elif dep == "ROOT": norm_dep = "ROOT"
    elif dep == "aux": norm_dep = "aux"
    else: norm_dep = "others"
    return norm_dep


def flatten_fillins(doc, indxes, fillins_by_idxes, is_return_text=True):
    """Use index and corresponding fill ins to create a modified string.
    Useful for creating blanks.

    Args:
        doc (Doc): spacy doc
        indxes ([int, int][]): the indexes where the text should be modified
        fillins_by_idxes (str[][]): a list of candidate fillins at each space.
        is_return_text (bool, optional): If true, return the first possible 
        changed text. Otherwise return a list. Defaults to False.

    Returns:
        [type]: [description]
    """
    text_arr = []
    prev = 0
    zipped = list(zip(indxes, fillins_by_idxes))
    zipped = sorted(zipped, key=lambda z: z[0][0])
    for (start, end), fillins in zipped:
        if type(fillins) == str:
            fillins = [fillins]
        space = " " if end == 0 or doc[end-1].text_with_ws.endswith(' ') else ""
        text_arr.append([doc[prev:start].text_with_ws])
        text_arr.append([f.strip() + space for f in fillins])
        prev = end
    #if len(doc) < prev:
    text_arr.append([doc[prev:].text])
    otexts = ["".join(s) for s in itertools.product(*text_arr)]
    if is_return_text:
        return otexts[0].strip() if otexts else ""
    else:
        return otexts.strip()


class _WhitespaceSpacyTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = _WhitespaceSpacyTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        words = text.strip().split(" ")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def create_processor(is_space_tokenizer):
    nlp = spacy.load("en_core_web_sm")
    if is_space_tokenizer:
        nlp.tokenizer = _WhitespaceSpacyTokenizer(nlp.vocab)
    return nlp