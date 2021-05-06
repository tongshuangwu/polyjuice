from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from munch import Munch
import scipy
import math
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from sentence_transformers import SentenceTransformer
import difflib
import numpy as np

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


def load_pipeline_by_task(model_name):
    pipename = "text2text-generation"
    tokenizer_class, model_class = (AutoTokenizer, AutoModelForSeq2SeqLM)
    tokenizer = tokenizer_class.from_pretrained("t5-base")
    model = model_class.from_pretrained(model_name)
    return pipeline(pipename, model=model, tokenizer="t5-base", framework="pt", device=0)


def generate_batch(examples, generator, temperature=1, 
    num_beams=None, n=3, top_p=0.9, do_sample=True, batch_size=128, **kwargs):
    preds_list = []
    with torch.no_grad():
        for e in (range(0, len(examples), batch_size)):
            preds_list += generator(
                examples[e:e+batch_size],
                temperature=temperature,
                return_tensors=True,
                num_beams=num_beams,
                top_p=top_p,
                max_length=1000,
                early_stopping=None if num_beams is None else True,
                do_sample=num_beams is None and do_sample,
                num_return_sequences=n, **kwargs
            )
    return preds_list

def _normalize_chunk(doc, start_idx, end_idx):
    # end idx is not included: [start, end)
    # the function is for normalizing the chunk so it does not start/end with punctuation
    punctuation = "''\"!, -.:;<=>?\^_|~”’ "
    if not doc: return None
    end_idx = min([len(doc), end_idx])
    start_idx = max([0, start_idx])
    while start_idx < end_idx and (doc[start_idx].text in punctuation or doc[start_idx].is_punct or doc[start_idx].is_space):
        start_idx += 1
    while start_idx < end_idx and (doc[end_idx-1].text in punctuation or doc[end_idx-1].is_punct or doc[start_idx].is_space):
        end_idx -= 1
    return doc[start_idx:end_idx]
def is_punct_phrase(idxes, doc):
    return idxes[1] - idxes[0] == 0 or len(_normalize_chunk(doc, idxes[0], idxes[1]))==0


#########################################################################
### helper functions for computing polyjuice changes
#########################################################################

def complete_tree_pair(aidx, bidx, adoc, bdoc, ops, adoc_range=None, bdoc_range=None):
    """this function extend the subphrases to their matching subtrees.

    Returns:    
        [Span, Span]: the extended aspan, and bspan
    """
    if is_punct_phrase(aidx, adoc) and is_punct_phrase(bidx, bdoc):
        return adoc[aidx[0]:aidx[1]], bdoc[bidx[0]:bidx[1]]
    for full_tree in [False, True]:
        aidx_extend = complete_one_tree(aidx, adoc, range_idxes=adoc_range, full_tree=full_tree)
        bidx_extend = complete_one_tree(bidx, bdoc, range_idxes=bdoc_range, full_tree=full_tree)
        # this part allows overflow
        _bidx_candidate = list(bidx_extend) + [match_idx_in_eops(idx, ops, "from", True) for idx in aidx_extend]
        bidx_new = [max(0, min(_bidx_candidate)), min(len(bdoc), max(_bidx_candidate))]        
        _aidx_candidate = list(aidx_extend) + [match_idx_in_eops(idx, ops, "to", True) for idx in bidx_new]
        aidx_new = [max(0, min(_aidx_candidate)), min(len(adoc), max(_aidx_candidate))]

        if not is_punct_phrase(aidx_new, adoc) and not is_punct_phrase(bidx_new, bdoc):
            break
    return adoc[aidx_new[0]:aidx_new[1]], bdoc[bidx_new[0]:bidx_new[1]]


def compute_edit_ops(adoc, bdoc, only_return_edit=False):
    stopwords = ["the", "an", "a", "be", ""]
    eops_raw = difflib.SequenceMatcher(
        a=[t.text for t in adoc],
        b=[t.text for t in bdoc]).get_opcodes()
    eops = []
    for op,f1,f2,t1,t2 in eops_raw:
        if only_return_edit and (
            op == "equal" or 
                (_normalize_chunk(adoc, f1, f2).lemma_.lower() in stopwords and \
                _normalize_chunk(bdoc, t1, t2).lemma_.lower() in stopwords)):
            continue
        s1, s2 = complete_tree_pair([f1, f2], [t1, t2], adoc, bdoc, eops_raw)
        #print(op,f1,f2,t1,t2)
        #print(s1)
        #print(s2)
        #print(adoc[f1:f2])
        #print(bdoc[2:2])
        eops.append(Munch(op=op,
            # all are spans
            fromz_core=adoc[f1:f2], toz_core=bdoc[t1:t2],
            fromz_full=s1, toz_full=s2,
        ))
    return eops

def complete_one_tree(span_idxes, doc, range_idxes=None, full_tree=True):
    """Compute the complete subtree for a given span

    Args:
        span_idxes ([int, int]): the indexes of the given span to be extended
        doc (Doc): The spacy doc
        range_idxes ([int, int]], optional): the maixmum indexs. Defaults to None.
        full_tree (bool, optional): if use the entire tree. Defaults to True.

    Returns:
        [int, int]: the extended tree
    """
    span_start, span_end = span_idxes
    if span_end - span_start == 0:
        return [span_start, span_end]
    if all([doc[i].is_punct for i in range(span_start, span_end)]):
        return [span_start, span_start]
    span = doc[span_start: span_end]
    if full_tree:
        left_edge, right_edge = range_idxes if range_idxes else [0, len(doc)]
        for token in span:
            if any([t.is_ancestor(token) for t in span]):
                continue
            head = token.head
            left_edge = max(left_edge, head.left_edge.i)
            right_edge = min(right_edge, head.right_edge.i)
    else:
        all_tokens = set()
        for token in span:
            head = token.head
            subtrees = set([head, token]) #(set(head.subtree) - set(token.subtree)) | set([token])
            all_tokens |= subtrees
        #left_edge = max(left_edge, curr_left_edge)
        #right_edge = min(right_edge, curr_right_edge)
        left_edge = min([a.i for a in all_tokens] + [span_start])
        right_edge = max([a.i for a in all_tokens] + [span_end-1])
    while left_edge < span_start and doc[left_edge].is_punct:
        left_edge += 1
    while right_edge > span_end and doc[right_edge].is_punct:
        right_edge -= 1
    left_idx, right_idx = left_edge, right_edge
    if left_idx <= right_idx: # or root == span.root.head:
        return [left_idx, right_idx+1]
    else:
        return [span_start, span_end]

def _convert_eops_munch_to_arr(eops_munch):
    return [(e.op, e.fromz_core.start, e.fromz_core.end, e.toz_core.start, e.toz_core.end) for e in eops_munch]

def match_idx_in_eops(source_idx, ops, source_str="from", allow_overflow=False):
    sidx_start = 1 if source_str=="from" else 3
    tidx_start = 3 if source_str=="from" else 1
    sidx_end, tidx_end = sidx_start + 1, tidx_start + 1
    target_idx = -1
    if allow_overflow and source_idx == ops[-1][sidx_start+1]:
        return ops[-1][tidx_end]
    for op in ops:
        if (source_idx >= op[sidx_start] and source_idx < op[sidx_end]) or \
            (source_idx == op[sidx_start] and source_idx == op[sidx_end]):
            if op[tidx_start] == op[tidx_end]:
                target_idx = op[tidx_start]
            else:
                target_idx = source_idx - op[sidx_start] + op[tidx_start]
            if not allow_overflow and target_idx >= ops[-1][tidx_end]-1:
                target_idx = ops[-1][tidx_end]-1
            break
    return target_idx


#########################################################################
### compute similarity
#########################################################################
def load_distance_scorer(is_cuda):
    distance_scorer = SentenceTransformer('stsb-distilbert-base')
    if is_cuda: distance_scorer.to('cuda')
    return distance_scorer

def compute_sent_cosine_distance(s1, s2, similarity_scorer):
    s1 = [s1] if type(s1) == str else s1
    s2 = [s2] if type(s2) == str else s2
    s1_embed, s2_embed = similarity_scorer.encode(s1), similarity_scorer.encode(s2)
    distances = scipy.spatial.distance.cdist(s1_embed, s2_embed, "cosine")[0][0]
    # np.unravel_index(distances.argmin(), distances.shape)
    return distances

#########################################################################
### compute perplexity
#########################################################################

def load_perplex_scorer(is_cuda):
    return LMScorer.from_pretrained("gpt2", device="cuda:0" if is_cuda else "cpu", batch_size=1)

def reduce_perplex_prob(log_probs, log=False, reduce="prod"):
    tlen = log_probs.shape[0]
    if reduce == "prod":
        score = log_probs.sum()
    elif reduce == "mean":
        score = log_probs.logsumexp(0) - math.log(tlen)
    elif reduce == "gmean":
        score = log_probs.mean(0)
    elif reduce == "hmean":
        score = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
    else:
        raise ValueError("Unrecognized scoring strategy: %s" % reduce)
    if not log:
        score = score.exp()
    return score.item()

def normalize_score(log_score, slen, alpha=0.8):
    #Elephant in the Room: An Evaluation Framework for Assessing Adversarial Examples in NLP
    return log_score/math.pow((5+slen)/6, alpha)

def compute_sent_perplexity(sentences, perplex_scorer, log=True, reduce="prod", is_normalize=False):
    """Compute the sentence perplexity. For filtering.

    Args:
        sentences ([type]): [description]
        perplex_scorer ([type]): [description]
        log (bool, optional): [description]. Defaults to True.
        reduce (str, optional): [description]. Defaults to "prod".
        is_normalize (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    scores = []
    for sentence in sentences:
        score = perplex_scorer.sentence_score(sentence, reduce=reduce, log=log)
        lens = len(perplex_scorer.tokenizer(sentence)["input_ids"])
        if is_normalize:
            score = normalize_score(score, lens)
        scores.append(score)
    return scores

def filter_by_sent_perplexity(sentences, perplex_scorer, thred=20):
    scores = compute_sent_perplexity(sentences, perplex_scorer, log=True, reduce="prod", is_normalize=False)
    idxes = np.where(np.array(scores) <= thred)[0]
    filtered =  [sentences[i] for i in idxes]


def compute_phrase_perplexity(
    sentence_phrase_tuples, perplex_scorer, log=True, reduce="prod", is_normalize=False):
    scores = []
    sentence_phrase_tuples = sentence_phrase_tuples if type(sentence_phrase_tuples) != tuple else [sentence_phrase_tuples]
    if len(sentence_phrase_tuples) == 0:
        return scores
    outputs = perplex_scorer._tokens_log_prob([s[0] for s in sentence_phrase_tuples])
    for idx, (sentence, phrase) in enumerate(sentence_phrase_tuples):
        full_len = len(perplex_scorer.tokenizer(sentence)["input_ids"])
        if phrase:
            prefix_len = len(perplex_scorer.tokenizer(sentence.split(phrase)[0].strip())["input_ids"])
        else:
            prefix_len = 0
        phrase_len = len(perplex_scorer.tokenizer(phrase)["input_ids"])
        prefix_idx, phrase_idx = [0, prefix_len], [prefix_len, prefix_len+phrase_len]
        log_probs_all = outputs[idx][0]
        log_probs = log_probs_all[phrase_idx[0]:phrase_idx[1]]
        #print(sentence.split(phrase)[0].strip(), perplex_scorer.tokenizer(sentence.split(phrase)[0].strip()))
        #print(sentence, phrase, phrase_idx)
        full_sent_score = reduce_perplex_prob(log_probs_all, log=log, reduce=reduce)
        phrase_score = reduce_perplex_prob(log_probs, log=log, reduce=reduce)
        if is_normalize:
            full_sent_score = normalize_score(full_sent_score, full_len)
            phrase_score = normalize_score(phrase_score, phrase_len)
        scores.append((full_sent_score, phrase_score))
    return scores

def compute_delta_perplexity(edit_ops, perplex_scorer, is_normalize=False, **kwargs):
    """This is to compute the perplexity 

    Args:
        edit_ops ([type]): [description]
        perplex_scorer ([type]): [description]
        is_normalize (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    tuples = []
    #print(metadata.primary.acore.doc.text)
    #print(metadata.primary.bcore.doc.text)
    for op in edit_ops:
        aphrase, bphrase = (op.fromz_full, op.toz_full) if \
            op.op == "insert" or op.op == "delete" else (op.fromz_core, op.toz_core)
        asent, bsent = aphrase.doc, bphrase.doc
        tuples += [(asent.text, aphrase.text), (bsent.text, bphrase.text)]
    #print(tuples)
    scores = compute_phrase_perplexity(tuples, perplex_scorer, is_normalize=is_normalize)
    #print(scores)
    paired_scores = []
    for i in range(len(edit_ops)):
        # because of negative, it's i - i+1; lower the better.
        #print(scores[2*i])
        #print(scores[2*i+1])
        paired_scores.append(Munch(
            pr_sent=scores[2*i][0]-scores[2*i+1][0], 
            pr_phrase=scores[2*i][1]-scores[2*i+1][1]))
    paired_scores = sorted(paired_scores, key=lambda x: (
        max(x.pr_sent, x.pr_phrase)), reverse=True) # use the most ungrammar part as the 
    return paired_scores[0]

