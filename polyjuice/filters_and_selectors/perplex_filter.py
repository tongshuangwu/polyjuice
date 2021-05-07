from lm_scorer.models.auto import AutoLMScorer as LMScorer
import math
import numpy as np
from munch import Munch

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
