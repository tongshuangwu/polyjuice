import math
import numpy as np
from munch import Munch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import torch
from copy import deepcopy

#########################################################################
### compute perplexity
#########################################################################

def _add_special_tokens(text, tokenizer):
    return tokenizer.bos_token + text + tokenizer.eos_token

def _tokens_log_prob(texts, model, tokenizer, batch_size=128, is_cuda=True):
        outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outputs.extend(_tokens_log_prob_for_batch(batch, model, tokenizer, is_cuda=is_cuda))
        return outputs
    
def _tokens_log_prob_for_batch(texts, model, tokenizer, is_cuda=True):
    device = "cuda" if is_cuda else "cpu"
    outputs = []
    texts = [_add_special_tokens(text, tokenizer) for text in deepcopy(texts)]
    #encoding = tokenizer.batch_encode_plus(texts, return_tensors='pt')
    encoding = tokenizer.batch_encode_plus(texts, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        #nopad_mask = ids != tokenizer.pad_token_id
        nopad_mask = ids != tokenizer.pad_token_id
        logits = model(ids, attention_mask=attention_mask)[0]
    
    for sent_index in range(len(texts)):
        sent_nopad_mask = nopad_mask[sent_index]
        sent_tokens = [tok 
            for i, tok in enumerate(encoding.tokens(sent_index))
            if sent_nopad_mask[i] and i != 0]
        sent_ids = ids[sent_index, sent_nopad_mask][1:]
        sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
        sent_logits[:, tokenizer.pad_token_id] = float("-inf")
        sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
        sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)
        #sent_log_probs = cast(torch.DoubleTensor, sent_log_probs)
        #sent_ids = cast(torch.LongTensor, sent_ids)

        output = (sent_log_probs.cpu().numpy(), sent_ids.cpu().numpy(), sent_tokens)
        outputs.append(output)
    return outputs

def load_perplex_scorer(model_id = 'gpt2', is_cuda=True):
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id, use_fast=True, add_special_tokens=False)
    device = "cuda" if is_cuda else "cpu"
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
    tokenizer.pad_token = "<|pad|>"
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model.to(device)
    return Munch(model=model, tokenizer=tokenizer)
    
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

def compute_sent_perplexity(
    sentences, perplex_scorer, log=True, reduce="prod", is_normalize=False, is_cuda=True):
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
    model, tokenizer = perplex_scorer.model, perplex_scorer.tokenizer
    outputs = _tokens_log_prob(sentences, model, tokenizer, is_cuda=is_cuda)
    for sent_log_prob, sent_ids, sent_tokens in outputs:
        score = reduce_perplex_prob(sent_log_prob, reduce=reduce, log=log)
        if is_normalize:
            score = normalize_score(score, len(sent_tokens))
        scores.append(score)
    return scores

def filter_by_sent_perplexity(sentences, perplex_scorer, thred=20, is_cuda=True):
    scores = compute_sent_perplexity(
        sentences, perplex_scorer, log=True, reduce="prod", is_normalize=False, is_cuda=is_cuda)
    idxes = np.where(np.array(scores) <= thred)[0]
    filtered =  [sentences[i] for i in idxes]

def compute_phrase_perplexity(
    sentence_phrase_tuples, perplex_scorer, 
    log=True, reduce="prod", is_normalize=False, is_cuda=True):
    scores = []
    sentence_phrase_tuples = sentence_phrase_tuples if type(sentence_phrase_tuples) != tuple else [sentence_phrase_tuples]
    if len(sentence_phrase_tuples) == 0:
        return scores
    model, tokenizer = perplex_scorer.model, perplex_scorer.tokenizer
    outputs = _tokens_log_prob([s[0] for s in sentence_phrase_tuples], model, tokenizer, is_cuda=is_cuda)
    for idx, (sentence, phrase) in enumerate(sentence_phrase_tuples):
        log_probs_all = outputs[idx][0]
        full_len = len(outputs[idx][1]) - 1
        if phrase:
            prefix_len = len(tokenizer(sentence.split(phrase)[0].strip())["input_ids"])
        else:
            prefix_len = 0
        phrase_len = len(tokenizer(phrase)["input_ids"])
        prefix_idx, phrase_idx = [0, prefix_len], [prefix_len, prefix_len+phrase_len]
        
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

def compute_delta_perplexity(edit_ops, perplex_scorer, is_normalize=False, is_cuda=True):
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
    edit_ops = [o for o in edit_ops if o.op != "equal"]
    for op in edit_ops:
        aphrase, bphrase = (op.fromz_full, op.toz_full) if \
            op.op == "insert" or op.op == "delete" else (op.fromz_core, op.toz_core)
        asent, bsent = aphrase.doc, bphrase.doc
        tuples += [(asent.text, aphrase.text), (bsent.text, bphrase.text)]
    #print(tuples)
    scores = compute_phrase_perplexity(tuples, perplex_scorer, 
        is_normalize=is_normalize, is_cuda=is_cuda)
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
