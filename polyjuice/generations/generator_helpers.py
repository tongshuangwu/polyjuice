from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
import itertools
from .special_tokens import BLANK_TOK, SEP_TOK, EMPTY_TOK, ANSWER_TOK, PERETURB_TOK
from .create_blanks import create_blanked_sents

def load_generator(model_path="uw-hai/polyjuice", is_cuda=True):
    return pipeline("text-generation", 
        model=AutoModelForCausalLM.from_pretrained(model_path), 
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        framework="pt", device=0 if is_cuda else -1)

def split_ctrl_code(text):
    r = re.search(r'\[(?P<code>[a-z]+)\](?P<text>.+)', text)
    if r:
        return r.group("code").strip(), r.group("text").strip()
    return "", text
def remove_blanks(text):
    try:
        before, answers = text.split(SEP_TOK)
        answers = [x.strip() for x in answers.split(ANSWER_TOK)][:-1]
        answers = [x if x != EMPTY_TOK else '' for x in answers]
        for a in answers:
            if a == '':
                before = re.sub(r' %s' % re.escape(BLANK_TOK), a, before, count=1)
            else:
                before = re.sub(r'%s' % re.escape(BLANK_TOK), a, before, count=1)
        return before, answers
    except:
        return text, []

def batched_generate(generator, 
    examples, 
    temperature=1, 
    num_beams=None,
    num_return_sequences=3, 
    do_sample=True, 
    batch_size=128, **kwargs):
    preds_list = []
    with torch.no_grad():
        for e in (range(0, len(examples), batch_size)):
            preds_list += generator(
                examples[e:e+batch_size],
                temperature=temperature,
                return_tensors=True,
                num_beams=num_beams,
                max_length=1000,
                early_stopping=None if num_beams is None else True,
                do_sample=num_beams is None and do_sample,
                num_return_sequences=num_return_sequences, 
                **kwargs
            )
    return preds_list

def generate_on_prompts(generator, prompts, temperature=1, 
    num_beams=None, n=3, do_sample=True, batch_size=128, **kwargs):
    preds_list = batched_generate(generator, prompts,
        temperature=temperature, n=n, 
        num_beams=num_beams, 
        do_sample=do_sample, batch_size=batch_size, **kwargs)
    if len(prompts) == 1:
        preds_list = [preds_list]
    preds_list_cleaned = []
    for prompt, preds in zip(prompts, preds_list):
        prev_list = set()
        for s in preds:
            total_sequence = s["generated_text"].split(PERETURB_TOK)[-1]
            normalized, _ = remove_blanks(total_sequence)
            input_ctrl_code, normalized = split_ctrl_code(normalized)
            prev_list.add((input_ctrl_code, normalized))
        preds_list_cleaned.append(list(prev_list))
    return preds_list_cleaned

def get_prompts(doc, ctrl_codes, blanked_sents, is_complete_blank=True):
    prompts = []
    for tag, bt in itertools.product(ctrl_codes, blanked_sents):
        sep_tok = SEP_TOK if bt and is_complete_blank else ""
        prompts.append(f"{doc.text.strip()} {PERETURB_TOK} [{tag}] {bt.strip()} {sep_tok}".strip())
    return prompts

"""
def perturb_one_doc(generator, doc, tags, indexes, temperature=1, 
    num_beams=None, n=3, top_p=0.9, do_sample=True, batch_size=128):
    prompts = get_prompts(doc, tags, indexes)
    return generate_on_prompts(
        generator, prompts, 
        temperature=temperature,
        n=n, 
        top_p=top_p, 
        do_sample=do_sample,
        batch_size=batch_size
    )
"""