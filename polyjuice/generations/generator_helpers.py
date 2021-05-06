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

def remove_tags(text):
    return re.sub(r'\[\w+\]', "", text).strip()
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

def batched_generate(generator, examples, temperature=1, 
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

def generate_on_prompts(generator, prompts, temperature=1, 
    num_beams=None, n=3, top_p=0.9, do_sample=True, batch_size=128):
    preds_list = batched_generate(prompts, generator, 
        temperature=temperature, n=n, 
        num_beams=num_beams, 
        do_sample=do_sample, batch_size=batch_size)
    if len(prompts) == 1:
        preds_list = [preds_list]
    preds_list_cleaned = []
    for prompt, preds in zip(prompts, preds_list):
        prev_list = set()
        for s in preds:
            total_sequence = s["generated_text"].split(PERETURB_TOK)[-1]
            normalized, answers = remove_blanks(total_sequence)
            normalized = remove_tags(normalized)
            prev_list.add(normalized)
        preds_list_cleaned.append(list(prev_list))
    return preds_list_cleaned

def get_prompts(doc, tags, indexes, is_append_sep_tok=True):
    blanked_texts = create_blanked_sents(doc, indexes)
    
    prompts = []
    for tag, bt in itertools.product(tags, blanked_texts):
        sep_tok = SEP_TOK if bt and is_append_sep_tok else ""
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