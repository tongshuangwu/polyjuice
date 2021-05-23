import numpy as np
from ..helpers import unify_tags, flatten_fillins
from .special_tokens import BLANK_TOK

def create_blanked_sents(doc, indexes=None):
    if indexes:
        if type(indexes[0]) == int: 
            indexes = [indexes]
        indexes_list = indexes #[indexes]
    else:
        indexes_list = get_random_idxes(
            doc, is_token_only=False, max_count=3)
    blanks = set([flatten_fillins(
        doc, indexes, [BLANK_TOK] * len(indexes)) \
        for indexes in indexes_list])
    return blanks

# the function for placing BLANKS.
def get_one_random_idx_set(
    doc, max_blank_block=3, req_dep=None, blank_type_prob=None, 
    pre_selected_idxes=None, is_token_only=False):
    if req_dep is not None:
        if type(req_dep) == str: req_dep = [req_dep]
        idx_range = [i for i, token in enumerate(doc) if token.dep_ in req_dep or unify_tags(token.dep_) in req_dep]
    else:
        idx_range = list(range(len(doc)))
    # only keep those pre_selected_idxes
    if pre_selected_idxes is not None:
        idx_range = [i for i in idx_range if i in pre_selected_idxes]
    max_blank_block = min(len(idx_range), max_blank_block)        
    #print(req_dep, idx_range)
    selected_indexes = []
    while max_blank_block > 0 and not selected_indexes:
        # if fixed the thing to change, then do one specific change
        n_perturb = np.random.choice(list(range(1, max_blank_block+1))) #if req_dep is None else 1
        replace_idx, total_run = -1, 1000
        while (total_run > 0 and n_perturb > 0): #and  len(span_and_edits) == 0:
            replace_idx = np.random.choice(idx_range)
            token = doc[replace_idx]
            if token.is_punct:
                total_run -= 1
                continue
            if blank_type_prob: p = blank_type_prob
            else:
                # if fixed the tree, then mostly use the tree
                if is_token_only:  p = [0.7, 0, 0.3]
                elif req_dep is None: p = [0.4, 0.35, 0.25]
                else: p = [0.1, 0.7, 0.2]
            is_replace_subtree = np.random.choice(["token", "subtree", "insert"], p=p)
            if is_replace_subtree == "subtree":
                start, end = token.left_edge.i, token.right_edge.i+1
            elif is_replace_subtree == "token":
                start, end = token.i, token.i+1
            else:
                start, end = token.i, token.i 
            if all([end < sstart or start > send for sstart, send in selected_indexes]):
                selected_indexes.append([start, end])
                n_perturb -= 1
            total_run -= 1
    return sorted(selected_indexes, key=lambda idx: (idx[0], idx[1]))


def get_random_idxes(doc, 
    pre_selected_idxes=None, 
    deps=None, is_token_only=False, 
    max_blank_block=3, max_count=None):
    unique_blanks = {str([[0, len(doc)]]): [[0, len(doc)]]}
    default_deps = [None, "", ["subj","obj"], ["aux", "ROOT"], ["conj", "modifier", "clause"]]
    if is_token_only: 
        unique_blanks = {}
    if deps is None: deps = default_deps
    for dep in deps:
        # for each different dep, get some blank
        rounds = 1 if dep is not None else 2
        if is_token_only:
            rounds = 5
        for _ in range(rounds):
            curr_idx = get_one_random_idx_set(
                doc, req_dep=dep, 
                max_blank_block=max_blank_block,
                pre_selected_idxes=pre_selected_idxes, 
                is_token_only=is_token_only) if dep != "" else None
            if curr_idx is not None:
                unique_blanks[str(curr_idx)] = curr_idx
    unique_blanks = list(unique_blanks.values())
    if max_count is not None:
        try:
            unique_blanks = list(np.random.choice(
                np.array(unique_blanks, dtype="object"), 
                min(len(unique_blanks), max_count), 
                replace=False))
        except:
            unique_blanks = unique_blanks[:max_count]
    return unique_blanks

