from spacy.tokens import Token, Span
from pattern.en import wordnet, number, numerals
import numpy as np
import itertools


####################################################################################
                ##### HELPER FUNCTIONS FOR SELECTING BLANKS ########
####################################################################################
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
    deps=None, is_token_only=False, max_counts=None):
    unique_blanks = {str([[0, len(doc)]]): [[0, len(doc)]]}
    default_deps = [None, "", ["subj","obj"], ["aux", "ROOT"], ["conj", "modifier", "clause"]]
    if is_token_only:
        unique_blanks = set()
    if deps is None: deps = default_deps
    for dep in deps:
        # for each different dep, get some blank
        rounds = 1 if dep is not None else 2
        if is_token_only:
            rounds = 5
        for r in range(rounds):
            curr_idx = get_one_random_idx_set(
                doc, req_dep=dep, 
                pre_selected_idxes=pre_selected_idxes, 
                is_token_only=is_token_only) if dep != "" else ""
            if curr_idx is not None:
                unique_blanks[str(curr_idx)] = curr_idx
    unique_blanks = list(unique_blanks.values())
    if max_counts is not None:
        unique_blanks = list(np.random.choice(unique_blanks, min(len(unique_blanks), max_counts), replace=False))
    return unique_blanks


####################################################################################
                ##### HELPER FUNCTIONS FOR META ########
####################################################################################
DEFAULT_NUM = 123456
NOUN_VERIATIONS = ["NOUN", "PROPN", "PRON"]

def all_synsets(word, pos=None):
    map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJECTIVE,
        'ADV': wordnet.ADVERB
        }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
    else:
        pos_list = [map[pos]]
    ret = []
    for pos in pos_list:
        ret.extend(wordnet.synsets(word, pos=pos))
    return ret

def clean_senses(synsets):
    return [x for x in set(synsets) if '_' not in x]
def all_possible_synonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        # if syn.synonyms[0] != word:
        #     continue
        ret.extend(syn.senses)
    return clean_senses(ret)

def all_possible_antonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        if not syn.antonym:
            continue
        for s in syn.antonym:
            ret.extend(s.senses)
    return clean_senses(ret)

def all_possible_hypernyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hypernyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)
def all_possible_hyponyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hyponyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)
def all_possible_related(words, pos=None, depth=1):
    all_syns = [y for word in words for y in all_synsets(word, pos=pos)]
    # all_syns = [all_synsets(x, pos=pos) for x in words]
    # all_syns = [x[0] for x in all_syns if x]
    # return all_syns
    # print(all_syns)
    all_ancestors = [wordnet.ancestor(s1, s2) for s1, s2 in itertools.combinations(all_syns, 2)]
    all_ancestors = [x for x in all_ancestors if x]
    # print(all_ancestors)
    mapz = {x.lexname: x for x in all_ancestors}
    all_ancestors = list(mapz.values())
    all_descendents = [y for x in all_ancestors for y in x.hyponyms(recursive=True, depth=depth)]
    ret = [y for x in all_descendents for y in x.senses]
    return clean_senses(ret)


def is_subphrase(subset, superset):
    return subset.start >= superset.start and subset.end <= superset.end

def is_child_phrase(subset, superset):
    #print("is_child_phrase", subset.root.head in superset, subset.start >= superset.start and subset.end <= superset.end)
    if subset.start >= superset.start and subset.end <= superset.end:
        return True
    if all([t.head in superset for t in get_top_tokens(subset)]):
        return True
    return False #

def _normalize_span(span):
    #print(span,type(span))
    
    if type(span) == Span: return _normalize_chunk(span.doc, span.start, span.end)
    if type(span) == Token: return _normalize_chunk(span.doc, span.i, span.i+1)
    return None

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

def is_noun(span):
    normalized_span = _normalize_span(span)
    if not normalized_span: return False
    if type(normalized_span) == Token: return normalized_span.pos_ in NOUN_VERIATIONS 
    else: return normalized_span.root.pos_ in NOUN_VERIATIONS
def get_wordnet_info (func, span):
    normalized_span = _normalize_span(span)
    return set(func(normalized_span.lemma_)).union(set(func(normalized_span.text.lower())))
def get_hypernyms(root):
    if not is_noun(root): return set()
    hypers = get_wordnet_info(all_possible_hypernyms, root)
    if root.is_stop: return set()
    if root.pos_ in ["NOUN", "PROPN"]: hypers.add("something")
    if root.pos_ in ["PRON", "PROPN"] or root.ent_type_ == "PERSON": hypers.add("someone")
    if root.ent_type_: hypers.add(root.ent_type_)
    return hypers

def is_passive(span):
    if not span: return False
    children_set = set()
    """
    for t in span:
        children_set.add(t.dep_)
        print(t, list(t.children))
        children_set.union([i.dep_ for i in t.children])
    print(span, children_set)
    """
    #print([t.dep_ for t in span])
    return "auxpass" in set([t.dep_ for t in span])#, span.root.lemma_
    #return passive and passive.issubset(set(["agent", "auxpass"]))

def get_passive_token(span, only_in_span=False):
    if not span: return None
    children_set = set()
    """
    for t in span:
        children_set.add(t.dep_)
        print(t, list(t.children))
        children_set.union([i.dep_ for i in t.children])
    print(span, children_set)
    """
    head = span.root
    if not only_in_span:
        for t in span:
            if t.dep_ == "auxpass": 
                head = t.head
                break
        #while (head.pos_ not in ["VERB", "AUX"] or head.dep_.endswith("comp")) and head != head.head: head = head.head
    #print("get_passive_token", span, head)
    if head.pos_ not in ["VERB", "AUX"]: #or head.dep_.endswith("comp"):
        return None
    else:
        return head.lemma_

def get_negations(span, is_return_token=False):
    general_negations = set([
        "rare", "not", "no", "nothing", "unlike", "unless", "nobody", 
        "never", "seem", "nothing", "neither", "nowhere", "hardly", "scarcely",
        "barely", ])
    negation_phrases = set([
        "rather than", "suppose to", "other than", "don't", "doesn't", "didn't"
        "use to", "should be", "can be",  "hard to", "isn't", "shouldn't"
    ])
    
    general_sets, head_sets = set(), set()
    if not span: general_sets, head_sets

    def add_negations(t):
        general_sets.add(t if is_return_token else t.lemma_)
    for t in span:
        if t.lemma_ in general_negations or t.dep_ == "neg":
            add_negations(t)
        elif "n't" in t.text.lower():
            add_negations(t)
        else:
            head_sets.add(t.lemma_)
    for n in negation_phrases:
        if n in span.lemma_:
            add_negations(span)
            head_sets = set()
    return general_sets, head_sets

def is_subjunctive(span):
    text = span.text.lower().replace("'ve", "have")
    lemma = span.lemma_.replace("'ve", "have")
    if not span: return False
    #print (span.text.lower(), span.lemma_)
    return span and (
        "should have" in text or  \
    "could have" in text or "should have" in lemma or  "could have" in lemma)

def get_tense(span, only_in_span=False):
    if not span: return None, None
    # get the verb
    head = span.root
    if not only_in_span:
        while (head.pos_ not in ["VERB", "AUX"] or head.dep_.endswith("comp")) and head != head.head: head = head.head
    # childrens 
    if head.pos_ not in ["VERB", "AUX"] or head.dep_.endswith("comp"):
        return None, None
    #print(span, head, head.dep_, list(head.children))
    if "will" in [c.lemma_ for c in list(head.children)+[head]]: return "future", head.lemma_
    sub_verb = [c for c in head.children if c.lemma_ in ["can", "do"]]
    core = sub_verb[0] if sub_verb else head
    return "past" if core.tag_ in ["VBD", "VBN"] else "present", core.lemma_

def get_quantifiers(span):
    # determine if it's changing quantifier
    # case 1: words occur: all, every, some, no, nothing, something, at most, at least
    # case 2: is changing nummod/quantmod or the root of the change is NUM

    def get_number(r):
        r = r.lower()
        #print(r)
        if r == "zero" or str(r) == "0": return 0
        elif number(r) > 0: return number(r)
        #elif r == "both": return 2
        elif r == "dozen": return 12
        elif r in ["no", "nobody"]: return 0
        #elif r in ["all", "every", "any", "each"]: return 40404
        #elif r in ["few"]: return 50505
        #elif r in ["many", "much"]: return 20202
        #elif r in ["some", "several", "certain"]: return 30303
        #elif r in ["other", "another"]: return 10101
        return -1
    general_quantifiers = set([
        "all", "every", "some", "any", "several", "certain", "each", "both", "many","much",
        "no", "dozen", "lot", "few", "other", "another", "nobody", "most", "least"])
    if not span:
        return set(), set(), ""
    # only allow it if we know it's a modifier change
    tops = span #get_top_tokens(span)
    #if span.root.pos_ != "NUM" and all([unify_tags(s) != "modifier" for s in tops]):
    #    return set(), set()
    # and t.ent_type_ in ["PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]
    # and t.head.ent_type_ in ["PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]
    #print([(t.lemma_, t.pos_, t.dep_) for t in tops])
    related = set([t.lemma_ for t in tops if (t.lower_ in general_quantifiers and \
        (t.dep_ in ["det"] or unify_tags(t.dep_) in ["subj", "obj"])) or \
        (t.dep_ in ["nummod", "quantmod"]) or \
        (t.pos_ == "NUM" ) or \
        (t.head.pos_ == "NUM" and unify_tags(t.dep_) in ["modifier"])
        ])
    #print(tops)
    # also extract the numbers
    numbers = set([get_number(r) for r in related if get_number(r) >= 0])
    #print(related, numbers)
    #if "a" in related: numbers.add(1)
    #if "an" in related: numbers.add(1)
    if "both" in related: numbers.add(2)
    if "no" in related or "none" in related: numbers.add(0)
    if "dozen" in related: numbers.add(12)
    #print(span, related, numbers)
    related = set([r for r in related if get_number(r) == -1])
    return related.union(numbers), numbers, span.root.lemma_.lower()

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

def unify_pos(pos):
    #if pos in ["VERB", "AUX"]: return "VERB"
    if pos in NOUN_VERIATIONS: return "NOUN"
    elif pos.endswith("CONJ") or pos == "ADP": return "ADP"
    return pos

def get_top_tokens(span, is_filter_stop=False):
    normalized_span = _normalize_span(span)
    roots = []
    if not normalized_span: 
        return roots
    for token in normalized_span:
        if is_filter_stop and (token.is_stop or token.is_punct):
            continue
        if any([t.is_ancestor(token) for t in normalized_span]):
            continue
        roots.append(token)
    return roots

def is_single_change(span):
    include_tokens = [t for t in _normalize_span(span) if not t.dep_ in ["det", "prt"] and not t.is_punct]
    top_tokens = get_top_tokens(span)
    #print("is_single_change", include_tokens)
    #print("is_single_change", top_tokens)
    return len(include_tokens) <= 1 and len(top_tokens) <= 1

def is_same_structure(t1, t2):
    a, b = get_top_tokens(t1), get_top_tokens(t2)
    #print("is_same_structure", t1, t2)
    dep1 = set([unify_tags(r.dep_) for r in a])
    dep2 = set([unify_tags(r.dep_) for r in b])
    dep_intersect = dep1.intersection(dep2) - set(["aux"])
    pos_intersect = set([unify_pos(r.pos_) for r in a]).intersection(set([unify_pos(r.pos_) for r in b]))
    #print("is_same_structure", dep1, dep2)
    head1 = set([(unify_tags(r.head.dep_), unify_tags(r.head.pos_)) for r in a])
    head2 = set([(unify_tags(r.head.dep_), unify_tags(r.head.pos_)) for r in b])
    head_intersect = head1 == head2 #and len(set([unify_tags(r.head.dep_) for r in a])-set(["ROOT"])) > 0
    #print(dep_intersect, head_intersect)
    #print(a, [r.head for r in a], [unify_pos(r.pos_) for r in a], [r.dep_ for r in a])
    #print(b, [r.head for r in b], [unify_pos(r.pos_) for r in b], [r.dep_ for r in b])
    #print(head1)
    #print(dep_intersect)
    #print(pos_intersect)
    #print(head1, head_intersect)
    #print(t1, is_single_change(t1))
    #print(t2, is_single_change(t2))
    #if "ROOT" in dep_intersect and is_single_change(t1) and is_single_change(t2):
    #    return True
    #elif "ROOT" not in dep_intersect:
    return  len(dep_intersect) or head_intersect
    if "ROOT" not in dep2 and "ROOT" not in dep1:
        return  (len(pos_intersect) > 0 and head_intersect) or len(dep_intersect) > 0
    else:
        return  (len(dep_intersect) > 0 and len(pos_intersect) > 0) and head_intersect

