import difflib
from ..helpers import _normalize_chunk
from munch import Munch

#########################################################################
### helper functions for computing polyjuice changes
#########################################################################
def is_punct_phrase(idxes, doc):
    return idxes[1] - idxes[0] == 0 or len(_normalize_chunk(doc, idxes[0], idxes[1]))==0

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
