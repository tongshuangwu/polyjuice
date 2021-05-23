from nltk.translate.bleu_score import sentence_bleu
from zss import simple_distance, Node
import numpy as np
import edit_distance
from munch import Munch
import itertools

def compute_lev_distance(doc1, doc2):
    sm = edit_distance.SequenceMatcher(
        a=[t.text for t in doc1], b=[t.text for t in doc2])
    return 1-sm.ratio()

def add_node_to_root(root):
    if not root or not list(root.children):
        return None
    curr_node = Node(root.text.lower())
    for c in root.children:
        cnode = add_node_to_root(c)
        if cnode:
            curr_node.addkid(cnode)
    return curr_node
        
def get_nodes(doc):
    span = list(doc.sents)[0]
    return add_node_to_root(span.root)

def compute_tree_edit_distance(doc1, doc2):
    try:
        dist = simple_distance(get_nodes(doc1), get_nodes(doc2))
        return dist #/ len(doc1)
    except: return 0

def compute_closeness(docs, base_doc, sentence_similarity=None):
    sem_dist, tree_dist, edit_dist = [], [], []
    for doc in docs:
        if sentence_similarity:
            sem_dist.append(sentence_similarity(base_doc.text, doc.text))
        tree_dist.append(compute_tree_edit_distance(base_doc, doc))
        edit_dist.append(compute_lev_distance(base_doc, doc))
    return Munch(
        sem_dist=np.mean(sem_dist), 
        tree_dist=np.mean(tree_dist),
        edit_dist=np.mean(edit_dist), 
    )

def compute_self_bleu(docs, base_doc, kwargs=None):
    # if should just be augments around one example.
    scores = []
    if len(docs) == 0:
        return Munch(bleu4=1)
    included = []
    data_points = []
    for doc in docs:
        included.append(doc.text)
        data_points.append([d.text for d in doc])
    included.append(base_doc)
    data_points.append([d.text for d in base_doc])

    points = list(itertools.combinations(range(len(included)), 2))
    for i, j in points:
        scores.append(sentence_bleu([data_points[i]], data_points[j] ))
    return Munch(bleu4=np.mean(scores))