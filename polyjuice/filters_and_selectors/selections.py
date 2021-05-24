from typing import Callable, List, Dict, Tuple
from munch import Munch
from ..compute_perturbs import compute_edit_ops, SentenceMetadata
from ..helpers import unify_tags
import numpy as np
import itertools
from copy import deepcopy
def submodular_fn(candidate, chosen, distances):
    return np.mean(distances[candidate, chosen])

def greedy(submodular_fn, distances, k, chosen=[]):
    chosen = deepcopy(chosen)
    all_items = list(range(distances.shape[0]))
    current_value = 0
    z = 0
    while len(chosen) != k:
        best_gain = 0
        best_item = all_items[0]
        for i in all_items:
            gain = submodular_fn(i, chosen, distances)
            if gain > best_gain:
                best_gain = gain
                best_item = i
        chosen.append(best_item)
        all_items.remove(best_item)
        current_value += best_gain
        #print(z, current_value)
        z += 1
    return chosen

def similarity_on_metas(meta1: SentenceMetadata, meta2: SentenceMetadata):
    # if the cores are exactly the same:
    is_same_cores = meta1.primary.acore.lemma_ == meta2.primary.acore.lemma_ and \
        meta1.primary.bcore.lemma_ == meta2.primary.bcore.lemma_
    # change if the tag is the same
    is_same_tag = meta1.primary.tag == meta2.primary.tag
    is_same_dep = unify_tags(meta1.primary.dep) == unify_tags(meta2.primary.dep)
    if is_same_cores and is_same_tag: return 1
    if is_same_tag and is_same_dep: return 0.75
    if is_same_tag: return 0.5
    if is_same_dep: return 0.1
    return 0.0001

def select_diverse_perturbations(
    orig_and_perturb_pairs: List[Tuple[str, str]],
    nsamples: int,
    compute_sent_cosine_distance: Callable,
    process: Callable) -> List[Tuple[str, str]]:
    """Having each perturbation be represented by its token changes, 
    control code, and dependency tree strcuture, we greedily select the 
    ones that are least similar to those already selected. This tries 
    to avoid redundancy in common perturbations such as black -> white.

    Args:
        orig_and_perturb_pairs (List[Tuple[str, str]]):
            A list of (orig, perturb) text pair. Not necessarily the 
            all the same orig text.
        nsamples (int): Number of samples to select
        compute_sent_cosine_distance (Callable): Distance function.
        process (Callable): The spaCy processor.

    Returns:
        List[Tuple[str, str]]: A subsample of (orig, perturb) pairs.
    """
    texts = set([s for s in set(np.concatenate([list(p) for p in orig_and_perturb_pairs]))])
    doc_mapper = {text: process(text) for text in texts}
    metas = []
    for orig_text, perturb_text in orig_and_perturb_pairs:
        orig_doc, perturb_doc = doc_mapper[orig_text], doc_mapper[perturb_text]
        eops = compute_edit_ops(orig_doc, perturb_doc)
        meta = SentenceMetadata(eops)
        meta.compute_metadata(sentence_similarity=compute_sent_cosine_distance)
        if meta.primary:
            metas.append(meta)
    
    distances = np.zeros((len(metas), len(metas)))
    idxes = list(range(len(metas)))
    for i, j in itertools.combinations(idxes, 2):
        distance = 1 - similarity_on_metas(metas[i], metas[j])
        distances[i, j] = distance
        distances[j, i] = distance
    picked = greedy(submodular_fn, distances=distances, k=nsamples)
    return [(metas[i].primary.acore.doc.text, metas[i].primary.bcore.doc.text) for i in picked]

def extract_predict_label(raw_pred):
    raw_pred = sorted(raw_pred, key=lambda r: -r["score"])
    if raw_pred:
        return raw_pred[0]["label"]
    return None
PredFormat = List[Dict[str, float]]
def select_surprise_explanations(
    orig_text: str,
    perturb_texts: List[str],
    orig_pred: PredFormat, 
    perturb_preds: List[PredFormat], 
    feature_importance_dict: Dict[str, float], 
    processor: Callable, 
    agg_feature_importance_thred: float=0.1) -> List[Dict]:
    """Select surprising perturbations based on a feature importance map. 
    we estimate the expected change in prediction with feature attributions, 
    and select counterfactuals that violate these expectations, i.e., examples 
    where the real change in prediction is large even though importance scores 
    are low, and examples where the change is small but importance scores are high.

    Args:
        orig_doc (Doc): The original text.
        perturb_docs (List[Doc]): The perturbed texts.
        orig_pred (PredFormat): Prediction of the original text, in the format of 
            [{label: str, score: float}]
        perturb_preds (List[PredFormat]): Prediction of the perturbed texts,
            Each entry in the format of [{label: str, score: float}]
        feature_importance_dict (Dict[str, float]): The precomputed feature map.
        agg_feature_importance_thred (float, Optional): A threshold for classifying
            surprises.
        process (Callable): The spaCy processor.

    Returns:
        List[Dict]: Return the selected surprises in the format of:
        {
            case, [str]: Suprise filp | Suprise unflip, 
            pred, [str]: the prediction for the perturbation,
            changed_features, List[str]: the changed features
            perturb_doc: the perturbed doc))
        }
    """
    # compute whether the perturbation is a surprising one.
    def is_surprise(
        delta_pred_prob,
        is_pred_changed,
        agg_feature_importance,
        agg_feature_importance_thred=0.1):
        # is an important feature, but changing it only has a small effect
        if agg_feature_importance >= agg_feature_importance_thred and abs(agg_feature_importance) > abs(delta_pred_prob):
            return not is_pred_changed
        if agg_feature_importance <= agg_feature_importance_thred and abs(agg_feature_importance) < abs(delta_pred_prob):
            return is_pred_changed
        return False

    orig_doc = processor(orig_text)
    perturb_docs = [processor(perturb) for perturb in perturb_texts]
    orig_pred_dict = {p["label"]: p["score"] for p in orig_pred}
    orig_pred = extract_predict_label(orig_pred)
    # a dict that groups perturbations by the words changed
    perturbed_word_dict = {}
    for perturbs_pred, perturb_doc in zip(perturb_preds, perturb_docs):
        perturbs_pred_dict = {p["label"]: p["score"] for p in perturbs_pred}
        perturb_pred = extract_predict_label(perturbs_pred)
        eops = compute_edit_ops(orig_doc, perturb_doc, only_return_edit=True)
        # actual prediction change
        is_pred_changed = orig_pred != perturb_pred
        delta_pred_prob = abs(perturbs_pred_dict[orig_pred] - orig_pred_dict[orig_pred])
        # get the changed parts
        all_changed_features = []
        for eop in eops:
            try: tokens = eop.fromz_core if eop.op != "insert" else [eop.fromz_full.root]
            except: tokens = []
            all_changed_features += [t.text for t in tokens]
        agg_feature_importance=sum([feature_importance_dict[t] for t in all_changed_features])

        surprise = is_surprise(
                delta_pred_prob, is_pred_changed, 
                agg_feature_importance, agg_feature_importance_thred)
        for t in all_changed_features:
            if t not in perturbed_word_dict:
                perturbed_word_dict[t] = [] # initialize the list
            perturbed_word_dict[t].append(Munch(
                perturb_doc = perturb_doc,
                pred=perturb_pred,
                delta_pred_prob=delta_pred_prob,
                changed_features=all_changed_features,
                agg_feature_importance=agg_feature_importance,
                is_pred_changed=is_pred_changed,
                is_surprise=surprise,
                # equally spread the impact on delta to each of the changed feature
                weight=1/len(all_changed_features)
            ))

    selected_features = []
    # find the surprising flip and suripring unflip separately.
    for is_pred_changed in [True, False]:
        sorted_features = []
        # find abnormal tokens: token has small SHAP weights, but the delta is large on average; Or the other way around
        for feature, group in perturbed_word_dict.items():
            grouped_perturbations = [g for g in group if g.is_pred_changed==is_pred_changed]
            if not grouped_perturbations: continue
            feature_imp = feature_importance_dict[feature]
            deltas = [feature_imp] + [k.delta_pred_prob for k in grouped_perturbations]
            weights = [1] + [k.weight for k in grouped_perturbations]
            sorted_features.append((feature, feature_imp - np.average(deltas, weights=weights)))
        # always prioritize the ones with the largest difference
        sorted_features = sorted(sorted_features, key=lambda s: -abs(s[1]))
        for feature, _ in sorted_features:
            grouped_perturbations = [g for g in perturbed_word_dict[feature] if 
                                    g.is_pred_changed==is_pred_changed and g.is_surprise]
            if not grouped_perturbations: continue
            grouped_perturbations = sorted(
                grouped_perturbations, 
                key=lambda p: (
                    -1 * p.weight,
                    # if surprising flipp, select the ones with the least prediction change
                    #(1 if is_pred_changed else -1) *
                    #p.delta_pred_prob * p.weight
                    -1 * abs(p.delta_pred_prob - p.agg_feature_importance)
                ))
                    # the delta between the prediction change, and the aggregated SHAP weights of all changed tokens
                    #abs(p.delta_pred_prob - p.agg_feature_importance) ))
            selected_features.append(Munch(
                case=f"Suprise {'flip' if is_pred_changed else 'unflip'}", 
                pred=grouped_perturbations[0].pred,
                changed_features=grouped_perturbations[0].changed_features,
                perturb_text=grouped_perturbations[0].perturb_doc.text))
            break
    return selected_features