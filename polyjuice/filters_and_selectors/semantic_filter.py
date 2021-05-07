import scipy
from sentence_transformers import SentenceTransformer

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
