import torch
import numpy as np
from spacy.tokens import Doc
from typing import List, Dict, Tuple
from .filters_and_selectors import \
    load_perplex_scorer, \
    compute_sent_cosine_distance, \
    compute_delta_perplexity, \
    compute_sent_perplexity, \
    select_diverse_perturbations, \
    select_surprise_explanations, PredFormat
from .generations import \
    get_prompts, \
    load_generator, \
    get_random_idxes, \
    create_blanked_sents, \
    generate_on_prompts, \
    RANDOM_CTRL_CODES, ALL_CTRL_CODES
from .compute_perturbs import compute_edit_ops, SentenceMetadata
from .compute_perturbs.compute_edit_ops import compute_edit_ops

from .helpers import create_processor


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class Polyjuice(object):
    def __init__(self, model_path: str="uw-hai/polyjuice", is_cuda: bool=True) -> None:
        """The wrapper.

        Args:
            model_path (str, optional): The path to the generator.
                Defaults to "uw-hai/polyjuice". 
                No need to change this, unless you retrained a model
            is_cuda (bool, optional): whether to use cuda. Defaults to True.
        """
        # generator
        self.generator = None
        # validators
        self.polyjuice_generator_path = model_path
        self.perplex_scorer = None
        self.distance_scorer = None
        self.generator = None
        self.spacy_processor = None

        self.is_cuda = is_cuda and torch.cuda.is_available()

    def validate_and_load_model(self, model_name: str) -> bool:
        """Validate whether the generator or scorer are loaded.
        If not, load the model.

        Args:
            model_name (str): the identifier of the loaded part.
                Should be [generator, perplex_scorer].

        Returns:
            bool: If the model is successfully load.
        """
        if getattr(self, model_name, None):
            return True
        else:
            loader = getattr(self, f"_load_{model_name}", None)
            return loader and loader()

    def _load_perplex_scorer(self):
        logger.info("Setup perplexity scorer.")
        self.perplex_scorer = load_perplex_scorer(is_cuda=self.is_cuda)
        return True
    def _load_generator(self):
        logger.info("Setup Polyjuice.")
        self.generator = load_generator(
            self.polyjuice_generator_path, is_cuda=self.is_cuda)
        return True
    def _load_spacy_processor(self, is_space_tokenizer: bool=False):
        logger.info("Setup SpaCy processor.")
        self.spacy_processor = create_processor(is_space_tokenizer)
        return True

    ##############################################
    # validation
    ##############################################
    def _process(self, sentence: str):
        if not self.validate_and_load_model("spacy_processor"): return None
        return self.spacy_processor(str(sentence))

    def _compute_sent_cosine_distance(self, s1: str, s2: str):
        if not self.validate_and_load_model("distance_scorer"): return 1
        return compute_sent_cosine_distance(s1, s2, self.distance_scorer, is_cuda=self.is_cuda)

    def _compute_delta_perplexity(self, eops):
        if not self.validate_and_load_model("perplex_scorer"): return None
        return compute_delta_perplexity(eops, self.perplex_scorer, is_cuda=self.is_cuda)

    def _compute_sent_perplexity(self, sentence: str):
        if not self.validate_and_load_model("perplex_scorer"): return None
        return compute_sent_perplexity([sentence], self.perplex_scorer, is_cuda=self.is_cuda)[0]
    
    ##############################################
    # apis
    ##############################################

    def detect_ctrl_code(self, 
        orig: Tuple[str, Doc], 
        perturb: Tuple[str, Doc],
        eops: List=None) -> str:
        """Detect the perturbation type.

        Args:
            orig (Tuple[str, Doc]): The perturb-from sentence,
                either in str or SpaCy doc.
            perturb (Tuple[str, Doc]): The perturb-to sentence,
                either in str or SpaCy doc.
            eops (List, Optional): The editing operations, output of
                `compute_edit_ops` in 
                `polyjuice.compute_perturbs.compute_edit_ops`.
                If None, will be computed.
        Returns:
            str: The extracted code. If cannot be identified, return None.
        """
        orig = self._process(orig) if type(orig) == str else orig
        perturb = self._process(perturb) if type(perturb) == str else perturb
        if orig.text == perturb.text:
            return "equal"
        if eops is None:
            eops = compute_edit_ops(orig, perturb)
        meta = SentenceMetadata(eops)
        meta.compute_metadata(
            sentence_similarity=self._compute_sent_cosine_distance)
        if meta.primary and meta.primary.tag in ALL_CTRL_CODES:
            return meta.primary.tag
        else:
            return None

    def get_random_blanked_sentences(self, 
        sentence: Tuple[str, Doc], 
        pre_selected_idxes: List[int]=None,
        deps: List[str]=None,
        is_token_only: bool=False,
        max_blank_sent_count: int=3,
        max_blank_block: int=1) -> List[str]:
        """Generate some random blanks for a given sentence

        Args:
            sentence (Tuple[str, Doc]): The sentence to be blanked,
                either in str or SpaCy doc.
            pre_selected_idxes (List[int], optional): 
                If set, only allow blanking a preset range of token indexes. 
                Defaults to None.
            deps (List[str], optional): 
                If set, only select from a subset of dep tags. Defaults to None.
            is_token_only (bool, optional):
                blank sub-spans or just single tokens. Defaults to False.
            max_blank_sent_count (int, optional): 
                maximum number of different blanked sentences. Defaults to 3.
            max_blank_block (int, optional): 
                maximum number of blanks per returned sentence. Defaults to 1.

        Returns:
            List[str]: blanked sentences
        """
        if type(sentence) == str:
            sentence = self._process(sentence)
        indexes = get_random_idxes(
            sentence, 
            pre_selected_idxes=pre_selected_idxes,
            deps=deps,
            is_token_only=is_token_only,
            max_count=max_blank_sent_count,
            max_blank_block=max_blank_block
        )
        blanked_sents = create_blanked_sents(sentence, indexes)
        return blanked_sents

    def perturb(self, 
        orig_sent: Tuple[str, Doc], 
        blanked_sent: Tuple[str, List[str]]=None,
        is_complete_blank: bool=False, 
        ctrl_code: Tuple[str, List[str]]=None, 
        perplex_thred: int=10,
        num_perturbations: int=3,
        verbose: bool=False, 
        #is_include_metadata: bool=True,
        **kwargs) -> List[str]:
        """The primary perturbation function. Running example:
        Original sentence: 
            "It is great for kids."

        Args:
            orig_sent (Tuple[str, Doc]): 
                Original sentence, either in the form of str or SpaCy Doc.
            blanked_sents (Tuple[str, List[str]], optional): 
                sentences that contain blanks, e.g., "It is [BLANK] for kids."
                Defaults to None. If is None, the blank will be automatically placed.
                If is "" or incomplete form like "It is", set `is_complete_blank` to
                True below to allow the model to generate where to blank.
            is_complete_blank (bool, optional): 
                Whether the blanked sentence is already complete or not. 
                Defaults to False.
            ctrl_codes (Tuple[str, List[str]], optional): 
                The ctrl code (can be a list). Defaults to None. 
                If is None, will automatically become [resemantic, lexical,
                negation, insert, delete]. 
            perplex_thred (int, optional): 
                Perplexity filter for fluent perturbations. 
                we score both x and x' with GPT-2, and filter x' when the 
                log-probability (on the full sentence or the perturbed chunks) 
                decreases more than {perplex_thred} points relative to x
                Defaults to 5. If None, will skip filter.
            num_perturbations: 
                Num of max perturbations to collect. Defaults to 3.
            is_include_metadata: 
                Whether to return text, or also include other metadata and perplex score.
                Default to True.
            **kwargs: 
                The function can also take arguments for huggingface generators, 
                like top_p, num_beams, etc.
        Returns:
            List[str]: The perturbations.
        """
        logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
        if not self.validate_and_load_model("generator"): return []
        orig_doc = self._process(orig_sent) if type(orig_sent) == str else orig_sent
        if blanked_sent:
            blanked_sents = [blanked_sent] if type(blanked_sent) == str else blanked_sent
        else:
            blanked_sents = self.get_random_blanked_sentences(orig_doc.text)
        if ctrl_code:
            ctrl_codes = [ctrl_code] if type(ctrl_code) == str else ctrl_code
            if not set(ctrl_codes).issubset(ALL_CTRL_CODES):
                logger.error(f"{set(ctrl_codes)-ALL_CTRL_CODES} is not a valid ctrl code. Please choose from {ALL_CTRL_CODES}.")
            is_filter_code = True
        else:
            ctrl_codes = RANDOM_CTRL_CODES
            is_filter_code = False
        prompts = get_prompts(
            doc=orig_doc,
            ctrl_codes=ctrl_codes, 
            blanked_sents=blanked_sents, 
            is_complete_blank=is_complete_blank)
        if verbose:
            logger.info("Generating on these prompts:")
            for p in prompts: logger.info(f" | {p}")
        generated = generate_on_prompts(
            generator=self.generator, prompts=prompts, **kwargs)
        merged = list(np.concatenate(generated))

        validated_set = []
        for _, generated in merged:
            # skip 
            if generated in validated_set or generated.lower() == orig_doc.text.lower(): 
                continue
            is_vaild = True
            generated_doc = self._process(generated)
            eop = compute_edit_ops(orig_doc, generated_doc)
            if perplex_thred is not None:
                pp = self._compute_delta_perplexity(eop)
                is_vaild = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
            if is_vaild and is_filter_code:
                ctrl = self.detect_ctrl_code(orig_doc, generated_doc, eop)
                is_vaild = is_vaild and ctrl is not None and ctrl in ctrl_codes
            if is_vaild:
                validated_set.append(generated)
                #validated_changes.append(Munch(
                #    text=generated, doc=generated_doc, meta=meta, perplex=pp))
        if num_perturbations is None:
            num_perturbations = 1000
        sampled = np.random.choice(validated_set,
            min(num_perturbations, len(validated_set)), replace=False)
        return [str(s) for s in sampled]


    def select_diverse_perturbations(self, 
        orig_and_perturb_pairs: List[Tuple[str, str]],
        nsamples: int)-> List[Tuple[str, str]]:
        """Having each perturbation be represented by its token changes, 
        control code, and dependency tree strcuture, we greedily select the 
        ones that are least similar to those already selected. This tries 
        to avoid redundancy in common perturbations such as black -> white.

        Args: 
            orig_and_perturb_pairs (List[Tuple[str, str]]):
                A list of (orig, perturb) text pair. Not necessarily the 
                all the same orig text.
            nsamples (int): Number of samples to select

        Returns:
            List[Tuple[str, str]]: A subsample of (orig, perturb) pairs.
        """

        return select_diverse_perturbations(
            orig_and_perturb_pairs=orig_and_perturb_pairs,
            nsamples=nsamples,
            compute_sent_cosine_distance=self._compute_sent_cosine_distance,
            process=self._process)
        
    def select_surprise_explanations(self,
        orig_text: str,
        perturb_texts: List[str],
        orig_pred: PredFormat, 
        perturb_preds: List[PredFormat], 
        feature_importance_dict: Dict[str, float],
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
        Returns:
            List[Dict]: Return the selected surprises in the format of:
            {
                case, [str]: Suprise filp | Suprise unflip, 
                pred, [str]: the prediction for the perturbation,
                changed_features, List[str]: the changed features
                perturb_doc: the perturbed doc))
            }
        """
        return select_surprise_explanations(
            orig_text=orig_text,
            perturb_texts=perturb_texts,
            orig_pred=orig_pred,
            perturb_preds=perturb_preds,
            feature_importance_dict=feature_importance_dict,
            processor=self._process,
            agg_feature_importance_thred=agg_feature_importance_thred
        )