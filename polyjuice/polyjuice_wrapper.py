import torch
from spacy.tokens import Doc
from typing import Tuple, List
import numpy as np
from .filters_and_selectors import \
    load_perplex_scorer, \
    compute_sent_cosine_distance, \
    compute_delta_perplexity, \
    compute_sent_perplexity
from .generations import \
    get_prompts, \
    load_generator, \
    get_random_idxes, \
    create_blanked_sents, \
    generate_on_prompts, \
    RANDOM_TAGS, ALL_TAGS
from .compute_perturbs import compute_edit_ops, SentenceMetadata
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
        self.perplex_scorer = load_perplex_scorer(self.is_cuda)
        return True
    def _load_generator(self):
        logger.info("Setup Polyjuice.")
        self.generator = load_generator(
            self.polyjuice_generator_path, self.is_cuda)
        return True
    def _load_spacy_processor(self, is_space_tokenizer: bool=True):
        logger.info("Setup SpaCy processor.")
        self.spacy_processor = create_processor(is_space_tokenizer)
        return True

    ##############################################
    # validation
    ##############################################
    def _process(self, sentence: str):
        if not self.validate_and_load_model("spacy_processor"): return None
        return self.spacy_processor(sentence)

    def _compute_sent_cosine_distance(self, s1: str, s2: str):
        if not self.validate_and_load_model("distance_scorer"): return 1
        return compute_sent_cosine_distance(s1, s2, self.distance_scorer)

    def _compute_delta_perplexity(self, eops):
        if not self.validate_and_load_model("perplex_scorer"): return None
        return compute_delta_perplexity(eops, self.perplex_scorer)

    def _compute_sent_perplexity(self, sentence: str):
        if not self.validate_and_load_model("perplex_scorer"): return None
        return compute_sent_perplexity([sentence], self.perplex_scorer)[0]
    
    def get_random_blanked_sentences(self, 
        sentence: Tuple[str, Doc], 
        pre_selected_idxes: List[int]=None,
        deps: List[str]=None,
        is_token_only: bool=False,
        max_blank_sent_count: int=3,
        max_blank_block: int=1) -> List[str]:
        """Generate some random blanks for a given sentence

        Args:
            sentence (Tuple[str, Doc]): [description]
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
            if not set(ctrl_codes).issubset(ALL_TAGS):
                logger.error(f"{set(ctrl_codes)-ALL_TAGS} is not a valid ctrl code. Please choose from {ALL_TAGS}.")
            is_filter_code = True
        else:
            ctrl_codes = RANDOM_TAGS
            is_filter_code = False
        prompts = get_prompts(
            doc=orig_doc,
            ctrl_codes=ctrl_codes, 
            blanked_sents=blanked_sents, 
            is_complete_blank=is_complete_blank)
        generated = generate_on_prompts(
            generator=self.generator, prompts=prompts, **kwargs)
        merged = list(np.concatenate(generated))

        validated_changes = []
        for input_ctrl, generated in merged:
            is_vaild = True
            generated_doc = self._process(generated)
            eop = compute_edit_ops(orig_doc, generated_doc)
            if perplex_thred is not None:
                pp = self._compute_delta_perplexity(eop)
                is_vaild = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
            if is_vaild and is_filter_code:
                meta = SentenceMetadata(eop)
                meta.compute_metadata(
                    sentence_similarity=self._compute_sent_cosine_distance)
                is_vaild = is_vaild and meta.primary and meta.primary.tag == input_ctrl
            if is_vaild and generated != orig_doc and not generated in validated_changes:
                validated_changes.append(generated)
        return list(np.random.choice(
            validated_changes, 
            min(num_perturbations, len(validated_changes)), replace=False))