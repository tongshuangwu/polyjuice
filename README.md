# Polyjuice

This repository contains code for testing NLP Models as described in the following paper:  
>[Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models](https://homes.cs.washington.edu/~wtshuang/static/papers/2021-acl-polyjuice.pdf)  
> Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, Daniel S. Weld
> Association for Computational Linguistics (ACL), 2021

Bibtex for citations:
```bibtex
@inproceedings{polyjuice:acl21,
    title = "{P}olyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models",
    author = "Tongshuang Wu and Marco Tulio Ribeiro and Jeffrey Heer and Daniel S. Weld",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```

## Installation

From Pypi:
```bash
pip install polyjuice_nlp
```

From source:
```bash
git clone git@github.com:tongshuangwu/polyjuice.git
cd polyjuice
pip install -e .
```

Polyjuice depends on [SpaCy](https://spacy.io/) and [Huggingface Transformers](https://huggingface.co/). To use most functions, please also install the following:

```bash
# install pytorch, as here: https://pytorch.org/get-started/locally/#start-locally
pip install torch
# The SpaCy language package
python -m spacy download en_core_web_sm
```

## Perturbation

```py
from polyjuice import Polyjuice
# initiate a wrapper.
# model path is defaulted to our portable model:
# https://huggingface.co/uw-hai/polyjuice
# No need to change this unless you are using customized model
pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)

# the base sentence
text = "It is great for kids."

# perturb the sentence with one line:
# When running it for the first time, the wrapper will automatically
# load related models, e.g. the generator and the perplexity filter.
perturbations = pj.perturb(text)
```

### More advanced APIs

Please see the documents in X for more explanations.

To perturb with more controls,
```py
perturbations = pj.perturb(
    orig_sent=text,
    # can specify where to put the blank. Otherwise, it's automatically selected.
    # Can be a list or a single sentence.
    blanked_sent="It is [BLANK] for kids.",
    # can also specify the ctrl code (a list or a single code.)
    # The code should be from 'resemantic', 'restructure', 'negation', 'insert', 'lexical', 'shuffle', 'quantifier', 'delete'.
    ctrl_code="negation",
    # Customzie perplexity score. 
    perplex_thred=5,
    # number of perturbations to return
    num_perturbations=1,
    # the function also takes in additional arguments for huggingface generators.
    num_beams=3
)

# return: ['It is bad for kids too.',
# "It 's great for kids.",
# 'It is great even for kids.']
```

To get randomly placed blanks,
```py
perturbations = pj.perturb(
    orig_sent=text,
    # can specify where to put the blank. Otherwise, it's automatically selected.
    # Can be a list or a single sentence.
    blanked_sent=["It is [BLANK] for kids.", "It is great for [BLANK]."],
    # can also specify the ctrl code (a list or a single code.)
    # The code should be from 'resemantic', 'restructure', 'negation', 'insert', 'lexical', 'shuffle', 'quantifier', 'delete'.
    ctrl_code="negation",
    # Customzie perplexity score. 
    perplex_thred=20,
    # number of perturbations to return
    num_perturbations=3,
    # the function also takes in additional arguments for huggingface generators.
    num_beams=3
)
# return: [
# 'It is not great for kids.', 
# 'It is great for kids but not for anyone.',
# 'It is great for kids but not for any adults.']
```


## Selection

For selecting diverse and surprising perturbations (for augmentation and explanation experiments in our paper), please see the notebook demo.
