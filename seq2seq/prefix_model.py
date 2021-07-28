
import argparse
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

#from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import MBartTokenizer, T5ForConditionalGeneration
import transformers
if transformers.__version__=="3.2.0":
    from transformers.modeling_bart import shift_tokens_right
else:
    from transformers.models.bart.modeling_bart import shift_tokens_right

from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_params,
    #get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    #save_git_info,
    use_task_specific_params,
)


# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
print(sys.path)
from prefix_base import PrefixTransformer  # noqa


logger = logging.getLogger(__name__)


class PrefixSummarizationModule(PrefixTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, config=None, hparams=None, tokenizer=None, seq2seq_model=None, **kwargs):
        super().__init__(config=config, hparams=hparams, tokenizer=tokenizer, seq2seq_model=seq2seq_model, **kwargs)

        #save_git_info(self.hparams.output_dir)
        #self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        #self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        #pickle_save(self.hparams, self.hparams_save_path)
        #self.step_count = 0
        #self.metrics = defaultdict(list)
        #self.model_type = self.config.model_type
        #self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        #self.hparams.git_sha = get_git_info()["repo_sha"]

        #self.decoder_start_token_id = None  # default to config
        #if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
        #    self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
        #    self.model.config.decoder_start_token_id = self.decoder_start_token_id
        #self.dataset_class = (
        #    Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        #)
        self.eval_beams = self.hparams.eval_beams
        #assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"

        #self.eval_max_length = self.hparams.val_max_target_length

        #self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        #self.training_acc_across_batches_at_curr_epoch = []


    def forward(self, input_ids, **kwargs):
        #TODO:only keep second condition
        if isinstance(self.model, T5ForConditionalGeneration):
            return self.model(input_ids, **kwargs)
        else:
            return self.model(input_ids, gpt2_model=self.seq2seq_model, use_cache=False, use_prefix=True, **kwargs)

    def generate(
        self,
        input_ids,
        **model_kwargs
    ) -> torch.LongTensor:

        bsz = input_ids.size(0)
        prefix_prompt = self.model.get_prompt(bsz=bsz, sample_size=self.eval_beams)
        return self.seq2seq_model.generate(
            input_ids,
            past_key_values=prefix_prompt,
            use_prefix=True,
            **model_kwargs,
        )






  


