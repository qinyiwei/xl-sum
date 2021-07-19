# from transformers import Trainer
import torch
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, PretrainedBartModel
from transformers import T5PreTrainedModel
from torch import  nn
import transformers
if transformers.__version__=="3.2.0":
    from transformers.modeling_bart import shift_tokens_right
else:
    from transformers.models.bart.modeling_bart import shift_tokens_right

import numpy as np
import random

# fix the random seed
def seed_everything(seed=11747):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class PrefixTuningT5(T5PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.num_decoder_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'use_infix'):
            assert (config.use_infix == False), "do not support use infix"	

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        deep_param = self.use_deep

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            assert self.lowdata == False, "do not support low data:{}".format(self.lowdata)

        if hasattr(config, 'lowdata_token'):
            assert self.lowdata_token is None, "do not support low_data_token:{}".format(self.lowdata_token)


        if not self.optim_prefix:
            assert False, "only surport optim_prefix!"
        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # DIFFERENT PARAMETRIZATION:
            if not deep_param:
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                #TODO: delete this sentence after debug
                #self.load_state_dict(torch.load("/home/yiweiq/initial_weights.ckp"))

            else:
                print('UNDER PARAMETRIZATION DEEP 1')

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))



    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.model.encoder.forward_with_encoder_past(temp_input).past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)

        self.use_self_prefix = False#True
        self.use_cross_prefix = False#True
        self.use_encoder_prefix = False#True

        if self.use_self_prefix:
            temp_control = self.wte(input_tokens)              #[torch.Size([16, 200, 768])] bsz, num input_tokens, embd_size
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb=768*2*6 [torch.Size([16, 200, 9216])]
            bsz, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                self.match_n_embd) #torch.Size([16, 200, 12, 12, 64]), bsz,seqlen, 6*2, 12, 64
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)   #6*(torch.Size([2, 16, 12, 200, 64])), 6*(2,bsz,12,seqlen,64)


        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        #for i, key_val in enumerate(past_key_values):
        for i in range(self.match_n_layer):
            if transformers.__version__=="3.2.0":
                temp_dict = {}
                if self.use_self_prefix:
                    key_val = past_key_values[i]
                    temp_dict['self'] = {"prev_key": key_val[0].contiguous(),
                                        "prev_value": key_val[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                        }
                if self.use_cross_prefix:
                    key_val2 = past_key_values2[i]
                    temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                    "prev_value": key_val2[1].contiguous(),
                                                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                    }
                if self.use_encoder_prefix:
                    key_val_enc = past_key_values_enc[i]
                    temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                            "prev_value": key_val_enc[1].contiguous(),
                                            "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                            }
                result.append(temp_dict)

            else:
                temp_tuple = ()
                if self.use_self_prefix:
                    key_val = past_key_values[i]
                    temp_tuple += (key_val[0].contiguous(),key_val[1].contiguous(),)
                else:
                    temp_tuple += (None, None,)
                if self.use_cross_prefix:
                    key_val2 = past_key_values2[i]
                    temp_tuple += (key_val2[0].contiguous(),key_val2[1].contiguous(),)
                else:
                    temp_tuple += (None, None,)
                if self.use_encoder_prefix:
                    key_val_enc = past_key_values_enc[i]
                    temp_tuple += (key_val_enc[0].contiguous(),key_val_enc[1].contiguous(),torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool(),)
                else:
                    temp_tuple += (None, None,)
                result.append(temp_tuple)
            
        return None
        #return result

    def forward(self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, **kwargs)

        return output


class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'use_infix'):
            assert (config.use_infix == False), "do not support use infix"	

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        deep_param = self.use_deep

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            assert self.lowdata == False, "do not support low data:{}".format(self.lowdata)

        if hasattr(config, 'lowdata_token'):
            assert self.lowdata_token is None, "do not support low_data_token:{}".format(self.lowdata_token)

            

        if not self.optim_prefix:
            assert False, "only surport optim_prefix!"
        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # DIFFERENT PARAMETRIZATION:
            if not deep_param:
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                #TODO: delete this sentence after debug
                self.load_state_dict(torch.load("/home/yiweiq/initial_weights.ckp"))

            else:
                print('UNDER PARAMETRIZATION DEEP 1')

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))



    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.model.encoder.forward_with_encoder_past(temp_input).past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)

        self.use_self_prefix = True
        self.use_cross_prefix = True
        self.use_encoder_prefix = True

        if self.use_self_prefix:
            temp_control = self.wte(input_tokens)              #[torch.Size([16, 200, 768])] bsz, num input_tokens, embd_size
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb=768*2*6 [torch.Size([16, 200, 9216])]
            bsz, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                self.match_n_embd) #torch.Size([16, 200, 12, 12, 64]), bsz,seqlen, 6*2, 12, 64
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)   #6*(torch.Size([2, 16, 12, 200, 64])), 6*(2,bsz,12,seqlen,64)


        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        #for i, key_val in enumerate(past_key_values):
        for i in range(self.match_n_layer):
            if transformers.__version__=="3.2.0":
                temp_dict = {}
                if self.use_self_prefix:
                    key_val = past_key_values[i]
                    temp_dict['self'] = {"prev_key": key_val[0].contiguous(),
                                        "prev_value": key_val[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                        }
                if self.use_cross_prefix:
                    key_val2 = past_key_values2[i]
                    temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                    "prev_value": key_val2[1].contiguous(),
                                                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                    }
                if self.use_encoder_prefix:
                    key_val_enc = past_key_values_enc[i]
                    temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                            "prev_value": key_val_enc[1].contiguous(),
                                            "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                            }
                result.append(temp_dict)

            else:
                temp_tuple = ()
                if self.use_self_prefix:
                    key_val = past_key_values[i]
                    temp_tuple += (key_val[0].contiguous(),key_val[1].contiguous(),)
                else:
                    temp_tuple += (None, None,)
                if self.use_cross_prefix:
                    key_val2 = past_key_values2[i]
                    temp_tuple += (key_val2[0].contiguous(),key_val2[1].contiguous(),)
                else:
                    temp_tuple += (None, None,)
                if self.use_encoder_prefix:
                    key_val_enc = past_key_values_enc[i]
                    temp_tuple += (key_val_enc[0].contiguous(),key_val_enc[1].contiguous(),torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool(),)
                else:
                    temp_tuple += (None, None,)
                result.append(temp_tuple)
            
        return result

    def forward(self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, **kwargs)

        return output