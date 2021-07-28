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
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5,):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.num_decoder_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.use_encoder_prefix = config.use_encoder_prefix
        self.use_cross_prefix = config.use_cross_prefix
        self.use_self_prefix = config.use_self_prefix

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen


        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

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

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        assert hasattr(config, 'mid_dim'), "need to initialize mid_dim"    
        self.mid_dim = config.mid_dim

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

  
        self.mode_para = 0
        print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
        print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))


        if self.lowdata and self.lowdata_token is not None:
                low_data_init = config.low_data_init
                if low_data_init == 1:
                    print('IN THE LOW DATA SETTING, EXPLORE INITIALIZATION FOR DIRECT OPTIM...')
                    # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p22
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
                    self.control_trans = self.lowdata_init_train1(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
                    print(self.control_trans.shape)
                elif low_data_init == 2:
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first')
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5

                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    # sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)

                elif low_data_init == 3:
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    #self.preseqlen = len(self.lowdata_token[0])
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                          'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))

                    self.input_tokens = torch.arange(self.preseqlen).long()
                    if self.use_self_prefix:
                        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                        self.control_trans = nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                    self.get_prompt = self.get_prompt_p5

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
        else:
            # DIFFERENT PARAMETRIZATION:
            low_data_init = 0
            print('UNDER PARAMETRIZATION 1')
            self.input_tokens = torch.arange(self.preseqlen).long()
            if self.use_self_prefix:
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

            self.get_prompt = self.get_prompt_p5

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

        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


        if low_data_init == 2:
            self.lowdata_init_train2(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
        elif low_data_init == 3:
            print('use pt for this tensor', torch.LongTensor(self.lowdata_token))
            self.lowdata_init_train3(gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token))


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)

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
                    temp_tuple += (key_val_enc[0].contiguous(),key_val_enc[1].contiguous(),)
                else:
                    temp_tuple += (None, None,)
                result.append(temp_tuple)
    
        #return None
        return result


    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.encoder(temp_input,use_cache=True).past_key_values

    def lowdata_init_train3(self, gpt2, sample_input, epochs=200): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            src_ids = sample_input.to(gpt2.device)
            tgt_ids = sample_input.to(gpt2.device)
            decoder_input_ids = gpt2._shift_right(tgt_ids)
            print(decoder_input_ids.shape, decoder_input_ids)


            for i in range(decoder_input_ids.size(-1)):
                output = gpt2(src_ids, decoder_input_ids=decoder_input_ids[:, :i+1], use_cache=True,
                               use_prefix=False, return_dict=True)
                output = output.past_key_values

            if self.use_self_prefix:
                self_full_key = torch.cat([ll[0] for ll in output])
                self_full_val = torch.cat([ll[1] for ll in output])
                self_full = torch.cat([self_full_val, self_full_key])
                print('gold self', self_full.shape)

            if self.use_cross_prefix:
                encdec_full_key = torch.cat([ll[2] for ll in output])
                encdec_full_val = torch.cat([ll[3] for ll in output])
                encdec_full = torch.cat([encdec_full_val, encdec_full_key])
                print('gold_encdec', encdec_full.shape)

            if self.use_encoder_prefix:
                encoder_full_past = self.get_encoder_output(gpt2, src_ids)
                encoder_full_key = torch.cat([ll[0] for ll in encoder_full_past])
                encoder_full_val = torch.cat([ll[1] for ll in encoder_full_past])
                encoder_full = torch.cat([encoder_full_val, encoder_full_key])
                print('gold_encoder', encdec_full.shape)


        list_param = self.parameters()
        # print(list_param)
        optimizer_temp = torch.optim.Adam(list_param, lr=0.00003)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            if self.use_self_prefix:
                self_our_key = torch.cat([ll[0] for ll in our_prompt])
                self_our_val = torch.cat([ll[1] for ll in our_prompt])
                self_our = torch.cat([self_our_val, self_our_key])
                # print('our_self', self_our.shape)

            if self.use_cross_prefix:
                encdec_our_key = torch.cat([ll[2] for ll in our_prompt])
                encdec_our_val = torch.cat([ll[3] for ll in our_prompt])
                encdec_our = torch.cat([encdec_our_val, encdec_our_key])
                # print(encdec_full.shape, encdec_our.shape)
                # print('our_encdec', encdec_our.shape)

            if self.use_encoder_prefix:
                encoder_our_key = torch.cat([ll[4] for ll in our_prompt])
                encoder_our_val = torch.cat([ll[5] for ll in our_prompt])
                encoder_our = torch.cat([encoder_our_val, encoder_our_key])
                # print('our_encoder', encoder_our.shape)

            # our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = 0
            if self.use_self_prefix:
                loss += loss_metrics(self_our.to(gpt2.device), self_full)
            if self.use_cross_prefix:
                loss += loss_metrics(encdec_our.to(gpt2.device), encdec_full)
            if self.use_encoder_prefix:
                loss += loss_metrics(encoder_our.to(gpt2.device), encoder_full )
            print(loss)
            loss.backward()
            optimizer_temp.step()
            #self.control_trans.zero_grad()
            self.zero_grad()
        return

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

        bsz = input_ids.shape[0]


        past_key_values_prompt = self.get_prompt(bsz=bsz)
        #past_key_values_prompt = None
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
























