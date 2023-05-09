from transformers.models.bart.modeling_bart import * 
from wenet.utils.cmvn import load_cmvn
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.encoder import TransformerEncoder as SpTransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.ctc import CTC
import yaml
import torch.nn as nn
import torch

MIN = -1e32

def create_speech_encoder(config_file):
    cmvn_file = 'data/sp_ner/global_cmvn_mel80'
    mean, istd = load_cmvn(cmvn_file, True)
    global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    input_dim = 80
    encoder_type = 'transformer'
    # config_file = 'conf/bart_speech_encoder.yaml'
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    encoder = SpTransformerEncoder(input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
    return encoder

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
    
class End2EndSpeechNERBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        I should pass the speech mel feats(sp_encoder_inputs) to self.sp_encoder, and get the speech repr and speech mask as input embeds and attention mask.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        # here different from raw bart
        hidden_states = inputs_embeds
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class End2EndSpeechNERBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class End2EndSpeechNERBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = End2EndSpeechNERBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()
        
class BartSpeechNER(nn.Module):
    def __init__(self,
                bart_tokenizer_path,
                bart_asrcorr_path,
                sp_config_path,
                add_ctc=False,
                ctc_weight=0.1,
                ctc_blank_id=1,
                device='0'):
        super().__init__()
        self.add_ctc = add_ctc
        self.ctc_weight = ctc_weight
        self.ctc_blank_id = ctc_blank_id
        self.sp_encoder = create_speech_encoder(sp_config_path)
        
        self.bart_tokenizer_path = bart_tokenizer_path
        self.bart_asrcorr_path = bart_asrcorr_path
        from transformers import AutoConfig
        self.config = AutoConfig.from_pretrained(self.bart_tokenizer_path)

        # self.sp_proj = nn.Linear(self.sp_encoder.output_size(), self.config.d_model)
        # if self.bart_model_path == 'None':
        #     self.model = BartModel(self.config)
        # else:
        #     self.model = BartModel.from_pretrained(self.bart_model_path)

        # if self.bart_model_path == 'None':
        #     self.model = End2EndSpeechNERBartModel(self.config)
        # else:
        #     self.model = End2EndSpeechNERBartModel.from_pretrained(self.bart_model_path)

        if self.bart_asrcorr_path == 'None':
            # init a new bart model
            self.model = BartASRCorrection('None', self.bart_tokenizer_path)
        else:
            self.model = BartASRCorrection('None', self.bart_tokenizer_path)
            print(f'loading bartasrcorr from {self.bart_asrcorr_path}')
            self.model.load_state_dict(torch.load(self.bart_asrcorr_path))
            use_cuda = device != '-1' and torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            self.model = self.model.to(device)
        self.encoder, self.decoder = self.model.encoder, self.model.decoder

        if self.add_ctc:
            self.ctc = CTC(self.config.vocab_size, self.config.d_model, blank_id=self.ctc_blank_id)
        
        self.classifier = self.model.classifier

        # self.classifier = nn.Linear(self.config.d_model, self.config.vocab_size)
        # self.classifier.weight = self.model.shared.weight
        # self.criterion = LabelSmoothingLoss(size=self.config.vocab_size, padding_idx=-1, smoothing=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sp_feat, sp_feat_lens):
        """
        Returns:
            ~torch.Tensor:
                Representations for the src sentences of the shape ``[batch_size, seq_len, d_model]``.
            ~torch.Tensor:
                [batch_size, seq_len]
        """
        encoder_out, encoder_mask = self.sp_encoder(sp_feat, sp_feat_lens)
        # encoder_out = self.sp_proj(encoder_out)
        attention_mask = encoder_mask.squeeze(1)
        x = self.encoder(inputs_embeds=encoder_out, attention_mask=attention_mask)[0]
        return x, attention_mask

    def loss(self, x, tgt, src_mask, tgt_mask):
        shifted = torch.full_like(tgt, self.config.decoder_start_token_id)
        shifted[:, 1:] = tgt[:, :-1]
        y = self.decoder(input_ids=shifted,
                        attention_mask=tgt_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=src_mask)[0]
        # tgt_mask[:, 0] = 0
        s_y = self.classifier(y)
        # att_loss = self.criterion.new_forward(s_y, tgt, pad_word_id=self.config.pad_token_id, mask=tgt_mask.bool())
        att_loss = self.criterion(s_y[tgt_mask], tgt[tgt_mask])
        ctc_loss = None
        if self.add_ctc:
            # bos is masked, so just need to -1
            real_tgt_len = tgt_mask.sum(1)-1
            ctc_loss = self.ctc(x, src_mask.sum(1), tgt[:, 1:], real_tgt_len)
            loss = self.ctc_weight * ctc_loss + (1-self.ctc_weight) * att_loss
            return loss, att_loss, ctc_loss
        else:
            return att_loss, att_loss, ctc_loss

    def getparamgroup_in_first_step(self, if_second_stage=False):
        if not if_second_stage:
            name_lst = [name for name, p in self.named_parameters() if name.startswith("sp_encoder")]
            param_lst = [p for name, p in self.named_parameters() if name.startswith("sp_encoder")]

            if self.add_ctc:
                name_lst += [name for name, p in self.named_parameters() if name.startswith("ctc")]
                param_lst += [p for name, p in self.named_parameters() if name.startswith("ctc")]

            name_lst += [name for name, p in self.named_parameters() if "embed_positions" in name]
            param_lst += [p for name, p in self.named_parameters() if "embed_positions" in name]

            name_lst += [name for name, p in self.named_parameters() if name.startswith("model.model.encoder.layers.0.self_attn.k_proj") or name.startswith("model.model.encoder.layers.0.self_attn.v_proj") or name.startswith("model.model.encoder.layers.0.self_attn.q_proj")]
            param_lst += [p for name, p in self.named_parameters() if name.startswith("model.model.encoder.layers.0.self_attn.k_proj") or name.startswith("model.model.encoder.layers.0.self_attn.v_proj") or name.startswith("model.model.encoder.layers.0.self_attn.q_proj")]

            # name_lst += [name for name, p in self.named_parameters() if name.startswith("sp_proj")]
            # param_lst += [p for name, p in self.named_parameters() if name.startswith("sp_proj")]
            # name_lst += [name for name, p in self.named_parameters() if name.startswith("classifier")]
            # param_lst += [p for name, p in self.named_parameters() if name.startswith("classifier")]
            return param_lst, name_lst

        else:
            name_lst = [name for name, p in self.named_parameters() if name.startswith("sp_encoder")]
            param_lst = [p for name, p in self.named_parameters() if name.startswith("sp_encoder")]

            if self.add_ctc:
                name_lst += [name for name, p in self.named_parameters() if name.startswith("ctc")]
                param_lst += [p for name, p in self.named_parameters() if name.startswith("ctc")]

            name_lst += [name for name, p in self.named_parameters() if "embed_positions" in name]
            param_lst += [p for name, p in self.named_parameters() if "embed_positions" in name]

            name_lst += [name for name, p in self.named_parameters() if name.startswith("model.model.encoder.layers.0.self_attn.k_proj") or name.startswith("model.model.encoder.layers.0.self_attn.v_proj") or name.startswith("model.model.encoder.layers.0.self_attn.q_proj")]
            param_lst += [p for name, p in self.named_parameters() if name.startswith("model.model.encoder.layers.0.self_attn.k_proj") or name.startswith("model.model.encoder.layers.0.self_attn.v_proj") or name.startswith("model.model.encoder.layers.0.self_attn.q_proj")]

            # intermediate_n_lst = [name for name, p in self.named_parameters() if name.startswith("model.encoder")]
            # intermediate_p_lst = [p for name, p in self.named_parameters() if name.startswith("model.encoder")]

            # other_param_lst = [p for name, p in self.named_parameters() if name not in name_lst and name not in intermediate_n_lst]
            # other_p_name = [name for name, p in self.named_parameters() if name not in name_lst and name not in intermediate_n_lst]

            # return param_lst, name_lst, intermediate_p_lst, intermediate_n_lst, other_param_lst, other_p_name


            other_param_lst = [p for name, p in self.named_parameters() if name not in name_lst]
            other_p_name = [name for name, p in self.named_parameters() if name not in name_lst]
            return param_lst, name_lst, other_param_lst, other_p_name

    def decode(self, x, src_mask, beam_size, max_len=100, topk=1):
        batch_size, *_ = x.shape
        n_words = self.config.vocab_size
        # repeat the src inputs beam_size times
        # [batch_size * beam_size, ...]
        x = x.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *x.shape[1:])
        src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, *src_mask.shape[1:])
        # initialize the tgt inputs by <bos>
        # [batch_size * beam_size, seq_len]
        tgt = x.new_full((batch_size * beam_size, 1), self.config.bos_token_id, dtype=torch.long)
        # [batch_size * beam_size]
        active = src_mask.new_ones(batch_size * beam_size)
        # [batch_size]
        batches = tgt.new_tensor(range(batch_size)) * beam_size
        # accumulated scores
        scores = x.new_full((batch_size, beam_size), MIN).index_fill_(-1, tgt.new_tensor(0), 0).view(-1)

        def rank(scores, mask, k, length_penalty=1.0):
            scores = scores / mask.sum(-1).unsqueeze(-1) ** length_penalty
            return scores.view(batch_size, -1).topk(k, -1)[1]

        for t in range(1,  max_len + 1):
            tgt_mask = tgt.ne(self.config.pad_token_id)
            s_y = self.decoder(input_ids=torch.cat((torch.full_like(tgt[:, :1], self.config.eos_token_id), tgt), 1)[active],
                                   attention_mask=torch.cat((torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1)[active],
                                   encoder_hidden_states=x[active],
                                   encoder_attention_mask=src_mask[active])[0]
            # [n_active, n_words]
            s_y = self.classifier(s_y[:, -1]).log_softmax(-1)
            # only allow finished sequences to get <pad>
            # [batch_size * beam_size, n_words]
            s_y = x.new_full((batch_size * beam_size, n_words), MIN).masked_scatter_(active.unsqueeze(-1), s_y)
            s_y[~active, self.config.pad_token_id] = 0
            # [batch_size * beam_size, n_words]
            scores = scores.unsqueeze(-1) + s_y
            # [batch_size, beam_size]
            cands = rank(scores, tgt_mask, beam_size)
            # [batch_size * beam_size]
            scores = scores.view(batch_size, -1).gather(-1, cands).view(-1)
            # beams, tokens = cands // n_words, cands % n_words
            beams, tokens = cands.div(n_words).floor().long(), (cands % n_words).view(-1, 1)
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            # [batch_size * beam_size, seq_len + 1]
            tgt = torch.cat((tgt[indices], tokens), 1)
            active = tokens.ne(tokens.new_tensor((self.config.eos_token_id, self.config.pad_token_id))).all(-1)

            if not active.any():
                break
        cands = rank(scores.view(-1, 1), tgt.ne(self.config.pad_token_id), topk)
        return tgt[(batches.unsqueeze(-1) + cands).view(-1)].view(batch_size, topk, -1)

class BartASRCorrection(nn.Module):
    def __init__(self,
                bart_path,
                bart_tokenizer_path=None):
        super().__init__()
        from transformers import AutoConfig
        if bart_path != "None":
            # self.model = BartModel.from_pretrained(bart_path)
            self.model = End2EndSpeechNERBartModel.from_pretrained(bart_path)
            self.config = AutoConfig.from_pretrained(bart_path)
        else:
            assert bart_tokenizer_path is not None
            self.config = AutoConfig.from_pretrained(bart_tokenizer_path)
            # self.model = BartModel(self.config)
            self.model = End2EndSpeechNERBartModel(self.config)
        self.encoder, self.decoder = self.model.encoder, self.model.decoder

        self.classifier = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.classifier.weight = self.model.shared.weight
        self.criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, words):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
        Returns:
            ~torch.Tensor:
                Representations for the src sentences of the shape ``[batch_size, seq_len, n_model]``.
        """
        return self.encoder(input_ids=words, attention_mask=words.ne(self.config.pad_token_id))[0]

    def loss(self, x, tgt, src_mask, tgt_mask):
        shifted = torch.full_like(tgt, self.config.decoder_start_token_id)
        shifted[:, 1:] = tgt[:, :-1]
        y = self.decoder(input_ids=shifted,
                        attention_mask=tgt_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=src_mask)[0]
        # tgt_mask[:, 0] = 0
        s_y = self.classifier(y)
        att_loss = self.criterion(s_y[tgt_mask.bool()], tgt[tgt_mask.bool()])
        return att_loss
    
    @torch.no_grad()
    def get_logits(self, src, tgt, src_mask, tgt_mask):
        x = self.forward(src)
        shifted = torch.full_like(tgt, self.config.decoder_start_token_id)
        shifted[:, 1:] = tgt[:, :-1]
        y = self.decoder(input_ids=shifted,
                        attention_mask=tgt_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=src_mask)[0]
        # [batch_size, len, n_vocab]
        s_y = self.classifier(y)
        # tgt_mask[:, 0] = 0
        att_loss = self.criterion(s_y[tgt_mask.bool()], tgt[tgt_mask.bool()])
        return s_y, att_loss

    def distill_loss(self, stu_x, tea_logits, tgt, src_mask, tgt_mask, tem=2.0, hard_weight=0.2):
        shifted = torch.full_like(tgt, self.config.decoder_start_token_id)
        shifted[:, 1:] = tgt[:, :-1]
        y = self.decoder(input_ids=shifted,
                        attention_mask=tgt_mask,
                        encoder_hidden_states=stu_x,
                        encoder_attention_mask=src_mask)[0]
        stu_logits = self.classifier(y)
        # tgt_mask[:, 0] = 0
        hard_loss = self.criterion(stu_logits[tgt_mask.bool()], tgt[tgt_mask.bool()])
        soft_loss = self.kl_criterion(F.log_softmax(stu_logits/tem, dim=-1)[tgt_mask.bool()], F.softmax(tea_logits/tem, dim=-1)[tgt_mask.bool()])
        return (1-hard_weight) * soft_loss + hard_weight * hard_loss

    def decode(self, x, src_mask, beam_size, max_len=100, topk=1):
        batch_size, *_ = x.shape
        n_words = self.config.vocab_size
        # repeat the src inputs beam_size times
        # [batch_size * beam_size, ...]
        x = x.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *x.shape[1:])
        src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, *src_mask.shape[1:])
        # initialize the tgt inputs by <bos>
        # [batch_size * beam_size, seq_len]
        tgt = x.new_full((batch_size * beam_size, 1), self.config.bos_token_id, dtype=torch.long)
        # [batch_size * beam_size]
        active = src_mask.new_ones(batch_size * beam_size)
        # [batch_size]
        batches = tgt.new_tensor(range(batch_size)) * beam_size
        # accumulated scores
        scores = x.new_full((batch_size, beam_size), MIN).index_fill_(-1, tgt.new_tensor(0), 0).view(-1)

        def rank(scores, mask, k, length_penalty=1.0):
            scores = scores / mask.sum(-1).unsqueeze(-1) ** length_penalty
            return scores.view(batch_size, -1).topk(k, -1)[1]

        for t in range(1,  max_len + 1):
            tgt_mask = tgt.ne(self.config.pad_token_id)
            s_y = self.decoder(input_ids=torch.cat((torch.full_like(tgt[:, :1], self.config.eos_token_id), tgt), 1)[active],
                                   attention_mask=torch.cat((torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1)[active],
                                   encoder_hidden_states=x[active],
                                   encoder_attention_mask=src_mask[active])[0]
            # [n_active, n_words]
            s_y = self.classifier(s_y[:, -1]).log_softmax(-1)
            # only allow finished sequences to get <pad>
            # [batch_size * beam_size, n_words]
            s_y = x.new_full((batch_size * beam_size, n_words), MIN).masked_scatter_(active.unsqueeze(-1), s_y)
            s_y[~active, self.config.pad_token_id] = 0
            # [batch_size * beam_size, n_words]
            scores = scores.unsqueeze(-1) + s_y
            # [batch_size, beam_size]
            cands = rank(scores, tgt_mask, beam_size)
            # [batch_size * beam_size]
            scores = scores.view(batch_size, -1).gather(-1, cands).view(-1)
            # beams, tokens = cands // n_words, cands % n_words
            beams, tokens = cands.div(n_words).floor().long(), (cands % n_words).view(-1, 1)
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            # [batch_size * beam_size, seq_len + 1]
            tgt = torch.cat((tgt[indices], tokens), 1)
            active = tokens.ne(tokens.new_tensor((self.config.eos_token_id, self.config.pad_token_id))).all(-1)

            if not active.any():
                break
        cands = rank(scores.view(-1, 1), tgt.ne(self.config.pad_token_id), topk)
        return tgt[(batches.unsqueeze(-1) + cands).view(-1)].view(batch_size, topk, -1)


        


