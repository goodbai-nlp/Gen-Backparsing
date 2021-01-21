"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
from torch import autograd

import onmt
from onmt.sublayer import PositionwiseFeedForward
import sys

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = onmt.sublayer.MultiHeadedAttention(heads, d_model, dropout=dropout)

        self.context_attn = onmt.sublayer.MultiHeadedAttention(heads, d_model, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer("mask", mask)

    def create_padding_variable(self, *shape):
        if torch.cuda.is_available():
            data = torch.zeros(*shape).to(device=torch.cuda.current_device())
        else:
            data = torch.zeros(*shape)
        # if gpu:
        #     data = data.to(self.config.device)
        var = autograd.Variable(data, requires_grad=False)
        # if gpu:
        #    var = var.to(config.device)
        return var

    def forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
        last_layer=False,
    ):
        dec_mask = None
        try:
            self.mask = self.mask.bool()
        except AttributeError:
            pass
        if step is None:
            dec_mask = torch.gt(
                tgt_pad_mask + self.mask[:, : tgt_pad_mask.size(-1), : tgt_pad_mask.size(-1)], 0
            )

        input_norm = self.layer_norm_1(inputs)

        query, top_attn, _ = self.self_attn(
            input_norm, input_norm, input_norm, mask=dec_mask, layer_cache=layer_cache, type="self"
        )

        # padding = self.create_padding_variable((query.size(0), 1, query.size(2)))

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, top_attn, mean_attn = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            type="context",
        )

        output = self.feed_forward(self.drop(mid) + query)

        return output, top_attn, mean_attn

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        a_drop,
        l_drop,
        h_drop,
        integrated,
        integrated_mode,
        embeddings,
    ):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = "transformer"
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.integrated = integrated
        self.integrated_mode = integrated_mode
        # Decoder State
        self.state = {}

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )

        if self.integrated:
            self.label_map = nn.Linear(64, d_model, bias=False)  # bad code, d_label = 64, to do
            self.align_gold_drop = nn.Dropout(a_drop)
            self.arc_hidden_drop = nn.Dropout(h_drop)
            self.label_drop = nn.Dropout(l_drop)

        if self.integrated and self.integrated_mode.startswith("cat"):
            self.align_gold_map = nn.Linear(d_model, d_model, bias=False)
            self.arc_map = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        if self.integrated:
            nn.init.eye_(self.label_map.weight)
            if self.integrated_mode.startswith("cat"):
                nn.init.eye_(self.align_gold_map.weight)
                nn.init.eye_(self.arc_map.weight)

    def init_state(self, src, src_enc):
        """ Init decoder state """
        self.state["src"] = src.transpose(0, 1).contiguous()            # [bsz, seq_len]
        self.state["src_enc"] = src_enc.transpose(0, 1).contiguous()    # [bsz, seq_len, H]
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 0)            # [bsz, seq_len]
        self.state["src_enc"] = fn(self.state["src_enc"], 0)    # [bsz, seq_len, H]
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, align=None, arc_hidden=None, label_emb=None, step=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        align: [tgt_len, bsz, H_dim]        # weight sum of memory bank according to tgt2src align attn
        arc_hidden: [tgt_len, bsz, H_dim]   # weight sum of ancestor hidden according to arc attn
        label_emb:  [tgt_len, bsz, L_dim]   # weight sum of in AMR label embs according to arc attn and label attn
        """
        if step == 0:
            self._init_cache(self.num_layers)

        src = self.state["src"]
        memory_bank = self.state["src_enc"]
        src_words = src
        tgt_words = tgt.transpose(0, 1)  # [B, Tgt_len (<s> included, last element excluded)]

        # print('tgt words', tgt_words.size(), tgt_words)
        # Initialize return variables.
        attns = {"std": [], "mean": []}

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()  # []

        if self.integrated:

            if align is not None:
                align_gold_inp = align.transpose(0, 1).contiguous()  # [B, Tgt_len, emb_dim]
                if self.integrated_mode == "cat_all":  # align do not use cat mode by default!
                    align_gold_inp = self.align_gold_map(align_gold_inp)
                output = output + self.align_gold_drop(align_gold_inp)

            if arc_hidden is not None:
                arc_hidden_inp = arc_hidden.transpose(0, 1).contiguous()  # [B, Tgt_len, emb_dim]
                if self.integrated_mode.startswith("cat"):
                    arc_hidden_inp = self.arc_map(arc_hidden_inp)
                output = output + self.arc_hidden_drop(arc_hidden_inp)

            if label_emb is not None:
                label_emb_inp = label_emb.transpose(0, 1).contiguous()
                label_emb_inp = self.label_map(label_emb_inp)
                output = output + self.label_drop(label_emb_inp)

        src_memory_bank = memory_bank  # [B, Src_len(</s> included), H]
        # print('emb_inp', output.size(), output)
        # print('Src_mem', src_memory_bank.size(), src_memory_bank)

        pad_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        # print('src_pad_mask',  src_pad_mask.size(),  src_pad_mask)
        # print('tgt_pad_mask',  tgt_pad_mask.size(),  tgt_pad_mask)
        # exit(0)

        for i in range(self.num_layers):
            output, top_attn, mean_attn = self.transformer_layers[i](
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=(
                    self.state["cache"]["layer_{}".format(i)] if step is not None else None
                ),
                step=step,
                last_layer=True if (i + 1 == self.num_layers) else False,
            )

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        dec_outs = output.transpose(0, 1).contiguous()      # [query_len, bsz, dim]
        top_attn = top_attn.transpose(0, 1).contiguous()    # [query_len, bsz, key_len]
        mean_attn = mean_attn.transpose(0, 1).contiguous()  #

        attns["std"] = top_attn
        attns["mean"] = mean_attn

        # print('ori_cache', self.state["cache"]["output"].size() if self.state["cache"]["output"] is not None else None, self.state["cache"]["output"])
        if step is not None:                                # In inference mode
            new_cache = (                                   # [bsz, seq_len, dim]
                torch.cat((self.state["cache"]["output"], output), dim=1)
                if self.state["cache"]["output"] is not None
                else output
            )  # [bsz, query_len+1, dim]
            self.state["cache"]["output"] = new_cache
            # print('new_cache', self.state["cache"]["output"].size(), self.state["cache"]["output"])
        return dec_outs, attns

    def _init_cache(self, num_layers):
        self.state["cache"] = {}
        self.state["cache"]["output"] = None
        for ll in range(num_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}  # source-side mem
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(ll)] = layer_cache
