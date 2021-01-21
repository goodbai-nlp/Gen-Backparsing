"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch import autograd

import onmt.constants as Constants

from onmt.transformer_encoder import TransformerEncoder
from onmt.transformer_decoder import TransformerDecoder

from onmt.embeddings import Embeddings
from utils.misc import use_gpu
from utils.logging import logger
from inputters.dataset import load_fields_from_vocab


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, biaffine):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.biaffine = biaffine

    def forward(
        self,
        src,
        tgt,
        structure,
        mask,
        align,
        lengths,
        use_0=False,
        gold_arc_attn=None,
        relation_mat=None,
    ):
        """
        :param src:
        :param tgt:
        :param structure:
        :param mask: for every sentence, mask is a dependency matrix with size [tgt_length, tgt_length]
        :param align: gold or predicted tgt2src alignment
        :param lengths:
        :return: the output of decoder and biaffine module if train
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        _, memory_bank, lengths = self.encoder(src, structure, lengths)  # src: ......<EOS>
        self.decoder.init_state(src, memory_bank)

        if align is not None:
            align_vec = torch.matmul(
                align, memory_bank.transpose(0, 1)
            )           # [batch, tgt_len (excluded <\s>), hidden_dim]
            padding = create_padding_variable((1, align_vec.size(0), align_vec.size(2)))
            align_vec_padded = torch.cat(
                [padding, align_vec.transpose(0, 1)], dim=0
            )  # [tgt_len+1, batch, hidden_dim]
        else:
            align_vec_padded = None

        if gold_arc_attn is not None and relation_mat is not None:
            # print("gold_arc_attn", gold_arc_attn.size(), gold_arc_attn.transpose(0,1)[:4,:,:5].size(),gold_arc_attn.transpose(0,1)[:4,:,:5]) # [bsz, tgt_len, tgt_len]

            emb_align_gold = (
                self.decoder.embeddings(tgt[1:]).transpose(0, 1).contiguous()
            )       # [bsz, tgt_len, H_emb]
            # emb_align_gold = (
            #     self.decoder.embeddings.word_embedding(tgt[1:]).transpose(0, 1).contiguous()
            # )       # [bsz, tgt_len, H_emb]
            avg_hidden = torch.matmul(gold_arc_attn, emb_align_gold)  # [bsz, tgt_len, H_emb]
            padding = create_padding_variable((1, avg_hidden.size(0), avg_hidden.size(2)))
            avg_hidden_padded = torch.cat(
                [padding, avg_hidden.transpose(0, 1)], dim=0
            )  # [tgt_len+1, bsz, H_emb]

            # print('avg_hidden_padded', avg_hidden_padded.size(), avg_hidden_padded[:5,:,:])

            label_embs = self.encoder.structure_embeddings(
                relation_mat
            )  # [bsz, tgt_len, tgt_len, R_dim]
            avg_arc_labels = torch.matmul(gold_arc_attn.unsqueeze(-2), label_embs).squeeze(
                -2
            )  # [bsz, tgt_len, R_dim]
            padding2 = create_padding_variable((1, avg_arc_labels.size(0), avg_arc_labels.size(2)))
            avg_label_padded = torch.cat(
                [padding2, avg_arc_labels.transpose(0, 1)], dim=0
            )  # [tgt_len+1, bsz, R_emb]

            # print('avg_label_padded', avg_label_padded.size(), avg_label_padded[:5,:,:])
        else:
            avg_hidden_padded, avg_label_padded = None, None

        dec_out, attns = self.decoder(
            tgt, align=align_vec_padded, arc_hidden=avg_hidden_padded, label_emb=avg_label_padded
        )

        # dec_out, attns = self.decoder(tgt, align=align_vec_padded, arc_hidden=None, label_emb=avg_label_padded)

        if mask is not None:
            bi_out, label_out = self.biaffine(dec_out[:-1], mask)
            return dec_out, attns, bi_out, label_out
        return dec_out, attns


# class Biaffine2(nn.Module):
#     '''
#     code for loss 1 in paper
#     '''
#     def __init__(self, in_size, dropout, out_size):
#         super(Biaffine2, self).__init__()
#         self.in_size=in_size
#         self.out_size=out_size

#         # code for MLP in paper
#         self.head_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
#         self.dep_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
#         self.label_head_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
#         self.label_dep_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
#         # arc
#         self.U = nn.Parameter(torch.Tensor(in_size, in_size))
#         self.W = nn.Parameter(torch.Tensor(2 * in_size))
#         self.b = nn.Parameter(torch.Tensor(1))

#         # label
#         self.label_U = nn.Parameter(torch.Tensor(out_size, in_size, in_size))
#         self.label_W_1 = nn.Parameter(torch.Tensor(in_size, out_size))
#         self.label_W_2 = nn.Parameter(torch.Tensor(in_size, out_size))
#         self.label_b = nn.Parameter(torch.Tensor(out_size))

#         self.reset_parameters()
#         self.gen_func = nn.LogSoftmax(dim=-1)

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.head_mlp[0].weight)
#         nn.init.constant_(self.head_mlp[0].bias, 0.)
#         nn.init.xavier_uniform_(self.dep_mlp[0].weight)
#         nn.init.constant_(self.dep_mlp[0].bias, 0.)
#         nn.init.xavier_uniform_(self.label_head_mlp[0].weight)
#         nn.init.constant_(self.label_head_mlp[0].bias, 0.)
#         nn.init.xavier_uniform_(self.label_dep_mlp[0].weight)
#         nn.init.constant_(self.label_dep_mlp[0].bias, 0.)

#         bound = 1 / math.sqrt(self.in_size*2)
#         nn.init.uniform_(self.W, -bound, bound)
#         nn.init.constant_(self.b, 0.)
#         nn.init.xavier_uniform_(self.U)

#         bound = 1 / math.sqrt(self.in_size)
#         nn.init.uniform_(self.label_W_1, -bound, bound)
#         nn.init.uniform_(self.label_W_2, -bound, bound)
#         nn.init.xavier_uniform_(self.label_U)
#         nn.init.constant_(self.label_b, 0.)

#     def bilinear_(self, head, dep):
#         out=torch.zeros([head.size(0),head.size(1),head.size(1),self.out_size]).cuda()
#         for k in range(self.label_U.shape[0]):
#             x=torch.matmul(head,self.label_U[k])
#             x=torch.matmul(x,dep.transpose(1,2))
#             out[:,:,:,k]=x
#         return out

#     def linear_(self, head, dep):
#         out=torch.zeros([head.size(1),head.size(0),head.size(1),self.out_size]).cuda()
#         x=torch.matmul(head, self.label_W_1)
#         y = torch.matmul(dep, self.label_W_2)
#         batch_size = x.size(0)
#         x=x.transpose(0,1)
#         for k in range(x.size(0)):
#             out[k]=x[k].view(batch_size,1,self.out_size)+y
#         out=out.transpose(0,1)
#         return out

#     def forward(self, input, mask):
#         '''
#         :param input: output of decoder
#         :param mask: dependency matrix of target sentence
#         :return:
#         '''
#         input = input.transpose(0, 1)
#         batch_size=input.size(0)
#         seq_len=input.size(1)
#         o_head = self.head_mlp(input)                   # [batch, seq_len, hidden]
#         o_dep = self.dep_mlp(input)
#         out = torch.matmul(o_head, self.U)              #
#         out = torch.matmul(out, o_dep.transpose(1, 2))  # [batch, seq_len, seq_len]
#         out_ = torch.matmul((torch.cat((o_head, o_dep), 2)), self.W).unsqueeze(2)   # [batch, seq_len, 1]
#         out = out + out_ + self.b                       # [batch, seq_len, seq_len]
#         out = F.log_softmax(out, 2)
#         out=out.view(batch_size,seq_len,seq_len, 1)

#         l_head = self.label_head_mlp(input)
#         l_dep = self.label_dep_mlp(input)
#         l_out = self.bilinear_(l_head, l_dep)
#         l_out = l_out + self.linear_(l_head, l_dep)+self.label_b
#         # print('l_out', l_out.size())                    # [batch, seq_len, seq_len, Label_vocab]

#         out=out[mask]                                   # [k]
#         out=out.sum()                                   # [1]
#         l_out = l_out[mask]                             # [k, Label_vocab]
#         # print('masked l_out',l_out.size())
#         l_out=self.gen_func(l_out)
#         # exit()
#         return out, l_out


class Biaffine(nn.Module):
    def __init__(self, in_size, dropout, out_size):
        super(Biaffine, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.head_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
        self.dep_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
        self.label_head_mlp = nn.Sequential(
            nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU()
        )
        self.label_dep_mlp = nn.Sequential(
            nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU()
        )

        self.arc_attn = BiLinear(n_in=self.in_size, bias_x=True, bias_y=False)
        self.rel_attn = BiLinear(n_in=self.in_size, n_out=self.out_size, bias_x=True, bias_y=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.head_mlp[0].weight)
        nn.init.constant_(self.head_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.dep_mlp[0].weight)
        nn.init.constant_(self.dep_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.label_head_mlp[0].weight)
        nn.init.constant_(self.label_head_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.label_dep_mlp[0].weight)
        nn.init.constant_(self.label_dep_mlp[0].bias, 0.0)

    def forward(self, input, mask):
        """
        :param input: output of decoder [seq_len, batch_size, H_dim]
        :param mask: dependency matrix of target sentence
        :return: masked arc attn, masked label attn
        """
        x = input.transpose(0, 1)
        arc_h = self.head_mlp(x)
        arc_d = self.dep_mlp(x)
        rel_h = self.label_head_mlp(x)
        rel_d = self.label_dep_mlp(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # out = F.log_softmax(s_arc, 2)                     # [batch_size, seq_len, seq_len]
        # out=out[mask]                                     # [k]
        # out=out.sum()                                     # [1]
        # l_out = s_rel[mask]                               # [k, Label_vocab]
        # l_out = F.log_softmax(l_out, -1)                  # [k, Label_vocab]

        # out = F.softmax(s_arc, 2)
        # l_out = F.softmax(s_rel, -1)                      # [batch_size, seq_len, seq_len, n_rels]
        return s_arc, s_rel


class BiLinear(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(BiLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


def build_embeddings(opt, word_dict, for_encoder="src"):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder == "src":
        embedding_dim = opt.src_word_vec_size  # 512
    elif for_encoder == "tgt":
        embedding_dim = opt.tgt_word_vec_size
    elif for_encoder == "structure":
        embedding_dim = 64

    word_padding_idx = word_dict.stoi[Constants.PAD_WORD]
    num_word_embeddings = len(word_dict)

    if for_encoder == "src" or for_encoder == "tgt":

        return Embeddings(
            word_vec_size=embedding_dim,
            position_encoding=opt.position_encoding,
            dropout=opt.dropout,
            word_padding_idx=word_padding_idx,
            word_vocab_size=num_word_embeddings,
            sparse=opt.optim == "sparseadam",
        )
    elif for_encoder == "structure":
        return Embeddings(
            word_vec_size=embedding_dim,
            position_encoding=False,
            dropout=opt.dropout,
            word_padding_idx=word_padding_idx,
            word_vocab_size=num_word_embeddings,
            sparse=opt.optim == "sparseadam",
        )


def build_encoder(opt, embeddings, structure_embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    return TransformerEncoder(
        opt.enc_layers,
        opt.enc_rnn_size,
        opt.heads,
        opt.transformer_ff,
        opt.dropout,
        embeddings,
        structure_embeddings,
    )


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return TransformerDecoder(
        opt.dec_layers,
        opt.dec_rnn_size,
        opt.heads,
        opt.transformer_ff,
        opt.dropout,
        opt.a_drop,
        opt.l_drop,
        opt.h_drop,
        opt.integrated,
        opt.integrated_mode,
        embeddings,
    )


def build_biaffine(opt, fields):
    return Biaffine(opt.dec_rnn_size, opt.dropout, len(fields["relation"].vocab))


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    fields = load_fields_from_vocab(checkpoint["vocab"])

    model_opt = checkpoint["opt"]

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)

    model.eval()
    model.generator.eval()
    return fields, model


def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # for backward compatibility
    if model_opt.enc_rnn_size != model_opt.dec_rnn_size:
        raise AssertionError(
            """We do not support different encoder and
                         decoder rnn sizes for translation now."""
        )

    # Bulid_structure
    structure_dict = fields["structure"].vocab
    structure_embeddings = build_embeddings(model_opt, structure_dict, for_encoder="structure")

    # Build encoder.
    src_dict = fields["src"].vocab
    src_embeddings = build_embeddings(model_opt, src_dict, for_encoder="src")
    encoder = build_encoder(model_opt, src_embeddings, structure_embeddings)

    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    tgt_embeddings = build_embeddings(model_opt, tgt_dict, for_encoder="tgt")

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        # print("[flog], src and tgt use shared embedding!")
        if src_dict != tgt_dict:
            raise AssertionError(
                "The `-share_vocab` should be set during " "preprocess if you use share_embeddings!"
            )

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_embeddings)
    # build biaffine.

    biaffine = build_biaffine(model_opt, fields)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda:0" if gpu else "cpu")
    model = NMTModel(encoder, decoder, biaffine)

    # Build Generator.
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab)), gen_func)
    if model_opt.share_decoder_embeddings:
        generator[0].weight = decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s)
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s)
            return s

        checkpoint["model"] = {fix_key(k): v for (k, v) in checkpoint["model"].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint["model"], strict=False)
        generator.load_state_dict(checkpoint["generator"], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            # for p in biaffine.generator.parameters():
            #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            # for p in biaffine.generator.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)

        if hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc
            )
        if hasattr(model.decoder, "embeddings"):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec
            )
    # pdb.set_trace()
    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    """ Build the Model """
    logger.info("Building model...")
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model


def create_padding_variable(*shape):
    if torch.cuda.is_available():
        data = torch.zeros(*shape).to(device=torch.cuda.current_device())
    else:
        data = torch.zeros(*shape)
    # if gpu:
    #     data = data.to(self.config.device)
    var = autograd.Variable(data, requires_grad=False)
    return var
