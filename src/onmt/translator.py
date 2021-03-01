# coding:utf-8
#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import onmt.opts as opts
import torch
import onmt.transformer as nmt_model
from inputters.dataset import build_dataset, OrderedIterator, make_features
from onmt.beam import Beam
from utils.misc import tile
import onmt.constants as Constants
import time
import copy

debug = False
# debug = True

def _tally_parameters(model):                   # Counting total parameters
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return n_params, enc, dec


def build_translator(opt):
    dummy_parser = configargparse.ArgumentParser(description="translate.py")
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model = nmt_model.load_test_model(opt, dummy_opt.__dict__)
    n_params, enc, dec = _tally_parameters(model)
    print("encoder: %d" % enc)
    print("decoder: %d" % dec)
    print("* number of parameters: %d" % n_params)

    translator = Translator(model, fields, opt)

    return translator


class Translator(object):
    def __init__(self, model, fields, opt, out_file=None):
        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.decode_extra_length = opt.decode_extra_length
        self.decode_min_length = opt.decode_min_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.out_file = out_file
        self.integrated_node = opt.integrated_node
        self.integrated_edge = opt.integrated_edge
        self.hidden_size = opt.hidden_size
        self.alpha = opt.alpha

        self.tgt_eos_id = fields["tgt"].vocab.stoi[Constants.EOS_WORD]
        self.tgt_bos_id = fields["tgt"].vocab.stoi[Constants.BOS_WORD]
        self.src_eos_id = fields["src"].vocab.stoi[Constants.EOS_WORD]

    def build_tokens(self, idx, side="tgt"):
        assert side in ["src", "tgt"], "side should be either src or tgt"
        vocab = self.fields[side].vocab

        if side == "tgt":
            eos_id = self.tgt_eos_id
        else:
            eos_id = self.src_eos_id

        tokens = []
        for tok in idx:
            if tok == eos_id:
                break
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
        return tokens

    def translate(
        self, src_data_iter, tgt_data_iter, structure_iter, batch_size, out_file=None, R2S=None
    ):

        data = build_dataset(
            self.fields,
            src_data_iter,
            tgt_data_iter,
            structure_iter,
            None,
            None,
            None,
            None,
            use_filter_pred=False,
        )
        self.relation_embs = (
            self.model.encoder.structure_embeddings(R2S) if R2S is not None else None
        )
        # for line in data:
        #   print(line.__dict__)    {src:  , indices:   structure: }

        def sort_translation(indices, translation):
            ordered_transalation = [None] * len(translation)
            for i, index in enumerate(indices):
                ordered_transalation[index] = translation[i]
            return ordered_transalation

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = OrderedIterator(
            dataset=data,
            device=cur_device,
            batch_size=batch_size,
            train=False,
            sort=True,
            sort_within_batch=True,
            shuffle=False,
        )

        start_time = time.time()
        print("Begin decoding ...")
        batch_count = 0
        all_translation = []

        for batch in data_iter:
            """
            batch
            [torchtext.data.batch.Batch of size 30]
            [.src]:('[torch.LongTensor of size 4x30]', '[torch.LongTensor of size 30]')
            [.indices]:[torch.LongTensor of size 30]
            [.structure]:[torch.LongTensor of size 30x4x4]
            """
            hyps, scores = self.translate_batch(batch)
            assert len(batch) == len(hyps)
            batch_translation = []
            for src_idx_seq, tran_idx_seq, score in zip(batch.src[0].transpose(0, 1), hyps, scores):
                src_words = self.build_tokens(src_idx_seq, side="src")
                src = " ".join(src_words)

                tran_words = self.build_tokens(tran_idx_seq, side="tgt")
                tran = " ".join(tran_words)

                batch_translation.append(tran)
                print("SOURCE: " + src + "\nOUTPUT: " + tran + "\n")

            for index, tran in zip(batch.indices.data, batch_translation):
                while len(all_translation) <= index:
                    all_translation.append("")
                all_translation[index] = tran
            batch_count += 1
            print("batch: " + str(batch_count) + "...")

        if out_file is not None:
            for tran in all_translation:
                out_file.write(tran + "\n")
        print("Decoding took %.1f minutes ..." % (float(time.time() - start_time) / 60.0))

    def translate_batch(self, batch):
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {
                inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """ Collect tensor parts associated to active instances. """

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def beam_decode_step(
            inst_dec_beams,
            len_dec_seq,
            inst_idx_to_position_map,
            n_bm,
            pre_align=None,
            pre_arc=None,
            pre_label=None,
            enc_mem=None,
        ):
            """ Decode and update beam status, and then return active beam idx """

            # len_dec_seq: i (starting from 0)

            def prepare_beam_dec_seq(inst_dec_beams):
                dec_seq = [b.get_last_target_word() for b in inst_dec_beams if not b.done]  #
                # dec_seq: [(nbeam)] * bsz
                dec_seq = torch.stack(dec_seq).to(self.device)  # [bsz, nbeam]
                dec_seq = dec_seq.view(1, -1)  # [1, bsz * nbeam]
                return dec_seq

            def predict_word(
                dec_seq,
                n_active_inst,
                n_bm,
                len_dec_seq,
                pre_align=None,
                pre_arc=None,
                pre_label=None,
                inst_dec_beams=None,
            ):
                if debug:
                    print('dec_input', dec_seq.size(), dec_seq)
                    # print('dec_membank', self.model.decoder.state["src_enc"].size(), self.model.decoder.state["src_enc"])
                dec_output, attns = self.model.decoder(
                    dec_seq,
                    align=pre_align,
                    arc_hidden=pre_arc,
                    label_emb=pre_label,
                    step=len_dec_seq,
                )
                
                # dec_output: [1, bsz*nbeam, H]
                biaffine_inp = self.model.decoder.state["cache"]["output"].transpose(0, 1)  # [T, bsz*nbeam, H]
                arc_attn, label_attn = self.model.biaffine(biaffine_inp, mask=None)
                align_attn = attns["mean"]  # [1, bsz*nbeam, src_len]
                if self.integrated_node:
                    align_attn = align_attn.transpose(0, 1)                             # [bsz*nbeam, 1, src_len]
                    softmax = torch.nn.Softmax(dim=-1)
                    align_attn = softmax(align_attn)                                    # [bsz*nbeam, 1, src_len]
                    # print("align_attn_new", align_attn.size(), align_attn)
                    mem_bank = self.model.decoder.state["src_enc"]                      # [bsz*nbeam, src_len, H]
                    # print("mem_bank", mem_bank.size())
                    pre_align = torch.matmul(align_attn, mem_bank).transpose(0, 1)      # [1, bsz*nbeam, H]
                else:
                    pre_align = None
                if self.integrated_edge:    
                    # print('align_attn', align_attn.size(), align_attn)
                    if arc_attn.size(1) > 1:    # step 2
                        arc_attn = arc_attn[:, -1:, :-1]                                # [bsz*nbeam, 1, T-1]
                        if debug:
                            print("ori_attn", arc_attn.size(), arc_attn)
                        
                        arc_softmax = torch.nn.Softmax(dim=2)
                        arc_attn = arc_softmax(arc_attn)                                # [bsz*nbeam, 1, T-1]
                        # arc_attn = torch.nn.functional.normalize(arc_attn, p=1, dim=-1)
                        if debug:
                            print('Arc_attn', arc_attn.size(), arc_attn)
                        dec_seq_lst = [
                            b.get_current_state() for b in inst_dec_beams if not b.done
                        ]  # bsz * [nbeam, T+1]
                        if debug:
                            print('dec_seq_lst', dec_seq.size(), dec_seq)
                        dec_seq_full = torch.stack(dec_seq_lst).to(self.device)           # [bsz, nbeam, T+1]
                        dec_seq_his = dec_seq_full.view(-1, dec_seq_full.size(2))[:, 1:]  # [bsz*nbeam, T]
                        if debug:
                            print("dec_seq_his", dec_seq_his.size(), dec_seq_his)
                        
                        pre_hiddens = self.model.decoder.embeddings(
                            dec_seq_his, step=arc_attn.size(1) - 1
                        )  # [bsz*nbeam, T, H]
                        if debug:
                            print('pre_hiddens', pre_hiddens.size(), pre_hiddens)
                        """ use predicted word's hidden state for weight avg"""
                        # pre_hiddens = self.model.decoder.state["cache"]["output"]     # [bsz*nbeam, T, H]
                        pre_arc = torch.matmul(arc_attn, pre_hiddens).transpose(0, 1)   # [1, bsz*nbeam, H]
                        if debug:
                            print('pre_arc', pre_arc.size(), pre_arc)

                        label_softmax = torch.nn.Softmax(dim=-1)
                        label_attn = label_softmax(label_attn)
                        # if debug:
                        #     print("label_attn_normed", label_attn.size(), label_attn)
                        pre_label_all = torch.matmul(
                            label_attn[:, -1:, :-1, :], self.relation_embs
                        )  # [bsz*nbeam, 1, T-1, R_size]
                        if debug:
                            print("Pre_label_all", pre_label_all.size())
                            print("arc_attn", arc_attn.size())
                        pre_label = (
                            torch.matmul(arc_attn.unsqueeze(-2), pre_label_all)
                            .squeeze(-2)
                            .transpose(0, 1)
                        )  # [1, bsz*nbeam, R_size]
                        # print("Label_inp_next", pre_label.size())
                else:
                    pre_arc = None
                    pre_label = None

                word_prob = self.model.generator(dec_output.squeeze(0))
                # word_prob: (bsz*nbeam, vocab_size)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                # word_prob: (bsz*nbeam, vocab_size)
                return word_prob, align_attn, arc_attn, label_attn, pre_align, pre_arc, pre_label

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                select_indices_array = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                        select_indices_array.append(
                            inst_beams[inst_idx].get_current_origin() + inst_position * n_bm
                        )
                if len(select_indices_array) > 0:
                    select_indices = torch.cat(select_indices_array)
                else:
                    select_indices = None
                return active_inst_idx_list, select_indices

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams)  # [1, bsz*nbeam]

            word_prob, align_attn, arc_attn, label_attn, pre_align, pre_arc, pre_label = predict_word(
                dec_seq, n_active_inst, n_bm, len_dec_seq, pre_align, pre_arc, pre_label, inst_dec_beams
            )
            # align_attn: torch.Size([1, bsz*nbeam, src_len]), arc_attn: [bsz*nbeam, T, T], new_label_attn: [bsz*nbeam, T, T, R_v]

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list, select_indices = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )

            if select_indices is not None:
                assert len(active_inst_idx_list) > 0
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )
                if self.integrated_node:
                    # print("align_attn_ori", align_attn.size())
                    # align_attn = align_attn.index_select(1, select_indices)
                    pre_align = pre_align.index_select(1, select_indices)
                    # print("pre_align_hidden_selected", pre_align_hidden)
                
                if self.integrated_edge:
                    # arc_attn = arc_attn.index_select(0, select_indices)  # []
                    pre_arc = pre_arc.index_select(1, select_indices)
                    if debug:
                        print('pre_arc_selected', pre_arc.size(), pre_arc)
                    # label_attn = label_attn.index_select(0, select_indices)  # []
                    pre_label = pre_label.index_select(1, select_indices)
                # print("pre_arc_attn_selected", arc_attn)
                # print("pre_label_attn_selected", label_attn)
                # print('active list after', len(active_inst_idx_list), active_inst_idx_list)
                # print('select idxs after', len(select_indices), select_indices)

                # print("align_attn_after", align_attn.size())
            # print('active list after', len(active_inst_idx_list), active_inst_idx_list)

            return active_inst_idx_list, align_attn, arc_attn, label_attn, pre_align, pre_arc, pre_label

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def collect_best_hypothesis_and_score(inst_dec_beams):
            hyps, scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                hyp, score = inst_dec_beams[inst_idx].get_best_hypothesis()
                hyps.append(hyp)
                scores.append(score)

            return hyps, scores

        with torch.no_grad():
            # -- Encode
            src_seq = make_features(batch, "src")
            # src: (seq_len_src, bsz)
            if debug:
                print("src_seq", src_seq)
            # threshold_m = torch.nn.Threshold(0.3, 0)

            structure = make_features(batch, "structure")
            structure = structure.transpose(0, 1)
            structure = structure.transpose(1, 2)
            # print(structure.size()) 30*4*4

            src_emb, src_enc, _ = self.model.encoder(src_seq, structure)
            # src_emb: (seq_len_src, bsz, emb_size)
            # src_enc: (seq_len_src, bsz, hid_size)
            self.model.decoder.init_state(src_seq, src_enc)
            src_len = src_seq.size(0)

            # -- Repeat data for beam search
            n_bm = self.beam_size
            n_inst = src_seq.size(1)  # batch_size
            self.model.decoder.map_state(lambda state, dim: tile(state, n_bm, dim=dim))

            pre_align_hidden, pre_label, pre_arc = None, None, None
            
            if self.integrated_node:
                pre_align_hidden = (
                    torch.zeros(size=(n_inst * n_bm, self.hidden_size))
                    .unsqueeze(0)
                    .to(src_emb.device)
                )
                assert pre_align_hidden.dim() == 3

            if self.integrated_edge:
                pre_label = (
                    torch.zeros(size=(n_inst * n_bm, 64)).unsqueeze(0).to(src_emb.device)
                )  # [1, bsz*nbeam, R]
                pre_arc = (
                    torch.zeros(size=(n_inst * n_bm, self.hidden_size))
                    .unsqueeze(0)
                    .to(src_emb.device)
                )  # [1, bsz*nbeam, H]
                # assert pre_align_hidden.dim() == 3
                assert pre_label.dim() == 3 and pre_arc.dim() == 3
                
            # -- Prepare beams
            decode_length = src_len + self.decode_extra_length
            decode_min_length = 0
            if self.decode_min_length >= 0:
                decode_min_length = src_len - self.decode_min_length

            inst_dec_beams = [
                Beam(
                    n_bm,
                    decode_length=decode_length,
                    minimal_length=decode_min_length,
                    bos_id=self.tgt_bos_id,
                    eos_id=self.tgt_eos_id,
                    device=self.device,
                    alpha=self.alpha,
                )
                for _ in range(n_inst)
            ]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # -- Decode
            step = 0
            for len_dec_seq in range(0, decode_length):
                step += 1
                if debug:
                    print("step:", step)
                    if step > 20:
                        exit()

                active_inst_idx_list, pre_align_attn, arc_attn, label_attn, pre_align_hidden, pre_arc, pre_label = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    inst_idx_to_position_map,
                    n_bm,
                    pre_align=pre_align_hidden,
                    pre_arc=pre_arc,
                    pre_label=pre_label,
                    enc_mem=src_enc,
                )

                # arc_attn [bsz*nbeam, T, T], label_attn [bsz*nbeam, T, T, V]

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        batch_hyps, batch_scores = collect_best_hypothesis_and_score(inst_dec_beams)
        return batch_hyps, batch_scores
