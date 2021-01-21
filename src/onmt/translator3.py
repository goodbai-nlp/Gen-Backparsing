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
import json

def build_translator(opt):
    dummy_parser = configargparse.ArgumentParser(description='translate.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    fields, model = nmt_model.load_test_model(opt, dummy_opt.__dict__)
    translator = Translator(model, fields, opt)
    return translator

class Translator(object):
    def __init__(self, model, fields, opt, out_file=None):
        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.decode_extra_length = opt.decode_extra_length
        self.decode_min_length = opt.decode_min_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.out_file = out_file
        self.integrated = opt.integrated
        self.hidden_size = opt.hidden_size

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

    def translate(self, src_data_iter, tgt_data_iter, structure_iter, batch_size, out_file=None):
        # print(self.fields)
        # print('tgt_iter:', tgt_data_iter)
        data = build_dataset(self.fields, src_data_iter, tgt_data_iter, structure_iter, None, None, None, None, use_filter_pred=False)

        # for line in data:
        #   print(line.__dict__)    #{src:  , indices:   structure: }

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
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=True,
            sort_within_batch=True, shuffle=True)

        start_time = time.time()
        print("Begin decoding ...")
        batch_count = 0
        all_translation = []
        res_all = []
        
        for batch in data_iter:
            '''
            batch
            [torchtext.data.batch.Batch of size 30]
            [.src]:('[torch.LongTensor of size 4x30]', '[torch.LongTensor of size 30]')
            [.indices]:[torch.LongTensor of size 30]
            [.structure]:[torch.LongTensor of size 30x4x4]
            '''
            # print('batch', batch)
            # print('tgt', batch.tgt)
            attn = self.translate_batch(batch)
            ith_dict = {}
            # print('attn',type(attn))
            src_idx_seq = batch.src[0].transpose(0, 1)[0]
            # print('src_idx', src_idx_seq)
            tran_idx_seq = batch.tgt.transpose(0, 1)[0]
            print('tgt_idxs', tran_idx_seq)

            res_all.append(ith_dict)

            src_words = self.build_tokens(src_idx_seq, side='src')
            src = ' '.join(src_words)
            ith_dict['src'] = src
            tran_words = self.build_tokens(tran_idx_seq, side='tgt')
            tran = ' '.join(tran_words)

            ith_dict['tgt'] = tran
            ith_dict['attn'] = attn
            
            print("SOURCE: " + src + "\nOUTPUT: " + tran + "\n")
            batch_count += 1
            print("batch: " + str(batch_count) + "...")
            # if batch_count > 20:
            #     break

        if out_file is not None:
            for tran in all_translation:
                out_file.write(tran + '\n')
        print('Decoding took %.1f minutes ...' % (float(time.time() - start_time) / 60.))
        print(res_all)
        json.dump(res_all, open('trans_attn','w'), indent=4)

    def translate_batch(self, batch):
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def beam_decode_step(inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm, pre_align=None):
            ''' Decode and update beam status, and then return active beam idx '''

            # len_dec_seq: i (starting from 0)

            def prepare_beam_dec_seq(inst_dec_beams):
                dec_seq = [b.get_last_target_word() for b in inst_dec_beams if not b.done]
                # print('Dec_seq before', len(dec_seq))
                # if pre_align is not None:
                #     active_idx_lst = [idx for idx, b in enumerate(inst_dec_beams) if not b.done]
                    # print('Active_idx in prepare:', len(active_idx_lst), active_idx_lst)
                # dec_seq: [(beam_size)] * batch_size
                dec_seq = torch.stack(dec_seq).to(self.device)
                # dec_seq: (batch_size, beam_size)
                dec_seq = dec_seq.view(1, -1)
                # dec_seq: (1, batch_size * beam_size)
                return dec_seq

            def predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq, pre_align=None):
                # dec_seq: (1, batch_size * beam_size)
                # print("dec_seq:", dec_seq.size())
                # print("pre_align", pre_align.size())
                dec_output, attns = self.model.decoder(dec_seq, align=pre_align.transpose(0,1) if pre_align is not None else None, step=len_dec_seq)
                align_attn = attns['mean']
                # align_attn_model = attns['std'][:-1].transpose(0, 1)   # batch_size * tgt_len * sec_len
                # mid (batch_size, 1, hid_size)
                # print('mid', mid.size(), mid)
                # exit()
                # dec_output: (1, batch_size * beam_size, hid_size)
                word_prob = self.model.generator(dec_output.squeeze(0))
                # word_prob: (batch_size * beam_size, vocab_size)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                # word_prob: (batch_size, beam_size, vocab_size)

                return word_prob, align_attn

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                select_indices_array = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                        select_indices_array.append(inst_beams[inst_idx].get_current_origin() + inst_position * n_bm)
                if len(select_indices_array) > 0:
                    select_indices = torch.cat(select_indices_array)
                else:
                    select_indices = None
                return active_inst_idx_list, select_indices

            n_active_inst = len(inst_idx_to_position_map)

            # print('beams before', len(inst_dec_beams))
            dec_seq = prepare_beam_dec_seq(inst_dec_beams)
            # dec_seq: (1, batch_size * beam_size)
            # print('inst_dec_beams:', len(inst_dec_beams))
            
            # print('Dec_seq', dec_seq.size())
            # print('Pre_align', pre_align.size())
            
            # exit()
            word_prob, align_attn = predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq, pre_align)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list, select_indices = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            if select_indices is not None:
                assert len(active_inst_idx_list) > 0
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))
                # print("align_attn_ori", align_attn.size())
                align_attn = align_attn.index_select(1, select_indices)
                # print("align_attn_after", align_attn.size())
            # print('active list after', len(active_inst_idx_list), active_inst_idx_list)

            return active_inst_idx_list, align_attn

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def collect_best_hypothesis_and_score(inst_dec_beams):
            hyps, scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                hyp, score = inst_dec_beams[inst_idx].get_best_hypothesis()
                hyps.append(hyp)
                scores.append(score)

            return hyps, scores

        def get_padded_tensor(inp_tensor, active_inst_idx_list, batch_size=30):
            # inp_tensor [1, 'batch' *n_beam, H_size]
            res = []
            inp_tensor = inp_tensor.squeeze(0).view(-1, self.beam_size, inp_tensor.size(2)) # ['batch', n_beam, H_size]
            idx2position_map = {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(active_inst_idx_list)}
            padding = torch.zeros(size=(self.beam_size, inp_tensor.size(2))).float().to(inp_tensor.device)                # [n_beam, H_size]

            for idx in range(batch_size):
                if idx not in idx2position_map:
                    res.append(padding)
                else:
                    res.append(inp_tensor[idx2position_map[idx]])
            return torch.stack(res).to(inp_tensor.device)
        
        with torch.no_grad():
            # -- Encode
            src_seq = make_features(batch, 'src')
            tgt_seq = make_features(batch, 'tgt')

            # src: (seq_len_src, batch_size)
            # print(src_seq.size(),  src_seq) #4*30

            structure = make_features(batch, 'structure')
            structure = structure.transpose(0, 1)
            structure = structure.transpose(1, 2)
            # print(structure.size()) 30*4*4

            src_emb, src_enc, _ = self.model.encoder(src_seq, structure)
            # src_emb: (seq_len_src, batch_size, emb_size)
            # src_end: (seq_len_src, batch_size, hid_size)
            self.model.decoder.init_state(src_seq, src_enc)
            dec_out, attns = self.model.decoder(tgt_seq[:-1],  None)
            attns = torch.nn.functional.softmax(attns['mean'][:-1].transpose(0, 1), -1)         # bsz, tgt_len, src_len

        return attns.cpu().numpy().tolist()