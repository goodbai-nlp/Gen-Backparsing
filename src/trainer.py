"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from utils.loss import build_loss_compute
from utils.logging import logger
from utils.report_manager import build_report_manager
from utils.statistics import Statistics
from utils.distributed import all_gather_list, all_reduce_and_rescale_tensors
from inputters.dataset import make_features
import torch.nn as nn
import torch
import json
import numpy as np
import torch.nn.functional as F


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = [build_loss_compute(model, fields["tgt"].vocab, opt), nn.NLLLoss(reduction="sum")]
    valid_loss = build_loss_compute(model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = opt.world_size
    integrated = opt.integrated
    integrated_mode = opt.integrated_mode
    attn_smoothing = opt.attn_smoothing
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = build_report_manager(opt)
    trainer = Trainer(
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size,
        shard_size,
        norm_method,
        grad_accum_count,
        n_gpu,
        gpu_rank,
        gpu_verbose_level,
        report_manager,
        model_saver=model_saver,
        integrated=integrated,
        integrated_mode=integrated_mode,
        attn_smoothing=attn_smoothing,
        use_0=opt.use_0,
        use_gumbel=opt.use_gumbel,
    )
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
        model(:py:class:`onmt.models.model.NMTModel`): translation model
            to train
        train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
           training loss computation
        valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
           training loss computation
        optim(:obj:`onmt.utils.optimizers.Optimizer`):
           the optimizer responsible for update
        trunc_size(int): length of truncated back propagation through time
        shard_size(int): compute loss in shards of this size for efficiency
        norm_method(string): normalization methods: [sents|tokens]
        grad_accum_count(int): accumulate gradients this many times.
        report_manager(:obj:`onmt.utils.ReportMgrBase`):
            the object that creates reports, or None
        model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
            used to save a checkpoint.
            Thus nothing will be saved if this parameter is None
    """

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        optim,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        grad_accum_count=1,
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        model_saver=None,
        integrated=False,
        integrated_mode="add",
        attn_smoothing="-1e5",
        use_0=False,
        use_gumbel=False,
    ):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss[0]
        self.train_relation_loss = train_loss[1]
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.integrated = integrated
        self.attn_smoothing = attn_smoothing
        self.use_0 = use_0
        self.use_gumbel = use_gumbel

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert (
                self.trunc_size == 0
            ), """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(
        self,
        train_iter_fct,
        valid_iter_fct,
        train_steps,
        valid_steps,
        ratio_alpha,
        ratio_beta,
        ratio_nmt,
        R2S=None,
    ):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info("Start training...")
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        normalization_relation = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        loss_all = []
        while step <= train_steps:
            loss_epoch = []
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    if self.gpu_verbose_level > 1:
                        logger.info("GpuRank %d: index: %d accum: %d" % (self.gpu_rank, i, accum))

                    # print('batch', batch)
                    # print('src', batch.src[0].size(), batch.src[0])
                    # print('tgt', batch.tgt.size(), batch.tgt)
                    # print('align_gold', batch.align_gold.size(), batch.align_gold)
                    # print('tgt mask', batch.mask.size(), batch.mask)
                    # print('tgt align', batch.align.size(), batch.align)

                    true_batchs.append(batch)

                    if self.norm_method == "tokens":
                        num_tokens = batch.tgt[1:].ne(self.train_loss.padding_idx).sum()
                        normalization += num_tokens.item()
                        tmp_relation = batch.relation[batch.relation != 1]
                        # print('tmp Norm_relation', tmp_relation.size(0))
                        normalization_relation += tmp_relation.size(0)
                    else:
                        normalization += batch.batch_size
                    # print('Normalization:',normalization)

                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        # print('Normalization_r', normalization_relation)
                        if self.gpu_verbose_level > 0:
                            logger.info(
                                "GpuRank %d: reduce_counter: %d \
                          n_minibatch %d"
                                % (self.gpu_rank, reduce_counter, len(true_batchs))
                            )
                        if self.n_gpu > 1:
                            normalization = sum(all_gather_list(normalization))
                            normalization_relation = sum(all_gather_list(normalization_relation))

                        losses = self._gradient_accumulation(
                            true_batchs,
                            normalization,
                            total_stats,
                            report_stats,
                            ratio_alpha,
                            ratio_beta,
                            ratio_nmt,
                        )

                        loss_epoch.append(losses)

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optim.learning_rate, report_stats
                        )

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        normalization_relation = 0
                        if step % valid_steps == 0:
                            if self.gpu_verbose_level > 0:
                                logger.info("GpuRank %d: validate step %d" % (self.gpu_rank, step))
                            valid_iter = valid_iter_fct()
                            valid_stats = self.validate(valid_iter)
                            if self.gpu_verbose_level > 0:
                                logger.info(
                                    "GpuRank %d: gather valid stat \
                              step %d"
                                    % (self.gpu_rank, step)
                                )
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            if self.gpu_verbose_level > 0:
                                logger.info(
                                    "GpuRank %d: report stat step %d" % (self.gpu_rank, step)
                                )
                            self._report_step(
                                self.optim.learning_rate, step, valid_stats=valid_stats
                            )

                        if self.gpu_rank == 0:
                            self._maybe_save(step)
                        step += 1
                        if step > train_steps:
                            break
            if self.gpu_verbose_level > 0:
                logger.info(
                    "GpuRank %d: we completed an epoch \
                    at step %d"
                    % (self.gpu_rank, step)
                )
            train_iter = train_iter_fct()

            loss_all.append(loss_epoch)

        with open(self.model_saver.base_path + "losses.json", "w") as f:
            json.dump(loss_all, f, indent=4)

        return total_stats

    def validate(self, valid_iter, R2S=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            src = make_features(batch, "src")
            _, src_lengths = batch.src

            tgt = make_features(batch, "tgt")

            structure = make_features(batch, "structure")
            structure = structure.transpose(0, 1)
            structure = structure.transpose(1, 2)

            align = make_features(batch, "align")
            align_mask = align - 2
            align[align <= 0] = -1e10  # can be tuned
            softmax = torch.nn.Softmax(dim=-1)
            align = softmax(align.float())

            # F-prop through the model.
            # outputs, attns = self.model(src, tgt, align_gold, structure, None, src_lengths)
            # if not self.integrated:
            #     align = None

            outputs, attns = self.model(
                src, tgt, structure, None, None, src_lengths, self.use_0, None, None
            )

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(
        self,
        true_batchs,
        normalization,
        total_stats,
        report_stats,
        ratio_alpha,
        ratio_beta,
        ratio_nmt,
    ):
        if self.grad_accum_count > 1:
            self.model.zero_grad()
        loss1, loss2, loss3, loss_total = 0, 0, 0, 0
        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            # dec_state = None
            src = make_features(batch, "src")
            _, src_lengths = batch.src
            # print('true src', src.size(), src)

            tgt_outer = make_features(batch, "tgt")
            # print('true tgt', tgt_outer.size())

            structure = make_features(batch, "structure")
            structure = structure.transpose(0, 1)
            structure = structure.transpose(1, 2)

            # print('True structure', structure.size())

            # bad code
            mask = make_features(batch, "mask")
            # print('Ori mask', mask)
            mask = mask - 2
            mask[mask <= 0] = 0

            arc_attn = torch.nn.functional.normalize(
                mask.float(), p=1, dim=-1
            )  # [bsz, tgt_len, tgt_len]
            mask = mask.byte()
            # print('new mask', mask)

            # ground truth label of biaffine relation
            relation = make_features(batch, "relation")  # [K, bsz]
            relation = relation.transpose(0, 1)
            relation = relation[relation != 1]  # padding
            # print('True relation', relation.size(), relation)

            relation_mat = make_features(batch, "relation2")  # [bsz, tgt_len , tgt_len]
            # print('relation mat', relation_mat)

            # ground truth of tgt2src attn
            align = make_features(batch, "align")
            align_mask = align - 2

            """ soft attn
            # align[align<=0] = -1e10     # can be tuned
            # softmax = torch.nn.Softmax(dim=-1)
            # align = softmax(align.float())
            """

            align[align < 0] = 0
            align = torch.nn.functional.normalize(align.float(), p=1, dim=-1)

            for j in range(0, target_size - 1, trunc_size):  #
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # print('truncated tgt,', tgt.size(), tgt)

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                if not self.integrated:
                    # print('Non-Integrated mode...')
                    align_gold, arc_attn, relation_mat = None, None, None
                    outputs, attns, p, rels = self.model(
                        src,
                        tgt,
                        structure,
                        mask,
                        align_gold,
                        src_lengths,
                        self.use_0,
                        arc_attn,
                        relation_mat,
                    )
                else:
                    # print('Integrated mode...')
                    align_inp = align 
                    gold_arc_attn = arc_attn  # [bsz, tgt_len, tgt_len]
                    outputs, attns, p, rels = self.model(
                        src,
                        tgt,
                        structure,
                        mask,
                        align if ratio_beta > 0 else None,
                        src_lengths,
                        self.use_0,
                        arc_attn if ratio_alpha > 0 else None,
                        relation_mat if ratio_alpha > 0 else None,
                    )

                # 3. Compute loss in shards for memory efficiency.
                batch_stats, nmt_loss = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j, trunc_size, self.shard_size, normalization, ratio_nmt
                )

                # exit()
                if relation.size(0) > 0 and ratio_alpha > 0:
                    # compute loss for label prediction

                    out = F.log_softmax(p, 2)  # [batch_size, seq_len, seq_len]
                    out = out[mask]  # [k]
                    p = out.sum()  # [1]
                    l_out = rels[mask]  # [k, Label_vocab]
                    rels = F.log_softmax(l_out, -1)  # [k, Label_vocab]

                    relation_loss = self.train_relation_loss(rels, relation)
                    # total loss of biaffine module
                    relation_loss = (-p + relation_loss) / relation.size(0)
                    relation_loss = relation_loss * ratio_alpha

                    # loss = relation_loss
                    # print('biaffine label loss', loss)
                if align.size(0) > 0 and ratio_beta > 0:
                    align_attn_model = attns["mean"][:-1].transpose(
                        0, 1
                    )  # batch_size * tgt_len * sec_len
                    # align_attn_model = attns['std'][:-1].transpose(0, 1)                  # batch_size * tgt_len * sec_len
                    if int(self.attn_smoothing) != 0:
                        align_attn_model[
                            align_mask == -1
                        ] = self.attn_smoothing  # Ignore padding value
                    attn_loss = my_cross_entropy(
                        align_attn_model, align, reduce="sum", use_onehot=False
                    )
                    attn_loss = attn_loss * ratio_beta / normalization

                if ratio_alpha > 0 and ratio_beta > 0:
                    loss = attn_loss + relation_loss
                if ratio_alpha > 0 and not ratio_beta > 0:
                    loss = relation_loss
                if not ratio_alpha > 0 and ratio_beta > 0:
                    loss = attn_loss
                if ratio_alpha > 0 or ratio_beta > 0:
                    loss.backward()

                loss1 += nmt_loss.item()
                loss2 += relation_loss.item() if ratio_alpha > 0 else 0
                loss3 += attn_loss.item() if ratio_beta > 0 else 0
                loss_total += loss1 + loss2 + loss3

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [
                            p.grad.data
                            for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        all_reduce_and_rescale_tensors(grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

        return (loss1, loss2, loss3, loss_total)

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)


def one_hot(label, n_class):
    y_one_hot = torch.zeros(label.size(0), label.size(1), n_class).scatter_(2, label, 1)
    return y_one_hot


def my_cross_entropy(input, target, reduce="mean", use_onehot=False):
    """
    @input: bsz * seq_len * n_class
    @target: bsz * seq_len if use_onehot=True
    @target: bsz * seq_len * n_class if use_onehot=False
    """
    if use_onehot:
        tgt_vec = one_hot(target, input.size(-1))
    else:
        tgt_vec = target

    loss_func = torch.nn.LogSoftmax(dim=-1)
    nll_input = loss_func(input)

    # attn_loss_cross_entropy = torch.mean(torch.sum(-(tgt_vec*nll_input), dim=-1), dim=-1)       # 句子内部是mean
    attn_loss_cross_entropy = torch.sum(
        torch.sum(-(tgt_vec * nll_input), dim=-1), dim=-1
    )  # 句子内部是sum

    if reduce == "mean":  # batch 内部
        attn_loss = torch.mean(attn_loss_cross_entropy, dim=0)
        return attn_loss
    elif reduce == "sum":
        attn_loss = torch.sum(attn_loss_cross_entropy, dim=0)
        return attn_loss
