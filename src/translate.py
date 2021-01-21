#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import codecs

from utils.logging import init_logger
from inputters.dataset import make_text_iterator_from_file
import onmt.opts as opts
from onmt.translator2 import build_translator
import json
import torch
import numpy as np


def main(opt):

    relation2structure = json.load(open(opt.data + "_R2S", "r"))
    R2S_idx_lst = [relation2structure[str(i)] for i in range(len(relation2structure.keys()))]
    device = torch.device("cuda:0")
    R2S_idx_Tensor = torch.from_numpy(np.array(R2S_idx_lst)).long().to(device)

    translator = build_translator(opt)
    out_file = codecs.open(opt.output, "w+", "utf-8")

    src_iter = make_text_iterator_from_file(opt.src)

    if opt.tgt is not None:
        tgt_iter = make_text_iterator_from_file(opt.tgt)
    else:
        tgt_iter = None

    if opt.structure is not None:
        structure_iter = make_text_iterator_from_file(opt.structure)
    else:
        structure_iter = None

    translator.translate(
        src_data_iter=src_iter,
        tgt_data_iter=tgt_iter,
        structure_iter=structure_iter,
        batch_size=opt.batch_size,
        out_file=out_file,
        R2S=R2S_idx_Tensor,
    )
    out_file.close()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="translate.py",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    logger.info("Input args: %r", opt)
    main(opt)

