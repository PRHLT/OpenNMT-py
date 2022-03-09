#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a user in an INMT session."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def get_prefix(hyp, ref):
    prefix = []
    for n in range(min(len(hyp), len(ref))):
        prefix.append(ref[n])
        if hyp[n] != ref[n]:
            break
    return ' '.join(prefix)


def simulate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser.validate_simulate_opts(opt)
    ArgumentParser.validate_inmt_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    with open(opt.src, "rb") as f:
        srcs = f.readlines()
    with open(opt.tgt, "rb") as f:
        refs = f.readlines()

    for n in range(len(srcs)):
        logger.info("Processing sentence %d." % n)
        src = srcs[n]
        ref = refs[n]
        score, hyp = translator.translate(src=[src], batch_size=1)

        while hyp[0] != ref:
            feedback = get_prefix(hyp[0], ref)
            score, hyp = translator.prefix_based_inmt(
                src=[src],
                prefix=[feedback]
                )


def _get_parser():
    parser = ArgumentParser(description='simulate.py')

    opts.config_opts(parser)
    opts.inmt_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    simulate(opt)


if __name__ == "__main__":
    main()
