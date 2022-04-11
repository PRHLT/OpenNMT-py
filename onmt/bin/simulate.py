#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a user in an INMT session."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def get_prefix(hyp, ref):
    prefix = []
    correction = False
    n = 0
    while not correction and n < len(ref):
        prefix.append(ref[n])
        if n >= len(hyp) or (hyp[n] != ref[n] and ref[n][-2:] != '@@'):
            correction = True
        n += 1
    return ' '.join(prefix), correction


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
        ref = refs[n].decode('utf-8').strip()
        score, hyp = translator.translate(src=[src], batch_size=1)

        if opt.inmt_verbose:
            print("Source: {0}".format(src.decode('utf-8').strip()
                                       .replace('@@ ', '')))
            print("Reference: {0}".format(ref.replace('@@ ', '')))
            print("Initial hypothesis: {0}".format(hyp[0][0]
                                                   .replace('@@ ', '')))
            print()

        cont = 1
        while hyp[0][0] != ref:
            feedback, correction = get_prefix(hyp[0][0].split(), ref.split())
            if not correction:  # End of sentence needed.
                break
            score, hyp = translator.prefix_based_inmt(
                src=[src],
                prefix=[feedback]
                )
            if opt.inmt_verbose:
                print("Prefix: {0}".format(feedback))
                print("Hypothesis {1}: {0}"
                      .format(hyp[0][0].replace('@@ ', ''), cont))
                print()
            cont += 1

        if opt.inmt_verbose:
            print('-------------------------------------------\n')

    print('Metric calculation to be implemented soon.')


def _get_parser():
    parser = ArgumentParser(description='simulate.py')

    opts.config_opts(parser)
    opts.inmt_opts(parser)
    opts.inmt_simulation_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    simulate(opt)


if __name__ == "__main__":
    main()
