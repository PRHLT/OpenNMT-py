#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""World-level autocompleter based on Segment-based INMT."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import json


def load_alignments(alignments_path):
    """
    Creates a dictionary with the word alignments between source and 
    target. The function receives the path to a file containing the 
    alignments generated using `tools/alignments.sh`.
    """
    alignments = {}

    for line in open(alignments_path):
        src, tgt, prob = tuple(line.split())
        if src in alignments:
            alignments[src][tgt] = float(prob)
        else:
            alignments[src] = {tgt: float(prob)}
    return alignments


def zero_context_autocompletion(src, seq, alignments):
    """
    Computes zero-context autocompletion using an alignment model.
    Params:
        src (str): Source sentence.
        seq (str): Sequence to autocomplete.
        alignments (dict): Alignments.
    """
    words = {}
    n = len(seq)
    for s in src.split():
        if s in alignments:
            for w, p in alignments[s].items():
                if w[:n] == seq:
                    if w in words:
                        if p > words[w]:
                            words[w] = p
                    else:
                        words[w] = p
    
    completion = seq
    prob = 0
    for w, p in words.items():
        if p > prob:
            prob = p
            completion = w
    return completion




def init_bpe(codes, separator):
    try:
        from subword_nmt import apply_bpe
        bpe_parser = apply_bpe.create_parser()
        bpe_args = bpe_parser.parse_args([
            '--codes', codes,
            '--separator', separator,
        ])
        return apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
    except ImportError:
        raise ImportError('subword_nmt requirement not satisfied.')


def word_level_autocompletion(opt):
    ArgumentParser.validate_autocomplete_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    sentences = json.load(open(opt.document))
    words = open(opt.predictions, 'w')
    matches = 0

    bpe = None if opt.bpe is None else init_bpe(opt.bpe, opt.bpe_separator)

    if opt.alignments is not None:
        alignments = load_alignments(opt.alignments)

    for n in range(len(sentences)):
        logger.info("Processing sentence %d." % n)
        if (opt.alignments is not None 
            and sentences[n]['context_type'] == 'zero_context'):
            completion = zero_context_autocompletion(sentences[n]['src'], 
                                                     sentences[n]['typed_seq'], 
                                                     alignments)
        else:
            completion = translator.word_level_autocompletion(
                src=sentences[n]['src'],
                left_context=sentences[n]['left_context'],
                right_context=sentences[n]['right_context'],
                typed_seq=sentences[n]['typed_seq'],
                bpe=bpe
                )
            if bpe is not None:
                completion = completion.replace(opt.bpe_separator + ' ',
                                                '').rstrip()
                completion = completion.replace(opt.bpe_separator, '').rstrip()
        words.write(completion + '\n')
        try:
            if completion == sentences[n]['target']:
                matches += 1
        except KeyError:
            pass
        if opt.wlac is not None:
            sentences[n]['target'] = completion

    if 'target' in sentences[0].keys():
        print(f'Acc: {matches / len(sentences):.3f}')

    if opt.wlac is not None:
        json.dump(sentences, open(opt.wlac, 'w'), indent=3)


def _get_parser():
    parser = ArgumentParser(description='autocomplete.py')

    opts.config_opts(parser)
    opts.autocomplete_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    word_level_autocompletion(opt)


if __name__ == "__main__":
    main()
