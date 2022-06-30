#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""World-level autocompleter based on Segment-based INMT."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import json


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


def parse_sentences(document):
    doc = []
    with open(document, "rb") as f:
        sentences = f.readlines()
    for sentence in sentences:
        doc.append(json.loads(sentence))
    return doc


def word_level_autocompletion(opt):
    ArgumentParser.validate_autocomplete_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    sentences = parse_sentences(opt.document)
    words = open(opt.predictions, 'w')
    matches = 0

    bpe = None if opt.bpe is None else init_bpe(opt.bpe, opt.bpe_separator)

    for n in range(len(sentences)):
        logger.info("Processing sentence %d." % n)
        src = (sentences[n]['src'] if bpe is None
               else bpe.process_line(sentences[n]['src']))
        left_context = (sentences[n]['left_context'] if bpe is None
                        else bpe.process_line(sentences[n]['left_context']))
        right_context = (sentences[n]['right_context'] if bpe is None
                         else bpe.process_line(sentences[n]['right_context']))
        typed_seq = (sentences[n]['typed_seq'] if bpe is None
                     else bpe.process_line(sentences[n]['typed_seq']))
        completion = translator.word_level_autocompletion(
            src=[src],
            left_context=left_context,
            right_context=right_context,
            typed_seq=typed_seq
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

    if 'target' in sentences[0].keys():
        print(f'Acc: {matches / len(sentences):.1f}')


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
