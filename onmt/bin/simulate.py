#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a user in an INMT session."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def compute_metrics(refs, mouse_actions, word_strokes, character_strokes):
    characters = sum([len(ref) for ref in refs])
    words = sum([len(ref.split()) for ref in refs])
    print('MAR: {0}'.format(round(mouse_actions / characters * 100, 1)))
    print('WSR: {0}'.format(round(word_strokes / words * 100, 1)))
    print('KSR: {0}'.format(round(character_strokes / characters * 100, 1)))


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest, y_longest = 0, 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                    y_longest = y
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest], x_longest - longest, y_longest - longest


def find_isles(s1, s2, s1_offset=0, s2_offset=0):
    if s1 == [] or s2 == []:
        return [], []
    
    com, s1_start, s2_start = longest_common_substring(s1, s2)
    len_common = len(com)
    if len_common == 0:
        return [], []

    s1_before = s1[:s1_start]
    s2_before = s2[:s2_start]
    s1_after  = s1[s1_start+len_common:]
    s2_after  = s2[s2_start+len_common:]
    before = find_isles(s1_before, s2_before, s1_offset, s2_offset) 
    after  = find_isles(s1_after, s2_after, s1_offset+s1_start+len_common, s2_offset+s2_start+len_common)

    return ( before[0] + [s1_offset+s1_start, com] + after[0],
             before[1] + [s2_offset+s2_start, com] + after[1])


def get_character_level_corrections_prefix(hyp, ref):
    prefix = []
    correction = ''
    n = 0
    while correction == '' and n < len(ref):
        if n >= len(hyp):
            correction += ref[n][0]
            break
        elif hyp[n] != ref[n]:
            for i in range(len(hyp[n])):
                # La referencia es mas pequenya
                if i >= len(ref[n]):
                    prefix.append(correction)
                    correction = ''
                    break
                correction += ref[n][i]
                # El error esta a mitad
                if ref[n][i] != hyp[n][i]:
                    break
            # La referencia es mas grande
            if hyp[n] == ref[n][:len(hyp[n])] and len(hyp[n]) < len(ref[n]):
                correction += ref[n][len(hyp[n])]
            break
        else:
            prefix.append(ref[n])
        n += 1

    prefix.append(correction)
    return prefix, correction


def character_level_simulate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser.validate_simulate_opts(opt)
    ArgumentParser.validate_inmt_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    with open(opt.src, "rb") as f:
        srcs = f.readlines()
    with open(opt.tgt, "rb") as f:
        refs = f.readlines()

    total_mouse_actions = 0
    total_character_strokes = 0

    for n in range(len(srcs)):
        logger.info("Processing sentence %d." % n)
        src = srcs[n]
        ref = refs[n].decode('utf-8').strip()
        translator.prefix = None
        score, hyp = translator.translate(src=[src], batch_size=1)

        old_feedback = ''
        eos = False

        if opt.inmt_verbose:
            print("Source: {0}".format(src.decode('utf-8').strip()
                                       .replace('@@ ', '')))
            print("Reference: {0}".format(ref.replace('@@ ', '')))
            print("Initial hypothesis: {0}".format(hyp[0][0]
                                                   .replace('@@ ', '')))
            print()

        cont = 1
        mouse_actions = 0
        character_strokes = 0
        while hyp[0][0] != ref and not eos:
            feedback, correction = get_character_level_corrections_prefix(
                hyp[0][0].split(), ref.split())

            mouse_actions_ = (1 if len(feedback) != len(old_feedback)+1
                                else 0)
            character_strokes_ = 1

            if correction == '':  # End of sentence needed.
                correction = 'EoS'
                character_strokes_ = 1
                eos = True
            score, hyp = translator.prefix_based_inmt(
                src=[src],
                prefix=[feedback]
                )
            if opt.inmt_verbose:
                print("Prefix: {0}".format(feedback))
                print("Correction: {0}".format(correction.replace('@@', '')))
                print("Hypothesis {1}: {0}"
                      .format(hyp[0][0].replace('@@ ', ''), cont))
                print("Reference: {0}".format(ref.replace('@@', '')))
                print('~~~~~~~~~~~~~~~~~~')
                print('Mouse actions: {0}'.format(mouse_actions_))
                print('Character strokes: {0}'.format(character_strokes_))
                print()
            cont += 1
            mouse_actions += mouse_actions_
            character_strokes += character_strokes_
            old_feedback = feedback

        if opt.inmt_verbose:
            print('------------------\n')
            print('Total mouse actions: {0}'.format(mouse_actions))
            print('Total character strokes: {0}'.format(character_strokes))
            print()
            print('-------------------------------------------\n')
        total_mouse_actions += mouse_actions
        total_character_strokes += character_strokes

    compute_metrics(refs, total_mouse_actions, 0, total_character_strokes)


def get_prefix(hyp, ref):
    prefix = []
    correction = ''
    n = 0
    while correction == '' and n < len(ref):
        prefix.append(ref[n])
        if n >= len(hyp) or hyp[n] != ref[n]:
            correction += ref[n]
        n += 1
    while prefix[-1][-2:] == '@@':
        prefix.append(ref[n])
        correction += ref[n]
        n += 1

    prefix += ['']
    return prefix, correction


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

    total_mouse_actions = 0
    total_word_strokes = 0
    total_character_strokes = 0

    for n in range(len(srcs)):
        logger.info("Processing sentence %d." % n)
        src = srcs[n]
        ref = refs[n].decode('utf-8').strip()
        translator.prefix = None
        score, hyp = translator.translate(src=[src], batch_size=1)

        old_feedback = ''
        eos = False

        if opt.inmt_verbose:
            print("Source: {0}".format(src.decode('utf-8').strip()
                                       .replace('@@ ', '')))
            print("Reference: {0}".format(ref.replace('@@ ', '')))
            print("Initial hypothesis: {0}".format(hyp[0][0]
                                                   .replace('@@ ', '')))
            print()

        cont = 1
        mouse_actions = 0
        word_strokes = 0
        character_strokes = 0
        while hyp[0][0] != ref and not eos:
            feedback, correction = get_prefix(hyp[0][0].split(), ref.split())
            isles = find_isles(hyp[0][0].split(), ref.split())
            print(isles)

            word_strokes_ = 1
            mouse_actions_ = (1 if len(feedback) != len(old_feedback) +1
                                else 0)
            character_strokes_ = len(correction)

            if correction == '':  # End of sentence needed.
                correction = 'EoS'
                character_strokes_ = 1
                eos = True
            score, hyp = translator.prefix_based_inmt(
                src=[src],
                prefix=[feedback]
                )
            if opt.inmt_verbose:
                print("Prefix: {0}".format(feedback))
                print("Correction: {0}".format(correction.replace('@@', '')))
                print("Hypothesis {1}: {0}"
                      .format(hyp[0][0].replace('@@ ', ''), cont))
                print('~~~~~~~~~~~~~~~~~~')
                print('Mouse actions: {0}'.format(mouse_actions_))
                print('Word strokes: {0}'.format(word_strokes_))
                print('Character strokes: {0}'.format(character_strokes_))
                print()
            cont += 1
            mouse_actions += mouse_actions_
            word_strokes += word_strokes_
            character_strokes += character_strokes_
            old_feedback = feedback

        if opt.inmt_verbose:
            print('------------------\n')
            print('Total mouse actions: {0}'.format(mouse_actions))
            print('Total word strokes: {0}'.format(word_strokes))
            print('Total character strokes: {0}'.format(character_strokes))
            print()
            print('-------------------------------------------\n')
        total_mouse_actions += mouse_actions
        total_word_strokes += word_strokes
        total_character_strokes += character_strokes

    compute_metrics(refs, total_mouse_actions,
                    total_word_strokes, total_character_strokes)


def _get_parser():
    parser = ArgumentParser(description='simulate.py')

    opts.config_opts(parser)
    opts.inmt_opts(parser)
    opts.inmt_simulation_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()

    if opt.character_level:
        character_level_simulate(opt)
    else:
        simulate(opt)


if __name__ == "__main__":
    main()
