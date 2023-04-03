#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a user in an INMT session."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
import sys
import random
from onmt.utils.arg_checker import check_arguments
from onmt.utils.parse import ArgumentParser
from onmt.constants import SegmentType

def compute_metrics(refs, mouse_actions, word_strokes, character_strokes):
    characters = sum([len(ref) for ref in refs])
    words = sum([len(ref.split()) for ref in refs])
    print('MAR: {0}'.format(round(mouse_actions / characters * 100, 1)))
    print('WSR: {0}'.format(round(word_strokes / words * 100, 1)))
    print('KSR: {0}'.format(round(character_strokes / characters * 100, 1)))


def generate_segment_list(feedback, correction):
    segments = []

    next_correction = None
    correction_index = 0
    if correction != '' and correction != []:
        next_correction = correction[correction_index]

    for segment in feedback:
        if next_correction!=None and next_correction[0] <= segment[0]:
            n_segment = [next_correction[1], next_correction[2]]
            segments.append(n_segment)

            correction_index += 1
            next_correction = correction[correction_index] if correction_index<len(correction) else None

        n_segment = [segment[1], segment[2]]
        segments.append(n_segment)

    if next_correction!=None:
        n_segment = [next_correction[1], next_correction[2]]
        segments.append(n_segment)

    return segments


def get_word_correction_bpe(pos, ref):
    ref_word = ref[pos]
    correction = [pos, [ref_word], SegmentType.GENERIC, pos]

    is_bpe_subword = (ref_word[-2:]=='@@')
    larger_than_reference = (pos+1 >= len(ref))
    while not larger_than_reference and is_bpe_subword:
        pos += 1
        ref_word = ref[pos]
        correction[1].append(ref_word)
        is_bpe_subword = (ref_word[-2:]=='@@')
        larger_than_reference = (pos+1 >= len(ref))
    return correction


def get_correction(character_level, hyp, ref):
    for n, ref_word in enumerate(ref): 

        larger_than_reference = (n >= len(hyp))
        if larger_than_reference:
            if character_level:
                return [n, hyp[:n] + [ref_word[0]], SegmentType.TO_COMPLETE, 0]
            return get_word_correction_bpe(n, ref)

        hyp_and_ref_not_equal = (ref_word != hyp[n])
        if hyp_and_ref_not_equal:
            if not character_level:
                return get_word_correction_bpe(n, ref)

            for m in range(len(ref_word)):
                if m >= len(hyp[n]) or hyp[n][m] != ref_word[m]:
                    chars = ref_word[:m+1]
                    if chars[-1] == '@':
                        chars += '@'
                        return [n, hyp[:n]+[chars]+[ref[n+1][0]], SegmentType.TO_COMPLETE, 0]
                    return [n, hyp[:n] + [chars], SegmentType.TO_COMPLETE, 0]
            return [n, ref, SegmentType.GENERIC, n]
    return ''


def correction_segments(feedback, hyp, ref, character_level=False, n_correction=0):
    hyp_end_last_segment = 0
    ref_end_last_segment = 0

    possible_corrections = []
    last_partial_hyp = None

    for segment in feedback:
        s_pos, s_com, s_typ = segment

        partial_hyp = hyp[ hyp_end_last_segment:s_pos ]
        partial_ref = ref[ ref_end_last_segment: ]

        common_pos = 0
        for i in range(len(partial_ref)):
            if partial_ref[i:i+len(s_com)] == s_com:
                common_pos = i
                break
        partial_ref = partial_ref[ :common_pos ]

        correction = get_correction(character_level, partial_hyp, partial_ref)
        if correction != '' and correction != []:
            correction[0] += hyp_end_last_segment
            correction[3] += ref_end_last_segment
            if possible_corrections:
                if last_partial_hyp == []:
                    possible_corrections = possible_corrections[:-1]
                else:
                    possible_corrections[-1][2] = SegmentType.DIFFERENT
                    possible_corrections[-1][1] = [last_partial_hyp[0]]
            possible_corrections.append(correction)

            last_correction = (len(possible_corrections) > n_correction)
            if last_correction:
                return possible_corrections

        last_partial_hyp = partial_hyp
        hyp_end_last_segment = s_pos+len(s_com)
        ref_end_last_segment += common_pos+len(s_com)

    partial_hyp = hyp[ hyp_end_last_segment: ]
    partial_ref = ref[ ref_end_last_segment: ]

    correction = get_correction(character_level, partial_hyp, partial_ref)
    if correction != '' and correction!=[]:
        correction[0] += hyp_end_last_segment
        correction[3] += ref_end_last_segment
        if possible_corrections:
                if last_partial_hyp == []:
                    possible_corrections = possible_corrections[:-1]
                else:
                    possible_corrections[-1][2] = SegmentType.DIFFERENT
                    possible_corrections[-1][1] = [last_partial_hyp[0]]
        possible_corrections.append(correction)
    return possible_corrections


def remove_bpe_from_list(sentence):
    noBPE_sentence = []
    codes_sentence = {}

    pos = 0
    while pos < len(sentence):
        last_token_was_subword = (pos>0 and noBPE_sentence[-1][-2:]=='@@')
        if last_token_was_subword:
            noBPE_sentence[-1] = noBPE_sentence[-1][:-2] + sentence[pos]
            codes_sentence[len(noBPE_sentence)-1][1] = pos
        else:
            noBPE_sentence.append(sentence[pos])
            codes_sentence[len(noBPE_sentence)-1] = [pos, pos]
        pos += 1

    return noBPE_sentence, codes_sentence


def longest_common_substring(s1, s2):
    noBPE_s1, codes_s1 = remove_bpe_from_list(s1)
    noBPE_s2, codes_s2 = remove_bpe_from_list(s2)

    m = [[0] * (1 + len(noBPE_s2)) for _ in range(1 + len(noBPE_s1))]
    longest, x_longest, y_longest = 0, 0, 0
    for x in range(1, 1 + len(noBPE_s1)):
        for y in range(1, 1 + len(noBPE_s2)):
            if noBPE_s1[x - 1] == noBPE_s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                    y_longest = y
            else:
                m[x][y] = 0


    s1_longest = 0 if x_longest==0 else codes_s1[x_longest-1][1]+1
    s2_longest = 0 if y_longest==0 else codes_s2[y_longest-1][1]+1
    ss_longest = 0 if longest==0 else codes_s1[x_longest-1][1]+1 - codes_s1[x_longest-longest][0]

    return (s1[s1_longest - ss_longest: s1_longest], s1_longest - ss_longest,
            s2_longest - ss_longest)


def get_segments(s1, s2, s1_offset=0, s2_offset=0):
    if s1 == [] or s2 == []:
        return [], []

    com, s1_start, s2_start = longest_common_substring(s1, s2)
    len_common = len(com)
    if len_common == 0:
        return [], []

    s1_before = s1[:s1_start]
    s2_before = s2[:s2_start]
    s1_after = s1[s1_start+len_common:]
    s2_after = s2[s2_start+len_common:]
    before = get_segments(s1_before, s2_before, s1_offset, s2_offset)
    after = get_segments(s1_after, s2_after, s1_offset+s1_start+len_common,
                         s2_offset+s2_start+len_common)

    return (before[0] + [[s1_offset+s1_start, com, SegmentType.GENERIC]] + after[0],
            before[1] + [[s2_offset+s2_start, com, SegmentType.GENERIC]] + after[1])


def merge_segments(feedback, correction, s1, s2):
    times = 0
    if feedback:
        s2_last_pos = len(feedback[0][1])
        previous = feedback[0]

        for segment in feedback[1:]:
            s_pos, s_com, s_typ = segment

            if s_typ == SegmentType.TO_COMPLETE or previous[2] == SegmentType.TO_COMPLETE:
                previous = segment
                continue

            common_pos = 0
            for i in range(s2_last_pos, len(s2)):
                if s2[i:i+len(s_com)] == s_com:
                    common_pos = i
                    break

            merged_segments = previous[1] + s_com
            refered_version = s2[common_pos-len(previous[1]) : common_pos+len(s_com)]
            if merged_segments == refered_version:
                
                len_diff = s_pos - (previous[0] + len(previous[1]))
                previous[1] = merged_segments
                pos = feedback.index(segment)
                for x in feedback[pos+1:]:
                    x[0]-=len_diff
                feedback.pop(pos)
                times += 1
            else:
                previous = segment
                s2_last_pos = common_pos+len(s_com)

        if feedback[0][0] != 0 and s2[:len(feedback[0][1])] == feedback[0][1]:
            if not correction or correction=='' or correction[0]>feedback[0][0]:                
                feedback[0][0] = 0
                feedback[0][2] = SegmentType.PREFIX
                times += 1

    return feedback, times


def new_segments(feedback, s1, s2):
    new_feedback = []
    new_segments = []
    s1_list, s2_list = [0], [0]
    for idx, segment in enumerate(feedback):
        s_pos, s_com, s_typ = segment

        common_pos = 0
        prt_s2 = s2[s2_list[-1]:]
        for i in range(len(prt_s2)):
            if prt_s2[i:i+len(s_com)] == s_com:
                common_pos = i
                break

        segments, b = get_segments(s1[s1_list[-1]:s_pos], prt_s2[:common_pos], s1_list[-1], s2_list[-1])
        if segments:
            [new_feedback.append(x) for x in segments]
            [new_segments.append(x) for x in segments]
        new_feedback.append(segment)

        s1_list.append(s_pos+len(s_com))
        s2_list.append(s2_list[-1]+common_pos+len(s_com))

    segments, _ = get_segments(s1[s1_list[-1]:], s2[s2_list[-1]:], s1_list[-1], s2_list[-1])
    if segments:
        [new_feedback.append(x) for x in segments]

    return new_feedback, new_segments


def expand_one_segment(s1, s2, segment, s1_offset=0, s2_offset=0):
    """
    longest common substring where s3 is present
    """
    s_pos, s_com, s_typ = segment
    s_pos -= s1_offset
    common_pos = 0
    for i in range(len(s2)):
        if s2[i:i+len(s_com)] == s_com:
            common_pos = i
            break

    prefix = s1[:s_pos]
    for i in range(1, len(prefix)+1):
        offset_1 = s_pos-1
        offset_2 = common_pos-1

        if offset_1<0 or offset_2<0:
            break
        if s1[offset_1]!=s2[offset_2]:
            break

        s_com = [s1[offset_1]] + s_com
        s_pos -= 1
        common_pos -= 1

    suffix = s1[s_pos+len(s_com):]
    for i in range(len(suffix)):
        offset_1 = s_pos + len(s_com)
        offset_2 = common_pos + len(s_com)

        if offset_1>=len(s1) or offset_2>=len(s2):
            break
        if s1[offset_1]!=s2[offset_2]:
            break

        s_com += [s1[offset_1]]

    return [s_pos+s1_offset, s_com, s_typ], s_pos+len(s_com)+s1_offset, common_pos+len(s_com)+s2_offset


def expand_segments(feedback, s1, s2):
    s1_list, s2_list = [0], [0]
    for idx, segment in enumerate(feedback):
        last_pos = feedback[idx+1][0] if idx < len(feedback)-1 else len(s1)
        segment, last_s1, last_s2 = expand_one_segment(s1[s1_list[-1]:last_pos], s2[s2_list[-1]:], segment, s1_list[-1], s2_list[-1])

        times = times if not changes else times+1
        feedback[idx] = segment
        s1_list.append(last_s1)
        s2_list.append(last_s2)

    return feedback


def compute_mouse_actions(segments):
    actions = 0
    for segment in segments:
        actions += 2 if len(segment[-1]) > 1 else 1
    return actions


def segment_based_simulation(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser.validate_simulate_opts(opt)
    ArgumentParser.validate_inmt_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    with open(opt.src, "rb") as f:
        srcs = f.readlines()
    with open(opt.tgt, "rb") as f:
        refs = f.readlines()

    print(opt.max_length)

    total_mouse_actions = 0
    total_word_strokes = 0
    total_character_strokes = 0

    for n in range(len(srcs)):
        #if n<429:
        #    continue
        #if n>429:
        #    sys.exit()

        logger.info("Processing sentence %d." % n)
        src = srcs[n]
        ref = refs[n].decode('utf-8').strip()

        translator.prefix = None
        translator.out_segments = []
        score, hyp = translator.translate(src=[src], batch_size=1)

        eos = False

        if opt.inmt_verbose:
            print("Source: {0}".format(src.decode('utf-8').strip()
                                       .replace('@@ ', '')))
            print("Reference: {0}".format(ref.replace('@@ ', '')))
            print("Initial hypothesis: {0}".format(hyp[0][0].replace('@@ ', '')))
            #print("Reference: {0}".format(ref))
            #print("Initial hypothesis: {0}".format(hyp[0][0]))
            print()

        cont = 1
        mouse_actions = 0
        word_strokes = 0
        character_strokes = 0

        feedback = []
        correction = []
        c_correction = None
        while hyp[0][0] != ref and not eos:
            mouse_actions_ = 0


            # 1) Si no hay segmentos guardados el usuario hace una primera seleccion
            if not feedback:
                feedback, _ = get_segments(hyp[0][0].split(), ref.split())
                mouse_actions_ += compute_mouse_actions(feedback)
            else:
                correction = [x for x in feedback if x[2]==SegmentType.TO_COMPLETE]
                correction = correction[0] if len(correction)>0 else correction
            
            # 2) Comprobamos si la corrección ha finalizado
            if correction:
                # 2.1) Se ha generado la palabra que queriamos
                if hyp[0][0].split()[correction[0]:correction[0]+len(c_correction)] == c_correction:
                    correction[1] = c_correction
                    correction[2] = SegmentType.GENERIC
                    mouse_actions_ += 1
                    correction = []
                    c_correction = None

                # 2.2) La palabra que queriamos rellenar se encuentra entre dos segmentos
                else:
                    pos_1 = 0
                    for segment in feedback:
                        pos_2 = segment[0]
                        if correction[0] >= pos_1 and correction[0] < pos_2:
                            if c_correction in hyp[0][0].split()[pos_1:pos_2]:
                                correction = []
                                c_correction = None
                            break
                        pos_1 = pos_2 + len(segment[1])

            # 3) El usuario añade nuevos segmentos que se hayan podido crear            
            if not correction:
                feedback, new_ones = new_segments(feedback, hyp[0][0].split(), ref.split())
                mouse_actions_ += compute_mouse_actions(new_ones)

            # 4) El usuario fusiona segmentos
            if not correction:
                feedback, times = merge_segments(feedback, correction, hyp[0][0].split(), ref.split())
                mouse_actions_ += times*2
            feedback   = [x for x in feedback if x[2]!=SegmentType.TO_COMPLETE]

            # 5) El usuario realiza la correccion
            if not correction:
                correction = correction_segments(feedback, hyp[0][0].split(), ref.split(), opt.character_level)
                correction_pos = 0
                c_correction = None

                some_correction = (correction!='' and correction!=[])
                if some_correction:
                    ref_list = ref.split()
                    correction_pos = correction[-1][3]
                    c_correction = [ ref_list[correction_pos] ]
                
                    correction_has_bpe = (c_correction[-1][-2:]=='@@')
                    while correction_has_bpe:
                        correction_pos = correction_pos+1
                        c_correction.append(ref_list[correction_pos])
                        correction_has_bpe = (correction_pos+1<len(ref_list) and c_correction[-1][-2:]=='@@')

            else:
                pos = correction[0]
                len_previous_correction = len(''.join(correction[1]).replace('@@',''))
                correction = get_correction(opt.character_level, correction[1], c_correction)
                len_current_correction = len(''.join(correction[1]).replace('@@',''))
                correction[0] = pos
                if len_current_correction > len_previous_correction +1:
                    mouse_actions_ += 1
                correction = [correction]

            word_strokes_ = 1
            character_strokes_ = 1             
            some_correction_performed = not (correction=='' or not correction)
            if some_correction_performed and not opt.character_level:
                character_strokes_ = len(''.join(correction[-1][1]).replace('@@', ''))

            segment_list = generate_segment_list(feedback, correction)
            if correction == []:  # End of sentence needed.
                correction = 'EoS'
                eos = True

            score, hyp = translator.segment_based_inmt(
                src=[src],
                segment_list=segment_list
                )
            feedback = translator.get_segments()

            if opt.inmt_verbose:
                print("Segments: {0}".format(' || '.join([' '.join(segment[0]).replace('@@ ', '') for segment in segment_list])))
                print("Correction: {0}".format(''.join(correction[-1][1]).replace('@@', '') if len(correction) > 0 and isinstance(correction[-1], list) else correction ))
                print("Reference: {0}".format(ref.replace('@@ ', '')))
                print("Hypothesis {1}: {0}".format(hyp[0][0].replace('@@ ', ''), cont))
                #print("Segments: {0}".format(' || '.join([' '.join(segment[0]) for segment in segment_list])))
                #print("Correction: {0}".format(''.join(correction[1]) if isinstance(correction, list) else correction))
                #print("Reference: {0}".format(ref))
                #print("Hypothesis {1}: {0}".format(hyp[0][0], cont))
                print('~~~~~~~~~~~~~~~~~~')
                print('Mouse actions: {0}'.format(mouse_actions_))
                print('Word strokes: {0}'.format(word_strokes_))
                print('Character strokes: {0}'.format(character_strokes_))
                print()
            cont += 1
            mouse_actions += mouse_actions_
            word_strokes += word_strokes_
            character_strokes += character_strokes_

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


def character_level_prefix_based_simulation(opt):
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

            mouse_actions_ = (1 if len(feedback) != len(old_feedback)+1 else 0)
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


def prefix_based_simulation(opt):
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

            word_strokes_ = 1
            mouse_actions_ = (1 if len(feedback) != len(old_feedback) + 1
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
    opt = check_arguments(opt)

    if opt.segment:
        segment_based_simulation(opt)

    elif opt.character_level:
        character_level_prefix_based_simulation(opt)
    else:
        prefix_based_simulation(opt)


if __name__ == "__main__":
    main()
