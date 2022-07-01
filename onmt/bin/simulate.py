#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a user in an INMT session."""
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.constants import SegmentType


def compute_metrics(refs, mouse_actions, word_strokes, character_strokes):
    characters = sum([len(ref) for ref in refs])
    words = sum([len(ref.split()) for ref in refs])
    print('MAR: {0}'.format(round((mouse_actions / characters) * 100, 1)))
    print('WSR: {0}'.format(round((word_strokes / words) * 100, 1)))
    print('KSR: {0}'.format(round((character_strokes / characters) * 100, 1)))


def generate_segment_list(feedback, correction):
    segments = []
    corrected = True if (correction == '' or not correction) else False

    for segment in feedback:
        if not corrected and correction[0] <= segment[0]:
            n_segment = [correction[1], correction[2]]
            segments.append(n_segment)
            corrected = True
        n_segment = [segment[1], segment[2]]
        segments.append(n_segment)
    if not corrected:
        n_segment = [correction[1], correction[2]]
        segments.append(n_segment)

    return segments


def get_correction(character_level, hyp, ref):
    for n in range(len(ref)):
        if n >= len(hyp):
            return ([n, [ref[n][0]], SegmentType.TO_COMPLETE, n] if character_level
                    else [n, [ref[n]], SegmentType.GENERIC, n])
        if ref[n] != hyp[n]:
            if not character_level:
                return [n, [ref[n]], SegmentType.GENERIC, n]
            for m in range(len(ref[n])):
                if m >= len(hyp[n]) or hyp[n][m] != ref[n][m]:
                    chars = ref[n][:m+1]
                    if chars[-1] == '@':
                        chars += '@'
                        return [n, [chars], SegmentType.GENERIC, n]
                    return [n, [chars], SegmentType.TO_COMPLETE, n]
            return [n, [ref[n]], SegmentType.GENERIC, n]
    return ''


def correction_segments(feedback, s1, s2, character_level=False):
    s1_list, s2_list = [0], [0]
    for segment in feedback:
        s_pos, s_com, s_typ = segment

        common_pos = 0
        prt_s2 = s2[s2_list[-1]:]
        for i in range(len(prt_s2)):
            if prt_s2[i:i+len(s_com)] == s_com:
                common_pos = i
                break

        correction = get_correction(character_level, s1[s1_list[-1]:s_pos], prt_s2[:common_pos])
        if correction != '' and correction != []:
            correction[0] += s1_list[-1]
            correction[3] += s2_list[-1]
            return correction

        s1_list.append(s_pos+len(s_com))
        s2_list.append(s2_list[-1]+common_pos+len(s_com))
    correction = get_correction(character_level, s1[s1_list[-1]:], s2[s2_list[-1]:])
    if correction != '' and correction!=[]:
        correction[0] += s1_list[-1]
        correction[3] += s2_list[-1]
        return correction
    return ''


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
    return (s1[x_longest - longest: x_longest], x_longest - longest,
            y_longest - longest)


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
    if feedback:
        previous = feedback[0]

        for segment in feedback[1:]:
            s_pos, s_com, s_typ = segment

            common_pos = 0
            for i in range(len(s2)):
                if s2[i:i+len(s_com)] == s_com:
                    common_pos = i
                    break

            merged_segments = previous[1] + s_com
            refered_version = s2[common_pos-len(previous[1]):common_pos+len(s_com)]
            if merged_segments == refered_version:
                
                len_diff = s_pos - (previous[0] + len(previous[1]))
                previous[1] = merged_segments
                pos = feedback.index(segment)
                for x in feedback[pos+1:]:
                    x[0]-=len_diff
                feedback.pop(pos)
            else:
                previous = segment

        if feedback[0][0] != 0 and s2[:len(feedback[0][1])] == feedback[0][1]:
            if not correction or correction=='' or correction[0]>feedback[0][0]:                
                feedback[0][0] = 0
                feedback[0][2] = SegmentType.PREFIX

    return feedback


def new_segments(feedback, s1, s2):
    new_feedback = []
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
        new_feedback.append(segment)

        s1_list.append(s_pos+len(s_com))
        s2_list.append(s2_list[-1]+common_pos+len(s_com))

    segments, _ = get_segments(s1[s1_list[-1]:], s2[s2_list[-1]:], s1_list[-1], s2_list[-1])
    if segments:
        [new_feedback.append(x) for x in segments]

    return new_feedback


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

    total_mouse_actions = 0
    total_word_strokes = 0
    total_character_strokes = 0

    try:
        for n in range(len(srcs)):
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
                print("Initial hypothesis: {0}".format(hyp[0][0]
                                                       .replace('@@ ', '')))
                print()

            cont = 1
            mouse_actions = 0
            word_strokes = 0
            character_strokes = 0

            feedback = []
            correction = []
            c_correction = None
            while hyp[0][0] != ref and not eos:
                #print("==================================================================")
                #print(feedback)
                # 1) Si no hay segmentos guardados el usuario hace una primera seleccion
                if not feedback:
                     feedback, _ = get_segments(hyp[0][0].split(), ref.split())
                    #feedback = new_segments(feedback, hyp[0][0].split(), ref.split())
                else:
                    correction = [x for x in feedback if x[2]==SegmentType.TO_COMPLETE]
                    correction = correction[0] if correction else correction
                    feedback   = [x for x in feedback if x[2]!=SegmentType.TO_COMPLETE]

                # 2) Comprobamos si la corrección ha finalizado
                if correction:
                    # 2.1) Se ha generado la palabra que queriamos
                    if correction[1] == [c_correction]:
                        correction[2] = SegmentType.GENERIC
                        added = False
                        for idx, segment in enumerate(feedback):
                            if correction[0]<segment[0]:
                                feedback.insert(idx, correction)
                                added = True
                                break
                        if not added:
                            feedback.append(correction)
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
                #print(correction, c_correction)

                #print("FEEDBACK {}: \n{}".format(cont-1, feedback))
                # 3) Con los segmentos actuales el usuario intenta extender
                feedback = expand_segments(feedback, hyp[0][0].split(), ref.split())
                #print("FIXED {}: \n{}".format(cont-1, feedback))

                # 4) El usuario añade nuevos segmentos que se hayan podido crear
                feedback = new_segments(feedback, hyp[0][0].split(), ref.split())
                #print("ADDED {}: \n{}".format(cont-1, feedback))

                # 5) El usuario fusiona segmentos
                feedback = merge_segments(feedback, correction, hyp[0][0].split(), ref.split())
                #print("MERGED {}: \n{}".format(cont-1, feedback))

                # 6) El usuario realiza la correccion
                if not correction:
                    correction = correction_segments(feedback, hyp[0][0].split(), ref.split(), opt.character_level)
                    c_correction = ref.split()[correction[3]] if correction != '' else None
                else:
                    pos = correction[0]
                    correction = get_correction(opt.character_level, correction[1], [c_correction])
                    correction[0] = pos
                #print(correction, c_correction)

                segment_list = generate_segment_list(feedback, correction)
                #print(segment_list)

                word_strokes_ = 1
                mouse_actions_ = compute_mouse_actions(feedback)
                character_strokes_ = 1 if (correction == '' or not correction) else len(correction[1])

                if correction == '':  # End of sentence needed.
                    correction = 'EoS'
                    eos = True
                score, hyp = translator.segment_based_inmt(
                    src=[src],
                    segment_list=segment_list
                    )
                feedback = translator.get_segments()

                if opt.inmt_verbose:
                    print("Segments: {0}".format(' || '.join([' '.join(segment[0]) 
                                                    for segment in segment_list])))
                    #print("Correction: {0}".format(''.join(correction[1])
                    #                               .replace('@@', '')))
                    #print("Reference: {0}".format(ref.replace('@@ ', '')))
                    #print("Hypothesis {1}: {0}"
                    #      .format(hyp[0][0].replace('@@ ', ''), cont))
                    print("Correction: {0}".format(''.join(correction[1]) if correction else ''))
                    print("Reference: {0}".format(ref))
                    print("Hypothesis {1}: {0}"
                          .format(hyp[0][0], cont))
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
    except KeyboardInterrupt: 
        compute_metrics(refs[:n], total_mouse_actions,
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

    if opt.segment:
        segment_based_simulation(opt)

    elif opt.character_level:
        character_level_prefix_based_simulation(opt)
    else:
        prefix_based_simulation(opt)


if __name__ == "__main__":
    main()
