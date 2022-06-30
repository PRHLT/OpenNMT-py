#!/usr/bin/env python
""" Translator Class and builder """
import codecs
import os
import time
import numpy as np
from itertools import count, zip_longest

import torch
import copy

from onmt.constants import DefaultTokens
import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.inputters.text_dataset import InferenceDataIterator
from onmt.translate.beam_search import BeamSearch, BeamSearchLM, BeamSearchINMT
from onmt.translate.greedy_search import GreedySearch, GreedySearchLM
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.constants import ModelTask, SegmentType


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, "w+", "utf-8")

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    model_opt.model_task = ModelTask.INMT
    if model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        translator = GeneratorLM.from_opt(
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    elif model_opt.model_task == ModelTask.INMT:
        translator = INMTTranslator.from_opt(
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
        translator = Translator.from_opt(
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Inference(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        fields,
        src_reader,
        tgt_reader,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=0,
        random_sampling_topp=0.0,
        random_sampling_temp=1.0,
        stepwise_penalty=None,
        dump_beam=False,
        block_ngram_repeat=0,
        ignore_when_blocking=frozenset(),
        replace_unk=False,
        ban_unk_token=False,
        tgt_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        copy_attn=False,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
        seed=-1,
    ):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = (
            torch.device("cuda", self._gpu)
            if self._use_cuda
            else torch.device("cpu")
        )

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking
        }
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.tgt_prefix = tgt_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if (
            self.global_scorer.has_cov_pen
            and not self.model.decoder.attentional
        ):
            raise ValueError(
                "Coverage penalty requires an attentional decoder."
            )
        self.out_file = out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
        cls,
        model,
        fields,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
    ):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        # TODO: maybe add dynamic part
        cls.validate_task(model_opt.model_task)

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_topp=opt.random_sampling_topp,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            ban_unk_token=opt.ban_unk_token,
            tgt_prefix=opt.tgt_prefix,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed,
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(
        self,
        batch,
        memory_bank,
        src_lengths,
        src_vocabs,
        use_src_map,
        enc_states,
        batch_size,
        src,
    ):
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                src_vocabs,
                batch.src_map if use_src_map else None,
            )
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs

    def translate_dynamic(
        self,
        src,
        transform,
        src_feats={},
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table=""
    ):

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError("Prefix should be feed to tgt if -tgt_prefix.")

        data_iter = InferenceDataIterator(src, tgt, src_feats, transform)

        data = inputters.DynamicDataset(
            self.fields,
            data=data_iter,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
        )

        return self._translate(
            data,
            tgt=tgt,
            batch_size=batch_size,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table,
            dynamic=True,
            transform=transform)

    def translate(
        self,
        src,
        src_feats={},
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_feats: See :func`self.src_reader.read()`.
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError("Prefix should be feed to tgt if -tgt_prefix.")

        src_data = {
            "reader": self.src_reader,
            "data": src,
            "features": src_feats
        }
        tgt_data = {
            "reader": self.tgt_reader,
            "data": tgt,
            "features": {}
        }
        _readers, _data = inputters.Dataset.config(
            [("src", src_data), ("tgt", tgt_data)]
        )

        data = inputters.Dataset(
            self.fields,
            readers=_readers,
            data=_data,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
        )

        return self._translate(
            data,
            tgt=tgt,
            batch_size=batch_size,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table)

    def _translate(
        self,
        data,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        dynamic=False,
        transform=None
    ):

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False,
        )

        xlation_builder = onmt.translate.TranslationBuilder(
            data,
            self.fields,
            self.n_best,
            self.replace_unk,
            tgt,
            self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[: self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [
                    " ".join(pred) for pred in trans.pred_sents[: self.n_best]
                ]
                if self.report_align:
                    align_pharaohs = [
                        build_align_pharaoh(align)
                        for align in trans.word_aligns[: self.n_best]
                    ]
                    n_best_preds_align = [
                        " ".join(align) for align in align_pharaohs
                    ]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(
                            n_best_preds, n_best_preds_align
                        )
                    ]

                if dynamic:
                    n_best_preds = [transform.apply_reverse(x)
                                    for x in n_best_preds]
                all_predictions += [n_best_preds]
                self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if align_debug:
                    tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score(
                "PRED", pred_score_total, pred_words_total
            )
            self._log(msg)
            if tgt is not None:
                msg = self._report_score(
                    "GOLD", gold_score_total, gold_words_total
                )
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log(
                "Average translation time (s): %f"
                % (total_time / len(all_predictions))
            )
            self._log(
                "Tokens per second: %f" % (pred_words_total / total_time)
            )

        if self.dump_beam:
            import json

            json.dump(
                self.translator.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )
        return all_scores, all_predictions

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [
            best.tolist() for bests in predictions for best in bests
        ]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype,
            device=device,
        ).T
        bos_tensor = torch.full(
            [paded_tgt.size(0), 1], bos, dtype=dtype, device=device
        )
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1)
        )  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = "%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name,
                avg_score,
                name,
                ppl,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        src_vocabs,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None,
    ):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(
                dec_out.view(-1, dec_out.size(2)),
                attn.view(-1, attn.size(2)),
                src_map,
            )
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset,
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        return log_probs, attn

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        raise NotImplementedError

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map
    ):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        batch,
        batch_size,
        src,
        src_lengths,
        src_vocabs,
        use_src_map,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions
            )
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results


class Translator(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.SEQ2SEQ:
            raise ValueError(
                f"Translator does not support task {task}."
                f" Tasks supported: {ModelTask.SEQ2SEQ}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        batch_tgt_idxs = self._align_pad_prediction(
            predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx
        )
        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_eos_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(
            dec_in, memory_bank, memory_lengths=src_lengths, with_align=True
        )

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(
            alignment_attn, prediction_mask, src_lengths, n_best
        )
        return alignement

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    batch_size=batch.batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(
                batch, src_vocabs, decode_strategy
            )

    def _run_encoder(self, batch):
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths
        )
        if src_lengths is None:
            assert not isinstance(
                memory_bank, tuple
            ), "Ensemble decoding only supported for text data"
            src_lengths = (
                torch.Tensor(batch.batch_size)
                .type_as(memory_bank)
                .long()
                .fill_(memory_bank.size(0))
            )
        return src, enc_states, memory_bank, src_lengths

    def _translate_batch_with_strategy(
        self, batch, src_vocabs, decode_strategy
    ):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        gold_score = self._gold_score(
            batch,
            memory_bank,
            src_lengths,
            src_vocabs,
            use_src_map,
            enc_states,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        target_prefix = batch.tgt if self.tgt_prefix else None
        (
            fn_map_state,
            memory_bank,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(
            memory_bank, src_lengths, src_map, target_prefix=target_prefix
        )
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
            )

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map
    ):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in,
            memory_bank,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores


class INMTTranslator(Translator):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.phrase_table = {}
        self.last_word_idx = {}
        self.prefix = None

        self.segments = []
        self.out_segments = []

    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.INMT:
            raise ValueError(
                f"Translator does not support task {task}."
                f" Tasks supported: {ModelTask.INMT}"
            )

    def get_segments(self):
        """
        TODO:
        Return the segments
        [POS, COM, TYPE]
        """
        return self.out_segments

    def word_level_autocompletion(
            self,
            src,
            left_context,
            right_context,
            typed_seq,
            src_feats={},
            tgt=None,
            batch_size=1,
            batch_type="sents",
            attn_debug=False,
            align_debug=False,
            phrase_table=""):
        segments = []
        if left_context != '':
            segments.append([left_context.split(), SegmentType.GENERIC])
        segments.append([typed_seq.split(), SegmentType.TO_COMPLETE])
        if right_context != '':
            segments.append([right_context.split(), SegmentType.GENERIC])

        self.segment_based_inmt(src=src, segment_list=segments)
        for segment in self.get_segments():
            if segment[-1] == SegmentType.TO_COMPLETE:
                return ' '.join(segment[1])

        return ''

    def segment_based_inmt(
        self,
        src,
        segment_list,
        src_feats={},
        tgt=None,
        batch_size=1,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        ):

        src_data = {
            "reader": self.src_reader,
            "data": src,
            "features": src_feats
        }
        tgt_data = {
            "reader": self.tgt_reader,
            "data": tgt,
            "features": {}
        }
        _readers, _data = inputters.Dataset.config(
            [("src", src_data), ("tgt", tgt_data)]
        )

        data = inputters.Dataset(
            self.fields,
            readers=_readers,
            data=_data,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
        )

        self.last_word_idx = {}
        self.out_segments = []
        self.prefix = None
        self.phrase_table = {}
        tgt_vocab = self._tgt_vocab.itos

        self.segments = []
        for segment in segment_list:
            segment_comm = segment[0]
            segment_type = segment[1]
            segment_phrase_table = {}
            new_segment = [[], segment_type, segment_phrase_table] # [[IDS], TYPE, PHRASE_TABLE]
            #|==============================================================|
            #|                 SegmentType.GENERIC
            #|==============================================================|
            if segment_type == SegmentType.GENERIC:
                segment_indices = new_segment[0]
                segment_offset = 1 if segment_comm[0]=='<s>'else 0

                for pos, word in enumerate(segment_comm):
                    try:
                        index_value = tgt_vocab.index(word)
                    except ValueError:
                        index_value = 0
                    segment_indices.append(index_value)

                    if index_value == 0:
                        segment_phrase_table[pos] = word
            #|==============================================================|
            #|                 SegmentType.TO_COMPLETE
            #|==============================================================|
            elif segment_type == SegmentType.TO_COMPLETE:
                segment_indices = new_segment[0]

                for pos, word in enumerate(segment_comm[:-1]):
                    try:
                        index_value = tgt_vocab.index(word)
                    except ValueError:
                        index_value = 0
                    segment_indices.append(index_value)

                    if index_value == 0:
                        segment_phrase_table[pos] = word

                last_word = segment_comm[-1]
                last_word_idx = []
                for idx, word in enumerate(tgt_vocab):
                    if word.startswith(last_word):
                        last_word_idx.append(idx)
                if len(last_word_idx) == 0:
                    last_word_idx = [0]
                    segment_phrase_table[len(segment_comm)-1] = last_word
                segment_indices.append(last_word_idx)
            #|==============================================================|
            #|                 SegmentType.PREFIX
            #|==============================================================|
            elif segment_type == SegmentType.PREFIX:
                segment_indices = [2]
                for pos, word in enumerate(segment_comm):
                    try:
                        index_value = tgt_vocab.index(word)
                    except ValueError:
                        index_value = 0
                    segment_indices.append(index_value)

                    if index_value == 0:
                        self.phrase_table[pos] = word
                segment_indices.append(3)
                self.prefix = torch.tensor([[[i]] for i in segment_indices])

                out_segment = [0, len(segment_comm), segment_type]
                self.out_segments.append(out_segment)
                continue
            #|==============================================================|
            self.segments.append(new_segment)

        """
        if self.segments[0][0][0] == 2:
            first_segment = self.segments.pop(0)
            type_segment = first_segment[1]
            if type_segment==SegmentType.GENERIC:
                self.prefix = torch.tensor([[[i]] for i in first_segment[0][:]])
            elif type_segment==SegmentType.TO_COMPLETE:
                self.prefix = torch.tensor([[[i]] for i in first_segment[0][:-1]])
                self.last_word_idx[len(self.prefix)] = first_segment[0][-1]
        """

        return self.translate(src, src_feats, tgt, batch_size, batch_type,
            attn_debug, align_debug, phrase_table)


    def prefix_based_inmt(
        self,
        src,
        prefix,
        src_feats={},
        tgt=None,
        batch_size=1,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        ):
        new_word = False
        if(prefix[0][-1] == ''):
            prefix[0] = prefix[0][:-1]
            new_word = True

        src_data = {
            "reader": self.src_reader,
            "data": src,
            "features": src_feats
        }
        tgt_data = {
            "reader": self.tgt_reader,
            "data": prefix,
            "features": {}
        }
        _readers, _data = inputters.Dataset.config(
            [("src", src_data), ("tgt", tgt_data)]
        )

        data = inputters.Dataset(
            self.fields,
            readers=_readers,
            data=_data,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False,
        )

        for batch in data_iter:
            self.prefix = batch.tgt
            self.phrase_table = {}
            for position, id in enumerate(batch.tgt[1:-1]):
                if id == 0:
                    word = prefix[0][position]
                    self.phrase_table[position] = word

        self.last_word_idx = {}
        if not new_word:
            last_word_pos = len(prefix[0])
            last_word = prefix[0][-1]
            len_last_word = len(last_word)
            for idx, vocab in enumerate(self._tgt_vocab.itos):
                if(vocab[:len_last_word]==last_word):
                    if last_word_pos not in self.last_word_idx:
                        self.last_word_idx[last_word_pos] = []
                    self.last_word_idx[last_word_pos].append(idx)

        return self.translate(src, src_feats, tgt, batch_size, batch_type,
                              attn_debug, align_debug, phrase_table)


    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    batch_size=batch.batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearchINMT(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(
                batch, src_vocabs, decode_strategy
            )


    def _translate_batch_with_strategy(
        self, batch, src_vocabs, decode_strategy
        ):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        gold_score = self._gold_score(
            batch,
            memory_bank,
            src_lengths,
            src_vocabs,
            use_src_map,
            enc_states,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        target_prefix = self.prefix

        (
            fn_map_state,
            memory_bank,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(
            memory_bank, src_lengths, src_map, target_prefix=target_prefix
        )
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        def _decode_forward_translations(step_start=0, next_segment=None):
            # Hacemos una copia de diferentes variables para no editarlas y que luego el
            # modelo siga funcionando como si no hubiera ocurrido este paso
            copy_decode_strategy = copy.deepcopy(decode_strategy)
            copy_memory_lengths = copy.deepcopy(memory_lengths)
            copy_memory_bank = copy.deepcopy(memory_bank)
            copy_src_map = copy.deepcopy(src_map)
            model_state = copy.deepcopy(self.model.decoder.state)

            # Numero maximo de pasos en el futuro que se va a realizar
            max_N = 5

            # Preparamos una variable para guardar las traducciones y otra para los scores
            forward_hyp_trans = []
            forward_hyp_score = []

            for step in range(max_N):
                decoder_input = copy_decode_strategy.current_predictions.view(1, -1, 1)

                log_probs, attn = self._decode_and_generate(
                    decoder_input,
                    copy_memory_bank,
                    batch,
                    src_vocabs,
                    memory_lengths=copy_memory_lengths,
                    src_map=copy_src_map,
                    step=step+step_start,
                    batch_offset=copy_decode_strategy.batch_offset,
                )

                copy_decode_strategy.advance(log_probs, attn, self.last_word_idx)

                any_finished = copy_decode_strategy.is_finished.any()
                if any_finished:
                    copy_decode_strategy.update_finished()
                    if copy_decode_strategy.done:
                        break

                select_indices = copy_decode_strategy.select_indices

                if any_finished:
                    # Reorder states.
                    if isinstance(copy_memory_bank, tuple):
                        copy_memory_bank = tuple(
                            x.index_select(1, select_indices) for x in copy_memory_bank
                        )
                    else:
                        copy_memory_bank = copy_memory_bank.index_select(1, select_indices)

                    copy_memory_lengths = copy_memory_lengths.index_select(0, select_indices)

                    if copy_src_map is not None:
                        copy_src_map = copy_src_map.index_select(1, select_indices)

                if parallel_paths > 1 or any_finished:
                    self.model.decoder.map_state(
                        lambda state, dim: state.index_select(dim, select_indices)
                    )

                mask = [False]*parallel_paths
                if next_segment != None and next_segment[1] == SegmentType.GENERIC:
                    new_words_list = copy_decode_strategy.alive_seq[:,-1].tolist()
                    new_words_list = np.asarray(new_words_list)
                    mask = new_words_list==next_segment[0][0]

                elif next_segment != None and next_segment[1] == SegmentType.TO_COMPLETE:
                    new_words_list = copy_decode_strategy.alive_seq[:,-1].tolist()
                    new_words_list = np.asarray(new_words_list)
                    next_word = next_segment[0][0]

                    if isinstance(next_word, list):
                        for possible_word in next_word:
                            current_mask = new_words_list==possible_word
                            mask = np.logical_or(mask, current_mask)
                    else:
                        mask = new_words_list==next_word

                if True in mask:
                    forward_hyp_trans = [copy_decode_strategy.alive_seq[mask,:]]
                    forward_hyp_score = [copy_decode_strategy.topk_scores[:,mask]]
                    break


                forward_hyp_trans.append(copy_decode_strategy.alive_seq)
                forward_hyp_score.append(copy.copy(copy_decode_strategy.topk_scores))

            # Dejamos el modelo tal y como estaba
            self.model.decoder.state = model_state

            return forward_hyp_trans, forward_hyp_score

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):

            # Miramos cual es la longitud del prefijo
            len_prefix = -1
            if decode_strategy.target_prefix is not None:
                len_prefix = decode_strategy.target_prefix.size(1) -1

            # Solo vamos a probar a añadir un segmento si se cumplen dos casos
            # No quedan palabras en el prefijo a añadir
            # Aun quedan segmentos
            if step >= len_prefix and len(self.segments) > 0:
                next_segment = self.segments.pop(0)
                next_segment_comm = next_segment[0]
                next_segment_type = next_segment[1]
                next_segment_phrs = next_segment[2]

                forward_hyp_trans, forward_hyp_score = _decode_forward_translations(step, next_segment)

                # Seleccionamos aquella traduccion de mayor calidad
                best_hyp = []
                max_prob = -np.Inf
                for n_word, beam_scores in enumerate(forward_hyp_score):
                    # Normalizamos el score
                    beam_scores = beam_scores[0]/(n_word+step+1)
                    for idx, score in enumerate(beam_scores):
                        if score > max_prob:
                            max_prob = score
                            best_hyp = forward_hyp_trans[n_word][idx].tolist()
                best_hyp = best_hyp[step+1:]

                pos_next_segment = -1
                #|==============================================================|
                #|                 SegmentType.TO_COMPLETE
                #|==============================================================|
                if next_segment_type == SegmentType.GENERIC:
                    for pos, _ in enumerate(best_hyp):
                        prt_best_hyp = best_hyp[pos : pos+len(next_segment_comm)]
                        prt_next_com = next_segment_comm[:len(prt_best_hyp)]

                        if prt_best_hyp == prt_next_com:
                            pos_next_segment = pos
                            break

                    if pos_next_segment == -1:
                        pos_next_segment = len(best_hyp)
                    new_prefix = best_hyp[:pos_next_segment] + next_segment_comm + [3]

                    decode_strategy.maybe_update_next_target_tokens(step, new_prefix)
                    if next_segment_phrs:
                        next_segment_phrs = dict([(step + k + pos_next_segment, v) for k, v in next_segment_phrs.items()])
                        self.phrase_table.update(next_segment_phrs)
                #|==============================================================|
                #|                 SegmentType.TO_COMPLETE
                #|==============================================================|
                elif next_segment_type == SegmentType.TO_COMPLETE:
                    for pos, _ in enumerate(best_hyp):
                        prt_best_hyp = best_hyp[pos : pos+len(next_segment_comm)-1]
                        prt_next_com = next_segment_comm[:-1][:len(prt_best_hyp)]

                        if prt_best_hyp == prt_next_com:
                            if pos+len(prt_best_hyp) < len(best_hyp):
                                if best_hyp[pos+len(prt_best_hyp)] in next_segment_comm[-1]:
                                    pos_next_segment = pos
                                    break
                                continue
                            pos_next_segment = pos
                            break

                    if pos_next_segment == -1:
                        pos_next_segment = len(best_hyp)
                    new_prefix = best_hyp[:pos_next_segment] + next_segment_comm[:-1]

                    if next_segment_comm[-1] != [0]:
                        self.last_word_idx[step+1+len(new_prefix)] = next_segment_comm[-1]
                    else:
                        new_prefix += [0]

                    new_prefix += [3]
                    decode_strategy.maybe_update_next_target_tokens(step, new_prefix)

                    if next_segment_phrs:
                        next_segment_phrs = dict([(k+step+len(new_prefix)-2, v) for k, v in next_segment_phrs.items()])
                        self.phrase_table.update(next_segment_phrs)

                #|==============================================================|
                out_segment = [step+pos_next_segment, len(next_segment_comm), next_segment_type]
                self.out_segments.append(out_segment)

            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
            )

            decode_strategy.advance(log_probs, attn, self.last_word_idx)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )


    def _translate(
        self,
        data,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        dynamic=False,
        transform=None
        ):

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False,
        )

        xlation_builder = onmt.translate.InteractiveTranslationBuilder(
            data,
            self.fields,
            self.n_best,
            self.replace_unk,
            tgt,
            self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[: self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [
                    " ".join(pred) for pred in trans.pred_sents[: self.n_best]
                ]
                if self.report_align:
                    align_pharaohs = [
                        build_align_pharaoh(align)
                        for align in trans.word_aligns[: self.n_best]
                    ]
                    n_best_preds_align = [
                        " ".join(align) for align in align_pharaohs
                    ]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(
                            n_best_preds, n_best_preds_align
                        )
                    ]

                if dynamic:
                    n_best_preds = [transform.apply_reverse(x)
                                    for x in n_best_preds]
                all_predictions += [n_best_preds]
                self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if align_debug:
                    tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score(
                "PRED", pred_score_total, pred_words_total
            )
            self._log(msg)
            if tgt is not None:
                msg = self._report_score(
                    "GOLD", gold_score_total, gold_words_total
                )
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log(
                "Average translation time (s): %f"
                % (total_time / len(all_predictions))
            )
            self._log(
                "Tokens per second: %f" % (pred_words_total / total_time)
            )

        if self.dump_beam:
            import json

            json.dump(
                self.translator.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )

        if self.out_segments:
            hyp = all_predictions[0][0].split()
            for segment in self.out_segments:
                segment_pos = segment[0]
                segment_len = segment[1]
                segment[1] = hyp[segment_pos : segment_pos+segment_len]

        return all_scores, all_predictions


    def translate(self,
        src,
        src_feats={},
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        return super(INMTTranslator, self).translate(src, src_feats, tgt, batch_size, batch_type, attn_debug, align_debug, phrase_table)

class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.LANGUAGE_MODEL:
            raise ValueError(
                f"GeneratorLM does not support task {task}."
                f" Tasks supported: {ModelTask.LANGUAGE_MODEL}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

    def translate(
        self,
        src,
        src_feats={},
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        if batch_size != 1:
            warning_msg = ("GeneratorLM does not support batch_size != 1"
                           " nicely. You can remove this limitation here."
                           " With batch_size > 1 the end of each input is"
                           " repeated until the input is finished. Then"
                           " generation will start.")
            if self.logger:
                self.logger.info(warning_msg)
            else:
                os.write(1, warning_msg.encode("utf-8"))

        return super(GeneratorLM, self).translate(
            src,
            src_feats,
            tgt,
            batch_size=1,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table,
        )

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearchLM(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    batch_size=batch.batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearchLM(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(
                batch, src_vocabs, decode_strategy
            )

    @classmethod
    def split_src_to_prevent_padding(cls, src, src_lengths):
        min_len_batch = torch.min(src_lengths).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch < src.size(0):
            target_prefix = src[min_len_batch:]
            src = src[:min_len_batch]
            src_lengths[:] = min_len_batch
        return src, src_lengths, target_prefix

    def tile_to_beam_size_after_initial_step(self, fn_map_state, log_probs):
        if fn_map_state is not None:
            log_probs = fn_map_state(log_probs, dim=1)
            self.model.decoder.map_state(fn_map_state)
            log_probs = log_probs[-1]
        return log_probs

    def _translate_batch_with_strategy(
        self, batch, src_vocabs, decode_strategy
    ):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) split src into src and target_prefix to avoid padding.
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        src, src_lengths, target_prefix = self.split_src_to_prevent_padding(
            src, src_lengths
        )

        # (2) init decoder
        self.model.decoder.init_state(src, None, None)
        gold_score = self._gold_score(
            batch,
            None,
            src_lengths,
            src_vocabs,
            use_src_map,
            None,
            batch_size,
            src,
        )

        # (3) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        (
            fn_map_state,
            src,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(
            src,
            src_lengths,
            src_map,
            target_prefix=target_prefix,
        )

        # (4) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = (
                src
                if step == 0
                else decode_strategy.current_predictions.view(1, -1, 1)
            )

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                None,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths.clone(),
                src_map=src_map,
                step=step if step == 0 else step + src_lengths[0].item(),
                batch_offset=decode_strategy.batch_offset,
            )

            if step == 0:
                log_probs = self.tile_to_beam_size_after_initial_step(
                    fn_map_state, log_probs)

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            memory_lengths += 1
            if any_finished:
                # Reorder states.
                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                # select indexes in model state/cache
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map
    ):
        tgt = batch.tgt
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        log_probs, attn = self._decode_and_generate(
            src,
            None,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold_scores = log_probs.gather(2, tgt)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores
