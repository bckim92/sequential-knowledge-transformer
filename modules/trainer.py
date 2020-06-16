import os
from collections import defaultdict
import random
from pprint import PrettyPrinter

import tensorflow as tf
import numpy as np
import language_evaluation
from tqdm import tqdm
import colorful
from sklearn.metrics import accuracy_score

from data import vocabulary as data_vocab
from data.wizard_of_wikipedia import (
    WowDatasetReader, PARLAI_KNOWLEDGE_SEPARATOR, BERT_KNOWLEDGE_SEPARATOR
)
from utils.etc_utils import check_none_gradients, check_nan_gradients
from models import BaseModel
from modules.from_parlai import normalize_answer

pformat = PrettyPrinter().pformat


class Trainer(object):
    def __init__(self,
                 model: BaseModel,
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 mirrored_strategy: tf.distribute.Strategy = None,
                 enable_function: bool = True,
                 preprocess_fn = lambda x: x):
        self.model = model
        self.optimizer = optimizer
        self.mirrored_strategy = mirrored_strategy
        self.enable_function = enable_function
        self.preprocess_fn = preprocess_fn

        self._batch_size = model.hparams.batch_size
        self._num_gpus = model.hparams.num_gpus

        self.train_step_fn = self._get_train_step_fn()
        self.test_step_fn = self._get_test_step_fn()

        if self.enable_function:
            self.train_step_fn = tf.function(self.train_step_fn)
            self.test_step_fn = tf.function(self.test_step_fn)

    def train_step(self, example):
        return self.train_step_fn(self.model, self.optimizer, example)

    def test_step(self, example):
        return self.test_step_fn(self.model, example)

    def test_loop(self, dataset, num_steps, epoch, mode):
        results_dict = defaultdict(list)
        dataset_iter = iter(dataset)
        test_tqdm = tqdm(range(num_steps), ncols=70, desc=f"Epoch {epoch} (test {mode})")
        for i, current_step in enumerate(test_tqdm):
            example = next(dataset_iter)
            step_result = self.test_step(example)
            for key, value in step_result.items():
                results_dict[key].append(value.numpy())

        for key, value in results_dict.items():
            if results_dict[key][0].shape == ():
                results_dict[key] = np.array(value)
            else:
                results_dict[key] = np.concatenate(value, axis=0)

        return results_dict

    def _get_train_step_fn(self):
        def _train_step(model, optimizer, example):
            example = self.preprocess_fn(example)
            with tf.GradientTape() as tape:
                output_dict = model(example)
                # XXX : need to exactly compute loss per gpu
                batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                    in output_dict else self._batch_size
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
                grads = tape.gradient(output_dict['loss'], model.trainable_variables)
                check_none_gradients(grads, model.trainable_variables, model.hparams.ignore_none_gradients)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return output_dict

        def _dist_train_step(model, optimizer, example):
            with self.mirrored_strategy.scope():
                output_dict = self.mirrored_strategy.experimental_run_v2(_train_step, (model, optimizer, example))
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = self.mirrored_strategy.reduce(
                            tf.distribute.ReduceOp.SUM, value, axis=None)
                    else:
                        value = self.mirrored_strategy.experimental_local_results(value)
                        if key == 'num_valid_turns':  # XXX : change this condition to check whether it is scalar
                            value = tf.stack(value, axis=0)
                        else:
                            value = tf.concat(value, axis=0)
                        output_dict[key] = value
            return output_dict

        return _dist_train_step if self.mirrored_strategy else _train_step

    def _get_test_step_fn(self):
        def _test_step(model, example):
            example = self.preprocess_fn(example)
            output_dict = model(example, training=False)
            # XXX : need to exactly compute loss per gpu
            batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                in output_dict else self._batch_size
            for key, value in output_dict.items():
                if 'loss' in key:
                    output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
            return output_dict

        def _dist_test_step(model, example):
            with self.mirrored_strategy.scope():
                output_dict = self.mirrored_strategy.experimental_run_v2(_test_step, (model, example))
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = self.mirrored_strategy.reduce(
                            tf.distribute.ReduceOp.SUM, value, axis=None)
                    else:
                        value = self.mirrored_strategy.experimental_local_results(value)
                        if key == 'num_valid_turns':
                            value = tf.stack(value, axis=0)
                        else:
                            value = tf.concat(value, axis=0)
                        output_dict[key] = value
            return output_dict

        return _dist_test_step if self.mirrored_strategy else _test_step


def run_wow_evaluation(results_dict, checkpoint_dir, mode):
    global_step = int(tf.compat.v1.train.get_global_step())
    if 'episode_mask' in results_dict:
        episode_mask = results_dict['episode_mask']
    else:
        episode_mask = None

    trim_fn = _trim_after_eos
    knowledge_separator = BERT_KNOWLEDGE_SEPARATOR

    predictions = trim_fn(results_dict['predictions'], mask=episode_mask)
    answers = trim_fn(results_dict['answers'], mask=episode_mask)
    contexts = trim_fn(results_dict['context'], mask=episode_mask)
    knowledge_sent_gts = trim_fn(results_dict['knowledge_sent_gt'], mask=episode_mask)
    knowledge_sent_preds = trim_fn(results_dict['knowledge_sent_pred'], mask=episode_mask)

    # XXX: Dump outputs

    # Show examples
    show_indices = random.sample(range(len(predictions)), 10)
    for index in show_indices:
        prediction = predictions[index]
        answer = answers[index]
        knowledge_sent_gt = knowledge_sent_gts[index]
        knowledge_sent_pred = knowledge_sent_preds[index]
        tqdm.write(f"{index} ({mode}).")
        tqdm.write(f"(knowledge_gt) {knowledge_sent_gt}")
        tqdm.write(f"(knowledge_pred) {knowledge_sent_pred}")
        tqdm.write(f"(gt) {answer}")
        tqdm.write(f"(pred) {prediction}\n\n")

    # Evaluation
    rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
    perplexity = np.exp(np.mean(results_dict['gen_loss']))
    total_loss = np.mean(results_dict['loss'])
    knowledge_accuracy = accuracy_score(
        np.zeros(results_dict['knowledge_predictions'].shape, dtype=np.int32),
        results_dict['knowledge_predictions'], sample_weight=episode_mask)

    rouge_result = rouge_evaluator.run_evaluation(predictions, answers)
    loss_result = {'perplexity': perplexity,
                   'total_loss': total_loss,
                   'accuracy': knowledge_accuracy,}

    # Optional metrics
    if 'knowledge_loss' in results_dict:
        knowledge_loss = np.mean(results_dict['knowledge_loss'])
        loss_result['knowledge_loss'] = knowledge_loss
    if 'kl_loss' in results_dict:
        kl_loss = np.mean(results_dict['kl_loss'])
        loss_result['kl_loss'] = kl_loss
    if 'multi_responses' and 'multi_gt_knowledge_sentences' in results_dict:
        rouge_result, loss_result = add_multi_results(
            results_dict, rouge_result, loss_result, predictions, episode_mask, trim_fn)

    log_dict = {}
    log_dict.update(rouge_result)
    log_dict.update(loss_result)

    summaries = {
        f"{mode}_test_loss": loss_result,
        f"{mode}_rouge": rouge_result
    }

    return summaries, log_dict


def _trim_after_eos(sentences, replace_unk=False, mask=None):
    if mask is not None:
        assert len(sentences) == len(mask), "sentences and mask should have same length"

    trimmed_sentences = []
    for i, sentence in enumerate(sentences):
        if mask is not None and not mask[i]:
            continue
        # Convert bytes array to utf-8 array
        sentence = np.char.decode(sentence.astype(np.bytes_), 'UTF-8')

        try:
            eos_idx = int(np.where(sentence == data_vocab._BERT_SEP)[0][0])
            trimmed_sentence = ' '.join(sentence[:eos_idx])
        except IndexError:
            trimmed_sentence = ' '.join(sentence)

        if replace_unk:
            trimmed_sentence = trimmed_sentence.replace(data_vocab._BERT_UNK, '_[UNK]')

        trimmed_sentences.append(trimmed_sentence)
    return trimmed_sentences

def add_multi_results(results_dict, rouge_result, loss_result, predictions, episode_mask, trim_fn):
    multi_responses = results_dict['multi_responses']
    num_responses = results_dict['num_responses'][episode_mask]
    multi_gt_knowledge_sentences = results_dict['multi_gt_knowledge_sentences']
    knowledge_sent_preds = results_dict['knowledge_sent_pred'][episode_mask]

    multi_rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1,
                                                            tokenization_fn=normalize_answer,
                                                            average=False)
    multi_rouge_results_list = []
    multi_accuracy_list = []
    for i in range(multi_responses.shape[1]):
        # choose best rouge scores among multi responses
        responses = trim_fn(multi_responses[:, i], mask=episode_mask)
        multi_rouge_result = multi_rouge_evaluator.run_evaluation(predictions, responses)
        multi_rouge_result['rouge1'][0] = multi_rouge_result['rouge1'][0] * (num_responses > i)
        multi_rouge_result['rouge2'][0] = multi_rouge_result['rouge2'][0] * (num_responses > i)
        multi_rouge_result['rougeL'][0] = multi_rouge_result['rougeL'][0] * (num_responses > i)
        multi_rouge_results_list.append(multi_rouge_result)

        # knowledge accuracy
        gt_knowledge_sentences = multi_gt_knowledge_sentences[:, i][episode_mask]
        knowledge_min_length = min(gt_knowledge_sentences.shape[-1], knowledge_sent_preds.shape[-1])
        multi_accuracy_list.append(np.logical_not(np.logical_not(
            gt_knowledge_sentences[:,:knowledge_min_length] == \
            knowledge_sent_preds[:,:knowledge_min_length]).sum(axis=1)))
    multi_rouge1_results = np.stack([x['rouge1'][0] for x in multi_rouge_results_list], axis=0)
    multi_rouge2_results = np.stack([x['rouge2'][0] for x in multi_rouge_results_list], axis=0)
    multi_rougeL_results = np.stack([x['rougeL'][0] for x in multi_rouge_results_list], axis=0)
    multi_rouge1_results = np.transpose(multi_rouge1_results, [1,0])
    multi_rouge2_results = np.transpose(multi_rouge2_results, [1,0])
    multi_rougeL_results = np.transpose(multi_rougeL_results, [1,0])
    multi_rouge1_max_indices = np.argmax(multi_rouge1_results, axis=1)
    max_multi_rouge1_results = np.max(multi_rouge1_results, axis=1)

    range_indices = np.arange(len(multi_rouge1_max_indices))
    max_multi_rouge2_results = multi_rouge2_results[range_indices, multi_rouge1_max_indices]
    max_multi_rougeL_results = multi_rougeL_results[range_indices, multi_rouge1_max_indices]

    multi_rouge1 = sum(max_multi_rouge1_results) / len(max_multi_rouge1_results)
    multi_rouge2 = sum(max_multi_rouge2_results) / len(max_multi_rouge2_results)
    multi_rougeL = sum(max_multi_rougeL_results) / len(max_multi_rougeL_results)
    rouge_result['rouge1_multi_responses'] = multi_rouge1
    rouge_result['rouge2_multi_responses'] = multi_rouge2
    rouge_result['rougeL_multi_responses'] = multi_rougeL

    # accuracy
    multi_accuracies = np.transpose(np.stack(multi_accuracy_list, axis=0), [1,0])
    multi_accuracies = multi_accuracies.sum(axis=1).astype(bool)
    multi_accuracy = sum(multi_accuracies) / len(multi_accuracies)
    loss_result['accuracy_multi_responses'] = multi_accuracy

    # perplexity
    multi_perplexity = np.exp(np.mean(results_dict['multi_gen_loss']))
    loss_result['perplexity_multi_responses'] = multi_perplexity

    return rouge_result, loss_result

__all__ = (
    'run_wow_evaluation',
)
