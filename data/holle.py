import json
import pickle
import os
import re
from collections import OrderedDict
import random
import sys
from operator import itemgetter

import tensorflow as tf
import numpy as np
import colorlog
from tqdm import tqdm

from data.dataset_reader import (
    string_split, list_of_string_split,
    list_of_list_of_string_split, tensor_pad,
    _scalar, _vector, _matrix, _tensor,
)
from data import vocabulary as data_vocab
from data.wizard_of_wikipedia import WowDatasetReader
from modules.from_parlai import download_from_google_drive
from official.bert import tokenization

HOLLE_RANDOM_SEED = 12345
_PUNCS_RE = re.compile(r'[^\w\s]')

_PLOT = 0
_REVIEW = 1
_COMMENTS = 2
_FACT_TABLE = 3
LABEL_ID2STR = {
    _PLOT: 'plot',
    _REVIEW: 'review',
    _COMMENTS: 'comments',
    _FACT_TABLE: 'fact_table'
}
_MAX_NUM_MULTI = 14

def _remove_duplicate(a_list):
    return list(OrderedDict.fromkeys(a_list))


def _f1_score(true_set, pred_set, eps=sys.float_info.epsilon):
    precision = len(true_set.intersection(pred_set)) / (float(len(pred_set)) + eps)
    recall = len(true_set.intersection(pred_set)) / (float(len(true_set)) + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score


def _check_continuity(bool_list):
    """Check if all matches are adjoint"""
    matched_indices = [idx for idx, is_match in enumerate(bool_list) if is_match]
    return all(a + 1 == b for a, b in zip(matched_indices[:-1], matched_indices[1:])), matched_indices


class HolleDatasetReader(WowDatasetReader):
    iterator_shapes = {
        "context": _matrix(),
        "response": _matrix(),
        "chosen_topic": _matrix(),
        "knowledge_sentences": _tensor(),
        "episode_length": _scalar(),
        "responses": _tensor(),
        "gt_knowledge_sentences": _tensor()
    }
    iterator_types = {
        "context": tf.string,
        "response": tf.string,
        "chosen_topic": tf.string,
        "knowledge_sentences": tf.string,
        "episode_length": tf.int32,
        "responses": tf.string,
        "gt_knowledge_sentences": tf.string,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sent_tok = self._set_sent_tok()
        self._datapath = os.path.join(self._cache_dir, 'holle')
        os.makedirs(self._datapath, exist_ok=True)

    def _read(self, mode: str, batch_size: int) -> tf.data.Dataset:
        episodes, dictionary = self._load_and_preprocess_all(mode)
        num_episodes = len(episodes)
        num_examples = sum([len(episode) for episode in episodes])
        num_iters = int(num_episodes / batch_size)

        if mode == 'train':
            self._dictionary = dictionary

        def _gen():
            for episode in episodes:
                examples = {'context': [],
                            'response': [],
                            'chosen_topic': [],
                            'knowledge_sentences': [],
                            'responses': [],
                            'gt_knowledge_sentences': []}
                for idx, example in enumerate(episode):
                    if idx == self._max_episode_length:
                        break

                    examples['context'].append(example['context'])
                    examples['response'].append(example['response'])
                    examples['chosen_topic'].append(example['chosen_topic'])

                    if self._knowledge_truncate > 0 and mode == 'train':  # Do not truncate in test time
                        knowledge_sentences = example['knowledge_sentences']
                        num_knowledges = min(len(knowledge_sentences), self._knowledge_truncate)
                        keepers = list(range(1, len(knowledge_sentences)))
                        random.shuffle(keepers)
                        keepers = [0] + keepers[:num_knowledges-1]
                        sentences = itemgetter(*keepers)(knowledge_sentences)
                        examples['knowledge_sentences'].append('\n'.join(sentences))
                    else:
                        knowledge_sentences = example['knowledge_sentences']
                        examples['knowledge_sentences'].append('\n'.join(knowledge_sentences))
                        examples['responses'].append('\n'.join(example['responses']))
                        examples['gt_knowledge_sentences'].append('\n'.join(example['gt_knowledge_sentences']))

                examples['episode_length'] = len(examples['context'])

                yield examples

        def _parse_fn(example):
            for key, value in self.iterator_shapes.items():
                dims = len(value)
                if dims == len(_scalar()):
                    pass
                elif dims == len(_matrix()):
                    sentences, lengths = list_of_string_split(example[key])
                    example[key] = sentences
                    example[f"{key}_length"] = tf.cast(lengths, tf.int32)
                elif dims == len(_tensor()) and key == 'knowledge_sentences':
                    list_of_sentences, sentence_lengths, num_sentences = \
                        list_of_list_of_string_split(example[key])
                    if self._max_knowledge > 0:
                        # Truncate length of each knowledge sentences
                        list_of_sentences = list_of_sentences[:, :, :self._max_knowledge]
                        sentence_lengths = tf.minimum(sentence_lengths, self._max_knowledge)
                    example[key] = list_of_sentences
                    example[f"{key}_length"] = tf.cast(sentence_lengths, tf.int32)
                    example[f"num_{key}"] = tf.cast(num_sentences, tf.int32)
                elif dims == len(_tensor()) and key in ['responses', 'gt_knowledge_sentences']:
                    list_of_sentences, sentence_lengths, num_sentences = \
                        list_of_list_of_string_split(example[key])
                    if self._max_length > 0:
                        max_length = self._max_length + 1 if key == 'responses' else self._max_knowledge
                        list_of_sentences = list_of_sentences[:, :, :max_length]
                        sentence_lengths = tf.minimum(sentence_lengths, max_length)
                    example[key] = list_of_sentences
                    example[f"{key}_length"] = tf.cast(sentence_lengths, tf.int32)
                    example[f"num_{key}"] = tf.cast(num_sentences, tf.int32)
                    if self._pad_to_max:
                        # XXX : (maybe bug...?) tf.function with dynamic input is extremely slower than
                        # static inputs. Therefore, convert dynamic to static.
                        episode_max_length = self._max_episode_length
                        max_length = self._max_length + 1 if key == 'responses' else self._max_knowledge
                        example[key] = tensor_pad(example[key], [episode_max_length, _MAX_NUM_MULTI, max_length])
                        example[f"{key}_length"] = tensor_pad(example[f"{key}_length"], [episode_max_length, _MAX_NUM_MULTI])
                        example[f"num_{key}"] = tensor_pad(example[f"num_{key}"], [episode_max_length])

                else:
                    raise ValueError

            if self._max_length > 0:
                example['response'] = example['response'][:, :(self._max_length + 1)]
                example['response_length'] = tf.minimum(example['response_length'], self._max_length + 1)
                example['context'] = example['context'][:, :(self._max_length + 1)]
                example['context_length'] = tf.minimum(example['context_length'], self._max_length + 1)

            if self._pad_to_max:
                # XXX : (maybe bug...?) tf.function with dynamic input is extremely slower than
                # static inputs. Therefore, convert dynamic to static.
                episode_max_length = self._max_episode_length
                example['context']                    = tensor_pad(example['context'], [episode_max_length, self._max_length + 1])
                example['response']                   = tensor_pad(example['response'], [episode_max_length, self._max_length + 1])
                example['chosen_topic']               = tensor_pad(example['chosen_topic'], [episode_max_length, 38])
                example['context_length']             = tensor_pad(example['context_length'], [episode_max_length])
                example['response_length']            = tensor_pad(example['response_length'], [episode_max_length])
                example['chosen_topic_length']        = tensor_pad(example['chosen_topic_length'], [episode_max_length])
                if mode == 'train':
                    example['num_knowledge_sentences']    = tensor_pad(example['num_knowledge_sentences'], [episode_max_length])
                    example['knowledge_sentences_length'] = tensor_pad(example['knowledge_sentences_length'], [episode_max_length, self._knowledge_truncate])
                    example['knowledge_sentences']        = tensor_pad(example['knowledge_sentences'], [episode_max_length, self._knowledge_truncate, self._max_knowledge])
                else:
                    example['num_knowledge_sentences']    = tensor_pad(example['num_knowledge_sentences'], [episode_max_length])
                    example['knowledge_sentences_length'] = tensor_pad(example['knowledge_sentences_length'], [episode_max_length, 175])
                    example['knowledge_sentences']        = tensor_pad(example['knowledge_sentences'], [episode_max_length, 175, self._max_knowledge])

            return example

        dataset = tf.data.Dataset.from_generator(_gen, self.iterator_types)
        if mode == 'train':
            dataset = dataset.shuffle(self._buffer_size).repeat(self._num_epochs)
        else:
            dataset = dataset.repeat(1)
        dataset = dataset.map(_parse_fn, num_parallel_calls=15)

        padded_shapes = {**self.iterator_shapes,
                         'context_length': _vector(),
                         'response_length': _vector(),
                         'chosen_topic_length': _vector(),
                         'knowledge_sentences_length': _matrix(),
                         'num_knowledge_sentences': _vector(),
                         'episode_length': _scalar(),
                         'responses_length': _matrix(),
                         'num_responses': _vector(),
                         'gt_knowledge_sentences_length': _matrix(),
                         'num_gt_knowledge_sentences': _vector()}

        drop_remainder = False if mode == 'train' else True
        batched_dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)

        return batched_dataset, num_iters

    @staticmethod
    def remove_pad(example):
        episode_max_length = tf.reduce_max(example['episode_length'])
        context_length = tf.reduce_max(example['context_length'])
        response_length = tf.reduce_max(example['response_length'])
        topic_length = tf.reduce_max(example['chosen_topic_length'])
        knowledge_length = tf.reduce_max(example['knowledge_sentences_length'])
        num_knowledges = tf.reduce_max(example['num_knowledge_sentences'])
        responses_length = tf.reduce_max(example['responses_length'])
        num_responses = tf.reduce_max(example['num_responses'])
        gt_knowledge_sentences_length = tf.reduce_max(example['gt_knowledge_sentences_length'])
        num_gt_knowledge_sentences = tf.reduce_max(example['num_gt_knowledge_sentences'])

        sliced_example = {}
        sliced_example['episode_length'] = example['episode_length']
        sliced_example['context_length'] = example['context_length'][:, :episode_max_length]
        sliced_example['response_length'] = example['response_length'][:, :episode_max_length]
        sliced_example['chosen_topic_length'] = example['chosen_topic_length'][:, :episode_max_length]
        sliced_example['num_knowledge_sentences'] = example['num_knowledge_sentences'][:, :episode_max_length]
        sliced_example['context'] = example['context'][:, :episode_max_length, :context_length]
        sliced_example['response'] = example['response'][:, :episode_max_length, :response_length]
        sliced_example['chosen_topic'] = example['chosen_topic'][:, :episode_max_length, :topic_length]
        sliced_example['knowledge_sentences_length'] = example['knowledge_sentences_length'][:, :episode_max_length, :num_knowledges]
        sliced_example['knowledge_sentences'] = example['knowledge_sentences'][:, :episode_max_length, :num_knowledges, :knowledge_length]
        sliced_example['responses'] = example['responses'][:, :episode_max_length, :, :]
        sliced_example['responses_length'] = example['responses_length'][:, :episode_max_length, :]
        sliced_example['num_responses'] = example['num_responses'][:, :episode_max_length]
        sliced_example['gt_knowledge_sentences'] = example['gt_knowledge_sentences'][:, :episode_max_length, :, :]
        sliced_example['gt_knowledge_sentences_length'] = example['gt_knowledge_sentences_length'][:, :episode_max_length, :]
        sliced_example['num_gt_knowledge_sentences'] = example['num_gt_knowledge_sentences'][:, :episode_max_length]

        return sliced_example

    def _load_and_preprocess_all(self, mode: str):
        self._download_data(mode)

        if os.path.exists(self._get_preprocessed_fname(mode)):
            episodes_fname = self._get_preprocessed_fname(mode)
            colorlog.info(f"Load preprocessed holle from {episodes_fname}")
            with open(episodes_fname, 'r') as fp:
                episodes = []
                for line in fp:
                    episodes.append(json.loads(line))
            dictionary = tokenization.FullTokenizer(self._vocab_fname)
            return episodes, dictionary

        # Load raw dataset
        raw_fname = os.path.join(self._datapath, f'{mode}_data.json')
        with open(raw_fname, 'r') as fp:
            episodes = json.load(fp)
        if mode != 'test':
            episodes = self._to_wow_format(episodes, mode)
        else:
            multi_fname = os.path.join(self._datapath, 'multi_reference_test.json')
            with open(multi_fname, 'r') as fp:
                multi_responses = json.load(fp)
            episodes = self._to_wow_format_multi(episodes, multi_responses, mode)

        dictionary = tokenization.FullTokenizer(self._vocab_fname)

        return self._preprocess_episodes(episodes, dictionary, mode)

    def _download_data(self, mode: str):
        if mode == 'train':
            fname = 'train_data.json'
            gd_id = '1XLrXU2_64FBVt3-3UwdprdyAGXOIc8ID'
        elif mode == 'test':
            fname = 'test_data.json'
            gd_id = '1hSGhG0HyZSvwU855R4FsnDRqxLursPmi'
            multi_fname = 'multi_reference_test.json'
            multi_gd_id = '1BIQ8VbXdndRSDaCkPEruaVv_8WegWeok'
        else:
            ValueError("Mode must be one of 'train' and 'test'")

        full_path = os.path.join(self._datapath, fname)
        if not os.path.exists(full_path):
            colorlog.info(f"Download {fname} to {full_path}")
            download_from_google_drive(gd_id, full_path)

        if mode == 'test':
            multi_full_path = os.path.join(self._datapath, multi_fname)
            if not os.path.exists(multi_full_path):
                colorlog.info(f"Download {fname} to {multi_full_path}")
                download_from_google_drive(multi_gd_id, multi_full_path)

    def _set_sent_tok(self):
        import spacy
        sent_tok = spacy.load('en_core_web_sm')
        return sent_tok

    def _to_wow_format(self, raw_episodes, mode):
        colorlog.info("Convert holle dataset to wow format")
        episodes = []
        for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
            episode = []
            for example_idx in range(0, len(raw_episode['chat']), 2):
                if example_idx + 1 < len(raw_episode['chat']):
                    chosen_topic = raw_episode['movie_name']
                    response = raw_episode['chat'][example_idx + 1]
                    knowledge_sentences = self._get_knowledge_sentences(
                        raw_episode,
                        episode_idx,
                        example_idx,
                        mode
                    )
                    checked_sentence = knowledge_sentences[0]
                    title = 'no_passages_used' if checked_sentence == 'no_passages_used' else chosen_topic
                    formatted_knowledge = '\n'.join([
                        chosen_topic + ' __knowledge__ ' + k
                        if k != 'no_passages_used'
                        else 'no_passages_used __knowledge__ no_passages_used'
                        for k in knowledge_sentences
                    ])

                    example = {
                        'text': raw_episode['chat'][example_idx],
                        'chosen_topic': chosen_topic,
                        'title': title,
                        'episode_num': episode_idx,
                        'example_num': example_idx // 2,
                        'checked_sentence': checked_sentence,
                        'knowledge': formatted_knowledge,
                    }
                    if mode == 'train':
                        example['labels'] = [response]
                    else:
                        example['eval_labels'] = [response]
                episode.append(example)
            episodes.append(episode)
        return episodes

    def _to_wow_format_multi(self, raw_episodes, multi_responses, mode):
        colorlog.info("Convert holle test dataset to wow format")
        episodes = []
        for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
            episode = []
            multi_cnt = 0
            for example_idx in range(0, len(raw_episode['chat']), 2):
                if example_idx + 1 < len(raw_episode['chat']):
                    chosen_topic = raw_episode['movie_name']
                    response = raw_episode['chat'][example_idx + 1]
                    knowledge_sentences = self._get_knowledge_sentences(
                        raw_episode,
                        episode_idx,
                        example_idx,
                        'test'
                    )
                    checked_sentence = knowledge_sentences[0]
                    title = 'no_passages_used' if checked_sentence == 'no_passages_used' else chosen_topic
                    formatted_knowledge = '\n'.join([
                        chosen_topic + ' __knowledge__ ' + k
                        if k != 'no_passages_used'
                        else 'no_passages_used __knowledge__ no_passages_used'
                        for k in knowledge_sentences
                    ])

                    example = {
                        'text': raw_episode['chat'][example_idx],
                        'chosen_topic': chosen_topic,
                        'title': title,
                        'episode_num': episode_idx,
                        'example_num': example_idx // 2,
                        'checked_sentence': checked_sentence,
                        'knowledge': formatted_knowledge,
                    }
                    example['eval_labels'] = [response]

                    # add multiple responses
                    example['multi_eval_labels'] = [response]
                    example['multi_checked_sentences'] = [checked_sentence]
                    if multi_cnt < len(raw_episode['chat']) // 2:
                        if f'ts_{episode_idx}_{multi_cnt}' in multi_responses.keys():
                            multi_response_id = f'ts_{episode_idx}_{multi_cnt}'
                            for multi_idx in range(len(multi_responses[multi_response_id]['responses'])):
                                raw_multi_response = multi_responses[multi_response_id]['responses'][multi_idx]
                                raw_multi_span = multi_responses[multi_response_id]['spans'][multi_idx]
                                if raw_multi_response != response:
                                    multi_response = _PUNCS_RE.sub('', str(raw_multi_response))
                                    multi_span = _PUNCS_RE.sub('', str(raw_multi_span))
                                    multi_knowledge_sentences = [_PUNCS_RE.sub('', str(x)) for x in knowledge_sentences]
                                    multi_knowledge_idx = self._get_best_match_idx(multi_span, multi_knowledge_sentences, multi_response)
                                    example['multi_eval_labels'].append(raw_multi_response)
                                    example['multi_checked_sentences'].append(knowledge_sentences[multi_knowledge_idx])
                            multi_cnt += 1
                episode.append(example)
            episodes.append(episode)
        return episodes

    def _preprocess_episodes(self, episodes, dictionary, mode):
        """
        Tokenize all the fields in Holl-E
        """
        colorlog.info("Preprocess holle dataset")
        tokenize = lambda x: ' '.join([str(data_vocab.BERT_CLS_ID)] +
            [str(y) for y in dictionary.convert_tokens_to_ids(dictionary.tokenize(x))] + [str(data_vocab.BERT_SEP_ID)])

        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, ncols=70)):
            new_examples = []
            for example_num, example in enumerate(episode):
                # Tokenize inputs and convert to tokens
                context = tokenize(example['text'])
                if mode == "train":
                    response = tokenize(example['labels'][0])
                else:
                    response = tokenize(example['eval_labels'][0])
                chosen_topic = tokenize(example['chosen_topic'])

                # Set up knowledge
                checked_knowledge = example['title'] + ' __knowledge__ ' + example['checked_sentence']
                knowledges = [checked_knowledge] + \
                    [k for k in example['knowledge'].rstrip().split('\n')]
                for idx, k in enumerate(knowledges[1:]):
                    if k == checked_knowledge:
                        break
                else:
                    # Sometimes, knowledge does not include checked_sentnece
                    idx = None
                    colorlog.warning("Knowledge does not include checked sentence.")
                if idx is not None:
                    del knowledges[idx + 1]

                # Tokenize knowledge
                knowledge_sentences = [tokenize(k) for k in knowledges]

                new_example = {'context': context,
                               'response': response,
                               'chosen_topic': chosen_topic,
                               'knowledge_sentences': knowledge_sentences,
                               'episode_num': episode_num,
                               'example_num': example_num}
                if 'multi_eval_labels' in example:
                    responses = [tokenize(response) for response in example['multi_eval_labels']]
                    new_example['responses'] = responses
                if 'multi_checked_sentences' in example:
                    gt_knowledge_sentences = [tokenize(example['title'] + ' __knowledge__ ' + checked_sentence)
                                              for checked_sentence
                                              in example['multi_checked_sentences']]
                    new_example['gt_knowledge_sentences'] = gt_knowledge_sentences
                new_examples.append(new_example)
            new_episodes.append(new_examples)
        if self._datapath:
            episodes_fname = self._get_preprocessed_fname(mode)
            colorlog.info(f"Cache preprocessed dataset to {episodes_fname}")
            with open(episodes_fname, 'w') as fp:
                for episode in new_episodes:
                    fp.write(json.dumps(episode) + '\n')

        return new_episodes, dictionary

    def _get_knowledge_sentences(self, raw_episode, episode_idx, example_idx, mode):
        # Handle special case
        if episode_idx == 5958 and mode == 'train':
            if example_idx in [0, 2]:
                return ['no_passages_used', 'Transformers: Aget of Extinction', '1']
            elif example_idx == 4 or example_idx == 8: # review
                return ['1', 'Transformers: Age of Extinction']
            elif example_idx == 6:
                return ['Transformers: Age of Extinction', '1']

        # Make GT and candidates
        knowledge_candidates = self._get_knowledge_candidates(raw_episode, example_idx)
        gt_knowledge, knowledge_candidates = self._get_gt_knowledge(
            raw_episode, knowledge_candidates, example_idx
        )
        for key, value in knowledge_candidates.items():
            knowledge_candidates[key] = _remove_duplicate(value)

        # Concat GT and candidates
        all_knowledge_sentences = [gt_knowledge]
        for candidates in knowledge_candidates.values():
            all_knowledge_sentences += candidates

        return all_knowledge_sentences

    def _get_knowledge_candidates(self, raw_episode, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        doc = raw_episode['documents']

        plot = self.validate_spacy_sentences(self._sent_tok(doc['plot']))
        review = self.validate_spacy_sentences(self._sent_tok(doc['review']))
        comments = doc['comments']
        fact_table = self._extract_fact_table(doc['fact_table'])
        knowledge_candidates = {
            'plot': plot,
            'review': review,
            'comments': comments,
            'fact_table': fact_table
        }

        return knowledge_candidates

    def _get_gt_knowledge(self, raw_episode, knowledge_candidates, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        label_str = LABEL_ID2STR.get(label, 'none')
        raw_gt_span = raw_episode['spans'][example_idx + 1]
        gt_span = _PUNCS_RE.sub('', raw_gt_span)
        raw_response = raw_episode['chat'][example_idx + 1]
        response = _PUNCS_RE.sub('', raw_response)

        # Find GT knowledge sentence
        if label_str == 'none':
            gt_knowledge = 'no_passages_used'
            gt_knowledge_idx = -1
        else:
            raw_label_candidates = knowledge_candidates[label_str]
            if label_str not in ['plot', 'review']:
                raw_label_candidates = _remove_duplicate(raw_label_candidates)
            label_candidates = [_PUNCS_RE.sub('', x) for x in raw_label_candidates]
            is_gt_in_cand = [gt_span in x for x in label_candidates]
            is_cand_in_gt = [x in gt_span for x in label_candidates]

            num_gt_in_cand = sum(is_gt_in_cand)
            num_cand_in_gt = sum(is_cand_in_gt)

            # Find matched candidate index
            if num_gt_in_cand == 1:  # Exact match
                gt_knowledge_idx = is_gt_in_cand.index(True)
            elif num_gt_in_cand > 1 or label in [_COMMENTS, _FACT_TABLE] or num_cand_in_gt == 0:
                # Find best match
                gt_knowledge_idx = self._get_best_match_idx(gt_span, label_candidates, response)
            elif num_cand_in_gt == 1:  # Inverse exact match
                gt_knowledge_idx = is_cand_in_gt.index(True)
            else:  # Span can exist over multiple sentences
                is_continue, matched_indices = _check_continuity(is_cand_in_gt)
                matched_words = ' '.join([label_candidates[idx] for idx in matched_indices])

                if is_continue and len(gt_span) > len(matched_words):
                    add_front = gt_span.split()[-1] == matched_words.split()[-1]
                    add_rear = gt_span.split()[0] == matched_words.split()[0]
                    index_to_add_front = [] if matched_indices[0] == 0 else [matched_indices[0] - 1]
                    if matched_indices[-1] + 1 == len(label_candidates):
                        index_to_add_rear = []
                    else:
                        index_to_add_rear = [matched_indices[-1] + 1]

                    if add_front:
                        matched_indices = index_to_add_front + matched_indices
                    elif add_rear:
                        matched_indices = matched_indices + index_to_add_rear
                    else:  # Add front & rear
                        matched_indices = index_to_add_front + matched_indices + \
                            index_to_add_rear
                    gt_knowledge_idx = matched_indices
                elif is_continue:
                    gt_knowledge_idx = matched_indices
                else:
                    gt_knowledge_idx = self._get_best_match_idx(
                        gt_span, label_candidates, response)

            # Get GT knowledge
            if isinstance(gt_knowledge_idx, int):
                gt_knowledge = raw_label_candidates[gt_knowledge_idx]
                gt_knowledge_idx = [gt_knowledge_idx]
            elif isinstance(gt_knowledge_idx, list):
                gt_knowledge = ' '.join(itemgetter(*gt_knowledge_idx)(raw_label_candidates))
            else:
                raise ValueError()

            # Remove GT from candidates
            for idx in sorted(gt_knowledge_idx, reverse=True):
                del raw_label_candidates[idx]
            knowledge_candidates[label_str] = raw_label_candidates

        return gt_knowledge, knowledge_candidates

    def _extract_fact_table(self, fact_table):
        if len(fact_table.keys()) == 2:
            return []

        awards = self.validate_sentences(fact_table['awards'])
        taglines = self.validate_sentences(fact_table['taglines'])
        similar_movies = self.validate_sentences(fact_table['similar_movies'])
        box_office = fact_table['box_office']
        if isinstance(box_office, str):
            box_office = [box_office if len(box_office) > 0 else []]
        else:
            box_office = []

        return awards + taglines + similar_movies + box_office

    def _get_best_match_idx(self, gt_span, label_candidates, response):
        gt_span_words = set(gt_span.split())
        response_words = set(response.split())
        label_words_candidates = [
            set(x.split()) for x in label_candidates
        ]

        f1_scores = []
        for label_words_candidate in label_words_candidates:
            f1_scores.append(_f1_score(gt_span_words, label_words_candidate))

        if sum(f1_scores) == 0.0:
            f1_scores = []
            for label_words_candidate in label_words_candidates:
                f1_scores.append(_f1_score(response_words, label_words_candidate))

        max_idx = f1_scores.index(max(f1_scores))

        return max_idx

    def validate_spacy_sentences(self, spacy_sentences):
        def _validate_sent(sent):
            if len(_PUNCS_RE.sub('', sent.text).strip()) > 1:
                return True
            else:
                False

        return [sent.text for sent in spacy_sentences.sents if _validate_sent(sent)]

    def validate_sentences(self, sentences):
        return [sent for sent in sentences if len(sent) > 0]
