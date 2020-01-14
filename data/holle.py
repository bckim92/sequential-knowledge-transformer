import json
import pickle
import os
import re
from collections import OrderedDict
import random
import sys
from operator import itemgetter

import numpy as np
import colorlog
from tqdm import tqdm

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sent_tok = self._set_sent_tok()
        self._datapath = os.path.join(self._cache_dir, 'holle')
        os.makedirs(self._datapath, exist_ok=True)

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
        episodes = self._to_wow_format(episodes, mode)

        dictionary = tokenization.FullTokenizer(self._vocab_fname)

        return self._preprocess_episodes(episodes, dictionary, mode)

    def _download_data(self, mode: str):
        if mode == 'train':
            fname = 'train_data.json'
            gd_id = '1XLrXU2_64FBVt3-3UwdprdyAGXOIc8ID'
        elif mode == 'test':
            fname = 'test_data.json'
            gd_id = '1hSGhG0HyZSvwU855R4FsnDRqxLursPmi'
        else:
            ValueError("Mode must be one of 'train' and 'test'")

        full_path = os.path.join(self._datapath, fname)
        if not os.path.exists(full_path):
            colorlog.info(f"Download {fname} to {full_path}")
            download_from_google_drive(gd_id, full_path)

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
                        chosen_topic + ' __knowledge__ ' + k for k in knowledge_sentences
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
