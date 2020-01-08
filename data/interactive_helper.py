import copy
import os
import json
import random

import tensorflow as tf
import numpy as np
from parlai.core.agents import Agent, create_agent

from data.dataset_reader import (
    tensor_pad
)
from data.vocabulary import BERT_PAD_ID


class TopicsGenerator(object):
    """Select topics from WoW dataset

    Code from ParlAI
    https://github.com/facebookresearch/ParlAI/blob/master/projects/wizard_of_wikipedia/mturk_evaluation_task/worlds.py
    """
    def __init__(self, datapath):
        self.topics_path = os.path.join(
            datapath, 'wizard_of_wikipedia/topic_splits.json'
        )
        self.load_topics()

    def load_topics(self):
        with open(self.topics_path) as f:
            self.data = json.load(f)
        self.seen_topics = self.data['train']
        self.unseen_topics = self.data['valid'] + self.data['test']

    def get_topics(self, seen=True, num=3):
        if seen:
            return random.sample(self.seen_topics, num)
        return random.sample(self.unseen_topics, num)


class InteractiveInputProcessor(object):
    def __init__(self,
                 tokenize_fn,
                 max_length=51,
                 max_knowledge=34,
                 max_episode_length=5,
                 max_topic_length=38,
                 max_num_knowledges=175):
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.max_knowledge = max_knowledge
        self.max_episode_length = max_episode_length
        self.max_topic_length = max_topic_length
        self.max_num_knowledges = max_num_knowledges

        self.context_history = []
        self.response_history = []
        self.knowledge_sentences_history = []
        self.chosen_topic_history = []

        self.need_response_update = False

    def update_and_get_input(
        self,
        context,
        knowledge_sentences,
        chosen_topic,
    ):
        if self.need_response_update == True:
            raise ValueError("update_response() must be called before re-calling update_and_get_input()")

        # Tokenize all inputs
        context = self.tokenize_fn(context)[:self.max_length + 1]
        response = self.tokenize_fn('dummy response')[:self.max_length + 1]
        knowledge_sentences = [self.tokenize_fn(s)[:self.max_knowledge] for s in knowledge_sentences[:self.max_num_knowledges]]
        chosen_topic = self.tokenize_fn(chosen_topic)[:self.max_topic_length]

        # Update history
        self.context_history.append(context)
        response_history = self.response_history + [response]
        self.knowledge_sentences_history.append(knowledge_sentences)
        self.chosen_topic_history.append(chosen_topic)
        self.context_history = self.context_history[-self.max_episode_length:]
        response_history = response_history[-self.max_episode_length:]
        self.knowledge_sentences_history = self.knowledge_sentences_history[-self.max_episode_length:]
        self.chosen_topic_history = self.chosen_topic_history[-self.max_episode_length:]
        episode_length = len(self.context_history)

        # Convert to tensor
        inputs = {}
        inputs['episode_length'] = tf.convert_to_tensor(episode_length, dtype=tf.int32)
        inputs['knowledge_sentences'], inputs['num_knowledge_sentences'], inputs['knowledge_sentences_length'] \
            = self.to_tensor(self.knowledge_sentences_history)
        inputs['context'], inputs['context_length'] = self.to_tensor(self.context_history)
        inputs['response'], inputs['response_length'] = self.to_tensor(response_history)
        inputs['chosen_topic'], inputs['chosen_topic_length'] = self.to_tensor(self.chosen_topic_history)

        # Pad to max (to convert dynamic input to static)
        inputs = self.pad_to_max(inputs)
        # Add batch dimension
        for key in inputs.keys():
            inputs[key] = tf.expand_dims(inputs[key], axis=0)

        self.need_response_update = True

        return inputs

    def update_response(self, response):
        if self.need_response_update:
            response = self.tokenize_fn(response)[:self.max_length + 1]
            self.response_history.append(response)
            self.response_history = self.response_history[-self.max_episode_length:]
            self.need_response_update = False

    def to_tensor(self, input_list):
        tensor = tf.ragged.constant(input_list).to_tensor(default_value=BERT_PAD_ID)
        rank = tensor.shape.rank
        assert rank <= 3, "Rank must be <= 3"

        if rank == 2:
            inputs_length = tf.convert_to_tensor([len(l) for l in input_list], dtype=tf.int32)
            return tensor, inputs_length
        elif rank == 3:
            num_inputs = tf.convert_to_tensor([len(l) for l in input_list], dtype=tf.int32)
            #inputs_length = tf.convert_to_tensor([[len(x) for x in l] for l in input_list], dtype=tf.int32)
            inputs_length = tf.ragged.constant([[len(x) for x in l] for l in input_list], dtype=tf.int32)
            inputs_length = inputs_length.to_tensor(default_value=0)
            return tensor, num_inputs, inputs_length

    def pad_to_max(self, input_tensors):
        input_tensors['context'] = tensor_pad(input_tensors['context'], [self.max_episode_length, self.max_length + 1])
        input_tensors['response'] = tensor_pad(input_tensors['response'], [self.max_episode_length, self.max_length + 1])
        input_tensors['chosen_topic'] = tensor_pad(input_tensors['chosen_topic'], [self.max_episode_length, self.max_topic_length])
        input_tensors['context_length'] = tensor_pad(input_tensors['context_length'], [self.max_episode_length])
        input_tensors['response_length'] = tensor_pad(input_tensors['response_length'], [self.max_episode_length])
        input_tensors['chosen_topic_length'] = tensor_pad(input_tensors['chosen_topic_length'], [self.max_episode_length])

        input_tensors['knowledge_sentences'] = tensor_pad(
            input_tensors['knowledge_sentences'],
            [self.max_episode_length, self.max_num_knowledges, self.max_knowledge]
        )
        input_tensors['knowledge_sentences_length'] = tensor_pad(
            input_tensors['knowledge_sentences_length'],
            [self.max_episode_length, self.max_num_knowledges]
        )
        input_tensors['num_knowledge_sentences'] = tensor_pad(
            input_tensors['num_knowledge_sentences'],
            [self.max_episode_length]
        )
        return input_tensors

    def get_dummy_input(self):
        # Dummy raw inputs
        knowledge_sentences = [
            'Austin John __knowledge__ Austin John Winkler (born October 25, 1981) is an American singer-songwriter best known for being the former lead singer of the American rock band Hinder.',
            'Austin John __knowledge__ Winkler was one of the founding members of Hinder and recorded a total of one EP, four studio albums and released twenty-four singles to radio while with them during his 12-year tenure with the band.',
            'Austin John __knowledge__ Since his departure from Hinder, Winkler has continued his career, but as a solo artist.',
            'Rupi Kaur __knowledge__ Rupi Kaur is a contemporary Canadian feminist poet, writer and spoken word artist based in Toronto.',
            'Rupi Kaur __knowledge__ She is popularly known as an Instapoet for the traction she gains online on her poems on Instagram.',
            'Rupi Kaur __knowledge__ She published a book of poetry and prose entitled "milk and honey" in 2015.',
            'Rupi Kaur __knowledge__ The book deals with themes of violence, abuse, love, loss, and femininity.',
            'Rebel Heart (Madonna album) __knowledge__ Rebel Heart is the thirteenth studio album by American singer and songwriter Madonna.',
            'Rebel Heart (Madonna album) __knowledge__ It was released on March 6, 2015, by Interscope Records.',
            'Rebel Heart (Madonna album) __knowledge__ Following the completion of the "MDNA" release and promotion, Madonna worked on the album throughout 2014, co-writing and co-producing it with various musicians, including Diplo, Avicii, and Kanye West.',
            'Rebel Heart (Madonna album) __knowledge__ She regularly uploaded pictures of her recording sessions on her Instagram account.',
            'Rebel Heart (Madonna album) __knowledge__ Unlike her previous efforts, which involved only a few people, working with a large number of collaborators posed problems for Madonna in keeping a cohesive sound and creative direction for the album.',
            'no_passages_used __knowledge__ no_passages_used',
        ]
        context = 'i love instagram'
        chosen_topic = 'Instagram'

        # To make sure preserve state
        context_history = self.context_history
        response_history = self.response_history
        knowledge_sentences_history = self.knowledge_sentences_history
        chosen_topic_history = self.chosen_topic_history
        need_response_update = self.need_response_update
        self.reset()

        # Make tensor
        dummy_input = self.update_and_get_input(
            context,
            knowledge_sentences,
            chosen_topic
        )

        # Rollback state
        self.context_history = context_history
        self.response_history = response_history
        self.knowledge_sentences_history = knowledge_sentences_history
        self.chosen_topic_history = chosen_topic_history
        self.need_response_update = need_response_update

        return dummy_input

    def reset(self):
        self.context_history = []
        self.response_history = []
        self.knowledge_sentences_history = []
        self.chosen_topic_history = []
        self.need_response_update = False


class WikiTfidfRetriever(object):
    def __init__(self, datapath, num_retrieved=7):
        """
        Setup TF-IDF retriever

        Most code are from following url
        https://github.com/facebookresearch/ParlAI/blob/51eada993206f5a5a264288acbddc45f33f219d8/projects/wizard_of_wikipedia/interactive_retrieval/interactive_retrieval.py
        """
        self.datapath = datapath
        self.num_retrieved = num_retrieved
        retriever, wiki_map = self._set_up_retriever()
        sent_tok = self._set_up_sent_tok()

        self.retriever = retriever
        self.sent_tok = sent_tok
        self.wiki_map = wiki_map

    def _set_up_sent_tok(self):
        """
        Set up sentence splitter
        """
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        # nltk-specific setup
        st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
        try:
            sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            sent_tok = nltk.data.load(st_path)
        return sent_tok

    def _set_up_retriever(self):
        retriever_opt = {
            'model_file': 'models:wikipedia_full/tfidf_retriever/model',
            'remove_title': False,
            'datapath': self.datapath,
            'override': {'remove_title': False},
        }
        retriever = create_agent(retriever_opt)

        wiki_map_path = os.path.join(
            self.datapath,
            'models/wizard_of_wikipedia/full_dialogue_retrieval_model',
            'title_to_passage.json'
        )
        if not os.path.exists(wiki_map_path):
            responder_opts = {
                'model_file': 'models:wizard_of_wikipedia/full_dialogue_retrieval_model/model',
                'model': 'projects.wizard_of_wikipedia.generator.agents:EndToEndAgent',
                'datapath': self.datapath,
            }
            try:
                responder = create_agent(responder_opts)
            except ModuleNotFoundError:
                pass
        with open(wiki_map_path, 'r') as fp:
            wiki_map = json.load(fp)

        return retriever, wiki_map

    def retrieve(self, context=None, response=None, chosen_topic=None):
        chosen_topic_txts = None
        apprentice_txts = None
        wizard_txts = None

        if chosen_topic:
            chosen_topic_txts = self._get_chosen_topic_passages(chosen_topic)
        if context:
            apprentice_act = {'text': context, 'episode_done': True}
            self.retriever.observe(apprentice_act)
            apprentice_txts = self._get_passages(self.retriever.act())
        if response:
            wizard_act = {'tet': response, 'episode_done': True}
            self.retriever.observe(wizard_act)
            wizard_txts = self._get_passages(self.retriever.act())

        combined_txt = ''
        if chosen_topic_txts:
            combined_txt += chosen_topic_txts
        if apprentice_txts:
            combined_txt += '\n' + apprentice_txts
        if wizard_txts:
            combined_txt += '\n' + wizard_txts

        return combined_txt.split('\n')

    def _get_chosen_topic_passages(self, chosen_topic):
        retrieved_txt_format = []
        if chosen_topic in self.wiki_map:
            retrieved_txt = self.wiki_map[chosen_topic]
            retrieved_txts = retrieved_txt.split('\n')

            if len(retrieved_txts) > 1:
                combined = ' '.join(retrieved_txts[2:])
                sentences = self.sent_tok.tokenize(combined)
                total = 0
                for sent in sentences:
                    if total >= 10:
                        break
                    if len(sent) > 0:
                        title = f"{chosen_topic} __knowledge__"
                        retrieved_txt_format.append(' '.join([title, sent]))
                        total += 1

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages

    def _get_passages(self, act):
        """Format passages retrieved by taking the first paragraph of the
        top `num_retrieved` passages.
        """
        retrieved_txt = act.get('text', '')
        cands = act.get('text_candidates', [])
        if len(cands) > 0:
            retrieved_txts = cands[:self.num_retrieved]
        else:
            retrieved_txts = [retrieved_txt]

        retrieved_txt_format = []
        for ret_txt in retrieved_txts:
            paragraphs = ret_txt.split('\n')
            if len(paragraphs) > 2:
                sentences = self.sent_tok.tokenize(paragraphs[2])
                for sent in sentences:
                    title = paragraphs[0]
                    title = f"{title} __knowledge__"
                    retrieved_txt_format.append(' '.join([title, sent]))

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages
