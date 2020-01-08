import numpy as np
import random

from data.vocabulary import convert_subword_to_word
from modules.trainer import _trim_after_eos

# INSTRUCTIONS
START_MSG = 'Let\'s talk about something through the chat!\
        You need to finish at least {} chat turns'
TOO_SHORT_MSG = 'Your message is too short, please make it more than {} words.'
TOO_LONG_MSG = 'Your message is too long, please make it less than {} words.'

# CHOOSING A TOPIC
PICK_TOPIC_MSG = 'To start, please select topic from below candidates (from 0 to {})"'
AFTER_PICK_TOPIC_MSG = 'Thank you for selecting a topic! Now, begin the conversation with our model about the topic.'

# FINISH
FINISHED_MSG = 'Chat is done! Let\'s move on to the next chat!\n\n'


class InteractiveWorld(object):
    """
    Playground for interactive demo

    Many code are from
    https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia/mturk_evaluation_task
    """
    def __init__(
        self,
        responder,
        input_processor,
        wiki_retriever,
        topics_generator,
        range_turn=(3, 5),
        max_turn=5,
    ):
        # Turn control
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.model_first = random.choice([True, False])

        # Model setup
        self.responder = responder
        self.input_processor = input_processor
        self.wiki_retriever = wiki_retriever
        self.topics_generator = topics_generator

    def run(self):
        seen = random.choice([True, False])
        num = random.choice([2, 3])
        topics = self.topics_generator.get_topics(seen=seen, num=num)

        # Select topic
        print(PICK_TOPIC_MSG.format(num))
        for idx, topic in enumerate(topics):
            print(f"{idx}. {topic}")
        print("(type 'q' to exit)")
        while True:
            key_input = input("> ")
            try:
                key_input = int(key_input)
                if key_input in list(range(num)):
                    break
                else:
                    print(f"You must choose number from 0 to {num}")
            except ValueError:
                if key_input == 'q':
                    import sys; sys.exit()
                print(f"You must choose number from 0 to {num}")

        chosen_topic = topics[key_input]
        print(AFTER_PICK_TOPIC_MSG)

        print(START_MSG.format(self.n_turn))
        prev_response = None
        while self.turn_idx < self.n_turn:
            response = self.step(chosen_topic, prev_response)
            prev_response = response
        print(FINISHED_MSG)

    def step(self, chosen_topic, prev_response):
        self.turn_idx += 1
        print("You are at turn {}....".format(self.turn_idx))

        # Context & response
        if self.model_first and self.turn_idx == 1:
            # If model_first and in first turn, context is chosen_topic
            context = chosen_topic
            knowledge_sentences = self.wiki_retriever.retrieve(
                chosen_topic=chosen_topic
            )
        else:
            # Else, human
            context = input("[You]: ")
            if context == "q":
                import sys; sys.exit()
            while self.is_msg_tooshortlong(context):
                context = input("[You]: ")

            if self.turn_idx == 1:
                # First human context is topic + context
                context = f"{chosen_topic} {context}"

            knowledge_sentences = self.wiki_retriever.retrieve(
                context=context, response=prev_response,
                chosen_topic=chosen_topic
            )

        # Model turn
        model_response = self.responder_step(
            context, knowledge_sentences, chosen_topic)

        print(f"[Model]: {model_response}")

        return model_response

    def responder_step(self, context, knowledge_sentences, chosen_topic):
        input_tensor = self.input_processor.update_and_get_input(
            context, knowledge_sentences, chosen_topic)
        model_output = self.responder.test_step(input_tensor)
        response = model_output['predictions'].numpy()[-1].reshape([1, -1])
        episode_mask = model_output['episode_mask'].numpy()[-1].reshape([1, -1])
        response = _trim_after_eos(response, mask=episode_mask)[0]
        response = convert_subword_to_word(response)
        self.input_processor.update_response(response)
        return response

    def is_msg_tooshortlong(self, response, th_min=3, th_max=20):
        msg_len = len(response.split(' '))
        if msg_len < th_min:
            print(TOO_SHORT_MSG.format(th_min))
            return True
        if msg_len > th_max:
            print(TOO_LONG_MSG.format(th_max))
            return True
        return False

    def reset(self):
        self.turn_idx = 0
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.model_first = random.choice([True, False])
        self.input_processor.reset()
