import os
import math
from pprint import PrettyPrinter
import random
import numpy as np

import torch
import sklearn
import tensorflow as tf
import better_exceptions
from tqdm import tqdm, trange
import colorlog
import colorful

from utils.etc_utils import set_logger, set_tcmalloc, set_gpus, check_none_gradients
from utils import config_utils, custom_argparsers
from models import MODELS
from modules.checkpoint_tracker import CheckpointTracker
from modules.trainer import run_wow_evaluation, Trainer
from modules.from_parlai import download_from_google_drive, unzip
from data.wizard_of_wikipedia import WowDatasetReader
from data.interactive_helper import (
    TopicsGenerator,
    WikiTfidfRetriever,
    InteractiveInputProcessor
)
from data.interactive_world import InteractiveWorld
from data import vocabulary as data_vocab

better_exceptions.hook()
_command_args = config_utils.CommandArgs()
pprint = PrettyPrinter().pprint


def main():
    # Argument passing/parsing
    args, model_args = config_utils.initialize_argparser(
        MODELS, _command_args, custom_argparsers.DialogArgumentParser)
    hparams, hparams_dict = config_utils.create_or_load_hparams(
        args, model_args, args.cfg)
    pprint(hparams_dict)

    if hparams.test_mode == 'wow':
        os.makedirs('./tmp', exist_ok=True)
        if not os.path.exists('tmp/wow_pretrained'):
            fname = 'wow_pretrained.zip'
            gd_id = '1lkF1QENr45j0vl-Oja3wEiqkxoNTxkXT'
            colorlog.info(f"Download pretrained checkpoint {fname}")
            download_from_google_drive(gd_id, os.path.join('tmp', fname))
            unzip('tmp', fname)
        ckpt_fname = os.path.join('tmp/wow_pretrained', 'ckpt-46070')
    else:
        raise ValueError("Only 'wow' is currently supported")

    # Set environment variables & gpus
    set_logger()
    set_gpus(hparams.gpus)
    set_tcmalloc()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set random seed
    #tf.random.set_seed(hparams.random_seed)
    #np.random.seed(hparams.random_seed)
    #random.seed(hparams.random_seed)

    # Set gpu
    assert hparams.num_gpus == 1
    mirrored_strategy = None

    # Make dataset reader
    os.makedirs(hparams.cache_dir, exist_ok=True)
    reader = WowDatasetReader(
        hparams.batch_size, hparams.num_epochs,
        buffer_size=hparams.buffer_size,
        bucket_width=hparams.bucket_width,
        max_length=hparams.max_length,
        max_episode_length=hparams.max_episode_length,
        max_knowledge=hparams.max_knowledge,
        knowledge_truncate=hparams.knowledge_truncate,
        cache_dir=hparams.cache_dir,
        bert_dir=hparams.bert_dir,
    )
    train_dataset, iters_in_train = reader.read('train', mirrored_strategy)
    test_dataset, iters_in_test = reader.read('test', mirrored_strategy)
    vocabulary = reader.vocabulary

    # Build model & optimizer & trainer
    model = MODELS[hparams.model](hparams, vocabulary)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.init_lr,
                                            clipnorm=hparams.clipnorm)
    trainer = Trainer(model, optimizer, mirrored_strategy,
                      hparams.enable_function,
                      WowDatasetReader.remove_pad)

    # Setup checkpoint
    global_step = tf.compat.v1.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     optimizer_step=global_step)
    train_example = next(iter(train_dataset))
    _ = trainer.train_step(train_example)
    checkpoint.restore(ckpt_fname)

    # Load retriever and input processor
    dictionary = reader._dictionary
    tokenize_fn = lambda x: [data_vocab.BERT_CLS_ID] \
        + dictionary.convert_tokens_to_ids(dictionary.tokenize(x)) \
        + [data_vocab.BERT_SEP_ID]
    input_processor = InteractiveInputProcessor(tokenize_fn, 5)

    # Compile graph
    colorlog.info("Compile model")
    dummy_input = input_processor.get_dummy_input()
    for _ in trange(5, ncols=70):
        trainer.test_step(dummy_input)

    # Module for interactive mode
    wiki_tfidf_retriever = WikiTfidfRetriever(hparams.cache_dir)
    topics_generator = TopicsGenerator(hparams.cache_dir)
    interactive_world = InteractiveWorld(
        responder=trainer,
        input_processor=input_processor,
        wiki_retriever=wiki_tfidf_retriever,
        topics_generator=topics_generator
    )

    # Loop!
    while True:
        interactive_world.run()
        interactive_world.reset()


if __name__ == '__main__':
    main()
