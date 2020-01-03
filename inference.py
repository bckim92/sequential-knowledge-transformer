import os
import math
from pprint import PrettyPrinter
import random
import numpy as np

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

better_exceptions.hook()
_command_args = config_utils.CommandArgs()
pprint = PrettyPrinter().pprint
pformat = PrettyPrinter().pformat
BEST_N_CHECKPOINTS = 5


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
    tf.random.set_seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    random.seed(hparams.random_seed)

    # For multi-gpu
    if hparams.num_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()  # NCCL will be used as default
    else:
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
    unseen_dataset, iters_in_unseen = reader.read('test_unseen', mirrored_strategy)
    vocabulary = reader.vocabulary

    # Build model & optimizer & trainer
    if mirrored_strategy:
        with mirrored_strategy.scope():
            model = MODELS[hparams.model](hparams, vocabulary)
            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.init_lr,
                                                 clipnorm=hparams.clipnorm)
    else:
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

    # Load
    train_example = next(iter(train_dataset))
    _ = trainer.train_step(train_example)
    #checkpoint.restore(ckpt_fname).assert_consumed()
    #checkpoint.restore(ckpt_fname).expect_partial()
    checkpoint.restore(ckpt_fname)

    # Test
    test_loop_outputs = trainer.test_loop(test_dataset, iters_in_test, 0, 'seen')
    unseen_loop_outputs = trainer.test_loop(unseen_dataset, iters_in_unseen, 0, 'unseen')

    test_summaries, log_dict = run_wow_evaluation(
        test_loop_outputs, hparams.checkpoint_dir, 'seen')
    unseen_summaries, unseen_log_dict = run_wow_evaluation(
        unseen_loop_outputs, hparams.checkpoint_dir, 'unseen')

    # Logging
    tqdm.write(colorful.bold_green("seen").styled_string)
    tqdm.write(colorful.bold_red(pformat(log_dict)).styled_string)
    tqdm.write(colorful.bold_green("unseen").styled_string)
    tqdm.write(colorful.bold_red(pformat(unseen_log_dict)).styled_string)


if __name__ == '__main__':
    main()
