import argparse


def str2list(v):
    return list(map(int, v.split(",")))


def str2bool(v):
    return v.lower() == "true"


class DialogArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We should not call _add_default_arguments here (will crash with add_subparsers)

    def _add_default_arguments(self):
        self.add_argument("--batch_size", type=int, default=128)
        self.add_argument("--gpus", type=str, default="0",
                            help="Comma separated (e.g. '0,1,2,')")
        self.add_argument("--random_seed", type=int, default=12345)
        self.add_argument("--other_info", type=str, default="debugging")

        # paths
        self.add_argument("--cache_dir", type=str, default="cached")
        self.add_argument("--checkpoint_base_dir", type=str, default="checkpoints")
        self.add_argument("--bert_dir", type=str, default="bert_pretrained/uncased_L-12_H-768_A-12")

        # train/data
        self.add_argument("--data_name", type=str, default="wizard_of_wikipedia")
        self.add_argument("--num_epochs", type=int, default=50)
        self.add_argument("--max_length", type=int, default=51,
                            help="Including EOS token")
        self.add_argument("--bucket_width", type=int, default=5)
        self.add_argument("--buffer_size", type=int, default=5000)
        self.add_argument("--ignore_none_gradients", type=bool, default=False)

        # wizard of wikipedia
        self.add_argument("--max_knowledge", type=int, default=34, help="ParlAI default setting is 32 (w/o GO and EOS)")
        self.add_argument("--knowledge_truncate", type=int, default=32, help="ParlAI default setting is 32")
        self.add_argument("--knowledge_teacher_forcing", type=str2bool, default=False)
        self.add_argument("--max_episode_length", type=int, default=5)

        # vocab/embedding
        self.add_argument("--vocab_size", type=int, default=30522)
        self.add_argument("--word_embed_size", type=int, default=768)
        self.add_argument("--embedding_dropout", type=float, default=0.2)

        # logging
        self.add_argument("--logging_step", type=int, default=20)
        self.add_argument("--evaluation_epoch", type=float, default=0.3)
        self.add_argument("--max_to_keep", type=int, default=5)
        self.add_argument("--keep_best_checkpoint", type=str2bool, default=True)

        # training
        self.add_argument("--init_lr", type=float, default=0.001)
        self.add_argument("--clipnorm", type=float, default=0.4)
        self.add_argument("--num_epochs_per_decay", type=float, default=15)
        self.add_argument("--lr_decay_factor", type=float, default=1.0)
        self.add_argument("--knowledge_label_smoothing", type=float, default=0.0)
        self.add_argument("--response_label_smoothing", type=float, default=0.0)
        self.add_argument("--enable_function", type=str2bool, default=True)

        # inference
        self.add_argument("--test_mode", type=str, default='wow')

        # uninitialized params
        self.add_argument("--num_gpus", type=int, default=0)
        self.add_argument("--checkpoint_dir", type=str, default="unset")
