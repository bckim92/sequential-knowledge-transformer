import collections
import functools
import os
import pathlib
import time
import sys
import argparse
from collections import namedtuple

from ruamel.yaml import YAML

__PATH__ = os.path.abspath(os.path.dirname(__file__))
DEFAULT_YML_FNAME = os.path.join(__PATH__, 'ymls/default.yml')


class CommandArgs:
    """Singleton version of collections.defaultdict
    """
    def __new__(cls):
        if not hasattr(cls, 'instance') or not cls.instance:
            cls.instance = collections.defaultdict(list)
        return cls.instance


def add_argument(*args, **kwargs):
    def decorator(func):
        _command_args = CommandArgs()
        _command_args[func.__name__].append((args, kwargs))

        @functools.wraps(func)
        def f(*args, **kwargs):
            ret = func(*args, **kwargs)
            return ret
        return f
    return decorator


def initialize_argparser(commands, command_args,
                         parser_cls=argparse.ArgumentParser):
    # Set default arguments
    parser = parser_cls(description=__doc__,
                        formatter_class=argparse.RawTextHelpFormatter)
    if hasattr(parser, "_add_default_arguments"):
        parser._add_default_arguments()
    parser.add_argument("--cfg", type=str, default="ymls/default.yml")
    subparsers = parser.add_subparsers(title='Available models', dest="model")

    # Set model-specific arguments
    sps = {}
    for (cmd, action) in commands.items():
        sp = subparsers.add_parser(cmd, help=action.__doc__)
        for (args, kwargs) in command_args.get(action.__name__, []):
            sp.add_argument(*args, **kwargs)
        sps[cmd] = sp

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args()
        model_args, _ = sps[args.model].parse_known_args()

    return args, model_args


def create_or_load_hparams(args, model_args, yaml_fname):
    args = vars(args)
    model_args = vars(model_args)

    HParams = namedtuple('HParams', args.keys())  # args already contain model_args
    hparams = HParams(**args)

    # Overwrite params from args (must be predefined in args)
    with open(yaml_fname, 'r') as fp:
        params_from_yaml = YAML().load(fp)
    if 'default' in params_from_yaml:
        for key, value in params_from_yaml['default'].items():
            hparams = hparams._replace(**{key: value})
    if 'model' in params_from_yaml:
        for key, value in params_from_yaml['model'].items():
            hparams = hparams._replace(**{key: value})

    # Set num_gpus, checkpoint_dir
    current_time = time.strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(hparams.checkpoint_base_dir,
                                  hparams.data_name,
                                  hparams.model,
                                  f"{current_time}_{hparams.other_info}")
    num_gpus = len(hparams.gpus.split(','))
    hparams = hparams._replace(num_gpus=num_gpus, checkpoint_dir=checkpoint_dir)

    # Save hparams to checkpoint dir (Separate default params and model params)
    model_keys = set(model_args.keys())
    default_keys = set(args.keys()) - model_keys
    dump_yaml_fname = os.path.join(checkpoint_dir, 'params.yml')
    dump_dict = {
        'model': {k: getattr(hparams, k) for k in model_keys},
        'default': {k: getattr(hparams, k) for k in default_keys},
    }
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(dump_yaml_fname, 'w') as fp:
        YAML().dump(dump_dict, fp)

    return hparams, dump_dict
