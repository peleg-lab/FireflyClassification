from argparse import ArgumentParser

from models.lightning_rnn import LITGRU
from models.model_runner import ModelRunner


def main(hparams):
    model_runner = ModelRunner(hparams)
    model_runner.run_rnn()


if __name__ == '__main__':
    # project-wide training arguments, eg
    parent_parser = ArgumentParser()
    parent_parser.add_argument('--test', '-t', action='store_true')
    subparsers = parent_parser.add_subparsers()
    # each LightningModule also defines arguments relevant to it

    # currently supported architectures:
    # 1. RNN GRU architecture on sequences
    parser_GRU = subparsers.add_parser("gru")
    parser_GRU = LITGRU.add_model_specific_args(parser_GRU)

    hyperparams = parent_parser.parse_args()
    main(hyperparams)
