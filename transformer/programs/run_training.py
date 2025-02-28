import yaml
import argparse

from transformer.dataset.dataset import BilingualDataset
from transformer.model.model import Transformer
from transformer.losses.loss import Loss
from transformer.trainer.trainer import Trainer


def run():
     parser = argparse.ArgumentParser(
        prog='Transformer runner',
        description='This program constructs a transformer module ' +
            'from a config file, and then runs the training ' +
            'in the configuration.',
        epilog='...'
    )

    parser.add_argument(
        'config_file', metavar='<str>.yaml', type=str,
        help='config file specification for a blipnet module.'
    )
    args = parser.parse_args()

    """Parse the config file and construct objects"""
    config = yaml.safe_load(args.config_file)
    dataset = BilingualDataset(config['dataset'])
    model = Transformer(config['model'])
    trainer = Trainer(config['trainer'])

    trainer.train(
        dataset,
        model
    )


if __name__ == "__main__":
    run()