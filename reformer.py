import argparse
import gin
import os
import jax
import trax
from trax.supervised import inputs

import numpy as onp
import jax.numpy as np
from scipy.special import softmax


import glob
import json
from tokenizers import ByteLevelBPETokenizer

from start_tpu import config
from config import train_config

parser = argparse.ArgumentParser(
    description='Tokenize a folder of text file(s)')

parser.add_argument('--data_folder', type=str, default='tokenized_data',
                    help='Data folder with 1 or more tokenized files')
parser.add_argument('--model_folder', type=str, default='model',
                    help='Folder For saving and loading the model')
parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


def gen_inputs(n_devices):
    max_length = int(65536 * 0.98)  # always leave a little padding
    folder = args.data_folder
    files = glob.glob(f'{folder}/*.npy')
    print(f'first start from {len(files)} files')
    while True:
        file = onp.random.choice(files, 1)[0]
        data = onp.load(file, allow_pickle=True)
        print(f'processing from {file}, {len(data)} examples in file')
        max_picks = int((len(data) * 0.7) / n_devices)
        indices = onp.arange(len(data))
        picks = onp.random.choice(
            indices, (max_picks, n_devices), replace=False)
        for id_list in picks:
            inputs = []
            mask = []
            for id_ in id_list:
                IDS = data[id_]
                if len(IDS) > max_length:
                    rand_start = onp.random.randint(0, len(IDS) - max_length)
                    IDS = IDS[rand_start:rand_start + max_length]

                PAD_AMOUNT = 65536 - len(IDS)  # same as axial_pos_shape
                pad_start = onp.random.choice(PAD_AMOUNT)
                inputs.append(onp.pad(IDS, (pad_start, PAD_AMOUNT - pad_start),
                                      mode='constant'))
                mask.append(onp.pad(onp.ones_like(IDS, dtype=onp.float32),
                                    (pad_start, PAD_AMOUNT - pad_start),
                                    mode='constant'))
            inputs = onp.stack(inputs)
            mask = onp.stack(mask)
            # for i in range(100):
            yield (inputs, inputs, mask)


def gen_validation_inputs(n_devices):
        # different validation each time but consistent across the run
    ids = next(gen_inputs(n_devices))
    while True:
        return ids


def create_fixed_training_schedule(lr=0.0001):
    # Yes, it does look unneceserily nested for passing a single float
    def FixedTrainingSchedule(*args, **kwargs):
        def learning_rate(step):
            return {'learning_rate': np.asarray(lr, dtype=np.float32)}
        return learning_rate


def train():
    output_dir = os.path.expanduser(f'{args.model_folder}/')
    trainer = trax.supervised.Trainer(
        model=trax.models.ReformerLM,
        loss_fn=trax.layers.CrossEntropyLoss,
        optimizer=trax.optimizers.Adam,
        lr_schedule=FixedTrainingSchedule,
        # lr_schedule=trax.lr.MultifactorSchedule,
        inputs=trax.supervised.inputs.Inputs(gen_inputs, gen_inputs2),
        output_dir=output_dir,
        has_weights=True)

    for _ in range(args.epochs):
        trainer.train_epoch(n_steps=args.steps_per_epoch, n_eval_steps=1)

if __name__ == '__main__':
    train()
