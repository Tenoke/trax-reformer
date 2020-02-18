import argparse
import gin
import glob
import jax
import os
import requests
import trax

from trax.supervised import inputs
import numpy as onp
import jax.numpy as np


from configs import train_config

parser = argparse.ArgumentParser(
    description='Tokenize a folder of text file(s)')

parser.add_argument('--data_folder', type=str, default='tokenized_data',
                    help='Data folder with 1 or more tokenized files')
parser.add_argument('--model_folder', type=str, default='model',
                    help='Folder For saving and loading the model')
parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--multi_factor_schedule',
                    default=False, action='store_true')
parser.add_argument('--tpu',
                    default=False, action='store_true')


args = parser.parse_args()

if args.tpu:
    if 'TPU_DRIVER_MODE' not in globals():
      url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver0.1-dev20191206'
      resp = requests.post(url)
      TPU_DRIVER_MODE = 1

    # The following is required to use TPU Driver as JAX's backend.
    from jax.config import config
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
    print(config.FLAGS.jax_backend_target)


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
        yield ids


def create_fixed_training_schedule(lr):
    # Yes, it does look unneceserily nested for passing a single float
    def FixedTrainingSchedule(*args, **kwargs):
        def learning_rate(step):
            return {'learning_rate': np.asarray(lr, dtype=np.float32)}
        return learning_rate
    return FixedTrainingSchedule


def train():
    gin.parse_config(train_config)
    schedule = create_fixed_training_schedule(args.learning_rate)
    if args.multi_factor_schedule:
        schedule = lr.MultifactorSchedule
    output_dir = os.path.expanduser(f'{args.model_folder}/')
    trainer = trax.supervised.Trainer(
        model=trax.models.ReformerLM,
        loss_fn=trax.layers.CrossEntropyLoss,
        optimizer=trax.optimizers.Adam,
        lr_schedule=schedule,
        inputs=trax.supervised.inputs.Inputs(gen_inputs, gen_validation_inputs),
        output_dir=output_dir,
        has_weights=True)

    for _ in range(args.epochs):
        trainer.train_epoch(n_steps=args.steps_per_epoch, n_eval_steps=1)


if __name__ == '__main__':
    train()
