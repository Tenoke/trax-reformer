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
from tokenizers import ByteLevelBPETokenizer


from configs import train_config, test_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample from a given model')

    parser.add_argument('--model_folder', type=str, default='model',
                        help='Folder For saving and loading the model')
    parser.add_argument('--tpu',
                        default=False, action='store_true')
    parser.add_argument('--prompt', type=str, default='',
                        help='Prompt for beginning the sampling e.g. {"title": "Sampling"')
    parser.add_argument('--length', type=int, default=512,
                        help='Maximum length of sample')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling Temperature')
    parser.add_argument('--top_k', type=int, default=0,)
    parser.add_argument('--exp2', default=False, action='store_true',
                        help='Use exp2 instead of exp during sampling')
    parser.add_argument('--meena_max', default=False, action='store_true',
                        help='pick the probabilities with highest max')
    parser.add_argument('--meena_combine', default=False, action='store_true',
                        help='use all probabilities from all samples (8 on TPU) at once')

    args = parser.parse_args()


tokenizer = ByteLevelBPETokenizer(
    '256bytebpe-res-vocab.json', '256bytebpe-merges.txt')


def fake_data(n_devices):
    data = onp.zeros((n_devices, 65536))
    while True:
        yield (data, data, data)


def create_fixed_training_schedule(lr=0.001):
        # Yes, it does look unneceserily nested for passing a single float
    def FixedTrainingSchedule(*args, **kwargs):
        def learning_rate(step):
            return {'learning_rate': np.asarray(lr, dtype=np.float32)}
        return learning_rate
    return FixedTrainingSchedule


def sample_single(model_weights, infer_state, jit_model_infer, length=2048, prompt=None, temperature=1.0, top_k=None, exp2=None,
           boost_top=None, meena_max=False, meena_combine=False):
    """Sample from the ReformerLM model
    example top_k = 32
    exp2 = True|False
    boost_top = tuple of how many to boost by how much - e.g. (5, 1.1).
    Doesnt boost the very top token or (5, 1.1, True) to include the top one
    meena_max = True|False - pick the probabilities with highest max
    meena_combine = True|False use all probabilities from all samples (8 on TPU) at once
    """
    # Token id 0 is the equivalent of a "start" token
    cur_inputs = np.zeros((trax.math.device_count(), 1, 1), dtype=np.int32)

    cur_state = infer_state
    rngs = trax.math.random.split(
        trax.math.random.get_prng(0), trax.math.device_count())
    all_samples = []

    if prompt is not None:
        prompt = np.asarray(
            [tokenizer.encode(prompt).ids] * trax.math.device_count())

    for iteration in range(length):
        logits, cur_state = jit_model_infer(
            cur_inputs,
            model_weights,
            cur_state,
            rngs)

        if prompt is not None and iteration < prompt.shape[1]:
            cur_samples = onp.array(prompt[:, iteration], dtype=int)
        else:
            logits = onp.array(logits)[:, 0, 0, :] / temperature

            # custom settings
            if exp2:
                probs = onp.exp2(logits)
            else:
                probs = onp.exp(logits)
            del logits
            if top_k:
                for prob in probs:
                    prob[prob.argsort()[:-top_k]] = 1e-12
                del prob
            if boost_top:
                boost_until = -1 if len(boost_top) < 3 else probs.shape[-1]
                for prob in probs:
                    sort = prob.argsort()
                    # boost all except the top one
                    prob[sort[-boost_top[0]:boost_until]] *= boost_top[1]
                del prob
            # make sure probabilities always add up to 1
            probs = probs / probs.sum(axis=1, keepdims=1)

            if meena_max:
                max = 0
                axis = 0
                for i, prob in enumerate(probs):
                    max_ = prob.max()
                    if max_ > max:
                        max = max_
                        axis = i

                sample = onp.random.choice(probs.shape[-1], p=probs[axis, :])
                cur_samples = [sample
                               for i in range(probs.shape[0])]
            elif meena_combine:
                sample = onp.random.choice(probs.shape[-1], p=probs[0, :])
                cur_samples = [sample
                               for i in range(probs.shape[0])]
            else:
                cur_samples = [onp.random.choice(probs.shape[-1], p=probs[i, :])
                               for i in range(probs.shape[0])]
        cur_samples = onp.array(cur_samples, dtype=int)
        all_samples.append(cur_samples)

        cur_inputs = np.array(cur_samples[:, None, None])
    all_samples = onp.stack(all_samples, -1)
    for ids in all_samples:
        print(tokenizer.decode(ids.tolist()).replace('\\n', '\n'))
        print('_____________________')
        if meena_combine or meena_max:
            # all samples are the same for those options
            break
    return all_samples


def sample(args):
    gin.parse_config(train_config)
    schedule = create_fixed_training_schedule()
    output_dir = os.path.expanduser(f'{args.model_folder}/')
    trainer = trax.supervised.Trainer(
        model=trax.models.ReformerLM,
        loss_fn=trax.layers.CrossEntropyLoss,
        optimizer=trax.optimizers.Adam,
        lr_schedule=schedule,
        inputs=trax.supervised.inputs.Inputs(
            fake_data, fake_data),
        output_dir=output_dir,
        has_weights=True)
    gin.parse_config(test_config)
    model_infer = trax.models.ReformerLM(mode='predict')

    # Prepare a jitted copy of the model.
    jit_model_infer = trax.layers.base._accelerate(
        model_infer._forward_internal, trax.math.device_count())
    # Set up the initial state for sampling.
    infer_state = model_infer.new_weights_and_state(
        trax.supervised.trainer_lib.ShapeDtype((1, 1), dtype=np.int32))[1]
    infer_state = trainer._for_n_devices(infer_state)

    model_weights = trainer._opt_state[0][0]
    del trainer
    del model_infer
    sample_single(model_weights=model_weights, infer_state=infer_state, jit_model_infer=jit_model_infer, length=args.length, prompt=args.prompt, temperature=args.temperature,
                  top_k=args.top_k, exp2=args.exp2, meena_combine=args.meena_combine,
                  meena_max=args.meena_max)


def main_sample(args):
    if args.tpu:
        if 'TPU_DRIVER_MODE' not in globals():
            url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(
                ':')[0] + ':8475/requestversion/tpu_driver0.1-dev20191206'
            resp = requests.post(url)
            TPU_DRIVER_MODE = 1
        # The following is required to use TPU Driver as JAX's backend.
        from jax.config import config
        config.FLAGS.jax_xla_backend = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + \
            os.environ['COLAB_TPU_ADDR']
        print(config.FLAGS.jax_backend_target)
    sample(args)


if __name__ == '__main__':
    main_sample(args)
