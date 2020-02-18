import argparse
import glob
import os
import json
import numpy as onp
from tokenizers import ByteLevelBPETokenizer

parser = argparse.ArgumentParser(
    description='Tokenize a folder of text file(s)')

parser.add_argument('--input_folder', type=str, required=True,
                    help='Input folder with 1 or more text files')
parser.add_argument('--output_folder', type=str, default='tokenized_data')
parser.add_argument('--files_start_with', type=str, default='',
                    help='Process only files starting with this string')
parser.add_argument('--remove_input', default=False, action='store_true',
                    help='Delete input file after tokenizing')

args = parser.parse_args()

input_files = glob.glob(f'{args.input_folder}/{args.files_start_with}*')
input_files = [x for x in input_files if os.path.isfile(x)]

print(input_files)

tokenizer = ByteLevelBPETokenizer(
    '256bytebpe-res-vocab.json', '256bytebpe-merges.txt')


def encode_data(file, max_per_n=10000):
    folder = args.output_folder
    with open(file, 'r') as f:
        # print(f.read())
        ids_n = 0
        largest_id = 0
        i = 0
        id_list = []
        for line in f:
            i += 1
            IDS = tokenizer.encode(line).ids
            IDS = onp.asarray(IDS, dtype=onp.int32)
            ids_n += len(IDS)
            largest_id = max(len(IDS), largest_id)
            print(largest_id)
            id_list.append(IDS)
            # print(id_list)
            if i > 0 and i % max_per_n == 0:
                # save every max_per_n lines
                onp.save(f'{folder}/{file[1:]}-{i}', id_list)
                print(f'{i} processed lines')
                id_list = []
                print(dict(ids_n=ids_n, largest_id=largest_id, name=folder))
        if len(id_list) > 16:
            # we skip if there's too litle for batching
            onp.save(f'{folder}/{file[1:]}-{i}', id_list)
    with open(f'{folder}/{file[1:]}-config', 'w') as out:
        json.dump(dict(ids_n=ids_n, largest_id=largest_id, name=folder), out)


if __name__ == '__main__':
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for i, file in enumerate(input_files):
        print(file)
        encode_data(file)
        if args.remove_input:
            os.remove(file)

    # combine configs - not used, just for sanity checking
    files = glob.glob(f'{args.output_folder}/*-config')
    config = {'ids_n': 0, 'largest_id': 0, 'name': args.output_folder}
    for file in files:
        with open(file) as json_file:
            temp_conf = json.load(json_file)
        config['ids_n'] += temp_conf['ids_n']
        config['largest_id'] = max(
            temp_conf['largest_id'], config['largest_id'])

    with open(f'config', 'w') as out:
        json.dump(f'{args.output_folder}/config', out)
    print(config)
    print('Finished tokenizing data')
