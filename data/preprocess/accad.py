import os
import pickle

from fire import Fire
import numpy as np
import random

MAX_FRAMES = 150
TRAIN_DIR_ROOT = 'data/motion3d/ACCAD/train'
TEST_DIR_ROOT = 'data/motion3d/ACCAD/test'


def prepare_train_test_split(path: str = 'data/raw/ACCAD.npz'):
    all_files = np.load(path, allow_pickle=True)['positions_3d'].item()

    raw_3d_data = {}
    for set_name in all_files.keys():
        for scene_name in all_files[set_name].keys():
            frames = all_files[set_name][scene_name]['positions_3d']
            frames = frames[:MAX_FRAMES * (len(frames) // MAX_FRAMES)]
            if len(frames) == 0:
                continue
            frames_groups = np.split(frames, len(frames) // MAX_FRAMES)
            raw_3d_data[set_name] = raw_3d_data.get(set_name, []) + frames_groups

    raw_3d_data = {k: v for k, v in raw_3d_data.items() if len(v)}

    sets = list(raw_3d_data.keys())
    random.shuffle(sets)
    num_train_scenes = int(0.8 * len(sets))
    train_scenes, test_scenes = sets[num_train_scenes:], sets[:num_train_scenes]
    os.makedirs(TRAIN_DIR_ROOT, exist_ok=True)
    os.makedirs(TEST_DIR_ROOT, exist_ok=True)
    files = 0
    for train_scene in train_scenes:
        for data in raw_3d_data[train_scene]:
            with open(os.path.join(TRAIN_DIR_ROOT, f'{files}.pickle'), 'wb') as handle:
                pickle.dump({'data_label': data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            files += 1

    files = 0
    for test_scene in test_scenes:
        for data in raw_3d_data[test_scene]:
            with open(os.path.join(TEST_DIR_ROOT, f'{files}.pickle'), 'wb') as handle:
                pickle.dump({'data_label': data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            files += 1


if __name__ == '__main__':
    random.seed(42)
    Fire(prepare_train_test_split)
