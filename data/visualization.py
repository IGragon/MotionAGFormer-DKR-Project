import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from fire import Fire

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]

ANGLE = np.pi / 2
ROTATION_MATRIX = np.array([
    [np.cos(ANGLE), - np.sin(ANGLE), 0],
    [np.sin(ANGLE), np.cos(ANGLE), 0],
    [0, 0, 1]
])


def load_checkpoint(checkpoint_path, config_path):
    args = get_config(config_path)
    model = load_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    checkpoint = {k[7:]: v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def plot_pose(points, x_lim, y_lim, z_lim):
    fig = plt.figure(figsize=(10, 10))
    is_2D_plot = points.shape[-1] == 2
    if is_2D_plot:
        axes = plt.gca()
    else:
        axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    if not is_2D_plot:
        axes.set_zlim(z_lim)
    for start, end in connections:
        xs = points[start][0], points[end][0]
        ys = points[start][1], points[end][1]
        if is_2D_plot:
            axes.plot(xs, ys)
        else:
            zs = points[start][2], points[end][2]
            axes.plot3D(xs, ys, zs)

    axes.scatter(points[..., 0], points[..., 1])
    plt.savefig('tmp.png')
    plt.close()
    im = cv2.imread('tmp.png')
    return im


def save_visualization(input_path='data/motion3d/ACCAD/test/1.pickle',
                       model_config_path='configs/accad/MotionAGFormer-base.yaml',
                       model_checkpoint_path='checkpoint/best_epoch.pth.tr',
                       output='output',
                       fps=50,
                       size=512):
    points = np.load(input_path, allow_pickle=True)['data_label']

    frames_2d = []
    frames_3d = []
    frames_3d_predicted = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    model = load_checkpoint(model_checkpoint_path, model_config_path)
    with torch.inference_mode():
        input_tensor = torch.tensor(points, device=next(model.parameters()).device, dtype=torch.float).unsqueeze(0)
        model_predictions = model(input_tensor).squeeze(0)
        model_predictions = model_predictions.cpu().numpy()
    points = points @ ROTATION_MATRIX
    model_predictions = model_predictions @ ROTATION_MATRIX

    x_min, x_max = points[..., 0].min() - 0.2, points[..., 0].max() + 0.2
    y_min, y_max = points[..., 1].min() - 0.2, points[..., 1].max() + 0.2
    z_min, z_max = points[..., 2].min() - 0.2, points[..., 2].max() + 0.2

    for i in tqdm(range(len(points)), desc='Processing frames'):
        fr = plot_pose(points[i][..., :2], [x_min, x_max], [y_min, y_max], None)
        fr = cv2.resize(fr, (size, size))
        frames_2d.append(fr)
        fr = plot_pose(points[i], [x_min, x_max], [y_min, y_max], [z_min, z_max])
        fr = cv2.resize(fr, (size, size))
        frames_3d.append(fr)
        fr = plot_pose(model_predictions[i], [x_min, x_max], [y_min, y_max], [z_min, z_max])
        fr = cv2.resize(fr, (size, size))
        frames_3d_predicted.append(fr)
    video_2d = cv2.VideoWriter(f'{output}_2d.mp4', fourcc, fps, (size, size))
    video_3d = cv2.VideoWriter(f'{output}_3d.mp4', fourcc, fps, (size, size))
    video_3d_pred = cv2.VideoWriter(f'{output}_3d_pred.mp4', fourcc, fps, (size, size))
    for image in frames_2d:
        video_2d.write(image)
    video_2d.release()
    for image in frames_3d:
        video_3d.write(image)
    video_3d.release()
    for image in frames_3d_predicted:
        video_3d_pred.write(image)
    video_3d_pred.release()
    os.remove('tmp.png')


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from utils.learning import load_model
    from utils.tools import get_config

    Fire(save_visualization)
