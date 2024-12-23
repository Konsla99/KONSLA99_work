#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
import shutil
import torchvision
import numpy as np
import subprocess as sp
from os import makedirs
from random import Random
from PIL import Image
from utils.loss_utils import l1_loss, ssim, scaled_l1_loss
from gaussian_renderer import render, network_gui
from lpipsPyTorch import lpips
import torch.utils.benchmark as benchmark
import torchvision.transforms.functional as tf
from pathlib import Path
import sys
import socket
from scene import Scene, GaussianModel, GaussianModelSQ #, GaussianModelVQ
from compress.decoders import LatentDecoder
from compress.inf_loss import EntropyLoss
from utils.general_utils import safe_state, sample_camera_order, mean_distances
import uuid
import hashlib
from collections import OrderedDict
from tqdm import tqdm
from utils.general_utils import DecayScheduler
from utils.image_utils import psnr, resize_image, downsample_image, blur_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams
# 논문 git의 코드가 미완성이라 추가
import yaml
from collections import defaultdict
import glob

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def run(cmd, print_err=True):
    try:
        return sp.check_output(cmd, shell=True, stderr=sp.STDOUT).decode('UTF-8').splitlines()
    except sp.CalledProcessError as e:
        # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        if print_err:
            print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        return [cmd.split()[-1]]
    
def prepare_output_and_logger(args, all_args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    # Create wandb logger
    if WANDB_FOUND and args.use_wandb:
        wandb_project = args.wandb_project
        wandb_run_name = args.wandb_run_name
        wandb_entity = args.wandb_entity
        wandb_mode = args.wandb_mode
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        # name = os.path.basename(args.model_path) if wandb_run_name is None else wandb_run_name
        name = os.path.basename(args.source_path)+'_'+str(id)
        wandb.init(
            project=wandb_project,
            name=name,
            entity=wandb_entity,
            config=all_args,
            sync_tensorboard=False,
            dir=args.model_path,
            mode=wandb_mode,
            id=id,
            resume=True
        )
    return tb_writer  

def render_set(model_path, iteration, views, gaussians, pipeline, background, save_images, use_amp):
    render_path = os.path.join(model_path, "ours_{}".format(iteration), "renders360")
    save_images=True

    if save_images:
        makedirs(render_path, exist_ok=True)

    preds, names = [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            rendering = render(view, gaussians, pipeline, background)["render"]
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        else:
            preds.append(rendering)
            names.append('{0:05d}'.format(idx))

    return preds, names

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, quantize: QuantizeParams, 
                wandb_enabled: bool, use_amp: bool):
    with torch.no_grad():
        quantize.use_shift = [bool(el) for el in quantize.use_shift]
        quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
        quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
        gaussians = GaussianModelSQ(dataset.sh_degree, quantize)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_ply=True)
        scene.gaussians.decode_latents()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        images = {}
        render_preds, render_names = render_set(dataset.model_path, scene.loaded_iter, scene.getRender360Cameras(), 
                                                gaussians, pipeline, background, dataset.save_images, use_amp)
        images["train"] = (render_preds, render_names)

        # 파일이 존재하는지 확인하고 절대 경로로 변환
        frames_path = os.path.abspath(os.path.join(dataset.model_path, 'ours_{}'.format(scene.loaded_iter), 'renders360')).replace('\\', '/')
        png_files = glob.glob(os.path.join(frames_path, '*.png'))
        png_files.sort()  # 파일들을 정렬합니다.

        # PNG 파일이 있는지 확인
        if len(png_files) > 0:
            # ffmpeg 명령어를 사용해 PNG 파일을 비디오로 인코딩
            cmd = f"ffmpeg -y -framerate 30 -i {frames_path}/%05d.png -vf \"scale=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p {frames_path}/360.mp4"
            try:
                sp.run(cmd, shell=True, check=True)
            except sp.CalledProcessError as e:
                print(f"FFmpeg command failed with error: {e}")
        else:
            print(f"Error: No PNG files found in {frames_path}")
    #-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" 옵션을 추가하여 가로(iw)와 세로(ih) 해상도를 각각 가장 가까운 짝수로 맞추도록 했습니다. 
    # 이렇게 하면, FFmpeg가 자동으로 홀수 해상도를 짝수로 변환합니다.
    # sp.run()에서 check=True를 추가하여 FFmpeg 명령어가 실패할 경우, 예외가 발생하도록 했습니다. 
    # 이렇게 하면, FFmpeg 명령어 실행 중 발생하는 오류를 쉽게 디버깅할 수 있습니다.
    # 이렇게 수정하면, FFmpeg 명령어에서 해상도를 짝수로 조정하고, PNG 파일들을 비디오로 인코딩하는 과정이 정상적으로 진행될 것입니다.

    return images, scene.loaded_iter

    return images, scene.loaded_iter

def render_fn(views, gaussians, pipeline, background, use_amp):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        for view in views:
            render(view, gaussians, pipeline, background)

if __name__ == "__main__":
    # config.yaml을 읽는 부분 # 기존코드 에러로 인해 필수
    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)


    # Set up command line argument parser

    #  code 만들다 말아서 수정함;;
    # parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    # qp = QuantizeParams(parser)
    
    # 수정함
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, config['model_params'])
    op = OptimizationParams(parser, config['opt_params'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_iteration", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])

    print('Running on ', socket.gethostname())
    print("Optimizing " + args.model_path)

    wandb_enabled=(WANDB_FOUND and lp.extract(args).use_wandb)
    tb_writer = prepare_output_and_logger(lp.extract(args), args)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if wandb_enabled:
        wandb.run.summary['GPU'] = torch.cuda.get_device_name(0).split()[-1]

    images, loaded_iter = render_sets(lp.extract(args), args.render_iteration, pp.extract(args), qp.extract(args), 
                                            wandb_enabled, op.extract(args).use_amp)

    # All done
    print("\nTraining complete.")
