import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader3d import get_dataloader, load_data, MotionDataset
from runner.train_mlp import train_step
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args
from diffusion import logger


def train_diffusion_model(args, dataloader, val_dataloader=None):
    print("creating model and diffusion...")
    logger.configure(
        dir=args.save_dir,
        format_strs=["stdout", "log", "csv", "tensorboard"]
    )
    args.arch = args.arch[len("diffusion_") :]

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus
    model, diffusion = create_model_and_diffusion(args)

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        # Move model to cuda:0 first before DataParallel
        model = model.to(torch.device('cuda:0'))
        model = torch.nn.DataParallel(model)
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    print("Training...")
    TrainLoop(args, model, diffusion, dataloader, val_dataloader).run_loop()
    print("Done.")


def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    motion_clean, motion_w_o = load_data(
        args.dataset_path,
        "train",
        keypointtype=args.keypointtype,
    )

    print("creating data loader...")
    dataset = MotionDataset(
        args.dataset,
        motion_clean,
        motion_w_o,
        input_motion_length=args.input_motion_length,
        use_dct=args.use_dct,
    )
    print("datasetsize:", len(dataset))

    dataloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    # args.lr_anneal_steps = (
    #    args.lr_anneal_steps // args.train_dataset_repeat_times
    # ) * len(
    #    dataloader
    # )  # the input lr_anneal_steps is by epoch, here convert it to the number of steps

    # Load validation dataset
    print("Loading validation dataset...")
    test_dataset_path = args.dataset_path.replace("mydataset", "test_dataset")
    val_motion_clean, val_motion_w_o = load_data(
        test_dataset_path,
        "train",  # Use "train" split mode for test_dataset folder
        keypointtype=args.keypointtype,
    )

    val_dataset = MotionDataset(
        args.dataset,
        val_motion_clean,
        val_motion_w_o,
        input_motion_length=args.input_motion_length,
        mode="test",
        use_dct=args.use_dct,
    )
    print("validation dataset size:", len(val_dataset))

    val_dataloader = get_dataloader(
        val_dataset, "test", batch_size=args.batch_size, num_workers=1
    )

    #model_type = args.arch.split("_")[0]
    train_diffusion_model(args, dataloader, val_dataloader)
    """
    if model_type == "diffusion":
        train_diffusion_model(args, dataloader)
    elif model_type == "mlp":
        train_mlp_model(args, dataloader)
    """

if __name__ == "__main__":
    main()
