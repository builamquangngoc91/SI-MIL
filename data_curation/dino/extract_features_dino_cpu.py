# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

input_image_size = 224


def extract_feature_pipeline(args):

    dataset_train = Dataset(args.data_path + "/train_list.pickle", args)
    dataset_test = Dataset(args.data_path + "/test_list.pickle", args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,  # Changed to False for CPU
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,  # Changed to False for CPU
        drop_last=False,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_test)} test imgs."
    )

    # ============ building network ... ============

    model = vits.__dict__[args.arch](
        img_size=input_image_size,
        patch_size=args.patch_size,
        num_classes=0,
    )

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    # CPU version - no model.cuda()
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )

    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(
        model, data_loader_train, use_cuda=False
    )  # Force CPU

    # save features
    if args.dump_features:
        torch.save(
            train_features, os.path.join(args.dump_features, "trainfeat_deep.pth")
        )

    del train_features

    print("Extracting features for test set...")
    test_features = extract_features(
        model, data_loader_test, use_cuda=False
    )  # Force CPU

    # save features
    if args.dump_features:
        torch.save(test_features, os.path.join(args.dump_features, "testfeat_deep.pth"))

    del test_features


@torch.no_grad()
def extract_features(model, data_loader, use_cuda):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.float()
        if use_cuda:
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)

        feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(
            len(data_loader.dataset), dtype=torch.long, device=feats.device
        )

        features.index_copy_(0, index, feats)
    return features


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_path, args):

        with open(list_path, "rb") as f:
            self.files_list = pickle.load(f)

        self.data_dir = args.data_dir

        self.to_tensor = albumentations.Compose(
            [ToTensorV2()],
        )

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]

        img = (
            cv2.cvtColor(
                cv2.imread(self.data_dir + "/" + temp_path)[:, :, :3], cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            / 255.0
        )  # albumentations

        return self.to_tensor(image=img)["image"], idx


###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracting features for downstream tasks")
    parser.add_argument(
        "--batch_size_per_gpu", default=100, type=int, help="Per-GPU batch-size"
    )

    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )

    parser.add_argument(
        "--use_cuda",
        default=False,
        type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM",
    )
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )

    parser.add_argument(
        "--dump_features", default="", type=str, help="Path to save features."
    )

    parser.add_argument(
        "--num_workers",
        default=5,
        type=int,
        help="Number of data loading workers per GPU.",
    )

    parser.add_argument("--data_path", default="", type=str)

    parser.add_argument("--use_avgpool", default=True, type=bool, help="avgpool.")

    parser.add_argument("--data_dir", default="", type=str, help="Dataset folder name")

    args = parser.parse_args()

    if not os.path.isdir(args.dump_features):
        os.mkdir(args.dump_features)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    extract_feature_pipeline(args)
