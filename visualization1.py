import numpy as np
import cv2, torch, os, argparse
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models  # 直接从官方的torchvision中到入库
from torchvision import transforms

from visualization import center_crop_img, ReshapeTransform
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision

from trainers.cmpa import CustomCLIP
from clip.model import CMPA

from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, \
    LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from trainers.cmpa import load_clip_to_cpu
from dassl.data import DataManager
from train import setup_cfg



if __name__ == "__main__":
    test_img_list = ['/home/zfy/auto-tmp/cmpa/select-image/oxford_flowers/jpg/image_00001.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/oxford_flowers/jpg/image_00002.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/oxford_pets/images/Abyssinian_1.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/oxford_pets/images/Abyssinian_102.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/stanford_cars/cars_train/00003.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/stanford_cars/cars_train/00004.jpg',
                     '/home/zfy/auto-tmp/cmpa/select-image/stanford_cars/cars_train/00010.jpg',
                     ]

    parser = argparse.ArgumentParser()  # 处理命令行参数的模块
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()  # 解析命令行参数，并将结果存储在args对象中
    cfg = setup_cfg(args)  # 模型结构配置文件cfg
    print('cfg:\n', cfg)
    print('*' * 20)

    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    file_path = args.dataset_config_file
    file_name = os.path.basename(file_path)
    dataset_name = os.path.splitext(file_name)[0]

    print(dataset_name)
    print(classnames)

    clip_model = load_clip_to_cpu(cfg)          ## this clip includes the cmpa instance constructed from class `cmpa` in file ./clip/model.py

    ## initialize the CustomCLIP
    model = CustomCLIP(cfg, classnames, clip_model)
    # print(model)

    # print("model structure:\n", model, type(model))

    # ## print every layer name of `model`
    # idx = 0
    # for name, param in model.named_parameters():
    #     print('{}->{}:\n{}\n'.format(idx, name, param.shape))
    #     idx += 1

    ## load model parameters from file
    weights_path = '/home/zfy/auto-tmp/cmpa/output_CCPL/' + str(dataset_name) +'/16shots/seed1/MultiModalPromptLearner/model.pth.tar-20'
    checkpoint = torch.load(weights_path)
    state_dict = checkpoint["state_dict"]
    epoch = checkpoint["epoch"]


    # Ignore fixed token vectors
    if "prompt_learner.token_prefix" in state_dict:
        del state_dict["prompt_learner.token_prefix"]

    if "prompt_learner.token_suffix" in state_dict:
        del state_dict["prompt_learner.token_suffix"]

    print("Loading weights to model " 'from "{}" (epoch = {})'.format( weights_path, epoch))
    # set strict=False
    model.load_state_dict(state_dict, strict=False)

    image_path = "/home/zfy/auto-tmp/cmpa/data/imagenet/images/train/n01443537/n01443537_10014.JPEG"
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    img_tensor = img.unsqueeze(0)
    # print(img_tensor.shape)

    label_tensor = torch.tensor([10])

    model = model.cuda()
    img_tensor, label_tensor = img_tensor.cuda(), label_tensor.cuda()
    ## forward

    # logit = model(img_tensor, label_tensor)

    layer = 12

    attn_visual_weights = model.get_last_layer(img_tensor, label_tensor)
    # print(attn_visual_weights.shape)
    for i, attn_weights in enumerate(attn_visual_weights):
        print("attn_weights shape: ", attn_weights.shape)
    #
    attentions = attn_visual_weights[layer-1]
    print(attentions.shape)
    attentions = attentions[:, :, 0:197, 0:197]
    print(attentions.shape)

    patch_size = 16
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    # print(w, h)
    # print(img.shape)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)


    # if args.threshold is not None:
    #     # we keep only a certain percentage of the mass
    #     val, idx = torch.sort(attentions)
    #     val /= torch.sum(val, dim=1, keepdim=True)
    #     cumval = torch.cumsum(val, dim=1)
    #     th_attn = cumval > (1 - args.threshold)
    #     idx2 = torch.argsort(idx)
    #     for head in range(nh):
    #         th_attn[head] = th_attn[head][idx2[head]]
    #     th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    #     # interpolate
    #     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    output_dir = "output/" + str(dataset_name)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    print(attentions.shape)
    # attentions = attentions.detach().cpu().numpy()
    # 插值
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].detach().cpu().numpy()

    print(attentions.shape)

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(output_dir, "L" + str(layer) + " attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
    print(image_path)
    threshold = None

    if threshold is not None:
        image = skimage.io.imread(os.path.join(output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j],
                              fname=os.path.join(output_dir, "mask_th" + str(threshold) + "_head" + str(j) + ".png"),
                              blur=False)
