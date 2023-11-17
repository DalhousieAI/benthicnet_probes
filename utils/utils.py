import argparse
import ast
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from networkx import relabel_nodes
from omegaconf import OmegaConf
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import (
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from utils.benthicnet.io import read_csv
from utils.constrained_ff import ConstrainedFFNNModel
from utils.linear_probe_pl import LinearProbe
from utils.multilabel_ff import MultiLabelFFNNModel
from utils.one_hot_probe_pl import OneHotLinearProbe

# 1. Definitions
_SAMPLE_HEADERS = [
    "CATAMI Biota",
    "CATAMI Substrate",
    "CATAMI Relief",
    "CATAMI Bedforms",
    "Colour-qualifier",
    "Biota Mask",
    "Substrate Mask",
    "Relief Mask",
    "Bedforms Mask",
]

_HEADER_ROOT_DICT = {
    "CATAMI Biota": "biota",
    "CATAMI Substrate": "substrate",
    "CATAMI Relief": "relief",
    "CATAMI Bedforms": "bedforms",
    "Colour-qualifier": "colour",
    "Biota Mask": "biota_mask",
    "Substrate Mask": "substrate_mask",
    "Relief Mask": "relief_mask",
    "Bedforms Mask": "bedforms_mask",
}

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
_SCHEDULERS = {"warmup_cosine", "reduce", "step", "exponential" "none"}

_BACKBONES = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "efficientnet-b0": models.efficientnet_b0,
    "efficientnet-b1": models.efficientnet_b1,
    "efficientnet-b2": models.efficientnet_b2,
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
}

_VIT_NUM_FEATURES = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
}


# 1. Base utility functions
def parser():
    parser = argparse.ArgumentParser(
        description="Parameters for benthicnet probe project"
    )
    # Required parameters
    parser.add_argument(
        "--train_cfg",
        type=str,
        required=True,
        help="set cfg file for training optimizer",
    )
    parser.add_argument("--nodes", type=int, required=True, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, required=True, help="number of gpus per node"
    )

    # Other parameters
    parser.add_argument(
        "--tar_dir",
        type=str,
        default="/gpfs/project/6012565/become_labelled/compiled_labelled_512px/tar",
        help="set directory for training tar file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="../data_csv/size_full_benthicnet.csv",
        help="set path for data csv",
    )
    parser.add_argument(
        "--colour_jitter",
        type=bool,
        default=False,
        help="turn on/off colour jitter for image augmentation",
    )
    parser.add_argument(
        "--enc_pth",
        type=str,
        default=None,
        help="set path for pre-trained encoder/backbone or determine encoder/backbone architecture",
    )
    parser.add_argument(
        "--fine_tune",
        type=bool,
        default=False,
        help="set fine tune mode (default: false)",
    )
    parser.add_argument(
        "--graph_pth",
        type=str,
        default="../graph_info/finalized_output.csv",
        help="set path for csv to generate output graph structures",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--random_partition",
        type=bool,
        default=False,
        help="bool flag to randomly partition data (default: False)",
    )
    parser.add_argument(
        "--test_mode",
        type=bool,
        default=False,
        help="sets bool flag to test only (loads head weights as well)",
    )
    parser.add_argument(
        "--name", type=str, default="benthicnet_hl", help="set name for the run"
    )
    parser.add_argument(
        "--windows",
        type=bool,
        default=False,
        help="set backend to gloo if running on Windows",
    )

    return parser.parse_args()


def one_hot_parser():
    parser = argparse.ArgumentParser(
        description="Parameters for benthicnet probe project"
    )
    # Required parameters
    parser.add_argument(
        "--train_cfg",
        type=str,
        required=True,
        help="set cfg file for training optimizer",
    )
    parser.add_argument("--nodes", type=int, required=True, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, required=True, help="number of gpus per node"
    )

    # Other parameters
    parser.add_argument(
        "--tar_dir",
        type=str,
        default="/gpfs/project/6012565/become_labelled/compiled_labelled_512px/tar",
        help="set directory for training tar file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/lustre06/project/6012565/isaacxu/benthicnet_probes/data_csv/ \
            one_hots/substrate_depth_2_data/substrate_depth_2_data.csv",
        help="set path for data csv",
    )
    parser.add_argument(
        "--colour_jitter",
        type=bool,
        default=False,
        help="turn on/off colour jitter for image augmentation",
    )
    parser.add_argument(
        "--enc_pth",
        type=str,
        default=None,
        help="set path for pre-trained encoder/backbone or determine encoder/backbone architecture",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--fine_tune",
        type=bool,
        default=False,
        help="set fine tune mode (default: false)",
    )
    parser.add_argument(
        "--random_partition",
        type=bool,
        default=False,
        help="bool flag to randomly partition data (default: False)",
    )
    parser.add_argument(
        "--test_mode",
        type=bool,
        default=False,
        help="sets bool flag to test only (loads head weights as well)",
    )
    parser.add_argument(
        "--name", type=str, default="benthicnet_one_hot", help="set name for the run"
    )
    parser.add_argument(
        "--windows",
        type=bool,
        default=False,
        help="set backend to gloo if running on Windows",
    )

    return parser.parse_args()


def get_df(in_path):
    df = read_csv(
        fname=in_path, expect_datetime=False, index_col=None, low_memory=False
    )
    return df


def set_seed(seed, performance_mode=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not performance_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 2. Graph related functions
# Generating ancestor matrix
# Modified from Coherent Hierarchical Multi-Label Classification Networks (https://github.com/EGiunchiglia/C-HMCNN)
# Under GPL-3.0 License
def gen_R_mat(G):
    n_nodes = G.number_of_nodes()
    R = np.zeros((n_nodes, n_nodes))
    np.fill_diagonal(R, 1)
    for i in range(n_nodes):
        descendants = list(nx.descendants(G, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.from_numpy(R)
    R = R.unsqueeze(0)
    return R


def process_node(node, root_dict):
    def append_connection(root, child, parent):
        root_dict["root"].append(root)
        root_dict["child"].append(child)
        root_dict["parent"].append(parent)

    def split_nodes(node, root=None):
        split_node = node.split(" > ")
        if not root:
            root = split_node[0].strip("[]").lower()
            child = " > ".join(split_node[1:])
            if len(split_node[1:-1]) == 0:
                parent = child
            else:
                parent = " > ".join(split_node[1:-1])
        else:
            child = " > ".join(split_node)
            if len(split_node[:-1]) == 0:
                parent = child
            else:
                parent = " > ".join(split_node[:-1])
        return root, child, parent

    def gen_roots(root_dict):
        child_list = root_dict["child"]
        child_indices = [i for i in range(len(child_list)) if child_list[i] == parent]
        roots_to_check = [root_dict["root"][i] for i in child_indices]
        return roots_to_check

    root, child, parent = split_nodes(node)
    append_connection(root, child, parent)

    roots_to_check = gen_roots(root_dict)

    while root not in roots_to_check:
        root, child, parent = split_nodes(parent, root)
        roots_to_check = gen_roots(root_dict)
        append_connection(root, child, parent)


# Using graph info to process dataframe
def parse_heads_and_masks(sample_row, Rs):
    for header in _SAMPLE_HEADERS:
        root = _HEADER_ROOT_DICT[header].split("_")[0]
        raw_indices = sample_row[header]
        non_hierarchical_head = isinstance(Rs[root], int)
        if isinstance(raw_indices, str) and len(raw_indices) > 0:
            indices = np.array(ast.literal_eval(raw_indices))
            if "Mask" in header:
                lab_array = torch.ones(len(Rs[root][0]))
                lab_array[indices] = 0
            else:
                if non_hierarchical_head:
                    lab_array = torch.zeros(Rs[root])
                else:
                    lab_array = torch.zeros(len(Rs[root][0]))
                lab_array[indices] = 1
        else:
            if non_hierarchical_head:
                lab_array = torch.zeros(Rs[root])
            else:
                lab_array = torch.zeros(len(Rs[root][0]))
        sample_row[header] = lab_array
    return sample_row


def process_data_df(data_df, Rs):
    processed_data_df = data_df.copy()
    processed_data_df = processed_data_df.apply(
        lambda row: parse_heads_and_masks(row, Rs), axis=1
    )
    return processed_data_df


def process_nodes(nodes_df):
    root_dict = {"root": [], "child": [], "parent": []}
    nodes = nodes_df["CATAMI"]
    nodes.apply(process_node, args=(root_dict,))

    parent_df = pd.DataFrame.from_dict(root_dict)
    return parent_df


def gen_root_graphs(nodes_f):
    ROOTS = ["biota", "substrate", "bedforms", "relief"]

    root_graphs = {}
    idx_to_node = {}
    node_to_idx = {}

    nodes = get_df(nodes_f)
    parent_df = process_nodes(nodes)
    for root in ROOTS:
        G = nx.DiGraph()
        graph_df = parent_df[parent_df["root"] == root]
        graph_df = graph_df.drop_duplicates().copy()
        index_map = pd.Series(
            graph_df.reset_index().index, index=graph_df["child"].values
        ).to_dict()

        g = nx.from_pandas_edgelist(df=graph_df, source="parent", target="child")

        G.update(g)
        relabel_nodes(G, index_map, copy=False)

        node_dict = {v: k for k, v in index_map.items()}

        root_graphs[root] = G
        idx_to_node[root] = node_dict
        node_to_idx[root] = {v: k for k, v in node_dict.items()}
    return root_graphs, idx_to_node, node_to_idx


# 3. Augmentation related functions
# Values take (from FastAutoAug - MIT license)
# https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    "eigval": [0.2175, 0.0188, 0.0045],
    "eigvec": [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ],
}


# Lighting Class (from FastAutoAug - MIT license)
# https://github.com/kakaobrain/fast-autoaugment
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Fine tune aug stack (from FastAutoAug - MIT License)
# https://github.com/kakaobrain/fast-autoaugment
def get_augs(colour_jitter: bool, input_size=224, size_size=256, use_benthicnet=True):
    imagenet_mean_std = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    benthicnet_mean_std = transforms.Normalize(
        # labelled_dataset_stats
        # mean=[0.363, 0.420, 0.344], std=[0.207, 0.210, 0.183]
        mean=[0.359, 0.413, 0.386],
        std=[0.219, 0.215, 0.209],
    )

    if use_benthicnet:
        default_mean_std = benthicnet_mean_std
    else:
        default_mean_std = imagenet_mean_std

    if colour_jitter:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (input_size, input_size), interpolation=Image.BICUBIC
                ),
                transforms.RandomResizedCrop(
                    input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                Lighting(0.1, _IMAGENET_PCA["eigval"], _IMAGENET_PCA["eigvec"]),
                default_mean_std,
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                default_mean_std,
            ]
        )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(size_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            default_mean_std,
        ]
    )
    return train_transforms, val_transforms


# 4. Construct dataloaders
# Note: num_workers in data loader/cfg file does not need to be the same as slrm cpu count
def construct_dataloaders(datasets, train_kwargs):
    # Expect datasets to be a list of datasets, where the first is the training set
    data_loader_list = []
    for i, dataset in enumerate(datasets):
        if i == 0:
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=train_kwargs.batch_size,
                num_workers=train_kwargs.num_workers,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
            )
            data_loader_list.append(train_dataloader)
        else:
            evaluation_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=train_kwargs.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=train_kwargs.num_workers,
            )
            data_loader_list.append(evaluation_loader)
    return data_loader_list


# 5. Building the model
def load_model_state(model, ckpt_path, origin=None, component="encoder"):
    # key = 'state_dict' for pre-trained models, 'model' for FB Imagenet
    alt_component_names = {
        "encoder": "backbone",
    }
    alt_component_name = alt_component_names.get(component, "")

    loaded_dict = torch.load(ckpt_path)

    if origin == "fb":
        key = "model"
    else:
        key = "state_dict"

    state = loaded_dict[key]
    loading_state = {}
    model_keys = model.state_dict().keys()

    if any(s in ckpt_path for s in ("mocov3", "mae", "vit")):
        loading_state = get_vit_state(model, state, model_keys, loading_state)
    else:
        for k in list(state.keys()):
            k_split = k.split(".")
            k_0 = k_split[0]
            if len(k_split) > 1:
                k_1 = k_split[1]
            else:
                k_1 = ""

            k_heads = ".".join([k_0, k_1])
            if k_0 == component or k_heads == component:
                k_to_check = k.replace(f"{component}.", "")
            elif k_0 == alt_component_name or k_heads == alt_component_name:
                k_to_check = k.replace(f"{alt_component_name}.", "")
            else:
                k_to_check = k

            if k_to_check in model_keys:
                loading_state[k_to_check] = state[k]
    print(
        f"Loading {len(loading_state.keys())} layers for {component}"
        " Expected layers (approx):\n\tViT base: 150\n\tViT Large: 294\n\tResNet-50: 320"
    )
    model.load_state_dict(loading_state, strict=False)
    print(f"Loaded {component} from {ckpt_path}.")

    return model


# Function for supporting ViT loading
def get_vit_state(model, state, model_keys, loading_state, reorder_pos_emb=True):
    # Remove default ImageNet head from requiring loading
    model_keys = list(model_keys)[:-2]
    state_list = list(state.items())
    if reorder_pos_emb:
        pos_emb = state_list[1]
        conv_proj_w = state_list[2]
        conv_proj_b = state_list[3]

        state_list[1] = conv_proj_w
        state_list[2] = conv_proj_b
        state_list[3] = pos_emb

    for i, key in enumerate(model_keys):
        try:
            assert model.state_dict()[key].shape == state_list[i][1].shape
            loading_state[key] = state_list[i][1]
        except AssertionError:
            print(
                f"\nViT layer {i} {key}, does not match loading state layer {state_list[i][0]}"
            )
            print(
                f"Expected shape: {model.state_dict()[key].shape}, "
                f"from loading state got shape: {state_list[i][1].shape}"
            )
            continue
    return loading_state


# Freeze model weights
def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


def construct_head(input_dim, hidden_dim, dropout, type, non_lin, R):
    if type == "HML":
        output_dim = len(R[0])
        head = ConstrainedFFNNModel(
            input_dim, hidden_dim, output_dim, dropout, R, non_lin
        )
    elif type == "ML":
        output_dim = R
        head = MultiLabelFFNNModel(input_dim, hidden_dim, output_dim, dropout)
    else:
        raise NotImplementedError
    return head


def process_scheduler(train_kwargs, optimizer, lr):
    scheduler_name = train_kwargs.scheduler.name

    if scheduler_name not in _SCHEDULERS:
        raise NotImplementedError(
            f"Scheduler {scheduler_name} is unknown and not implemented"
        )
    else:
        if scheduler_name == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=train_kwargs.scheduler.warmup_epochs,
                max_epochs=train_kwargs.max_epochs,
                warmup_start_lr=train_kwargs.scheduler.warmup_start_lr
                if train_kwargs.scheduler.warmup_epochs > 0
                else lr,
                eta_min=train_kwargs.scheduler.min_lr,
            )
        elif scheduler_name == "reduce":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=train_kwargs.scheduler.mode,
                factor=train_kwargs.scheduler.factor,
                patience=train_kwargs.scheduler.patience,
                threshold=train_kwargs.scheduler.threshold,
                threshold_mode=train_kwargs.scheduler.threshold_mode,
                cooldown=train_kwargs.scheduler.cooldown,
                min_lr=train_kwargs.scheduler.min_lr,
                eps=train_kwargs.scheduler.eps,
            )
        elif scheduler_name == "step":
            scheduler = MultiStepLR(optimizer, train_kwargs.scheduler.lr_decay_steps)
        elif scheduler_name == "exponential":
            scheduler = ExponentialLR(optimizer, train_kwargs.scheduler.gamma)
        elif scheduler_name == "none":
            return None
    return scheduler


# OmegaConfiguration default selection (from SoloLearn library - MIT license)
# https://github.com/vturrisi/solo-learn/blob/main/solo/utils/misc.py#L448
def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value


def print_model_layers(model):
    for name, module in model.named_modules():
        print(name, module)


def construct_model(
    train_kwargs, Rs, enc_pth=None, test_mode=False, fine_tune_mode=False
):
    # Prepare encoder - Note (for json): use "IMAGENET1K_V1" for ImageNet weights, null for None
    backbone = train_kwargs.backbone
    backbone_name = backbone.name
    enc = _BACKBONES[backbone_name](weights=backbone.weights)
    if enc_pth:
        origin = omegaconf_select(train_kwargs, "backbone.origin")
        enc = load_model_state(enc, enc_pth, origin=origin)
    else:
        print("No encoder weights loaded.")
    set_requires_grad(enc, backbone.grad)
    if "resnet" in backbone_name:
        features_dim = enc.inplanes
        enc.fc = nn.Identity()
    elif "vit" in backbone_name:
        features_dim = _VIT_NUM_FEATURES[backbone_name]
        enc.heads.head = nn.Identity()
    else:
        print("No adjusment to:", backbone_name)

    # Prepare heads
    heads = nn.ModuleDict()
    train_heads = train_kwargs.heads

    for head in train_heads:
        head_obj = train_heads[head]

        # Get hierarchical matrix or number of classes (for non-hierarchical heads)
        if head in Rs:
            R = Rs[head]
        else:
            R = head_obj.num_classes
            Rs[head] = R

        heads[head] = construct_head(
            input_dim=features_dim,
            hidden_dim=head_obj.hidden_dim,
            dropout=head_obj.dropout,
            type=head_obj.type,
            non_lin=head_obj.non_lin,
            R=R,
        )
        component = f"heads.{head}"
        if test_mode or fine_tune_mode:
            heads[head] = load_model_state(
                heads[head], enc_pth, origin=origin, component=component
            )

    # Combine parameters of encoder and heads
    encoder_params = enc.parameters()
    heads_params = [head.parameters() for head in heads.values()]

    all_params = list(encoder_params) + [p for params in heads_params for p in params]

    # Prepare optimizer
    optimizer_name = train_kwargs.optimizer.name
    lr = train_kwargs.optimizer.lr
    wd = train_kwargs.optimizer.weight_decay
    extra_optimizer_args = train_kwargs.optimizer.extra_optimizer_args
    optimizer = _OPTIMIZERS[optimizer_name](
        all_params, lr=lr, weight_decay=wd, **extra_optimizer_args
    )

    # Prepare scheduler
    scheduler = process_scheduler(train_kwargs, optimizer, lr)

    model = LinearProbe(enc, heads, optimizer, scheduler, Rs)
    return model


def construct_one_hot_model(
    train_kwargs, enc_pth=None, test_mode=False, fine_tune_mode=False
):
    # Prepare encoder - Note (for json): use "IMAGENET1K_V1" for ImageNet weights, null for None
    # Note: dims argument should not include input_dim, that is automatically obtained
    backbone = train_kwargs.backbone
    backbone_name = backbone.name
    enc = _BACKBONES[backbone_name](weights=backbone.weights)
    if enc_pth:
        origin = omegaconf_select(train_kwargs, "backbone.origin")
        enc = load_model_state(enc, enc_pth, origin=origin)
    else:
        print("No encoder weights loaded.")
    set_requires_grad(enc, backbone.grad)
    if "resnet" in backbone_name:
        features_dim = enc.inplanes
        enc.fc = nn.Identity()
    elif "vit" in backbone_name:
        features_dim = _VIT_NUM_FEATURES[backbone_name]
        enc.heads = nn.Identity()
    else:
        print("No adjusment to:", backbone_name)

    classifier = construct_head(
        input_dim=features_dim,
        hidden_dim=train_kwargs.dims[0:-1],
        dropout=train_kwargs.dropout,
        type="ML",
        non_lin=None,
        R=train_kwargs.dims[-1],
    )
    component = "classifier"
    if test_mode or fine_tune_mode:
        classifier = load_model_state(
            classifier, enc_pth, origin=origin, component=component
        )

    # Combine parameters of encoder and heads
    encoder_params = enc.parameters()
    classifier_params = classifier.parameters()

    all_params = list(encoder_params) + list(classifier_params)

    # Prepare optimizer
    optimizer_name = train_kwargs.optimizer.name
    lr = train_kwargs.optimizer.lr
    wd = train_kwargs.optimizer.weight_decay
    extra_optimizer_args = train_kwargs.optimizer.extra_optimizer_args
    optimizer = _OPTIMIZERS[optimizer_name](
        all_params, lr=lr, weight_decay=wd, **extra_optimizer_args
    )

    # Prepare scheduler
    scheduler = process_scheduler(train_kwargs, optimizer, lr)

    model = OneHotLinearProbe(enc, classifier, optimizer, scheduler)
    return model


# 6. One-hot relevant functions
def process_one_hot_df(data_df, col):
    data_df[col] = data_df[col].apply(lambda x: ast.literal_eval(x)[0])
    return data_df
