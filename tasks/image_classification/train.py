import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')
import torch.nn as nn
from tqdm.auto import tqdm

from data.custom_datasets import ImageNet, SARDClassificationDataset
from torchvision import datasets
from torchvision import transforms
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss # Used by CTM, LSTM
from utils.losses import attention_diversity_loss # Added for CTM attention diversity
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

from autoclip.torch import QuantileClip

import gc
import torchvision
torchvision.disable_beta_transforms_warning()


import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)

# Helper function for MCC
def calculate_mcc(tp, tn, fp, fn):
    """Calculates Matthews Correlation Coefficient."""
    # Cast to float64 to prevent overflow with large numbers
    tp = np.float64(tp)
    tn = np.float64(tn)
    fp = np.float64(fp)
    fn = np.float64(fn)

    numerator = (tp * tn) - (fp * fn)
    denominator_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator_sq == 0:
        return 0.0 # Avoid division by zero; MCC is 0 if any sum in denom is zero
    return numerator / np.sqrt(denominator_sq)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, default='ctm', choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    # Common
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet18-4', help='Type of backbone featureiser.')
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (CTM, LSTM).')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none', help='Type of positional embedding (CTM, LSTM).',
                        choices=['none',
                                 'learnable-fourier',
                                 'multi-learnable-fourier',
                                 'custom-rotational'])
    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).')
    parser.add_argument('--n_synch_out', type=int, default=512, help='Number of neurons to use for output synch (CTM only).')
    parser.add_argument('--n_synch_action', type=int, default=512, help='Number of neurons to use for observation/action synch (CTM only).')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory (CTM only).')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components (backbone, synapses if CTM).')
    parser.add_argument('--num_workers_train', type=int, default=1, help='Num workers training.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/scratch', help='Directory for logging.')
    parser.add_argument('--attention_diversity_coeff', type=float, default=0.0, help='Coefficient for attention diversity loss (CTM only).')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use.')
    parser.add_argument('--data_root', type=str, default='data/', help='Where to save dataset.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True, help='Should use strict reload for model weights.') # Added back
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')


    args = parser.parse_args()
    # Add SARD specific default for data_root if dataset is sard
    if args.dataset == 'sard' and args.data_root == 'data/': # if default 'data/' is used
        args.data_root = 'data/sard-search-and-rescue' # Default SARD path relative to workspace
        print(f"Dataset is SARD, changing data_root to default: {args.data_root}")
    return args


def get_dataset(dataset, root):
    if dataset=='imagenet':
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])

        class_labels = list(IMAGENET2012_CLASSES.values())

        train_data = ImageNet(which_split='train', transform=train_transform)
        test_data = ImageNet(which_split='validation', transform=test_transform)
    elif dataset=='cifar10':
        dataset_mean = [0.49139968, 0.48215827, 0.44653124]
        dataset_std = [0.24703233, 0.24348505, 0.26158768]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        train_transform = transforms.Compose(
            [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
            ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            normalize,
            ])
        train_data = datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(root, train=False, transform=test_transform, download=True)
        class_labels = ['air', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset=='cifar100':
        dataset_mean = [0.5070751592371341, 0.48654887331495067, 0.4409178433670344]
        dataset_std = [0.2673342858792403, 0.2564384629170882, 0.27615047132568393]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        train_transform = transforms.Compose(
            [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            normalize,
            ])
        train_data = datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(root, train=False, transform=test_transform, download=True)
        idx_order = np.argsort(np.array(list(train_data.class_to_idx.values())))
        class_labels = list(np.array(list(train_data.class_to_idx.keys()))[idx_order])
    elif dataset=='sard': # Added SARD dataset handling
        
        dataset_mean = [0.44452422857284546, 0.45209550857543945, 0.29234638810157776]
        dataset_std = [0.19780108332633972, 0.19067855179309845, 0.15231430530548096]

        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Basic augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        
        
        class_labels = ['no human', 'has human'] # Binary classification
        
        # The SARDClassificationDataset expects the root to be the base SARD directory 
        # e.g., 'data/sard-search-and-rescue' which contains 'train', 'valid', 'test' subdirs
        # The `root` argument to `get_dataset` should point to this base SARD directory.
        train_data = SARDClassificationDataset(root=root, split='train', transform=train_transform)

        # For SARD, often the Kaggle dataset splits into train/valid/test. 
        # If your SARD data only has 'train' and 'test', you might need to split 'train' further
        # or use the provided 'valid' set if available.
        # Assuming a 'valid' split exists for now, similar to train/test.
        # You may need to adjust this if your SARD directory structure is different or if 'valid' is missing.
        valid_root_path = os.path.join(os.path.dirname(root), 'valid') if not os.path.exists(os.path.join(root, 'valid')) and os.path.exists(os.path.join(os.path.dirname(root), 'valid')) else root
        if not os.path.exists(os.path.join(root, 'valid')):
            print(f"Warning: No 'valid' directory found directly in {root}. Trying to use test set as validation or you need to create a validation split.")
            # Option 1: Use the test set as validation if no dedicated validation set
            # test_data = SARDClassificationDataset(root=root, split='test', transform=test_transform)
            # Option 2: Create a validation split from training data (more involved)
            # For now, let's assume if 'valid' isn't in root, we might look for it one level up (common in Kaggle downloads)
            # Or, more simply, just try to load 'test' as the validation set if 'valid' is truly missing.
            # This part needs to be robust based on how the SARD data is structured in `args.data_root`
            # Trying to use the 'test' split for validation if 'valid' is not found. 
            # This may not be ideal for rigorous evaluation.
            test_data = SARDClassificationDataset(root=root, split='test', transform=test_transform)
        else:
            test_data = SARDClassificationDataset(root=root, split='valid', transform=test_transform) # Using 'valid' as test_data for consistency in var name

    else:
        raise NotImplementedError

    return train_data, test_data, class_labels, dataset_mean, dataset_std



if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'sard']

    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    
    # Data
    train_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(args.dataset, args.data_root)
    
    num_workers_test = 1 # Defaulting to 1, change if needed
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test, drop_last=False)
    
    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = len(class_labels)
    print(f"Number of output classes (args.out_dims) set to: {args.out_dims}")

    # For total reproducibility
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)

    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model} on {device}')

    # Build model conditionally
    model = None
    if args.model == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    elif args.model == 'lstm':
         model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs) 

    model.train()

    
    print(f'Total params: {sum(p.numel() for p in model.parameters())}')
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip parameters that don't require gradients
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    # Optimizer and scheduler (Common setup)
    if len(no_decay_names) and args.weight_decay!=0:
        optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay':args.weight_decay},
                                       {'params': no_decay_params, 'weight_decay':0}],
                                  lr=args.lr,
                                  eps=1e-8 if not args.use_amp else 1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    eps=1e-8 if not args.use_amp else 1e-6,
                                    weight_decay=args.weight_decay)
    

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError


    # Metrics tracking
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_mccs = [] if args.dataset == 'sard' or args.out_dims == 2 else None # MCC for SARD/binary
    test_mccs = [] if args.dataset == 'sard' or args.out_dims == 2 else None # MCC for SARD/binary
    iters = []
    # Conditional metrics for CTM/LSTM
    train_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None
    test_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None

    scaler = torch.cuda.amp.GradScaler("cuda", enabled=args.use_amp) if "cuda" in device else torch.amp.GradScaler("cpu", enabled=args.use_amp)

    # Reloading logic
    if args.reload:
        checkpoint_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(checkpoint_path):
            print(f'Reloading from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not args.strict_reload: print('WARNING: not using strict reload for model weights!')
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict_reload)
            print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']
                # Load common metrics
                train_losses = checkpoint['train_losses']
                test_losses = checkpoint['test_losses']
                train_accuracies = checkpoint['train_accuracies'] 
                test_accuracies = checkpoint['test_accuracies'] 
                iters = checkpoint['iters']

                # Load conditional metrics if they exist in checkpoint and are expected for current model
                if args.model in ['ctm', 'lstm']:
                    train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                    test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                if (args.dataset == 'sard' or args.out_dims == 2) and 'train_mccs' in checkpoint: # Load MCC if present
                    train_mccs = checkpoint['train_mccs']
                    test_mccs = checkpoint['test_mccs']

            else:
                print('Only reloading model!')

            if 'torch_rng_state' in checkpoint:
                # Reset seeds
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Conditional Compilation
    if args.do_compile:
        print('Compiling...')
        if hasattr(model, 'backbone'):
            model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)

        # Compile synapses only for CTM
        if args.model == 'ctm':
            model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)

    # Define class weights for SARD dataset if applicable
    sard_class_weights = None
    if args.dataset == 'sard':
        # For SARD: Class 0 ("no human") is minority, Class 1 ("has human") is majority (4.5:1 ratio)
        # For CrossEntropyLoss, weights should be inverse of class frequency
        # If class 1 is 4.5x more frequent than class 0, then:
        # weight_class_0 = 4.5, weight_class_1 = 1.0 (normalized)
        # This gives more importance to the minority class (no human)
        sard_class_weights = torch.tensor([4.5, 1.0], device=device)
        print(f"Using CrossEntropyLoss class weights for SARD: {sard_class_weights}")

    # Training
    iterator = iter(trainloader)


    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)

            loss = None
            accuracy = None
            # Model-specific forward and loss calculation
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                if args.do_compile: # CUDAGraph marking for clean compile
                     torch.compiler.cudagraph_mark_step_begin()

                batch_mcc = None # Initialize batch_mcc
                if args.model == 'ctm':
                    predictions, certainties, synchronisation = model(inputs)
                    loss_args = {'predictions': predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                    if args.dataset == 'sard':
                        loss_args['class_weights'] = sard_class_weights
                    loss, where_most_certain = image_classification_loss(**loss_args)
                    
                    if args.attention_diversity_coeff > 0.0 and hasattr(model, 'attention_weights_for_loss_buffer'):
                        attention_weights_buffer = model.attention_weights_for_loss_buffer
                        if attention_weights_buffer: # Check if the buffer is not empty
                            last_iter_attn_weights = attention_weights_buffer[-1]
                            # Squeeze QuerySeqLen dim: (B, H, 1, KeySeqLen) -> (B, H, KeySeqLen)
                            squeezed_attn_weights = last_iter_attn_weights.squeeze(2)
                            div_loss = attention_diversity_loss(squeezed_attn_weights)
                            loss = loss + args.attention_diversity_coeff * div_loss

                    # Accuracy for pbar
                    preds_for_acc = predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device),where_most_certain]

                    accuracy = (preds_for_acc == targets).float().mean().item()
                    pbar_desc = f'CTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})'
                    if args.dataset == 'sard' or args.out_dims == 2:
                        tp = torch.sum((preds_for_acc == 1) & (targets == 1)).item()
                        tn = torch.sum((preds_for_acc == 0) & (targets == 0)).item()
                        fp = torch.sum((preds_for_acc == 1) & (targets == 0)).item()
                        fn = torch.sum((preds_for_acc == 0) & (targets == 1)).item()
                        batch_mcc = calculate_mcc(tp, tn, fp, fn)
                        pbar_desc += f'. MCC={batch_mcc:0.3f}'

                elif args.model == 'lstm':
                    predictions, certainties, synchronisation = model(inputs)
                    loss_args = {'predictions': predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                    if args.dataset == 'sard':
                        loss_args['class_weights'] = sard_class_weights
                    loss, where_most_certain = image_classification_loss(**loss_args)
                    # LSTM where_most_certain will just be -1 because use_most_certain is False owing to stability issues with LSTM training
                    preds_for_acc = predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device),where_most_certain]
                    accuracy = (preds_for_acc == targets).float().mean().item()
                    pbar_desc = f'LSTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})'
                    if args.dataset == 'sard' or args.out_dims == 2:
                        tp = torch.sum((preds_for_acc == 1) & (targets == 1)).item()
                        tn = torch.sum((preds_for_acc == 0) & (targets == 0)).item()
                        fp = torch.sum((preds_for_acc == 1) & (targets == 0)).item()
                        fn = torch.sum((preds_for_acc == 0) & (targets == 1)).item()
                        batch_mcc = calculate_mcc(tp, tn, fp, fn)
                        pbar_desc += f'. MCC={batch_mcc:0.3f}'

                elif args.model == 'ff':
                    predictions = model(inputs)
                    if args.dataset == 'sard':
                        loss = nn.CrossEntropyLoss(weight=sard_class_weights)(predictions, targets)
                    else:
                        loss = nn.CrossEntropyLoss()(predictions, targets)
                    
                    preds_for_acc = predictions.argmax(1)
                    accuracy = (preds_for_acc == targets).float().mean().item()
                    pbar_desc = f'FF Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}'
                    if args.dataset == 'sard' or args.out_dims == 2:
                        tp = torch.sum((preds_for_acc == 1) & (targets == 1)).item()
                        tn = torch.sum((preds_for_acc == 0) & (targets == 0)).item()
                        fp = torch.sum((preds_for_acc == 1) & (targets == 0)).item()
                        fn = torch.sum((preds_for_acc == 0) & (targets == 1)).item()
                        batch_mcc = calculate_mcc(tp, tn, fp, fn)
                        pbar_desc += f'. MCC={batch_mcc:0.3f}'

            scaler.scale(loss).backward()

            if args.gradient_clipping!=-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            pbar.set_description(f'Dataset={args.dataset}. Model={args.model}. {pbar_desc}')


            # Metrics tracking and plotting (conditional logic needed)
            if (bi % args.track_every == 0 or bi == args.warmup_steps) and (bi != 0 or args.reload_model_only):

                iters.append(bi)
                current_train_losses = []
                current_test_losses = []
                current_train_accuracies = [] # Holds list of accuracies per tick for CTM/LSTM, single value for FF
                current_test_accuracies = [] # Holds list of accuracies per tick for CTM/LSTM, single value for FF
                current_train_accuracies_most_certain = [] # Only for CTM/LSTM
                current_test_accuracies_most_certain = [] # Only for CTM/LSTM
                current_train_mcc = None # For SARD/binary
                current_test_mcc = None  # For SARD/binary


                # Reset BN stats using train mode
                pbar.set_description('Resetting BN')
                model.train()
                for module in model.modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        module.reset_running_stats()

                pbar.set_description('Tracking: Computing TRAIN metrics')
                with torch.no_grad(): # Should use inference_mode? CTM/LSTM scripts used no_grad
                    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test)
                    all_targets_list = []
                    all_predictions_list = [] # List to store raw predictions (B, C, T) or (B, C)
                    all_predictions_most_certain_list = [] # Only for CTM/LSTM
                    all_losses = []
                    # For MCC (binary classification)
                    all_preds_for_mcc_list = [] # Store final predictions (argmax or direct if binary) for MCC

                    with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        for inferi, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                            # Model-specific forward and loss for evaluation
                            if args.model == 'ctm':
                                these_predictions, certainties, _ = model(inputs)
                                loss_args_eval = {'predictions': these_predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                                if args.dataset == 'sard':
                                    loss_args_eval['class_weights'] = sard_class_weights
                                loss, where_most_certain = image_classification_loss(**loss_args_eval)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy()) # Shape (B, T)
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy()) # Shape (B,)
                                if args.dataset == 'sard' or args.out_dims == 2:
                                    # For SARD/binary, predictions are (B, 2, T), want argmax over class dim for each tick for MCC later
                                    # Or if using most_certain, it's (B,) based on that tick
                                    all_preds_for_mcc_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())

                            elif args.model == 'lstm':
                                these_predictions, certainties, _ = model(inputs)
                                loss_args_eval = {'predictions': these_predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                                if args.dataset == 'sard':
                                    loss_args_eval['class_weights'] = sard_class_weights
                                loss, where_most_certain = image_classification_loss(**loss_args_eval)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy()) # Shape (B, T)
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy()) # Shape (B,)
                                if args.dataset == 'sard' or args.out_dims == 2:
                                     all_preds_for_mcc_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())

                            elif args.model == 'ff':
                                these_predictions = model(inputs)
                                if args.dataset == 'sard':
                                    loss = nn.CrossEntropyLoss(weight=sard_class_weights)(these_predictions, targets)
                                else:
                                    loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy()) # Shape (B,)
                                if args.dataset == 'sard' or args.out_dims == 2:
                                    all_preds_for_mcc_list.append(these_predictions.argmax(1).detach().cpu().numpy())

                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1 : break # Check condition >= N-1
                            pbar_inner.set_description(f'Computing metrics for train (Batch {inferi+1})')
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list) # Shape (N, T) or (N,)
                    train_losses.append(np.mean(all_losses))

                    if args.model in ['ctm', 'lstm']:
                        # Accuracies per tick for CTM/LSTM
                        current_train_accuracies = np.mean(all_predictions == all_targets[...,np.newaxis], axis=0) # Mean over batch dim -> Shape (T,)
                        train_accuracies.append(current_train_accuracies)
                        # Most certain accuracy
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)
                        current_train_accuracies_most_certain = (all_targets == all_predictions_most_certain).mean()
                        train_accuracies_most_certain.append(current_train_accuracies_most_certain)
                    else: # FF
                         current_train_accuracies = (all_targets == all_predictions).mean() # Shape scalar
                         train_accuracies.append(current_train_accuracies)
                
                    if args.dataset == 'sard' or args.out_dims == 2:
                        all_preds_for_mcc = np.concatenate(all_preds_for_mcc_list)
                        # all_targets is already (N,)
                        tp = np.sum((all_preds_for_mcc == 1) & (all_targets == 1))
                        tn = np.sum((all_preds_for_mcc == 0) & (all_targets == 0))
                        fp = np.sum((all_preds_for_mcc == 1) & (all_targets == 0))
                        fn = np.sum((all_preds_for_mcc == 0) & (all_targets == 1))
                        current_train_mcc = calculate_mcc(tp, tn, fp, fn)
                        train_mccs.append(current_train_mcc)

                del these_predictions
                

                # Switch to eval mode for test metrics (fixed BN stats)
                model.eval()
                pbar.set_description('Tracking: Computing TEST metrics')
                with torch.inference_mode(): # Use inference_mode for test eval
                    loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test)
                    all_targets_list = []
                    all_predictions_list = []
                    all_predictions_most_certain_list = [] # Only for CTM/LSTM
                    all_losses = []
                    all_preds_for_mcc_list_test = [] # For MCC on test set

                    with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                       for inferi, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                            # Model-specific forward and loss for evaluation
                            if args.model == 'ctm':
                                these_predictions, certainties, _ = model(inputs)
                                loss_args_eval = {'predictions': these_predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                                if args.dataset == 'sard':
                                    loss_args_eval['class_weights'] = sard_class_weights
                                loss, where_most_certain = image_classification_loss(**loss_args_eval)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())
                                if args.dataset == 'sard' or args.out_dims == 2:
                                    all_preds_for_mcc_list_test.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())

                            elif args.model == 'lstm':
                                these_predictions, certainties, _ = model(inputs)
                                loss_args_eval = {'predictions': these_predictions, 'certainties': certainties, 'targets': targets, 'use_most_certain': True}
                                if args.dataset == 'sard':
                                    loss_args_eval['class_weights'] = sard_class_weights
                                loss, where_most_certain = image_classification_loss(**loss_args_eval)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())
                                if args.dataset == 'sard' or args.out_dims == 2:
                                    all_preds_for_mcc_list_test.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())

                            elif args.model == 'ff':
                                these_predictions = model(inputs)
                                if args.dataset == 'sard':
                                    loss = nn.CrossEntropyLoss(weight=sard_class_weights)(these_predictions, targets)
                                else:
                                    loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())
                                if args.dataset == 'sard' or args.out_dims == 2:
                                    all_preds_for_mcc_list_test.append(these_predictions.argmax(1).detach().cpu().numpy())

                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                            pbar_inner.set_description(f'Computing metrics for test (Batch {inferi+1})')
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list)
                    test_losses.append(np.mean(all_losses))

                    if args.model in ['ctm', 'lstm']:
                        current_test_accuracies = np.mean(all_predictions == all_targets[...,np.newaxis], axis=0)
                        test_accuracies.append(current_test_accuracies)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)
                        current_test_accuracies_most_certain = (all_targets == all_predictions_most_certain).mean()
                        test_accuracies_most_certain.append(current_test_accuracies_most_certain)
                    else: # FF
                         current_test_accuracies = (all_targets == all_predictions).mean()
                         test_accuracies.append(current_test_accuracies)

                    if args.dataset == 'sard' or args.out_dims == 2:
                        all_preds_for_mcc_test = np.concatenate(all_preds_for_mcc_list_test)
                        tp = np.sum((all_preds_for_mcc_test == 1) & (all_targets == 1))
                        tn = np.sum((all_preds_for_mcc_test == 0) & (all_targets == 0))
                        fp = np.sum((all_preds_for_mcc_test == 1) & (all_targets == 0))
                        fn = np.sum((all_preds_for_mcc_test == 0) & (all_targets == 1))
                        current_test_mcc = calculate_mcc(tp, tn, fp, fn)
                        test_mccs.append(current_test_mcc)

                # Plotting (conditional)
                figacc = plt.figure(figsize=(10, 10))
                axacc_train = figacc.add_subplot(211)
                axacc_test = figacc.add_subplot(212)
                cm = sns.color_palette("viridis", as_cmap=True)

                if args.model in ['ctm', 'lstm']:
                    # Plot per-tick accuracy for CTM/LSTM
                    train_acc_arr = np.array(train_accuracies) # Shape (N_iters, T)
                    test_acc_arr = np.array(test_accuracies) # Shape (N_iters, T)
                    num_ticks = train_acc_arr.shape[1]
                    for ti in range(num_ticks):
                         axacc_train.plot(iters, train_acc_arr[:, ti], color=cm(ti / num_ticks), alpha=0.3)
                         axacc_test.plot(iters, test_acc_arr[:, ti], color=cm(ti / num_ticks), alpha=0.3)
                    # Plot most certain accuracy
                    axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')
                    axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')
                else: # FF
                    axacc_train.plot(iters, train_accuracies, 'k-', alpha=0.7, label='Accuracy') # Simple line
                    axacc_test.plot(iters, test_accuracies, 'k-', alpha=0.7, label='Accuracy')

                axacc_train.set_title('Train Accuracy')
                axacc_test.set_title('Test Accuracy')
                axacc_train.legend(loc='lower right')
                axacc_test.legend(loc='lower right')
                axacc_train.set_xlim([0, args.training_iterations])
                axacc_test.set_xlim([0, args.training_iterations])
                if args.dataset=='cifar10':
                    axacc_train.set_ylim([0.75, 1])
                    axacc_test.set_ylim([0.75, 1])



                figacc.tight_layout()
                figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                plt.close(figacc)

                figloss = plt.figure(figsize=(10, 5))
                axloss = figloss.add_subplot(111)
                axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]:.4f}')
                axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]:.4f}')
                axloss.legend(loc='upper right')
                axloss.set_xlim([0, args.training_iterations])
                axloss.set_ylim(bottom=0)

                figloss.tight_layout()
                figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                plt.close(figloss)

                # Plot MCC if SARD/binary
                if args.dataset == 'sard' or args.out_dims == 2:
                    figmcc = plt.figure(figsize=(10, 5))
                    axmcc = figmcc.add_subplot(111)
                    axmcc.plot(iters, train_mccs, 'b-', linewidth=1, alpha=0.8, label=f'Train MCC: {train_mccs[-1]:.4f}')
                    axmcc.plot(iters, test_mccs, 'r-', linewidth=1, alpha=0.8, label=f'Test MCC: {test_mccs[-1]:.4f}')
                    axmcc.legend(loc='lower right') # MCC can be negative, so lower right might be better
                    axmcc.set_xlim([0, args.training_iterations])
                    axmcc.set_ylim([-1, 1]) # MCC range is -1 to 1
                    axmcc.set_title('Train/Test MCC')
                    figmcc.tight_layout()
                    figmcc.savefig(f'{args.log_dir}/mcc.png', dpi=150)
                    plt.close(figmcc)

                # Conditional Visualization (Only for CTM/LSTM)
                if args.model in ['ctm', 'lstm']:
                    try: # For safety
                        
                        for i in range(3):
                            # For SARD dataset, create a specific visualization sample to track file paths
                            if args.dataset == 'sard':
                                # Create a deterministic sample from test_data for visualization
                                visualization_sample_index = i # Use first sample for consistency
                                if visualization_sample_index < len(test_data):
                                    inputs_viz_single, targets_viz_single = test_data[visualization_sample_index]
                                    inputs_viz = inputs_viz_single.unsqueeze(0).to(device)  # Add batch dimension
                                    targets_viz = torch.tensor([targets_viz_single]).to(device)
                                    viz_dataset = test_data
                                    viz_sample_index = visualization_sample_index
                                else:
                                    # Fallback to batch approach if sample index is out of range
                                    inputs_viz, targets_viz = next(iter(testloader))
                                    inputs_viz = inputs_viz.to(device)
                                    targets_viz = targets_viz.to(device)
                                    viz_dataset = None
                                    viz_sample_index = None
                            else:
                                # For non-SARD datasets, use the existing batch approach
                                inputs_viz, targets_viz = next(iter(testloader)) # Get a fresh batch
                                inputs_viz = inputs_viz.to(device)
                                targets_viz = targets_viz.to(device)
                                viz_dataset = None
                                viz_sample_index = None
                                if i > 0:
                                    continue

                            pbar.set_description('Tracking: Processing test data for viz')
                            predictions_viz, certainties_viz, _, pre_activations_viz, post_activations_viz, attention_tracking_viz = model(inputs_viz, track=True)

                            att_shape = (model.kv_features.shape[2], model.kv_features.shape[3])
                            attention_tracking_viz = attention_tracking_viz.reshape(
                                attention_tracking_viz.shape[0], 
                                attention_tracking_viz.shape[1], -1, att_shape[0], att_shape[1])

                            pbar.set_description('Tracking: Neural dynamics plot')
                            plot_neural_dynamics(post_activations_viz, 100, args.log_dir, axis_snap=True)

                            random_imgi = np.random.randint(0, inputs_viz.shape[0])

                            imgi = 0 # Visualize the first image in the batch
                            img_to_gif = np.moveaxis(np.clip(inputs_viz[imgi].detach().cpu().numpy()*np.array(dataset_std).reshape(len(dataset_std), 1, 1) + np.array(dataset_mean).reshape(len(dataset_mean), 1, 1), 0, 1), 0, -1)
                            
                            pbar.set_description('Tracking: Producing attention gif')
                            make_classification_gif(img_to_gif,
                                                    targets_viz[imgi].item(),
                                                    predictions_viz[imgi].detach().cpu().numpy(),
                                                    certainties_viz[imgi].detach().cpu().numpy(),
                                                    post_activations_viz[:,imgi], 
                                                    attention_tracking_viz[:,imgi], 
                                                    class_labels,
                                                    f'{args.log_dir}/{i}_{imgi}_attention.gif',
                                                    dataset=viz_dataset,
                                                    sample_index=viz_sample_index,
                                                    )
                            del predictions_viz, certainties_viz, pre_activations_viz, post_activations_viz, attention_tracking_viz
                    except Exception as e:
                        print(f"Visualization failed for model {args.model}: {e}")
                    
                    

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.train() # Switch back to train mode


            # Save model checkpoint (conditional metrics)
            if (bi % args.save_every == 0 or bi == args.training_iterations - 1) and bi != start_iter:
                pbar.set_description('Saving model checkpoint...')
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iteration': bi,
                    # Always save these
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies, # This is list of scalars for FF, list of arrays for CTM/LSTM
                    'test_accuracies': test_accuracies, # This is list of scalars for FF, list of arrays for CTM/LSTM
                    'iters': iters,
                    'args': args, # Save args used for this run
                    # RNG states
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                }
                # Conditionally add metrics specific to CTM/LSTM
                if args.model in ['ctm', 'lstm']:
                    checkpoint_data['train_accuracies_most_certain'] = train_accuracies_most_certain
                    checkpoint_data['test_accuracies_most_certain'] = test_accuracies_most_certain
                if args.dataset == 'sard' or args.out_dims == 2: # Save MCC if SARD/binary
                    checkpoint_data['train_mccs'] = train_mccs
                    checkpoint_data['test_mccs'] = test_mccs

                torch.save(checkpoint_data, f'{args.log_dir}/checkpoint.pt')

            pbar.update(1)
