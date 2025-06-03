import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
import os

class SortDataset(Dataset):
    def __init__(self, N):
       self.N = N
    def __len__(self):
        return 10000000
    def __getitem__(self, idx):
        data = torch.zeros(self.N).normal_()
        ordering = torch.argsort(data)
        inputs = data
        return (inputs), (ordering)

class QAMNISTDataset(Dataset):
    """A QAMNIST dataset that includes plus and minus operations on MNIST digits."""
    def __init__(self, base_dataset, num_images, num_images_delta, num_repeats_per_input, num_operations, num_operations_delta):
        self.base_dataset = base_dataset

        self.num_images = num_images
        self.num_images_delta = num_images_delta
        self.num_images_range = self._calculate_num_images_range()

        self.operators = ["+", "-"]
        self.num_operations = num_operations
        self.num_operations_delta = num_operations_delta
        self.num_operations_range = self._calculate_num_operations_range()

        self.num_repeats_per_input = num_repeats_per_input

        self.current_num_digits = num_images
        self.current_num_operations = num_operations

        self.modulo_base = 10

        self.output_range = [0, 9]

    def _calculate_num_images_range(self):
        min_val = self.num_images - self.num_images_delta
        max_val = self.num_images + self.num_images_delta
        assert min_val >= 1, f"Minimum number of images must be at least 1, got {min_val}"
        return [min_val, max_val]

    def _calculate_num_operations_range(self):
        min_val = self.num_operations - self.num_operations_delta
        max_val = self.num_operations + self.num_operations_delta
        assert min_val >= 1, f"Minimum number of operations must be at least 1, got {min_val}"
        return [min_val, max_val]

    def set_num_digits(self, num_digits):
        self.current_num_digits = num_digits

    def set_num_operations(self, num_operations):
        self.current_num_operations = num_operations

    def _get_target_and_question(self, targets):
        question = []
        equations = []
        num_digits = self.current_num_digits
        num_operations = self.current_num_operations

        # Select the initial digit
        selection_idx = np.random.randint(num_digits)
        first_digit = targets[selection_idx]
        question.extend([selection_idx] * self.num_repeats_per_input)
        # Set current_value to the initial digit (mod is applied in each operation)
        current_value = first_digit % self.modulo_base

        # For each operation, build an equation line
        for _ in range(num_operations):
            # Choose the operator ('+' or '-')
            operator_idx = np.random.randint(len(self.operators))
            operator = self.operators[operator_idx]
            encoded_operator = -(operator_idx + 1)  # -1 for '+', -2 for '-'
            question.extend([encoded_operator] * self.num_repeats_per_input)
            
            # Choose the next digit
            selection_idx = np.random.randint(num_digits)
            digit = targets[selection_idx]
            question.extend([selection_idx] * self.num_repeats_per_input)
            
            # Compute the new value with immediate modulo reduction
            if operator == '+':
                new_value = (current_value + digit) % self.modulo_base
            else:  # operator is '-'
                new_value = (current_value - digit) % self.modulo_base
            
            # Build the equation string for this step
            equations.append(f"({current_value} {operator} {digit}) mod {self.modulo_base} = {new_value}")
            # Update current value for the next operation
            current_value = new_value

        target = current_value
        question_readable = "\n".join(equations)
        return target, question, question_readable

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        images, targets = [],[]
        for _ in range(self.current_num_digits):
            image, target = self.base_dataset[np.random.randint(self.__len__())]
            images.append(image)
            targets.append(target)

        observations = torch.repeat_interleave(torch.stack(images, 0), repeats=self.num_repeats_per_input, dim=0)
        target, question, question_readable = self._get_target_and_question(targets)
        return observations, question, question_readable, target

class ImageNet(Dataset):
    def __init__(self, which_split, transform):
        """
        Most simple form of the custom dataset structure. 
        Args:
            base_dataset (Dataset): The base dataset to sample from.
            N (int): The number of images to construct into an observable sequence.
            R (int): number of repeats
            operators (list): list of operators from which to sample
            action to take on observations (str): can be 'global' to compute operator over full observations, or 'select_K', where K=integer.
        """
        dataset = load_dataset('imagenet-1k', split=which_split, trust_remote_code=True)

        self.transform = transform
        self.base_dataset = dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data_item = self.base_dataset[idx]
        image = self.transform(data_item['image'].convert('RGB'))
        target = data_item['label']
        return image, target
  
class MazeImageFolder(ImageFolder):
    """
    A custom dataset class that extends the ImageFolder class.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=Image.open, 
                 is_valid_file=None, 
                 which_set='train', 
                 augment_p=0.5,
                 maze_route_length=10, 
                 trunc=False,
                 expand_range=True):
        super(MazeImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)
        self.which_set = which_set
        self.augment_p = augment_p
        self.maze_route_length = maze_route_length
        self.all_paths = {}
        self.trunc = trunc
        self.expand_range = expand_range
        
        self._preload()
        print('Solving all mazes...')
        for index in range(len(self.preloaded_samples)):
            path = self.get_solution(self.preloaded_samples[index])
            self.all_paths[index] = path

    def _preload(self):
        preloaded_samples = []
        with tqdm(total=self.__len__(), initial=0, leave=True, position=0, dynamic_ncols=True) as pbar:
            
            for index in range(self.__len__()):
                pbar.set_description('Loading mazes')
                path, target = self.samples[index]
                sample = self.loader(path)   
                sample = np.array(sample).astype(np.float32)/255     
                preloaded_samples.append(sample)
                pbar.update(1)
                if self.trunc and index == 999: break
        self.preloaded_samples = preloaded_samples

    def __len__(self):
        if hasattr(self, 'preloaded_samples') and self.preloaded_samples is not None:
            return len(self.preloaded_samples)
        else:
            return super().__len__()
        
    def get_solution(self, x):
        x = np.copy(x)
        # Find start (red) and end (green) pixel coordinates
        start_coords = np.argwhere((x == [1, 0, 0]).all(axis=2))
        end_coords = np.argwhere((x == [0, 1, 0]).all(axis=2))

        if len(start_coords) == 0 or len(end_coords) == 0:
            print("Start or end point not found.")
            return None
        
        start_y, start_x = start_coords[0]
        end_y, end_x = end_coords[0]

        current_y, current_x = start_y, start_x
        path = [4] * self.maze_route_length

        pi = 0
        while (current_y, current_x) != (end_y, end_x):
            next_y, next_x = -1, -1  # Initialize to invalid coordinates
            direction = -1  # Initialize to an invalid direction


            # Check Up
            if current_y > 0 and ((x[current_y - 1, current_x] == [0, 0, 1]).all() or (x[current_y - 1, current_x] == [0, 1, 0]).all()):
                next_y, next_x = current_y - 1, current_x
                direction = 0

            # Check Down
            elif current_y < x.shape[0] - 1 and ((x[current_y + 1, current_x] == [0, 0, 1]).all() or (x[current_y + 1, current_x] == [0, 1, 0]).all()):
                next_y, next_x = current_y + 1, current_x
                direction = 1

            # Check Left
            elif current_x > 0 and ((x[current_y, current_x - 1] == [0, 0, 1]).all() or (x[current_y, current_x - 1] == [0, 1, 0]).all()):
                next_y, next_x = current_y, current_x - 1
                direction = 2
                
            # Check Right
            elif current_x < x.shape[1] - 1 and ((x[current_y, current_x + 1] == [0, 0, 1]).all() or (x[current_y, current_x + 1] == [0, 1, 0]).all()):
                next_y, next_x = current_y, current_x + 1
                direction = 3

            
            path[pi] = direction
            pi += 1
            
            x[current_y, current_x] = [255,255,255] # mark the current as white to avoid going in circles
            current_y, current_x = next_y, next_x
            if pi == len(path): 
                break

        return np.array(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        sample = np.copy(self.preloaded_samples[index])
        
        path = np.copy(self.all_paths[index])
        
        if self.which_set == 'train':
            # Randomly rotate -90 or +90 degrees
            if random.random() < self.augment_p:
                which_rot = random.choice([-1, 1])
                sample = np.rot90(sample, k=which_rot, axes=(0, 1))
                for pi in range(len(path)):
                    if path[pi] == 0: path[pi] = 3 if which_rot == -1 else 2
                    elif path[pi] == 1: path[pi] = 2 if which_rot == -1 else 3
                    elif path[pi] == 2: path[pi] = 0 if which_rot == -1 else 1
                    elif path[pi] == 3: path[pi] = 1 if which_rot == -1 else 0
                    

            # Random horizontal flip
            if random.random() < self.augment_p:
                sample = np.fliplr(sample)
                for pi in range(len(path)):
                    if path[pi] == 2: path[pi] = 3
                    elif path[pi] == 3: path[pi] = 2
                

            # Random vertical flip
            if random.random() < self.augment_p:
                sample = np.flipud(sample)
                for pi in range(len(path)):
                    if path[pi] == 0: path[pi] = 1
                    elif path[pi] == 1: path[pi] = 0
                
        sample = torch.from_numpy(np.copy(sample)).permute(2,0,1)
        
        blue_mask = (sample[0] == 0) & (sample[1] == 0) & (sample[2] == 1)

        sample[:, blue_mask] = 1
        target = path


        if not self.expand_range:
            return sample, target
        return (sample*2)-1, (target)

class ParityDataset(Dataset):
    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vector = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        vector = vector.float()
        negatives = (vector == -1).to(torch.long)
        cumsum = torch.cumsum(negatives, dim=0)
        target = (cumsum % 2 != 0).to(torch.long)
        return vector, target

# Helper function for SARDClassificationDataset
def _sard_has_human_in_label_file(label_file_path: str) -> bool:
    """Checks if a SARD label file contains any valid bounding box (indicating a human)."""
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # A valid line has 5 parts: class_id, x_center, y_center, width, height
                if len(parts) == 5:
                    try:
                        # Attempt to convert to float to ensure parts are numeric
                        # class_id should be int, but float conversion check is fine here
                        [float(p) for p in parts]
                        return True  # Found a valid bounding box line
                    except ValueError:
                        continue # Line parts are not all numeric
    except IOError:
        # Could be FileNotFoundError or other read issues
        return False
    return False  # No valid bounding box line found or file not readable


class SARDClassificationDataset(VisionDataset):
    """
    SARD (Search and Rescue Dataset) for binary classification: "has human" vs "no human".
    A "has human" label is assigned if the corresponding label file contains at least one bounding box.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # root is expected to be the path to the 'search-and-rescue' directory
        # e.g., /path/to/datasets/sard-search-and-rescue
        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        
        self.image_dir = os.path.join(self.root, self.split, 'images')
        self.label_dir = os.path.join(self.root, self.split, 'labels')
        
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        # Label directory might not exist or be fully populated, which is handled by _sard_has_human_in_label_file

        self.samples = []
        self._load_metadata()

    def _load_metadata(self):
        for img_filename in os.listdir(self.image_dir):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_dir, img_filename)
                
                # Determine label based on the corresponding .txt file in labels_dir
                label_filename = os.path.splitext(img_filename)[0] + '.txt'
                label_path = os.path.join(self.label_dir, label_filename)
                
                has_human = _sard_has_human_in_label_file(label_path)
                # Label: 1 if "has human", 0 if "no human"
                self.samples.append((img_path, 1 if has_human else 0))

        if not self.samples:
            print(f"Warning: No image samples found in {self.image_dir} for split {self.split}.")
    

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # This should ideally not happen if _load_metadata worked correctly
            raise FileNotFoundError(f"Image file not found during __getitem__: {img_path}")
        except Exception as e:
            raise IOError(f"Error opening image {img_path}: {e}")


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.samples)
