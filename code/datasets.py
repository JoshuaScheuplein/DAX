import ast
import random
import platform
from pathlib import Path

import pydicom
import tifffile

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import preprocess_image
from utils import preprocess_image_v2


##############################################
''' AzureDataset Version 2 '''
# Random I0-Normalization and Neglog-Transform
##############################################
class AzureDatasetV2(Dataset):

    def __init__(self, dataset_path: str, subtract_lowpass=False, transforms=None):
        
        self.dataset_path = dataset_path
        self.subtract_lowpass = subtract_lowpass
        self.transforms = transforms

        self.samples = list(Path(self.dataset_path).rglob('*.tiff'))
        random.seed(42) # Set the seed for reproducibility
        random.shuffle(self.samples) # Random inplace shuffling of the given dataset samples

        print(f"\nFound {len(self.samples)} samples in dataset '{self.dataset_path}'\n")
        for i in range(100):
            print(f"Sample {i+1}: {self.samples[i]}")
        print("...")

        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # Load projection image
        sample_path = self.samples[idx]
        image_data = tifffile.imread(sample_path)
        assert image_data.dtype == np.uint16
        
        # Apply preprocessing
        preprocessed_image, raw_image = preprocess_image_v2(image_data,
                                                            clahe=False,
                                                            subtract_lowpass=self.subtract_lowpass,
                                                            dtype='float32')
        assert preprocessed_image.dtype == np.float32

        # Apply data augmentation
        if self.transforms is not None:
            tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
            images = self.transforms(tensor_image)
        else:
            images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
        assert [(image.dtype == torch.float32) for image in images]

        ##################################################################
        # Convert 1-channel grayscale images to 3-channel grayscale images
        # to be compatible with resnet50 architecture ...
        for i, image in enumerate(images):
            C, H, W = image.shape
            assert C == 1
            expanded_image = torch.cat([image] * 3, dim=0)
            assert expanded_image.shape == (3, H, W)
            images[i] = expanded_image
        ##################################################################
        
        return images, [] # The empty list is just added for compatibility with original DINO implementation ...
    

''' AzureDataset Version 1.3 '''
class AzureDataset(Dataset):

    def __init__(self, dataset_path: str, subtract_lowpass=False, transforms=None):
        
        self.dataset_path = dataset_path
        self.subtract_lowpass = subtract_lowpass
        self.transforms = transforms

        self.samples = list(Path(self.dataset_path).rglob('*.tiff'))
        random.seed(42) # Set the seed for reproducibility
        random.shuffle(self.samples) # Random inplace shuffling of the given dataset samples

        print(f"\nFound {len(self.samples)} samples in dataset '{self.dataset_path}'\n")
        for i in range(100):
            print(f"Sample {i+1}: {self.samples[i]}")
        print("...")

        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # Load projection image
        sample_path = self.samples[idx]
        image_data = tifffile.imread(sample_path)
        assert image_data.dtype == np.uint16
        
        # Apply preprocessing
        preprocessed_image, raw_image = preprocess_image(image_data,
                                                         clahe=False,
                                                         subtract_lowpass=self.subtract_lowpass,
                                                         dtype='float32')
        assert preprocessed_image.dtype == np.float32

        # Apply data augmentation
        if self.transforms is not None:
            tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
            images = self.transforms(tensor_image)
        else:
            images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
        assert [(image.dtype == torch.float32) for image in images]

        ##################################################################
        # Convert 1-channel grayscale images to 3-channel grayscale images
        # to be compatible with resnet50 architecture ...
        for i, image in enumerate(images):
            C, H, W = image.shape
            assert C == 1
            expanded_image = torch.cat([image] * 3, dim=0)
            assert expanded_image.shape == (3, H, W)
            images[i] = expanded_image
        ##################################################################
        
        return images, [] # The empty list is just added for compatibility with original DINO implementation ...
    

''' AzureDataset Version 1.2 '''
# class AzureDataset(Dataset):

#     def __init__(self, dataset_csv_file: str, dataset_path: str, subtract_lowpass=False, transforms=None):
        
#         self.dataset_csv_file = dataset_csv_file
#         self.dataset_path = dataset_path
#         self.subtract_lowpass = subtract_lowpass
#         self.transforms = transforms

#         self.samples = pd.read_csv(self.dataset_csv_file, keep_default_na=False, sep=';')
#         # Random shuffling of the given dataset samples
#         self.samples = self.samples.sample(frac=1.0, replace=False, random_state=42, ignore_index=True)
#         print(f"\nFound {len(self.samples)} samples in CSV file '{self.dataset_csv_file}'")

#         for i in range(100):
#             sample = self.samples.iloc[i]
#             sample_path = Path(sample['Datastore']) / Path(sample['Datastore Subset']) / Path(sample['Dataset Filename'])
#             print(f"Sample {i+1}: View = {sample['View Index']+1}/{sample['Views']} / Path = '{sample_path}'")
#         print(...)

#         return

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
        
#         # Load projection image
#         sample = self.samples.iloc[idx]
#         sample_path = Path(sample['Datastore']) / Path(sample['Datastore Subset']) / Path(sample['Dataset Filename'])
#         sample_path = Path(self.dataset_path) / sample_path
        
#         image_data = tifffile.imread(sample_path)
#         shape = ast.literal_eval(sample['2D Shape'])
#         assert image_data.shape == shape and image_data.dtype == np.uint16
        
#         # Apply preprocessing
#         preprocessed_image, raw_image = preprocess_image(image_data,
#                                                          clahe=False,
#                                                          subtract_lowpass=self.subtract_lowpass,
#                                                          dtype='float32')
#         assert preprocessed_image.shape == shape and preprocessed_image.dtype == np.float32

#         # Apply data augmentation
#         if self.transforms is not None:
#             tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
#             images = self.transforms(tensor_image)
#         else:
#             images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
#         assert [(image.dtype == torch.float32) for image in images]

#         ##################################################################
#         # Convert 1-channel grayscale images to 3-channel grayscale images
#         # to be compatible with resnet50 architecture ...
#         for i, image in enumerate(images):
#             C, H, W = image.shape
#             assert C == 1
#             expanded_image = torch.cat([image] * 3, dim=0)
#             assert expanded_image.shape == (3, H, W)
#             images[i] = expanded_image
#         ##################################################################
        
#         return images, [] # The empty list is just added for compatibility with original DINO implementation ...
    

''' AzureDataset Version 1.1 '''
# class AzureDataset(Dataset):

#     def __init__(self, path: str, subtract_lowpass=False, transforms=None):
        
#         self.path = Path(path)
#         self.subtract_lowpass = subtract_lowpass
#         self.transforms = transforms

#         if self.path.is_dir():
#             self.sample_paths = sorted(list(self.path.rglob('*.tiff')))
#         else:
#             raise ValueError(f"Could not find the specified dataset in '{self.path}'")
        
#         global_view_index = 0
#         self.total_num_samples, self.total_num_views = 0, 0
#         self.samples, num_samples = [], len(self.sample_paths)
#         for sample_path in self.sample_paths:
#             file_name = str(sample_path).split("/")[-1] # sample_path is a 'PosixPath' object

#             sample_index = file_name.split("_")[1]
#             sample_index = int(sample_index.replace("n", ""))

#             num_views = file_name.split("_")[2]
#             num_views = num_views.replace("v", "")
#             num_views = int(num_views.replace(".tiff", ""))

#             sample_dicts = []
#             for view_index in range(num_views):
#                 view_dict = {
#                     "sample_path"       : sample_path,
#                     "num_views"         : num_views,
#                     "view_index"        : view_index,
#                     "num_samples"       : num_samples,
#                     "sample_index"      : sample_index,
#                     "global_view_index" : global_view_index
#                 }
#                 sample_dicts.append(view_dict)
#                 global_view_index += 1
#             self.samples.extend(sample_dicts)
        
#         print(f"\nFound {len(self.samples)} samples in '{self.path}'")
#         max_samples = 100
#         self.samples = self.samples[:max_samples]
#         print(f"\nUsing only the first {max_samples} single projection images contained in the dataset:")
#         for sample in self.samples:
#             print(f"Sample: '{sample['sample_path']}' - Sample Index: {sample['sample_index']} - View Index: {sample['view_index']} - Global Index: {sample['global_view_index']}")
        
#         return

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
        
#         # Load projection image
#         sample = self.samples[idx]
#         assert idx == sample['global_view_index']
#         file_path = sample['sample_path']
        
#         image_data = tifffile.imread(file_path)
#         view_index = int(sample['view_index'])
#         image = image_data[view_index, :, :] # [976, 976]
#         assert (0 <= np.min(image)) and (np.max(image) <= 65535)
#         assert (image.shape == (976,976)) and (image.dtype == np.uint16)
        
#         # Apply image preprocessing
#         preprocessed_image, raw_image = preprocess_image(image,
#                                                          clahe=False,
#                                                          subtract_lowpass=self.subtract_lowpass,
#                                                          dtype='float32')
#         assert (preprocessed_image.shape == (976,976)) and (raw_image.shape == (976,976))
#         assert (preprocessed_image.dtype == np.float32) and (raw_image.dtype == np.float32)

#         # Apply data augmentation
#         if self.transforms is not None:
#             tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
#             images = self.transforms(tensor_image)
#         else:
#             images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
#         assert [(image.dtype == torch.float32) for image in images]

#         ##################################################################
#         # Convert 1-channel grayscale images to 3-channel grayscale images
#         # to be compatible with resnet50 architecture ...
#         for i, image in enumerate(images):
#             C, H, W = image.shape
#             assert C == 1
#             expanded_image = torch.cat([image] * 3, dim=0)
#             assert expanded_image.shape == (3, H, W)
#             images[i] = expanded_image
#         ##################################################################
        
#         return images, [] # The empty list is just added for compatibility with original DINO implementation ...


##############################################
''' DummyDataset Version 2 '''
# Random I0-Normalization and Neglog-Transform
##############################################
class DummyDatasetV2(Dataset):

    def __init__(self, path: str, subtract_lowpass=False, transforms=None, use_multi_view=True, visualize=False):
        
        self.subtract_lowpass = subtract_lowpass
        self.transforms = transforms
        self.visualize = visualize

        self.path = Path(path)
        if self.path.is_file() and (self.path.suffix == '.txt'):
            with open(self.path, 'r') as f:
                self.samples = [line.strip() for line in f.readlines()]
            self.format = 'dicom'
        elif self.path.is_dir():
            self.samples = sorted(list(self.path.rglob('*.tiff')))
            self.format = 'tiff'
        else:
            raise ValueError(f"Could not load specified dataset {self.path}")
        
        if not use_multi_view: # Only use single-view projection images
            self.samples = [sample for sample in self.samples if "View" not in str(sample)] # type(sample) = Path()
        
        num_plot_samples = 200
        print(f"\nFile paths for the first {num_plot_samples} samples from {self.path}:\n")
        for sample in self.samples[:num_plot_samples]:
            print(sample)
        
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # Load projection image
        file_path = self.samples[idx]
        if self.format == 'dicom':
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array
        else:
            image = tifffile.imread(file_path)
        assert (0 <= np.min(image)) and (np.max(image) <= 65535)
        assert (image.dtype == np.uint16)
        # assert (image.shape == (976,976)) and (image.dtype == np.uint16) # also possible: (1360, 1360)
        
        # Apply image preprocessing
        preprocessed_image, raw_image = preprocess_image_v2(image,
                                                            clahe=False,
                                                            subtract_lowpass=self.subtract_lowpass,
                                                            dtype='float32')
        # assert (preprocessed_image.shape == (976,976)) and (raw_image.shape == (976,976)) # also possible: (1360, 1360)
        assert (preprocessed_image.dtype == np.float32) and (raw_image.dtype == np.float32)

        # Apply data augmentation
        if self.transforms is not None:
            tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
            images = self.transforms(tensor_image)
        else:
            images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
        assert [(image.dtype == torch.float32) for image in images]

        ##################################################################
        # Convert 1-channel grayscale images to 3-channel grayscale images
        # to be compatible with resnet50 architecture ...
        for i, image in enumerate(images):
            C, H, W = image.shape
            assert C == 1
            expanded_image = torch.cat([image] * 3, dim=0)
            assert expanded_image.shape == (3, H, W)
            images[i] = expanded_image
        ##################################################################

        if self.visualize:
            image_buffer, num_global_views, num_local_views = {}, 0, 0
            image_buffer['Raw_Image'] = np.stack([raw_image] * 3, axis=-1)
            image_buffer['Preprocessed_Image'] = np.stack([preprocessed_image] * 3, axis=-1)
            for im in images:
                im = im.permute(1, 2, 0) # [3,:,:] -> [:,:,3]
                assert (im.dtype == torch.float32) and (im.shape[-1] == 3)
                if im.shape[0:2] == (224,224):
                    num_global_views += 1
                    image_buffer[f'Global_View_{num_global_views}'] = im
                elif im.shape[0:2] == (96,96):
                    num_local_views += 1
                    image_buffer[f'Local_View_{num_local_views}'] = im
            return image_buffer, [] # The empty list is just added for compatibility with original DINO implementation ...
        
        return images, [] # The empty list is just added for compatibility with original DINO implementation ...
    

''' DummyDataset Version 1.2 '''
class DummyDataset(Dataset):

    def __init__(self, path: str, subtract_lowpass=False, transforms=None, use_multi_view=True, visualize=False):
        
        self.subtract_lowpass = subtract_lowpass
        self.transforms = transforms
        self.visualize = visualize

        self.path = Path(path)
        if self.path.is_file() and (self.path.suffix == '.txt'):
            with open(self.path, 'r') as f:
                self.samples = [line.strip() for line in f.readlines()]
            self.format = 'dicom'
        elif self.path.is_dir():
            self.samples = sorted(list(self.path.rglob('*.tiff')))
            self.format = 'tiff'
        else:
            raise ValueError(f"Could not load specified dataset {self.path}")
        
        if not use_multi_view: # Only use single-view projection images
            self.samples = [sample for sample in self.samples if "View" not in str(sample)] # type(sample) = Path()
        
        num_plot_samples = 200
        print(f"\nFile paths for the first {num_plot_samples} samples from {self.path}:\n")
        for sample in self.samples[:num_plot_samples]:
            print(sample)
        
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # Load projection image
        file_path = self.samples[idx]
        if self.format == 'dicom':
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array
        else:
            image = tifffile.imread(file_path)
        assert (0 <= np.min(image)) and (np.max(image) <= 65535)
        assert (image.dtype == np.uint16)
        # assert (image.shape == (976,976)) and (image.dtype == np.uint16) # also possible: (1360, 1360)
        
        # Apply image preprocessing
        preprocessed_image, raw_image = preprocess_image(image,
                                                         clahe=False,
                                                         subtract_lowpass=self.subtract_lowpass,
                                                         dtype='float32')
        # assert (preprocessed_image.shape == (976,976)) and (raw_image.shape == (976,976)) # also possible: (1360, 1360)
        assert (preprocessed_image.dtype == np.float32) and (raw_image.dtype == np.float32)

        # Apply data augmentation
        if self.transforms is not None:
            tensor_image = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))
            images = self.transforms(tensor_image)
        else:
            images = [torch.from_numpy(np.expand_dims(preprocessed_image, axis=0))]
        assert [(image.dtype == torch.float32) for image in images]

        ##################################################################
        # Convert 1-channel grayscale images to 3-channel grayscale images
        # to be compatible with resnet50 architecture ...
        for i, image in enumerate(images):
            C, H, W = image.shape
            assert C == 1
            expanded_image = torch.cat([image] * 3, dim=0)
            assert expanded_image.shape == (3, H, W)
            images[i] = expanded_image
        ##################################################################

        if self.visualize:
            image_buffer, num_global_views, num_local_views = {}, 0, 0
            image_buffer['Raw_Image'] = np.stack([raw_image] * 3, axis=-1)
            image_buffer['Preprocessed_Image'] = np.stack([preprocessed_image] * 3, axis=-1)
            for im in images:
                im = im.permute(1, 2, 0) # [3,:,:] -> [:,:,3]
                assert (im.dtype == torch.float32) and (im.shape[-1] == 3)
                if im.shape[0:2] == (224,224):
                    num_global_views += 1
                    image_buffer[f'Global_View_{num_global_views}'] = im
                elif im.shape[0:2] == (96,96):
                    num_local_views += 1
                    image_buffer[f'Local_View_{num_local_views}'] = im
            return image_buffer, [] # The empty list is just added for compatibility with original DINO implementation ...
        
        return images, [] # The empty list is just added for compatibility with original DINO implementation ...


''' DummyDataset Version 1.1 '''
# class DummyDataset(Dataset):

#     def __init__(self, file_path: str, convert_to_uint8=True, transform=None, visualize=False):
        
#         self.convert_to_unit8 = convert_to_uint8
#         self.transform = transform

#         with open(file_path, 'r') as f:
#             self.samples = [line.strip() for line in f.readlines()]
        
#         print(f"\nSelected {len(self.samples)} Samples for Dummy Dataset:\n")
#         for sample in self.samples:
#             print(sample)

#         random.shuffle(self.samples)

#         self.visualize = visualize
        
#         return

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
        
#         file_path = self.samples[idx]
#         ds = pydicom.dcmread(file_path)
#         image = ds.pixel_array
#         assert (image.shape == (976,976)) and (image.dtype == np.uint16)
#         assert (np.min(image) >= 0) and (np.max(image) <= 65535)
#         # np.uint16 covers the numbers from 0 to 65535

#         if self.convert_to_unit8:
#             image = (image / 65535 * 255).astype(np.uint8)
#             assert (image.shape == (976,976)) and (image.dtype == np.uint8)
#             assert (np.min(image) >= 0) and (np.max(image) <= 255)
#             # np.uint8 covers the numbers from 0 to 255
#         else:
#             raise ValueError("Only uint8 is supported yet!")
        
#         if self.visualize:
#             image_buffer = []
#             image_buffer.append(np.expand_dims(image, axis=0))
        
#         '''
#         Different modes:
#             -> L - 8-bit pixels, grayscale
#             -> I - 32-bit signed integer pixels
#             -> F - 32-bit floating point pixels
#         '''
#         image = Image.fromarray(image, mode='L')
#         assert type(image) == Image.Image
        
#         if self.transform is not None:
#             images = self.transform(image)
#         assert [(image.dtype == torch.float32) for image in images]

#         ##################################################################
#         # Convert 1-channel grayscale images to 3-channel grayscale images
#         # to be compatible with resnet50 architecture ...
#         for i, image in enumerate(images):
#             C, H, W = image.shape
#             assert C == 1
#             expanded_image = torch.cat([image] * 3, dim=0)
#             assert expanded_image.shape == (3, H, W)
#             images[i] = expanded_image
#         ##################################################################

#         '''
#         For RGB and RGBA images, Matplotlib supports float32 and uint8 data types.
#         For grayscale, Matplotlib supports only float32.
#         If your array data does not meet one of these descriptions, you need to rescale it.
#         '''
#         if self.visualize:
#             for im in images:
#                 image = im.numpy()
#                 # image = ((image * std) + mean)
#                 image = ((image * 0.19125) + 0.45302)
#                 image = np.clip(image, a_min=0.0, a_max=1.0)
#                 assert image.dtype == np.float32
#                 assert (np.min(image) >= 0.0) and  (np.max(image) <= 1.0)
#                 image_buffer.append(image)
#             images = image_buffer
        
#         return images, [] # The empty list is just added for compatibility issues ...
