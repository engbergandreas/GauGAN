import os
import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F


class CamVidDataset(data.Dataset):
    def __init__(self, img_dir, labels_dir, transform=None, target_transform=None):
        self.image_paths = make_dataset(img_dir)
        self.label_paths = make_dataset(labels_dir)

        self.transform = transform
        
        self.target_transform = target_transform
        
        assert len(self.image_paths) == len(self.label_paths), "The #images in %s and %s do not match. Is there something wrong?"

        self.dataset_size = len(self.label_paths)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) :
        #label
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        label_tensor = self.target_transform(label) * 255.0 #transform to make it into a tensor, but we also get it back from [0,1] range to [0,255] range
        label_tensor[label_tensor == 255] = 32  # not sure about this step(?) look at pix2pix_dataset.py line:64
        label_tensor = label_tensor.long() #transform it to long int instead of floats

        #image
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        image = image.convert('RGB')
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            #"seg_map": input_semantics,
            "label": label_tensor
        }

def make_dataset(dir) :
    images = []

    images = os.listdir(dir)
    #print(images)
    # annotations_files = [os.path.join(os.path.realpath("."), annotations_dir, x) for x in annotations_files]
    images = [os.path.join(dir, x) for x in images]
    # print(os.listdir(dir))
    # possible_filelist = os.path.join(dir, 'files.list')
    # if os.path.isfile(possible_filelist):
    #     with open(possible_filelist, 'r') as f:
    #         images = f.read().splitlines()
    #         return images

    return images