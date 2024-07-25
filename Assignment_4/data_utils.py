import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset
from PIL import Image


DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "data/broden1_224/images/"}



def load_images_from_folder(folder_path):
    resize_size = (224, 224)  # Target size for resizing
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor()
    ])
    images = []
    for i, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Ensure image is in RGB format
                img = transform(img)
                images.append(img)
                if len(images) == 1000:
                    return images
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return images

# def load_images_from_folder(folder_path):
#     # transform = transforms.ToTensor()
#     resize_size = (224, 224)  # Target size for resizing
#     transform = transforms.Compose([
#         transforms.Resize(resize_size),
#         transforms.ToTensor()
#     ])
#     i = 0
#     images = []
#     for filename in os.listdir(folder_path):
#         i += 1
#         # print(i)
#         try:
#             img_path = os.path.join(folder_path, filename)
#             with Image.open(img_path) as img:
#                 img = transform(img)
#                 images.append(img)
#                 if len(images) == 50:
#                     return images
#         except:
#             print("Error in processing file", i, filename)
#     return images

def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar', map_location=torch.device('mps') )['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif target_name == 'resnet18_imagenet':
        # Load a pre-trained ResNet-18 model with weights trained on the ImageNet dataset
        # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        target_name = 'ResNet18'
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))

    # elif "vit_b" in target_name:
    #     target_name_cap = target_name.replace("vit_b", "ViT_B")
    #     weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
    #     preprocess = weights.transforms()
    #     target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    
    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == 'imagenet':
        # data = load_dataset("imagenet-1k")
        # data = data["test"]
        folder_path = '/Users/nursulusagimbayeva/Downloads/TrustworthyML-24/Assignment_4/imagenet'
        data = load_images_from_folder(folder_path)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
        
    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass