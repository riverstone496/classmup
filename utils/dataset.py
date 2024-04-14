import os,json
import numpy as np
import torch
import torchvision
from torchvision.datasets.folder import ImageFolder, default_loader

from torch.utils.data.dataset import Subset
from torchvision import transforms
from utils.cutout import Cutout
from .autoaugment import CIFAR10Policy

def permute_classwise(dataset, class_permutations, img_size, num_classes):
    """ 各クラスごとに異なるパーミュテーションを適用する """
    permuted_data = dataset.data.clone()
    for i in range(num_classes):  # 10 classes
        indices = (dataset.targets == i)
        perm = class_permutations[i]
        permuted_data[indices] = dataset.data[indices].view(-1, img_size**2)[:, perm].view(-1, img_size, img_size)
    return permuted_data

class Dataset(object):
    def __init__(self, args):
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.dataset_shuffle,
                                                    pin_memory=True,
                                                    num_workers=args.num_workers)
        self.train_val_loader = torch.utils.data.DataLoader(dataset=self.train_val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=args.num_workers)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=args.num_workers)

        self.num_steps_per_epoch = len(self.train_loader)


    def create_transform(self,args):
        self.train_transform = transforms.Compose([])
        self.val_transform = transforms.Compose([])
        normalize = transforms.Normalize(   mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        if args.use_timm or 'vit_' in args.model or 'mixer_' in args.model or 'efficient' in args.model or 'gmlp' in args.model or 'dense' in args.model or 'inception' in args.model or 'convnext' in args.model or 'deit_' in args.model:
            self.train_transform.transforms.append(transforms.RandomResizedCrop(224))
            self.val_transform.transforms.append(transforms.Resize(256))
            self.val_transform.transforms.append(transforms.CenterCrop(224))
        else:
            if args.RandomCrop:
                self.train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            else:
                self.train_transform.transforms.append(transforms.Resize((32, 32)))
            self.val_transform.transforms.append(transforms.Resize((32, 32)))
        if args.RandomHorizontalFlip:
            self.train_transform.transforms.append(transforms.RandomHorizontalFlip())
        if args.CIFAR10Policy:
            self.train_transform.transforms.append(CIFAR10Policy())
            
        self.train_transform.transforms.append(transforms.ToTensor())
        self.train_transform.transforms.append(normalize)

        self.val_transform.transforms.append(transforms.ToTensor())
        self.val_transform.transforms.append(normalize)
    
    def _filter_dataset(self, dataset, num_classes):
        if num_classes == dataset.classes:
            return dataset
        class_indices = list(range(num_classes))
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in class_indices]
        return Subset(dataset, indices)

class MNIST(Dataset):
    def __init__(self, args, rotation_angle=0, permutate=False):
        self.num_classes = 10
        self.num_channels=1
        self.img_size = 28

        self.train_transform = transforms.Compose([])
        self.val_transform = transforms.Compose([])
        if rotation_angle>0:
            self.train_transform.transforms.append(transforms.RandomRotation((rotation_angle, rotation_angle)))
            self.val_transform.transforms.append(transforms.RandomRotation((rotation_angle, rotation_angle)))
        self.train_transform.transforms.append(transforms.ToTensor())
        self.train_transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
        self.val_transform.transforms.append(transforms.ToTensor())
        self.val_transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))

        self.train_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=False,
                                            transform=self.val_transform,
                                            download=True)
        
        if permutate:
            np.random.seed(args.seed)
            class_permutations = [np.random.permutation(self.img_size*self.img_size) for _ in range(self.num_classes)]
            self.train_dataset.data = permute_classwise(self.train_dataset, class_permutations, self.img_size, self.num_classes)
            self.train_val_dataset.data = permute_classwise(self.train_val_dataset, class_permutations, self.img_size, self.num_classes)
            self.val_dataset.data = permute_classwise(self.val_dataset, class_permutations, self.img_size, self.num_classes)

        ## split dataset
        if args.train_size != -1:
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)
            train_idx = indices[:args.train_size]
            self.train_dataset = Subset(self.train_dataset, train_idx)
            #self.train_val_dataset = Subset(self.train_val_dataset, train_idx)

        super().__init__(args)

class FashionMNIST(Dataset):
    def __init__(self, args, rotation_angle=0):
        self.num_classes = 10
        self.img_size = 28
        self.num_channels = 1

        self.train_transform = transforms.Compose([])
        if args.RandomAffine:
            self.train_transform.transforms.append(transforms.RandomAffine([-15,15], scale=(0.8, 1.2)))
        if rotation_angle>0:
            self.train_transform.transforms.append(transforms.RandomRotation((rotation_angle, rotation_angle)))
        self.train_transform.transforms.append(transforms.ToTensor())
        self.train_transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        self.train_dataset = torchvision.datasets.FashionMNIST(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.FashionMNIST(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.FashionMNIST(root='data/',
                                            train=False,
                                            transform=self.val_transform,
                                            download=True)
        
        if args.train_size != -1:
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)
            train_idx = indices[:args.train_size]
            self.train_dataset = Subset(self.train_dataset, train_idx)
            
        super().__init__(args)

class CIFAR10(Dataset):
    def __init__(self, args):
        self.num_classes = 10
        self.num_channels = 3
        self.img_size = 32

        self.create_transform(args)

        if args.cutout:
            if args.length is None:
                args.length=16
            self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        self.train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            download=True,
                                            transform=self.val_transform)
        
        ## split dataset
        if args.train_size != -1:
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)
            train_idx = indices[:args.train_size]
            self.train_dataset = Subset(self.train_dataset, train_idx)
            self.train_val_dataset = Subset(self.train_val_dataset, train_idx)
        super().__init__(args)

class CIFAR100(Dataset):
    def __init__(self, args, num_classes = 100):
        if num_classes != -1:
            self.num_classes = num_classes
        self.img_size = 32
        self.num_channels = 3
        self.create_transform(args)

        if args.cutout:
            if args.length is None:
                args.length=8
            self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        self.train_dataset = torchvision.datasets.CIFAR100(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.CIFAR100(root='data/',
                                            train=True,
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.CIFAR100(root='data/',
                                            train=False,
                                            transform=self.val_transform,
                                            download=True)
        
        if num_classes!=100:
            self.train_dataset = self._filter_dataset(self.train_dataset, num_classes=num_classes)
            self.train_val_dataset = self._filter_dataset(self.train_val_dataset, num_classes=num_classes)
            self.val_dataset = self._filter_dataset(self.val_dataset, num_classes=num_classes)

        if args.train_size != -1:
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)
            train_idx = indices[:args.train_size]
            self.train_dataset = Subset(self.train_dataset, train_idx)

        super().__init__(args)
    
class STL(Dataset):
    def __init__(self, args):
        self.num_classes = 10
        self.img_size = 32
        self.num_channels = 3
        self.create_transform(args)

        # train dataset
        self.train_dataset = torchvision.datasets.STL10(root='data/',
                                        split='train',
                                        transform=self.train_transform,
                                        download=True)
        # val dataset
        self.train_val_dataset = torchvision.datasets.STL10(root='data/',
                                        split='train',
                                        transform=self.val_transform,
                                        download=True)

        self.val_dataset = torchvision.datasets.STL10(root='data/',
                                        split='test',
                                        transform=self.val_transform,
                                        download=True)
        super().__init__(args)

class SVHN(Dataset):
    def __init__(self, args):
        self.num_classes = 10
        self.img_size = 32
        self.create_transform(args)

        if args.cutout:
            if args.length is None:
                args.length=20
            self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        # train dataset
        self.train_dataset = torchvision.datasets.SVHN(root='data/',
                                        split='train',
                                        transform=self.train_transform,
                                        download=True)

        self.extra_dataset = torchvision.datasets.SVHN(root='data/',
                                        split='extra',
                                        transform=self.train_transform,
                                        download=True)

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([self.train_dataset.data, self.extra_dataset.data], axis=0)
        labels = np.concatenate([self.train_dataset.labels, self.extra_dataset.labels], axis=0)
        self.train_dataset.data = data
        self.train_dataset.labels = labels

        # val dataset
        self.train_val_dataset = torchvision.datasets.SVHN(root='data/',
                                        split='train',
                                        transform=self.val_transform,
                                        download=True)

        self.val_extra_dataset = torchvision.datasets.SVHN(root='data/',
                                        split='extra',
                                        transform=self.val_transform,
                                        download=True)

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([self.train_val_dataset.data, self.val_extra_dataset.data], axis=0)
        labels = np.concatenate([self.train_val_dataset.labels, self.val_extra_dataset.labels], axis=0)
        self.train_val_dataset.data = data
        self.train_val_dataset.labels = labels

        self.val_dataset = torchvision.datasets.SVHN(root='data/',
                                        split='test',
                                        transform=self.val_transform,
                                        download=True)
        super().__init__(args)

class Cars(Dataset):
    def __init__(self, args):
        self.num_classes = 196
        self.img_size = 32
        self.create_transform(args)

        #if args.cutout:
        #    if args.length is None:
        #        args.length=8
        #    self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        self.train_dataset = torchvision.datasets.StanfordCars(root='data/',
                                            split='train',
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.StanfordCars(root='data/',
                                            split='train',
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.StanfordCars(root='data/',
                                            split='test',
                                            download=True,
                                            transform=self.val_transform,)
        super().__init__(args)

class Flowers(Dataset):
    def __init__(self, args):
        self.num_classes = 102
        self.img_size = 30
        self.create_transform(args)

        #if args.cutout:
        #    if args.length is None:
        #        args.length=8
        #    self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        self.train_dataset = torchvision.datasets.Flowers102(root='data/',
                                            split="train",
                                            download=True,
                                            transform=self.train_transform,)
        self.train_val_dataset = torchvision.datasets.Flowers102(root='data/',
                                            split="train",
                                            download=True,
                                            transform=self.val_transform,)
        self.val_dataset = torchvision.datasets.Flowers102(root='data/',
                                            split="val",
                                            download=True,
                                            transform=self.val_transform,)
        super().__init__(args)

class INat2019(Dataset):
    def __init__(self, args):
        self.num_classes = 102
        self.img_size = 30
        self.create_transform(args)

        if args.cutout:
            if args.length is None:
                args.length=8
            self.train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        self.train_dataset = INatDataset(root='data/',
                                            train=True,
                                            download=True,
                                            year=2019,
                                            transform=self.train_transform,)
        self.train_val_dataset = INatDataset(root='data/',
                                            train=True,
                                            download=True,
                                            year=2019,
                                            transform=self.val_transform,)
        self.val_dataset = INatDataset(root='data/',
                                        train=False,
                                        download=True,
                                        year=2019,
                                        transform=self.val_transform,)
        super().__init__(args)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2019, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))