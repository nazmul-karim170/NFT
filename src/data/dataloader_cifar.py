from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy

## Transformations 
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

def get_train_loader(opt):
    print('==> Preparing train data..')
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='../data', train=True, download=False)
    else:
        raise Exception('Invalid dataset')

    train_data   = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')

    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='../data', train=False, download=False)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad   = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    ## (Apart from label 0) Bad Test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    ## All clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader

def get_val_loader(opt, val_frac, perm=None):
    print('==> Preparing test data..')
    # tf_test = transforms.Compose([transforms.ToTensor()
    #                               ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='../data', train=True, download=False)
    else:
        raise Exception('Invalid Dataset')

    if perm is None:
        perm = np.arange(len(trainset))
        np.random.shuffle(perm)

    nb_val = int(val_frac * len(trainset))
    val_set = deepcopy(trainset)
    val_set.data = val_set.data[perm[:nb_val]]
    val_set.targets = np.array(val_set.targets)[perm[:nb_val]].tolist()

    test_data_bad   = DatasetBD(opt, full_dataset=val_set, inject_portion=1, transform=tf_train, mode='test')

    return test_data_bad

def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = tf_test

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='../data', train=True, download=False)
    else:
        raise Exception('Invalid Dataset')

    train_data_bad   = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       )

    return train_data_bad, train_bad_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = torch.Tensor(self.dataset[index][0])
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

class Dataset_Feature_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, mode = 'train', transform=None):
        
        if mode == 'train':
            self.data  = full_dataset['x_train']
            self.label = full_dataset['y_train']
        else:
            self.data  = full_dataset['x_test']
            self.label = full_dataset['y_test']

        self.transform = transform
        self.dataLen   = len(self.data)
        print("dataset size:", self.dataLen)

    def __getitem__(self, index):
        # image = np.transpose(self.dataset[index][0], (2,0,1))
        image = self.data[index]
        label = torch.Tensor(self.label[index]).long()
        # print(label)
        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label

    def __len__(self):
        return self.dataLen



class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # Select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # Change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width  = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.15
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        ## Load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        ## Load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        # print(np.shape(trg), np.shape(img))
        try:
            trg  = np.transpose(trg, (1, 2, 0))
            img_ = np.clip((img + trg).astype('uint8'), 0, 255)
        except:
            # trg  = np.transpose(trg, (0, 1, 2))
            img_ = np.clip((img + trg).astype('uint8'), 0, 255)            


        return img_
