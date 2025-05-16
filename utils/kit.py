import os

import torch
from PIL import Image
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg' 等其他后端
import matplotlib.pyplot as plt
import numpy as np

class Args:
    def __init__(self, image_size=32, num_channels=3, num_classes=10):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_trigger_set = 100
        self.device = 'cuda'
        self.test_bs = 512
        self.gpu = 0

# test one pattern-label pair
def test_one_pattern():
    path = "../result/VGG16-fast/"
    model_path = os.path.join(path, "model_last_epochs_75.pth")
    from utils.models import VGG16
    model = VGG16(Args())
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for idx in range(10):
        pattern_path = "../data/pattern/{}.png".format(idx)
        pattern_save_path = "../data/noisy_pattern/{}.png".format(idx)
        if not os.path.exists(os.path.dirname(pattern_save_path)):
            os.makedirs(os.path.dirname(pattern_save_path))
        pattern = Image.open(pattern_path).convert("RGB").resize((32, 32), Image.BILINEAR)
        arr = np.array(pattern).astype(np.float32)
        arr = (arr + np.random.randint(0, 255, arr.shape)).astype(np.float32) / 255 / 2
        Image.fromarray((arr * 255).astype(np.uint8)).save(pattern_save_path)

        # show the pattern
        axes[idx//5][idx%5].imshow(Image.fromarray((arr * 255).astype(np.uint8)))
        axes[idx//5][idx%5].axis("off")
        axes[idx//5][idx%5].set_title(f"Pattern {idx}")

        trigger_set_mean = [0.7141363, 0.72196555, 0.7009312 ]
        trigger_set_std = [0.1813506, 0.16413535, 0.18837212]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trigger_set_mean, trigger_set_std)
        ])
        pattern_tensor = transform(arr).unsqueeze(0).to("cuda")
        label = torch.tensor([idx]).to("cuda")
        with torch.no_grad():
            model.eval()
            output = model(pattern_tensor)
            print(output)
            pred = torch.argmax(output, dim=1)
            print("Predicted label: ", pred.item())
            print("True label: ", label.item())
            if pred.item() == label.item():
                print("Correct prediction!")
            else:
                print("Incorrect prediction!")

    plt.show()

def test_verify_watermark():
    path = "../result/VGG16-fast/"
    model_path = os.path.join(path, "model_last_epochs_75.pth")
    from utils.models import VGG16
    model = VGG16(Args())
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")
    trigger_set = generate_waffle_pattern(Args())
    # watermark_set = DataLoader(trigger_set, batch_size=16, shuffle=True)

    from torchvision.datasets import CIFAR10
    test_dataset = CIFAR10('../data/cifar10/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Resize((32, 32)),
                                 transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                           ]))

    from utils.test import test_img
    watermark_acc, _ = test_img(model, trigger_set, Args())
    print("Watermark accuracy: ", watermark_acc)

    acc, acc_top5 = test_img(model, test_dataset, Args())
    print("Test accuracy: ", acc)


def test_dict():
    stat_dict = {
        'num_channel':3,
        'num_classes':10,
    }
    print(*stat_dict.items())


if __name__== '__main__':
    images = []
    images_noisy = []
    for i in range(10):
        path = "../data/pattern/{}.png".format(i)
        pattern = Image.open(path)
        # pattern = pattern.convert("RGB")
        pattern = np.array(pattern)
        images.append(pattern)

        pattern = np.resize(pattern, (32, 32, 3))
        image = (pattern + np.random.randint(0, 255, (32, 32, 3))).astype(np.float32) / 2
        images_noisy.append(image.astype(np.uint8))

    # 使用matplotlib并列显示这些图片
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(images[i])
        axes[1, i].imshow(images_noisy[i])
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.show()

class NumpyLoader(Dataset):

    def __init__(self, x, y, transformer=None):
        self.x = x
        self.y = y
        self.transformer = transformer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        image = self.x[item]
        label = self.y[item]
        if self.transformer is not None:
            image = self.transformer(image)
        return image, label

# modified method
def generate_waffle_pattern(args):
    # np.random.seed(0)
    path = "../data/pattern/"
    base_patterns = []
    # 加载触发集图案
    for i in range(args.num_classes):
        pattern_path = os.path.join(path, "{}.png".format(i))
        pattern = Image.open(pattern_path)
        if args.num_channels == 1:
            pattern = pattern.convert("L")
        else:
            pattern = pattern.convert("RGB")
        pattern = pattern.resize((args.image_size, args.image_size), Image.BILINEAR)
        pattern = np.array(pattern)
        # pattern = np.resize(pattern, (args.image_size, args.image_size, args.num_channels))
        base_patterns.append(pattern)
    trigger_set = []
    trigger_set_labels = []
    label = 0
    # num_trigger_each_class 每个类别的触发器数量
    num_trigger_each_class = args.num_trigger_set // args.num_classes
    for pattern in base_patterns:
        for _ in range(num_trigger_each_class):
            image = (pattern + np.random.randint(0, 255, (args.image_size, args.image_size, args.num_channels)))\
                        .astype(np.float32) / 255 / 2
            trigger_set.append(image)
            trigger_set_labels.append(label)
        label += 1
    trigger_set = np.array(trigger_set)
    trigger_set_labels = np.array(trigger_set_labels)
    trigger_set_mean = np.mean(trigger_set, axis=(0, 1, 2))
    trigger_set_std = np.std(trigger_set, axis=(0, 1, 2))
    print(trigger_set_mean, trigger_set_std)
    dataset = NumpyLoader(trigger_set, trigger_set_labels, transformer=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(trigger_set_mean, trigger_set_std)
                                ]))
    return dataset


