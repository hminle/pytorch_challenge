import torchvision.transforms as transforms
import torchvision.datasets as dset


def check_dataset(dataroot, training=True):
    """
    Args:
        dataset (str): Name of the dataset to use. See CLI help for details
        dataroot (str): root directory where the dataset will be stored.
    Returns:
        dataset (data.Dataset): torchvision Dataset object
    """
    to_rgb = transforms.Lambda(lambda img: img.convert('RGB'))
    random_rotation = transforms.RandomRotation(45)
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    flip_horizontal = transforms.RandomHorizontalFlip()
    gray_scale = transforms.RandomGrayscale(p=0.3)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if training:
        dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([random_rotation,
                                                                                resize,
                                                                                crop,
                                                                                flip_horizontal,
                                                                                to_tensor,
                                                                                normalize]))
    else:
        dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([resize,
                                                                                crop,
                                                                                to_tensor,
                                                                                normalize]))
    return dataset
