try:
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except:
    pass
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gnpc.default_modules import *
from gnpc.preprocessing.patch_base import WSI


def image_transform():
    ops_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return ops_test


class WSITensor(WSI, Dataset):
    def __init__(
        self,
        fpath_wsi: str,
        fpath_patch: str,
        virchow_model=None,
    ):
        super().__init__(fpath_wsi)
        self._load_patch_info(fpath_patch)
        if virchow_model is not None:
            # https://huggingface.co/paige-ai/Virchow
            self.image_transform = create_transform(
                **resolve_data_config(virchow_model.pretrained_cfg, model=virchow_model)
            )
        else:

            self.image_transform = image_transform()

    def __getitem__(self, index):
        img, coor, center = self.get_img_patch(index, return_numpy=False)
        img_tensor = self.image_transform(img)
        top_left = torch.tensor(coor)
        center = torch.tensor(center)
        return img_tensor, top_left, center


def make_dataloader(
    fpath_patch, fpath_wsi, batch_size, virchow_model=None, num_workers=4
):
    dset = WSITensor(
        fpath_patch=fpath_patch, fpath_wsi=fpath_wsi, virchow_model=virchow_model
    )
    dloader = DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dloader
