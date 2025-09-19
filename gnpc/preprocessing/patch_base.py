import warnings
from abc import ABC

from gnpc.default_modules import *
from gnpc.utils.io import load_pickle


class WSI(ABC):
    def __init__(self, fpath_wsi):
        self.patch_info = {
            "wsi_name": None,
            "source_mag": None,
            "target_mag": None,
            "source_patch_size": None,
            "target_patch_size": None,
            "mask_downsample": 64,
            "min_foreground_percent": None,
            "patch": {},
        }
        self.fpath_wsi = fpath_wsi
        assert os.path.exists(
            self.fpath_wsi
        ), f"WSI file doesnt exist in the path {self.fpath_wsi}"
        self.wsi = ops.OpenSlide(self.fpath_wsi)
        self.fpath_patch = None
        self._parse_wsi_metadata()

    def _parse_wsi_metadata(self):
        base_mag = int(self.wsi.properties["openslide.objective-power"])
        mpp_x = round(float(self.wsi.properties["openslide.mpp-x"]), 2)
        mpp_y = round(float(self.wsi.properties["openslide.mpp-y"]), 2)
        if mpp_x != mpp_y:
            warnings.warn("mpp-x doesnt equal to mpp-y", UserWarning)
        if (base_mag == 40) and (mpp_x != 0.25):
            raise ValueError(f"Slide {base_mag}x not equal to {mpp_x} mpp")
        if (base_mag == 20) and (mpp_x != 0.5):
            raise ValueError(f"Slide {base_mag}x not equal to {mpp_x} mpp")
        self.base_mag = base_mag
        self.base_mpp = mpp_x
        self.wsi_dimensions = self.wsi.dimensions
        self.wsi_name = pathlib.Path(self.fpath_wsi).stem

    def __len__(self):
        return len(self.patch_info["patch"])

    def get_img_patch(
        self, idx: int, return_numpy: bool = False, return_target_size: bool = False
    ):
        coord = self.patch_info["patch"][idx]["top_left"]
        center = self.patch_info["patch"][idx]["center"]
        img = self.wsi.read_region(
            location=coord, level=0, size=self.patch_info["source_patch_size"]
        ).convert("RGB")
        if return_target_size:
            img = img.resize(self.patch_info["target_patch_size"])
        if return_numpy:
            return np.array(img), coord, center
        else:
            return img, coord, center

    def get_wsi_thumbnail(self, downsample):
        return self.wsi.get_thumbnail(
            [int(x // downsample) for x in self.wsi_dimensions]
        ).convert("RGB")

    def _load_patch_info(self, fpath_patch):
        self.fpath_patch = fpath_patch
        self.patch_info = load_pickle(fpath_patch)


class WSIVisual(WSI):
    def __init__(self, fpath_wsi):
        super().__init__(fpath_wsi)

    def visualise(self):
        pass

    def save_thumbnail(self, save_path, downsample=32, dpi=60, figsize=(20, 20)):
        thumbnail = self.get_wsi_thumbnail(downsample=downsample)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(thumbnail)
        ax.axis("off")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=False)
        plt.close()
