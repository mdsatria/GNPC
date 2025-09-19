import warnings
from functools import partial

from skimage import morphology
from tqdm.contrib.concurrent import thread_map

from gnpc.default_modules import *
from gnpc.preprocessing.patch_base import WSI
from gnpc.utils.io import *


class WSIPatchGenerator(WSI):
    def __init__(self, fpath_wsi: str, mask_config: dict, mask_downsample: int = 64):
        super().__init__(fpath_wsi)
        self.mask_config = mask_config
        self.mask_downsample = mask_downsample
        self.thumbnail = np.array(self.get_wsi_thumbnail(self.mask_downsample))
        self.binary_mask = self.__create_mask()

    def __create_mask(self):
        """
        return numpy array of thumbnail and mask
        """
        # thumbnail = np.array(self.get_wsi_thumbnail(self.mask_downsample))

        # correction if wsi size is small
        if self.thumbnail.size // 3 < 25_000:
            area_threshold = 64 * 64
        else:
            area_threshold = 256 * 256

        hsv_image = cv2.cvtColor(self.thumbnail, cv2.COLOR_RGB2HSV)
        s_channel = hsv_image[:, :, 1]
        s_channel = cv2.medianBlur(s_channel, self.mask_config["med_blur"])
        _, img_bw = cv2.threshold(
            s_channel,
            self.mask_config["threshold"],
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        img_bw = (img_bw == 0).astype(np.uint8)
        mask = morphology.remove_small_objects(
            img_bw == 0, min_size=16 * 16, connectivity=2
        )
        mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)
        mask = mask.astype(np.uint8) * 255

        if np.count_nonzero(mask == 255) > (0.95 * mask.size):
            warning_msg = f"NO TISSUE DETECTED in {self.wsi_name}!"
            warnings.warn(warning_msg)

        return mask

    def __is_tissue(
        self,
        patch_size: list,
        min_percent: int,
        check_white: bool,
        max_white_area: int,
        white_threshold: int,
        return_roi: bool,
        coords: list,
    ):
        # get coordinate region of mask which corresponding with current patch
        # coordinate input are from level 0 slide, thus it need to rescaled down according to self.patch_info["mask_downsample"]
        top_left_x, top_left_y = [int(x // self.mask_downsample) for x in coords]
        # get appropriate size region of the mask w.r.t. patch size at level 0
        patch_size = [int(x // self.mask_downsample) for x in patch_size]
        patch_area = patch_size[0] * patch_size[1]
        # coordinates are flipped, because numpy and openslide image indexing is the opposite
        roi = self.binary_mask[
            top_left_y : top_left_y + patch_size[1],
            top_left_x : top_left_x + patch_size[0],
        ]
        roi_foreground = roi == 255
        percentage_mask = (np.sum(roi_foreground) / patch_area) * 100
        is_mask_ok = percentage_mask > min_percent

        if check_white:
            # slow
            img = np.array(
                self.wsi.read_region(
                    coords, 0, self.patch_info["source_patch_size"]
                ).convert("RGB")
            )
            percentage_white = (np.sum(img > white_threshold) / np.size(img)) * 100
            is_not_background = percentage_white < max_white_area

            is_tissue = is_mask_ok and is_not_background
        else:
            is_tissue = is_mask_ok

        if is_tissue:
            if return_roi:
                return roi, True
            else:
                return True
        else:
            if return_roi:
                return roi, False
            else:
                return False

    # def get_wsi_thumbnail(self):
    #     return self.wsi.get_thumbnail(
    #         [int(x // self.mask_downsample) for x in self.wsi_dimensions]
    #     ).convert("RGB")

    def get_patch_coords(
        self,
        target_patch_size: list,
        target_mag: int,
        foreground_min_percent: int,
        overlap: int = 0,
        save_roi: bool = False,
        check_white: bool = False,
        max_white_area: int = 85,
        white_threshold: int = 200,
    ):

        assert (
            self.base_mag >= 20
        ), f"{self.wsi_name} is {self.base_mag} which is lower than 20"
        assert overlap >= 0

        target_scalling = int(self.base_mag / target_mag)
        patch_size = [x * target_scalling for x in target_patch_size]
        if overlap > 0:
            # coords_x = np.arange(0, self.wsi_dimensions[0], patch_size[0] - overlap)
            # coords_y = np.arange(0, self.wsi_dimensions[1], patch_size[1] - overlap)
            return NotImplemented
        else:
            coords_x = np.arange(0, self.wsi_dimensions[0], patch_size[0])
            coords_y = np.arange(0, self.wsi_dimensions[1], patch_size[1])

        self.patch_info["wsi_name"] = self.wsi_name
        self.patch_info["source_mag"] = self.base_mag
        self.patch_info["target_mag"] = target_mag
        self.patch_info["source_patch_size"] = patch_size
        self.patch_info["target_patch_size"] = target_patch_size
        self.patch_info["min_foreground_percent"] = foreground_min_percent
        self.patch_info["mask_downsample"] = self.mask_downsample

        top_left = []
        center = []
        for top_left_x in coords_x:
            center_x = top_left_x + (patch_size[0] // 2)
            for top_left_y in coords_y:
                center_y = top_left_y + (patch_size[1] // 2)
                top_left.append([top_left_x, top_left_y])
                center.append([center_x, center_y])

        partial_is_tissue = partial(
            self.__is_tissue,
            patch_size,
            foreground_min_percent,
            check_white,
            max_white_area,
            white_threshold,
            save_roi,
        )
        check_results = thread_map(
            partial_is_tissue,
            top_left,
            disable=True,
        )
        check_results = np.array(check_results)
        top_left = np.array(top_left)

        tissue_loc = np.where(
            check_results == True,
        )
        tissue_loc = tissue_loc[0]
        tissue_topleft = top_left[tissue_loc]
        tissue_center = np.array(center)[tissue_loc]
        for ix, tl in enumerate(tissue_topleft):
            self.patch_info["patch"][ix] = {
                "top_left": list(tl),
                "center": list(tissue_center[ix]),
            }

        background_loc = np.where(
            check_results == False,
        )
        background_loc = background_loc[0]
        background_topleft = top_left[background_loc]
        background_center = np.array(center)[background_loc]
        patch_background = {}
        for ix, tl in enumerate(background_topleft):
            patch_background[ix] = {
                "top_left": list(tl),
                "center": list(background_center[ix]),
            }
        self.patch_background = patch_background

    def save_all_patches(self, dir_save: str, pbar_tqdm=None, tqdm_msg=""):
        assert self.patch_info != None, "run get_patch_coords() first!"
        num_patch = self.__len__()

        def save_img(idx):
            img, coord, _ = self.get_img_patch(idx, return_numpy=False)
            fpath = os.path.join(
                dir_save, f"{self.wsi_name}_{idx}_{coord[0]}_{coord[1]}.jpg"
            )
            img.save(fpath)

        if pbar_tqdm is not None:
            pbar_tqdm.set_description(f"{tqdm_msg} saving {num_patch} patches")
        # else:
        # print(
        #     f"saving {num_patch} patch with size {self.patch_info['target_patch_size']} at {self.patch_info['target_mag']}x"
        # )

        thread_map(
            save_img,
            range(num_patch),
            disable=True,
        )

    def stitch_patch(self):
        assert self.patch_info != None, "run get_patch_coords() first!"
        # thumb = np.array(self.get_thumbnail(downsample))
        rgb_image = np.array(self.thumbnail)
        patch_size_level0 = self.patch_info["source_patch_size"]
        patch_size_stich = [int(x // self.mask_downsample) for x in patch_size_level0]
        # stitch_patch_size = [int(x // downsample) for x in patch_size_level0]
        coord_background = [
            self.patch_background[x]["top_left"] for x in self.patch_background.keys()
        ]
        for coord in coord_background:
            coords = [int(x // self.mask_downsample) for x in coord]
            start_x = coords[1]
            end_x = coords[1] + patch_size_stich[1]
            start_y = coords[0]
            end_y = coords[0] + patch_size_stich[0]

            black_pixel = np.zeros_like(rgb_image[start_x:end_x, start_y:end_y, :])
            rgb_image[start_x:end_x, start_y:end_y, :] = black_pixel

        return rgb_image

    def save_coord_info(self, fpath: str):
        assert self.patch_info != None, "run get_patch_coords() first!"
        save_pickle(self.patch_info, fpath)
