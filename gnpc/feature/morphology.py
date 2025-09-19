from matplotlib.patches import Polygon
from scipy.stats import kurtosis, skew

from gnpc.default_modules import *
from gnpc.preprocessing.patch_base import WSI
from gnpc.utils.io import *


def get_hovernet_label():
    dct_colour = {
        0: {"name": "no_label", "colour": "black"},
        1: {"name": "neoplasm", "colour": "r"},
        2: {"name": "inflammatory", "colour": "g"},
        3: {"name": "connective", "colour": "b"},
        4: {"name": "necrosis", "colour": "y"},
        5: {"name": "no_neo", "colour": "white"},
    }

    return dct_colour


class PatchMorph(WSI):
    def __init__(
        self,
        fpath_wsi: str,
        fpath_patch: str,
        dct_colour: dict = None,
    ):
        super().__init__(fpath_wsi)

        self.fpath_patch = fpath_patch
        assert (
            self.wsi_name == pathlib.Path(self.fpath_patch).stem
        ), f"wsi and coords file is not the same {self.wsi_name} {pathlib.Path(self.fpath_patch).stem}"

        self._load_patch_info(self.fpath_patch)
        self.coords_info = self.patch_info["patch"]
        self.patch_size = self.patch_info["source_patch_size"]
        self.target_patch_size = self.patch_info["target_patch_size"]
        if dct_colour == None:
            self.dct_colour = get_hovernet_label()
        else:
            self.dct_colour = dct_colour


class NucleiPatch(PatchMorph):
    def __init__(
        self,
        fpath_wsi: str,
        fpath_patch: str,
        fpath_nuclei: str,
        dct_colour: dict = None,
        round_scale: bool = False,
    ):
        super().__init__(
            fpath_wsi=fpath_wsi, fpath_patch=fpath_patch, dct_colour=dct_colour
        )
        self.fpath_nuclei = fpath_nuclei
        self.data_nuclei = load_pickle(self.fpath_nuclei)

        self.nuclei_mpp = self.data_nuclei["element-resolution"]["resolution"]

        self.nuclei_downsample = self.nuclei_mpp / self.base_mpp
        if round_scale:
            self.nuclei_downsample = round(self.nuclei_downsample)

        self.contain_nuclei = False
        if len(self.data_nuclei["elements"]) > 0:
            self.contain_nuclei = True

        self.nuclei_keys = np.array(list(self.data_nuclei["elements"].keys()))
        self.nuclei_centroids = (
            np.array(
                [self.data_nuclei["elements"][x]["centroid"] for x in self.nuclei_keys],
            )
            * self.nuclei_downsample
        )

    def show_patch_nuc_info(self):

        print(
            f"Patch size: {self.target_patch_size} at:{self.patch_info['target_mag']}x from base mag: {self.base_mag}"
        )
        print(
            f"Nuclei predicted at resolution:{self.nuclei_mpp} from base resolution:{self.base_mpp}"
        )

    def points_in_bbox(self, top_left):
        """
        Get all nuclei in patch, based on their centroids
        Return indices of nuclei (self.nuclei_keys)
        """
        assert self.contain_nuclei, f"no nuclei detected in {self.wsi_name}!"

        min_x, min_y = top_left
        max_x = min_x + self.patch_size[0]
        max_y = min_y + self.patch_size[0]

        # Extract x and y coordinates of points
        x, y = self.nuclei_centroids[:, 0], self.nuclei_centroids[:, 1]

        # Check if points lie within the bounding box using vectorized operations
        idx_in_bbox = np.logical_and(
            np.logical_and(min_x <= x, x <= max_x),
            np.logical_and(min_y <= y, y <= max_y),
        )
        return idx_in_bbox

    def get_nuclei_in_patch(
        self,
        patch_index,
        coordinate_in_patch=False,
        rounded_scalling=False,
        scalling=0,
    ):
        """
        Return all nuclei contour and centroid in patch-i
        If coordinate_in_patch is set to be False, all nuclei coordinate will be in their base mag/mpp
            Otherwise it will return on mag/mpp from extracted patch
        """

        # THIS IS QUICK FIX FOR VISUALISATION
        # due to rounding problem of scalling between base mpp and mpp for nuclei prediction
        if rounded_scalling:
            self.nuclei_downsample = round(self.nuclei_downsample)
        else:
            self.nuclei_downsample = self.nuclei_mpp / self.base_mpp
        if scalling > 0:
            self.nuclei_downsample = scalling
        self.nuclei_centroids = (
            np.array(
                [self.data_nuclei["elements"][x]["centroid"] for x in self.nuclei_keys],
            )
            * self.nuclei_downsample
        )
        ###############

        # get top left of selected patch
        top_left_patch = np.array(self.coords_info[patch_index]["top_left"])
        # get center of selected patch
        center_patch = np.array(self.coords_info[patch_index]["center"])
        # get indices of nuclei centroids inside the patch
        roi_nuclei_indices = np.where(self.points_in_bbox(top_left_patch) == True)[0]

        # get keys from detected nuclei
        roi_nuclei_keys = self.nuclei_keys[roi_nuclei_indices]
        # get contours from detected nuclei
        roi_nuclei_contours = [
            self.data_nuclei["elements"][x]["contour"] for x in roi_nuclei_keys
        ]
        roi_nuclei_centroids = [
            self.data_nuclei["elements"][x]["centroid"] for x in roi_nuclei_keys
        ]
        # get type of nuclei from detected nuclei
        roi_nuclei_type = [
            self.data_nuclei["elements"][x]["type"] for x in roi_nuclei_keys
        ]
        dct_nuc_in_patch = {}
        ix = 0
        for t, ct, ce in zip(
            roi_nuclei_type, roi_nuclei_contours, roi_nuclei_centroids
        ):
            # get true coordinate/base mag
            ct_s = np.array([[x[0], x[1]] for x in ct], dtype=np.int32)
            ct_s = (ct_s * self.nuclei_downsample) - top_left_patch

            ce_s = np.array([ce[0], ce[1]], dtype=np.int32)
            ce_s = (ce_s * self.nuclei_downsample) - top_left_patch

            if coordinate_in_patch:
                # get coordinate in patch mag
                factor_nuc_to_patch = (
                    self.patch_info["target_mag"] / self.patch_info["source_mag"]
                )

                ct_s = ct_s * factor_nuc_to_patch
                ce_s = ce_s * factor_nuc_to_patch

            dct_nuc_in_patch[ix] = {
                "contour": ct_s,
                "centroid": ce_s,
                "colour": self.dct_colour[t]["colour"],
                "type": t,
            }
            ix += 1

        return dct_nuc_in_patch, top_left_patch, center_patch

    def plot_nuclei_in_patch(
        self,
        index: int,
        exclude_type: list = [0],
        plot_contour: bool = True,
        plot_centroid: bool = False,
        plot_as_patch_size: bool = True,
        rounded_scalling: bool = False,
        scalling=0,
    ):
        img_patch, _, _ = self.get_img_patch(
            index, return_numpy=True, return_target_size=plot_as_patch_size
        )
        dct_nuc_in_patch, _, _ = self.get_nuclei_in_patch(
            index,
            coordinate_in_patch=plot_as_patch_size,
            rounded_scalling=rounded_scalling,
            scalling=scalling,
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(img_patch)
        for k, v in dct_nuc_in_patch.items():
            if v["type"] in exclude_type:
                pass
            else:
                if plot_contour:
                    polygon = Polygon(v["contour"], edgecolor=v["colour"], fill=None)
                    plt.gca().add_patch(polygon)
        if plot_centroid:
            centroid_to_plot = np.array(
                [dct_nuc_in_patch[x]["centroid"] for x in dct_nuc_in_patch.keys()]
            )

            centroid_colour_to_plot = [
                dct_nuc_in_patch[x]["colour"] for x in dct_nuc_in_patch.keys()
            ]
            plt.scatter(
                centroid_to_plot[:, 0],
                centroid_to_plot[:, 1],
                c=centroid_colour_to_plot,
                s=1,
            )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return dct_nuc_in_patch


class NucleiPatchMorphology(NucleiPatch):
    def __init__(
        self,
        fpath_wsi: str,
        fpath_patch: str,
        fpath_nuclei: str,
        morph_mpp: float,
        dct_colour: dict = None,
        round_scale: bool = True,
    ):
        super().__init__(
            fpath_wsi=fpath_wsi,
            fpath_patch=fpath_patch,
            fpath_nuclei=fpath_nuclei,
            dct_colour=dct_colour,
            round_scale=round_scale,
        )
        self.morph_mpp = morph_mpp
        self.morph_downsample = self.nuclei_mpp / self.morph_mpp
        self.feature_names = [
            "convex_area",
            "contour_area",
            "equiv_diameter",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
            "radius",
            "bbox_area",
            "roundness",
            "spindle",
            "eccentricity",
            "solidity",
            "orientation",
        ]

    def get_shapefeatures(self, contours):
        """
        contours is numpy array[[x1, y1], ... , [xn,yn]]
        """
        contours = (contours * self.morph_downsample).astype(np.int32)

        if type(contours) != np.ndarray:
            contours = np.array(contours)

        inst_box = np.array(
            [
                [np.min(contours[:, 1]), np.min(contours[:, 0])],
                [np.max(contours[:, 1]), np.max(contours[:, 0])],
            ]
        )

        bbox_h, bbox_w = inst_box[1] - inst_box[0]
        bbox_area = bbox_h * bbox_w
        contour_area = cv2.contourArea(contours)
        convex_hull = cv2.convexHull(contours)
        convex_area = cv2.contourArea(convex_hull)
        convex_area = convex_area if convex_area != 0 else 1
        solidity = float(contour_area) / convex_area
        equiv_diameter = np.sqrt(4 * contour_area / np.pi)
        if contours.shape[0] > 4:
            _, axes, orientation = cv2.fitEllipse(contours)
            major_axis_length = max(axes)
            minor_axis_length = min(axes)
        else:
            orientation = 0
            major_axis_length = 1
            minor_axis_length = 1
        perimeter = cv2.arcLength(contours, True)
        _, radius = cv2.minEnclosingCircle(contours)
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
        roundness = (4 * np.pi * contour_area) / (perimeter**2)
        spindle = 1 - (minor_axis_length / major_axis_length)
        res_value = [
            convex_area,
            contour_area,
            equiv_diameter,
            major_axis_length,
            minor_axis_length,
            perimeter,
            radius,
            bbox_area,
            roundness,
            spindle,
            eccentricity,
            solidity,
            orientation,
        ]
        res_value = [1e-6 if np.isnan(x) else x for x in res_value]
        res_value = [1e-6 if x == np.inf else x for x in res_value]

        result = {k: v for k, v in zip(self.feature_names, res_value)}
        return result

    def get_patch_morphology(self, index: int, return_per_type: bool = False):
        dct_contours, top_left_patch, center_patch = self.get_nuclei_in_patch(
            patch_index=index, coordinate_in_patch=False
        )
        dct_features = {"type": list(self.dct_colour.keys())}
        dct_features.update(
            {
                k: [np.nan for _ in list(self.dct_colour.keys())]
                for k in self.feature_names
            }
        )

        for k, v in dct_contours.items():
            temp_morph = self.get_shapefeatures(v["contour"])
            dct_features["type"].append(v["type"])
            for k in self.feature_names:
                dct_features[k].append(temp_morph[k])

        if return_per_type:
            raise NotImplementedError
        else:
            return dct_features, top_left_patch, center_patch

    def aggregate_feature_per_patch(
        self,
        index: int,
        exclude_type: list,
    ):

        dct_feature_per_type, topleft_patch, center_patch = self.get_patch_morphology(
            index=index
        )
        df_feature = pd.DataFrame(dct_feature_per_type)
        df_feature = df_feature.drop(columns="spindle")
        df_agg = df_feature[~df_feature["type"].isin(exclude_type)]
        agg_features = (
            df_agg.groupby("type")
            .agg(
                [
                    "mean",
                    "count",
                    "std",
                    kurtosis,
                    skew,
                    ("rangeval", lambda x: x.max() - x.min()),
                ]
            )
            .reset_index(drop=True)
            .values.ravel()
        )
        agg_features = np.nan_to_num(agg_features, nan=0)
        df_feature = df_feature.iloc[6:]

        return df_feature, agg_features, topleft_patch, center_patch
