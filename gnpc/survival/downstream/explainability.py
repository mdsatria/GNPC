import re

from gnpc.default_modules import *

plt.rcParams["font.family"] = "FreeSans"
plt.rcParams.update({"font.size": 12})


class AttentionDesc:
    def __init__(
        self,
        dir_feature: str,
        dir_att_score: str,
        slide_ids: list,
        result_fpath: str,
        csv_dataset: str,
        dir_slide: str = None,
        top_percent: int = 0,
        top_patch: int = 0,
        low_score_low_risk: bool = True,
        num_layer: int = 4,
    ):
        assert (top_percent == 0 and top_patch > 0) or (
            top_percent > 0 and top_patch == 0
        ), "Use either top_percent or top_patch, not both!"
        self.dir_feature = dir_feature
        self.dir_att_score = dir_att_score
        self.dir_slide = dir_slide
        self.slide_ids = sorted(slide_ids)
        self.result_fpath = result_fpath
        self.csv_dataset = csv_dataset
        self.ps_object = self._load_pkl(self.result_fpath)
        self.df_survival = pd.read_csv(self.csv_dataset)
        self.df_complete = self.combine_dataframe()
        self.nuclei_types = [
            "neoplasm",
            "inflammatory",
            "connective",
            "necrosis",
            "no_neoplasm",
        ]
        self.feature_names = [
            "area_convex",
            "area_contour",
            "eccentricity",
            "equiv_diameter",
            "major_axis_len",
            "minor_axis_len",
            "perimeter",
            "solidity",
            "orientation",
            "radius",
            "area_bbox",
            "roundness",
            "spindle",
        ]
        self.dct_feature_all_slides = None
        self.top_percent = top_percent
        self.top_patch = top_patch
        self.low_score_low_risk = low_score_low_risk
        self.num_layer = num_layer

    def _load_pkl(self, fpath):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        return data

    def get_top_patches(self, idx: int, layer: int):
        slide_id = self.slide_ids[idx]
        risk_group = self.df_complete[self.df_complete["wsi"] == slide_id][
            "group"
        ].values[0]

        fpath_att_score = os.path.join(self.dir_att_score, f"{slide_id}.pkl")
        data_score = self._load_pkl(fpath_att_score)

        coordinates = data_score["coordinate"]
        layer_info = data_score["attention"][f"layer{layer}"]
        include_nodes = coordinates[layer_info["include_nodes"]]

        selected_patch = np.array(
            [f"{self.dir_feature}/{slide_id}/{x[0]}_{x[1]}.pkl" for x in include_nodes]
        )
        # temp fix
        # exist_file = np.array([os.path.exists(x) for x in selected_patch])
        # exist_file = selected_patch
        # selected_patch = selected_patch[exist_file]
        # layer_score = layer_info["score"][exist_file]
        layer_score = layer_info["score"]
        # temp fix

        # if patch from high and low risk score are both picked from high attention region
        idx_sort = np.argsort(layer_score)[::-1]
        # if patch from high risk score is  picked from high attention region
        # and if patch from low risk score is  picked from low attention region
        if self.low_score_low_risk:
            if risk_group == "low":
                idx_sort = np.argsort(layer_score)

        selected_patch = selected_patch[idx_sort]
        if self.top_percent > 0:
            top_len = int(len(selected_patch) * (self.top_percent / 100))
        elif self.top_patch > 0:
            if len(selected_patch) >= self.top_patch:
                top_len = self.top_patch
            else:
                top_len = len(selected_patch)

        top_patch = selected_patch[:top_len]
        top_coordinates = coordinates[idx_sort][:top_len]

        return slide_id, top_coordinates, top_patch

    def get_top_features(self, idx: int, layer: int):
        slide_id = self.slide_ids[idx]
        _, _, top_patch = self.get_top_patches(idx=idx, layer=layer)

        num_nuclei_type = len(self.nuclei_types)
        dct_feat_nuc = {x + 1: [] for x in range(num_nuclei_type)}
        for f in top_patch:
            temp_ = self._load_pkl(f)
            for k, _ in dct_feat_nuc.items():
                # ignore zero vector/no nuclei detected
                if sum([sum(x) for x in (temp_[k])]) > 0:
                    dct_feat_nuc[k].append(temp_[k])

        for k, v in dct_feat_nuc.items():
            if len(v) == 0:
                dct_feat_nuc[k] = np.zeros((1, len(self.feature_names)))
            else:
                dct_feat_nuc[k] = np.concatenate(v)

        shape_feature = {}
        for index, (key, value) in enumerate(dct_feat_nuc.items()):
            shape_feature[key] = {
                "nuclei_type": self.nuclei_types[index],
                "feature_names": self.feature_names,
                "features": value,  # np.mean(value, axis=1),
            }

        return slide_id, shape_feature

    def get_layer_features(self, layer: int):
        dct_features = {}
        for idx, _ in enumerate(self.slide_ids):

            fname, temp_feat = self.get_top_features(idx=idx, layer=layer)

            dct_features[fname] = {}

            for nuc_id, nuc_name in enumerate(self.nuclei_types):
                dct_features[fname][nuc_name] = {}

                for feat_id, feat_name in enumerate(self.feature_names):
                    dct_features[fname][nuc_name][feat_name] = np.array([])

                    curr_value = temp_feat[nuc_id + 1]["features"][:, feat_id]
                    if np.sum(curr_value) > 0:
                        dct_features[fname][nuc_name][feat_name] = curr_value

        return dct_features

    def get_all_features(self, layer: int):
        if layer == -1:
            # dct_list = []
            for i in range(self.num_layer):
                temp = self.get_layer_features(layer=i + 1)
                if i == 0:
                    dct_features = temp
                else:
                    for slide_name, dct_nuc in temp.items():
                        for nuc_name, dct_feat in dct_nuc.items():
                            for feat_name, value in dct_feat.items():
                                val_old = dct_features[slide_name][nuc_name][feat_name]
                                new_val = np.concatenate((val_old, value))
                                dct_features[slide_name][nuc_name][feat_name] = new_val

        else:
            dct_features = self.get_layer_features(layer=layer)

        self.dct_feature_all_slides = dct_features

    def combine_dataframe(self):
        # combine dataframe, to get risk group with its slide name and patient id
        df_result = self.ps_object.df
        df_survival = self.df_survival[["patient", "wsi"]].copy(deep=True)

        group_per_wsi = []
        for patient in df_survival["patient"]:
            group_per_wsi.append(
                df_result[df_result["patient"] == str(patient)]["group"].values[0]
            )
        df_survival["group"] = group_per_wsi

        return df_survival

    def feature_per_group(
        self,
        feature_name: str,
        nuclei_name: str = "neoplasm",
        avg_per_slide: bool = True,
    ):
        assert self.dct_feature_all_slides is not None, "call get_all_features() first!"

        dct_group = {"high": [], "low": []}
        for k, v in self.dct_feature_all_slides.items():
            group_name = self.df_complete[self.df_complete["wsi"] == k]["group"].values[
                0
            ]
            current_value = v[nuclei_name][feature_name]
            if avg_per_slide:
                # return mean of nuclei feature per slide
                if len(current_value) > 0:
                    val = np.mean(current_value)
                    dct_group[group_name].append(val)
            else:
                # return all of nuclei feature
                if current_value is not None:
                    dct_group[group_name].append(current_value)

        if not avg_per_slide:
            dct_group["high"] = np.concatenate(dct_group["high"])
            dct_group["low"] = np.concatenate(dct_group["low"])

        return dct_group

    def get_top_patch_img(
        self,
        idx: int = None,
        slide_id: str = None,
        layer: int = 1,
        size: list = (1024, 1024),
        downsample: int = 2,
    ):
        assert (idx == None and slide_id != None) or (
            idx != None and slide_id == None
        ), "slide or index"
        if idx != None:
            slide_id = self.slide_ids[idx]
        if slide_id != None:
            idx = self.slide_ids.index(slide_id)
        wsi_fpath = os.path.join(self.dir_slide, f"{slide_id}.svs")
        wsi = ops.OpenSlide(wsi_fpath)

        slide_id_, coors, _ = self.get_top_patches(idx=idx, layer=layer)
        assert str(slide_id) == str(slide_id_), "something wrong"
        imgs = []
        for c in coors:
            img = wsi.read_region(c, 0, size).resize([x // downsample for x in size])
            imgs.append(img)
        return imgs
