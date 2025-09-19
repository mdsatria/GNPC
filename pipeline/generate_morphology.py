import re
import shutil

import path_project

# from functools import partial
from tqdm.contrib.concurrent import process_map

from gnpc.default_modules import *
from gnpc.feature.morphology import *
from gnpc.utils.io import *
from gnpc.utils.logger import LogMsg
from gnpc.utils.misc import StopWatch


def get_file_size_in_mb(file_path):
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb


def check_processed(path):
    if not os.path.exists(path):
        return False
    else:
        for root, dirs, files in os.walk(path):
            if len(files) == 0:
                os.rmdir(path)
                return False
            else:
                return True


def get_finished_slide(log_file):
    finished_slide = []
    with open(log_file, "r") as file:
        for line in file:
            match = re.search(r"\[([^\]]*)\]", line)
            if match:
                finished_slide.append(match[1])
    return finished_slide


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_slide", type=str)
    parser.add_argument("--dir_wsi", type=str)
    parser.add_argument("--dir_nuclei", type=str)
    parser.add_argument("--dir_patch", type=str)
    parser.add_argument("--dir_save", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--target_mpp", type=float)

    args = parser.parse_args()

    csv_slide = args.csv_slide
    dir_wsi = args.dir_wsi
    dir_nuclei = args.dir_nuclei
    dir_patch = args.dir_patch
    dir_save = args.dir_save
    log_file = args.log_file
    target_mpp = args.target_mpp

    slide_list = [
        pathlib.Path(x).stem for x in pd.read_csv(csv_slide)["wsi"].values.tolist()
    ]
    timer = StopWatch()

    counter = 0
    if os.path.exists(log_file):
        finished_slide = get_finished_slide(log_file)
        slide_list = [x for x in slide_list if x not in finished_slide]
        logger = LogMsg(fpath_log=log_file, is_append=True)
        counter = len(finished_slide)
    else:
        logger = LogMsg(fpath_log=log_file, is_append=False)

    if counter == 0:
        logger.msg(f"Generating morphology feature at {target_mpp}mpp")

    for ix, slide in enumerate(slide_list):
        timer.start()
        is_ok = True

        fpath_wsi = os.path.join(dir_wsi, f"{slide}.svs")
        fpath_patch = os.path.join(dir_patch, f"{slide}.pkl")
        fpath_nuclei = os.path.join(dir_nuclei, f"{slide}.pkl")

        dir_save_slide_raw = os.path.join(dir_save, f"{slide}", "raw")
        dir_save_slide_stat = os.path.join(dir_save, f"{slide}", "stat")

        nuclei_file_size = get_file_size_in_mb(fpath_nuclei)
        num_workers = 6
        if nuclei_file_size > 300:
            num_workers = 4
        if nuclei_file_size > 550:
            num_workers = 2
        if nuclei_file_size > 800:
            num_workers = 1

        log_msg = f"{ix+1+counter}/{len(slide_list)+counter}: [{slide}] "
        if not os.path.exists(fpath_nuclei):
            log_msg = f"{log_msg} NUCLEI not exist!"
            is_ok == False
        if not os.path.exists(fpath_wsi):
            log_msg = f"{log_msg} WSI not exist!"
            is_ok == False
        if not os.path.exists(fpath_patch):
            log_msg = f"{log_msg} PATCH not exist!"
            is_ok == False
        if is_ok:
            # try:
            nuclei_per_patch = NucleiPatchMorphology(
                fpath_wsi=fpath_wsi,
                fpath_nuclei=fpath_nuclei,
                fpath_patch=fpath_patch,
                morph_mpp=target_mpp,
            )

            def save_feature(index):
                df_feature, agg_features, topleft_patch, _ = (
                    nuclei_per_patch.aggregate_feature_per_patch(
                        index=index,
                        exclude_type=[0, 3, 4, 5],
                    )
                )
                fpath_raw = os.path.join(
                    dir_save_slide_raw,
                    f"{index}_{topleft_patch[0]}_{topleft_patch[1]}.csv",
                )
                fpath_percentile = os.path.join(
                    dir_save_slide_stat,
                    f"{index}_{topleft_patch[0]}_{topleft_patch[1]}.pkl",
                )
                save_pickle(agg_features, fpath_percentile)
                df_feature.to_csv(fpath_raw, index=False)

            if os.path.exists(dir_save_slide_raw):
                shutil.rmtree(dir_save_slide_raw)
            if os.path.exists(dir_save_slide_stat):
                shutil.rmtree(dir_save_slide_stat)

            os.makedirs(dir_save_slide_raw)
            os.makedirs(dir_save_slide_stat)
            contain_nuclei = False
            patch_indices = [x for x in range(len(nuclei_per_patch))]

            if nuclei_per_patch.contain_nuclei:
                process_map(
                    save_feature,
                    patch_indices,
                    max_workers=num_workers,
                )
                contain_nuclei = True

            log_msg = f"{log_msg} base-mpp:{nuclei_per_patch.base_mpp} nuclei-mpp:{nuclei_per_patch.nuclei_mpp} nuclei-factor:{nuclei_per_patch.morph_downsample}"
            if not contain_nuclei:
                log_msg = f"{log_msg} NO NUCLEI WAS DETECTED"

            # except:
            #     log_msg = f"{log_msg} SOMETHING WRONG!!!!"
        elapsed_time = timer.stop_and_count()
        logger.msg(f"{log_msg} in {elapsed_time:.4f}")
