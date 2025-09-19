import path_project

from gnpc.default_modules import *
from gnpc.preprocessing.patch_worker import *
from gnpc.utils.io import *
from gnpc.utils.logger import LogMsg


def main(
    csv_file: str,
    dir_output: str,
    save_coord: bool,
    save_thumb: bool,
    save_mask: bool,
    save_img: bool,
    do_stitch: bool,
    downsample: int,
    config_patch: dict,
    config_mask: dict,
):
    df = pd.read_csv(csv_file)
    slide_list = df["path"].values.tolist()

    dir_output_thumbnail = f"{dir_output}/thumbnail"
    dir_output_mask = f"{dir_output}/mask"
    dir_output_coords = f"{dir_output}/coords"
    dir_output_stich = f"{dir_output}/stitches"
    dir_output_patch = f"{dir_output}/patches"

    if not os.path.exists(dir_output_thumbnail) and save_thumb:
        os.makedirs(dir_output_thumbnail)
    if not os.path.exists(dir_output_mask) and save_mask:
        os.makedirs(dir_output_mask)
    if not os.path.exists(dir_output_coords) and save_coord:
        os.makedirs(dir_output_coords)
    if not os.path.exists(dir_output_stich) and do_stitch:
        os.makedirs(dir_output_stich)
    if not os.path.exists(dir_output_patch) and save_img:
        os.makedirs(dir_output_patch)

    logger = LogMsg(fpath_log=f"{dir_output}/process.log")
    save_json(
        config_patch,
        f"{dir_output}/config_patch.json",
    )
    save_json(
        config_mask,
        f"{dir_output}/config_mask.json",
    )

    logger.msg(
        f"Patches extracted at mag: {config_patch['target_mag']} size: {config_patch['target_patch_size']}"
    )

    for ix, fpath_wsi in enumerate(slide_list):

        patchObj = WSIPatchGenerator(
            fpath_wsi=fpath_wsi, mask_downsample=downsample, mask_config=config_mask
        )
        if save_thumb:
            thumbnail = Image.fromarray(patchObj.thumbnail)
            thumbnail.save(f"{dir_output_thumbnail}/{patchObj.wsi_name}.jpg")
        if save_mask:
            cv2.imwrite(
                f"{dir_output_mask}/{patchObj.wsi_name}.jpg", patchObj.binary_mask
            )
        patchObj.get_patch_coords(
            target_patch_size=config_patch["target_patch_size"],
            target_mag=config_patch["target_mag"],
            foreground_min_percent=config_patch["foreground_min_percent"],
            max_white_area=config_patch["max_white_area"],
            white_threshold=config_patch["white_threshold"],
        )
        if save_coord:
            fpath_patch = f"{dir_output_coords}/{patchObj.wsi_name}.pkl"
            patchObj.save_coord_info(fpath_patch)

        if save_img:
            dir_img = f"{dir_output_patch}/{patchObj.wsi_name}"
            if not os.path.exists(dir_img):
                os.makedirs(dir_img)
            patchObj.save_all_patches(dir_img)

        if do_stitch:
            # pbar_slide.set_description(f"{tqdm_msg}  -- stitching")
            fpath_stitch = f"{dir_output_stich}/{patchObj.wsi_name}.jpg"
            stich = patchObj.stitch_patch()
            Image.fromarray(stich).save(fpath_stitch)

        logger.msg(
            f"{ix+1}/{len(slide_list)} Patching slide:{patchObj.wsi_name} Base mag:{patchObj.base_mag} Num patches:{len(patchObj)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str)
    parser.add_argument("--dir_output", type=str)
    parser.add_argument("--downsample", type=int)
    parser.add_argument("--kernel_close", type=int)
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--med_blur", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--target_mag", type=int)
    parser.add_argument("--percent", type=float)
    parser.add_argument("--max_white_area", type=int)
    parser.add_argument("--white_threshold", type=int)
    parser.add_argument("--save_thumb", action="store_true", default=False)
    parser.add_argument("--save_mask", action="store_true", default=False)
    parser.add_argument("--save_coord", action="store_true", default=False)
    parser.add_argument("--save_img", action="store_true", default=False)
    parser.add_argument("--stitch", action="store_true", default=False)

    args = parser.parse_args()
    config_patch = {
        "target_patch_size": (args.patch_size, args.patch_size),
        "target_mag": args.target_mag,
        "foreground_min_percent": args.percent,
        "max_white_area": args.max_white_area,
        "white_threshold": args.white_threshold,
    }
    config_mask = {
        "kernel_close": args.kernel_close,
        "threshold": args.threshold,
        "med_blur": args.med_blur,
    }
    main(
        csv_file=args.csv,
        dir_output=args.dir_output,
        save_coord=args.save_coord,
        save_thumb=args.save_thumb,
        save_mask=args.save_mask,
        save_img=args.save_img,
        do_stitch=args.stitch,
        downsample=args.downsample,
        config_patch=config_patch,
        config_mask=config_mask,
    )
