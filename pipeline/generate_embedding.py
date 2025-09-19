import argparse

import path_project

from gnpc.default_modules import *
from gnpc.feature.embedding import *

# from gnpc.feature.foundation_model import *
from gnpc.preprocessing.patch_embedding import make_dataloader
from gnpc.utils.logger import LogMsg


def main(
    dir_patch,
    dir_wsi,
    dir_out,
    cohort,
    csv_wsi,
    model_name,
    wsi_ext="svs",
    batch_size=64,
):
    logger = LogMsg(
        fpath_log=os.path.join(dir_out, cohort, f"process_{model_name}.log")
    )
    fnames = pd.read_csv(csv_wsi)["wsi"].values.tolist()
    dir_target = os.path.join(dir_out, cohort, model_name)
    if not os.path.exists(dir_target):
        os.makedirs(dir_target)

    if model_name == "resnet50":
        model = load_model_resnet50()
    else:
        from gnpc.feature.foundation_model import (
            load_model_dinovit16,
            load_model_uni,
            load_model_virchow,
        )

        if model_name == "dino":
            model = load_model_dinovit16()
        elif model_name == "uni":
            model = load_model_uni()
        elif model_name == "virchow":
            model = load_model_virchow()

    for ix, fname in enumerate(fnames):
        f = pathlib.Path(fname).stem
        logger.msg(
            f"cohort:{cohort.upper()} model:{model_name.upper()} {ix+1}/{len(fnames)} {f}"
        )
        fpath_patch = os.path.join(dir_patch, f"{f}.pkl")
        fpath_wsi = os.path.join(dir_wsi, f"{f}.{wsi_ext}")
        fpath_embedding = os.path.join(dir_target, f"{f}.pkl")
        if not os.path.exists(fpath_embedding):
            if model_name == "virchow":
                dloader = make_dataloader(
                    fpath_patch=fpath_patch,
                    fpath_wsi=fpath_wsi,
                    batch_size=batch_size,
                    virchow_model=model,
                    num_workers=4,
                )
            else:
                dloader = make_dataloader(
                    fpath_patch=fpath_patch,
                    fpath_wsi=fpath_wsi,
                    batch_size=batch_size,
                    virchow_model=None,
                    num_workers=4,
                )

            coords, centroids, embeddings = infer_dloader(
                model=model, model_name=model_name, dloader=dloader, device="cuda"
            )
            save_embedding(f, fpath_embedding, coords, centroids, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_patch", type=str)
    parser.add_argument("--dir_wsi", type=str)
    parser.add_argument("--dir_save", type=str)
    parser.add_argument("--csv_wsi", type=str)
    parser.add_argument("--models", type=str)
    parser.add_argument("--cohort", type=str)

    args = parser.parse_args()

    models_name = [x.strip() for x in args.models.split(",")]

    for m in models_name:
        main(
            dir_patch=args.dir_patch,
            dir_wsi=args.dir_wsi,
            dir_out=args.dir_save,
            csv_wsi=args.csv_wsi,
            cohort=args.cohort,
            model_name=m,
        )
