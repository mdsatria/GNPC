from torchvision.models import resnet50

from gnpc.default_modules import *
from gnpc.utils.io import *


def load_model_resnet50(device: str | list = "cuda"):
    model = resnet50(weights="IMAGENET1K_V1")
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if device is not None:
        model = model.to(device)
    return model


def infer_dloader(model, model_name, dloader, device):

    coords = []
    centroids = []
    embeddings = []
    with torch.inference_mode():
        for _, batch in enumerate(dloader):

            img_batch, coord_batch, centroid_batch = batch
            img_batch = img_batch.to(device)
            embedding_batch = model(img_batch).detach().cpu()

            if model_name == "resnet50":
                embedding_batch = embedding_batch.view(
                    embedding_batch.shape[0], embedding_batch.shape[1]
                )
            elif model_name == "virchow":
                # https://huggingface.co/paige-ai/Virchow
                class_token = embedding_batch[:, 0]
                patch_tokens = embedding_batch[:, 1:]
                embedding_batch = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            if embedding_batch.ndim != 2:
                # debug
                print(embedding_batch.shape)

            coords.append(coord_batch)
            centroids.append(centroid_batch)
            embeddings.append(embedding_batch)

    coords = torch.cat(coords, dim=0).numpy()
    centroids = torch.cat(centroids, dim=0).numpy()
    embeddings = torch.cat(embeddings, dim=0).numpy()

    return coords, centroids, embeddings


def save_embedding(slide_name, fpath_save, coords, centroids, embeddings):
    # fpath_save = os.path.join(dir_save, f"{slide_name}.pkl")
    dct = {
        "slide": slide_name,
        "coords": coords,
        "centroids": centroids,
        "embeddings": embeddings,
    }
    save_pickle(dct, fpath_save)
