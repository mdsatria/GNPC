import timm
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer

from gnpc.default_modules import *

# from gnpc.utils.io import *


def load_model_uni(
    # https://github.com/mahmoodlab/UNI
    ckpt_path: str = "ckpts/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin",
    device: str | list = "cuda",
):
    uni_kwargs = {
        "model_name": "vit_large_patch16_224",
        "img_size": 224,
        "patch_size": 16,
        "init_values": 1e-5,
        "num_classes": 0,
        "dynamic_img_size": True,
    }
    model = timm.create_model(**uni_kwargs)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model = model.to(device)
    return model


def load_model_dinovit16(
    patch_size: int = 16,
    pretrained_model_path: str = "ckpts/ssl/dino_vit_small_patch16_ep200.torch",
    device: str | list = "cuda",
):
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(device)
    return model


def load_model_virchow(device: str | list = "cuda"):
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval()
    model.to(device)
    return model


# def infer_dloader(model, model_name, dloader, device):

#     coords = []
#     centroids = []
#     embeddings = []
#     with torch.inference_mode():
#         for _, batch in enumerate(dloader):

#             img_batch, coord_batch, centroid_batch = batch
#             img_batch = img_batch.to(device)
#             embedding_batch = model(img_batch).detach().cpu()

#             if model_name == "resnet50":
#                 embedding_batch = embedding_batch.view(
#                     embedding_batch.shape[0], embedding_batch.shape[1]
#                 )
#             elif model_name == "virchow":
#                 # https://huggingface.co/paige-ai/Virchow
#                 class_token = embedding_batch[:, 0]
#                 patch_tokens = embedding_batch[:, 1:]
#                 embedding_batch = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
#             if embedding_batch.ndim != 2:
#                 # debug
#                 print(embedding_batch.shape)

#             coords.append(coord_batch)
#             centroids.append(centroid_batch)
#             embeddings.append(embedding_batch)

#     coords = torch.cat(coords, dim=0).numpy()
#     centroids = torch.cat(centroids, dim=0).numpy()
#     embeddings = torch.cat(embeddings, dim=0).numpy()

#     return coords, centroids, embeddings


# def save_embedding(slide_name, fpath_save, coords, centroids, embeddings):
#     # fpath_save = os.path.join(dir_save, f"{slide_name}.pkl")
#     dct = {
#         "slide": slide_name,
#         "coords": coords,
#         "centroids": centroids,
#         "embeddings": embeddings,
#     }
#     save_pickle(dct, fpath_save)
