import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from gnpc.default_modules import *
from gnpc.preprocessing.patch_base import WSIVisual
from gnpc.utils.io import load_pickle


class WSIVisualGraph(WSIVisual):
    def __init__(self, fpath_wsi, fpath_graph, downsample=16):
        super().__init__(fpath_wsi)
        self.downsample = downsample
        self.fpath_graph = fpath_graph
        self.graph_data = torch.load(self.fpath_graph)

    def make_canvas(
        self,
        canvas: np.ndarray,
        nodes: np.ndarray,
        att_mask,
        edges: np.ndarray,
        edge_weight: np.ndarray,
        node_colors: tuple[int] = (255, 0, 0),
        node_size: int = 5,
        edge_colors: tuple[int] = (0, 0, 0),
        edge_size: int = 3,
    ) -> np.ndarray:
        if isinstance(node_colors, tuple):
            node_colors = [node_colors] * len(nodes)
        if isinstance(edge_colors, tuple):
            edge_colors = [edge_colors] * len(edges)

        # draw the edges
        def to_int_tuple(x: list) -> tuple[int, ...]:
            """Helper to convert to tuple of int."""
            return tuple(int(v) for v in x)

        for idx, ((src, dst), wgt) in enumerate(zip(edges, edge_weight)):
            src_ = to_int_tuple(nodes[src])
            dst_ = to_int_tuple(nodes[dst])
            color = to_int_tuple(edge_colors[idx])
            cv2.line(canvas, src_, dst_, color, thickness=round(edge_size * wgt) + 1)

        # draw the nodes

        for idx, node in enumerate(nodes):
            if len(att_mask) > 0:
                if idx in att_mask:
                    node_ = to_int_tuple(node)
                    color = to_int_tuple(node_colors[idx])
                    cv2.circle(canvas, node_, node_size, color, thickness=-1)
            else:
                node_ = to_int_tuple(node)
                color = to_int_tuple(node_colors[idx])
                cv2.circle(canvas, node_, node_size, color, thickness=-1)
        return canvas

    def visualise(
        self,
        att_mask=[],
        node_size=25,
        figsize=(20, 20),
        dpi=300,
        edge_size=3,
        node_color=None,
        save_path=None,
    ):
        slide_dimension = self.wsi.dimensions

        graph_data = {k: np.array(v) for k, v in self.graph_data.items()}
        graph = Data(**graph_data)

        graph.x = StandardScaler().fit_transform(graph.x)

        if node_color is None:
            node_colors = PCA(n_components=3).fit_transform(graph.x)[:, [1, 0, 2]]
            for channel in range(node_colors.shape[-1]):
                node_colors[:, channel] = (
                    1 - equalize_hist(node_colors[:, channel]) ** 2
                )
            node_colors = (node_colors * 255).astype(np.uint8)
        else:
            node_colors = np.array(
                [node_color for _ in range(graph.x.shape[0])]
            ).astype(np.uint8)

        plot_dimension = [int(x / self.downsample) for x in slide_dimension]
        node_coordinates = np.array(graph.coordinates) / self.downsample

        edges = graph.edge_index.T
        edge_weight = graph.edge_weight

        if len(att_mask) > 0:
            # node_coordinates = node_coordinates[att_mask]
            # edge_weight = edge_weight[att_mask]

            mask_edge = np.isin(edges, att_mask).all(axis=1)
            edges = edges[mask_edge]
            edge_weight = edge_weight[mask_edge]
            #     edges = edges[att_mask]

        thumbnail = np.array(self.wsi.get_thumbnail(plot_dimension))
        thumb_overlaid = self.make_canvas(
            thumbnail.copy(),
            node_coordinates,
            att_mask,
            edges,
            edge_weight,
            node_colors=node_colors,
            node_size=node_size,
            edge_size=edge_size,
        )

        fig, axes = plt.subplots(1, figsize=figsize, dpi=dpi)
        axes.imshow(thumb_overlaid)
        axes.axis("off")
        if save_path != None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=False)
            plt.close()
        else:
            plt.show()


class WSIVisualAttention(WSIVisual):
    def __init__(
        self,
        fpath_wsi,
        fpath_patch,
        fpath_mask,
        attention_info,
        downsample=None,
    ):
        super().__init__(fpath_wsi)

        self.fpath_mask = fpath_mask
        if downsample is None:
            self.downsample = self.patch_info["mask_downsample"]
        else:
            self.downsample = downsample

        self._load_patch_info(fpath_patch)
        if isinstance(attention_info, dict):
            self.all_attention = attention_info
        else:
            self.all_attention = load_pickle(attention_info)

        self.dct_attention = None
        self.score_min = None
        self.score_max = None

        self.thumbnail = self.get_wsi_thumbnail(self.downsample)
        self.thumb_size = [x // self.downsample for x in self.wsi.dimensions]

    def set_attention_layer(self, layer):
        self.dct_attention = self.all_attention["attention"][f"layer{layer}"]
        self.score_min = np.min(self.dct_attention["score"])
        self.score_max = np.max(self.dct_attention["score"])

    def _normalize_array(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    def _colour_gradation(self, cmap_name):
        rgba_colours = [
            eval(f"cm.{cmap_name}({i})")
            for i in np.linspace(0, 1, len(self.dct_attention["score"]))
        ]
        rgba_colours = np.array(rgba_colours)
        rgb_colours = rgba_colours[:, :3]
        rgb_colours = rgb_colours * 255.0
        sort_idx = np.argsort(self.dct_attention["score"])  # low score to high
        interpolated_colors = rgb_colours[sort_idx]

        return interpolated_colors

    def _get_heatmap(self, nodes_info, resize_factor=2, cmap_name="jet"):
        colours = self._colour_gradation(cmap_name)
        thumb_shape = np.array(self.thumbnail).shape
        canvas_threshold = np.full((thumb_shape), 255, dtype=np.uint8)
        canvas_heatmap = np.full((thumb_shape), 255, dtype=np.uint8)
        size_x, size_y = [
            int(x // (self.downsample / resize_factor))
            for x in self.patch_info["target_patch_size"]
        ]
        colour_foreground = [0, 0, 0]

        for ix, ix_node in enumerate(nodes_info):
            node_coord = self.all_attention["coordinate"][ix_node]
            node_coord = [x // self.downsample for x in node_coord]
            x_box, y_box = node_coord
            canvas_threshold[y_box : y_box + size_y, x_box : x_box + size_x, ...] = (
                colour_foreground
            )
            canvas_heatmap[y_box : y_box + size_y, x_box : x_box + size_x, ...] = (
                colours[ix]
            )

        return canvas_threshold, canvas_heatmap

    def get_sample_include(self, index=None):
        if index == None:
            index = random.choice(self.dct_attention["include_nodes"])
        coord = self.all_attention["coordinate"][index]
        patch = self.wsi.read_region((coord), 0, self.patch_info["target_patch_size"])
        return np.array(patch)

    def get_sample_exclude(self, index=None):
        if index == None:
            index = random.choice(self.exclude_nodes)
        coord = self.all_attention["coordinate"][index]
        patch = self.wsi.read_region((coord), 0, self.patch_info["target_patch_size"])
        return np.array(patch)

    def visualise(
        self,
        layer=1,
        alpha=0.4,
        blur_radius=20,
        high_attention=True,
        cmap_name="autumn",
    ):
        self.set_attention_layer(layer=layer)

        if high_attention:
            _, heatmap = self._get_heatmap(
                self.dct_attention["include_nodes"], cmap_name=cmap_name
            )
        else:
            _, heatmap = self._get_heatmap(
                self.dct_attention["exclude_nodes"], cmap_name=cmap_name
            )

        mask = Image.open(self.fpath_mask).convert("L")
        thumb = self.thumbnail.resize(mask.size)
        heatmap = Image.fromarray(heatmap).resize(mask.size)

        heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # remove background from heatmap
        mask = np.array(mask)
        heatmap = np.array(heatmap)
        foreground_pixel = mask == 0
        heatmap[foreground_pixel] = [255, 255, 255]

        heatmap = Image.fromarray(heatmap)
        output = Image.blend(thumb, heatmap, alpha=alpha)

        return output


class WSIVisualCluster(WSIVisual):
    def __init__(self, fpath_wsi, fpath_embedding, fpath_patch):
        super().__init__(fpath_wsi)
        self._load_patch_info(fpath_patch)
        self.fpath_embedding = fpath_embedding

        self.data_embedding = load_pickle(self.fpath_embedding)
        self.embedding = self.data_embedding["embeddings"]
        self.coords = self.data_embedding["coords"]

    def generate_rgb_colors(self, n, cmap_name="tab10"):
        cmap = colormaps.get_cmap(cmap_name)
        colors = []
        for i in range(n):
            color = cmap(i / (n - 1))[:3]  # Get RGB values (ignoring alpha)
            colors.append(color)

        return colors

    def _cluster_embedding(self, num_cluster, random_state):
        self.clusters_label = (
            KMeans(n_clusters=num_cluster, random_state=random_state, n_init="auto")
            .fit(self.embedding)
            .labels_
        )

    def _pca_embedding(self, random_state, n_components=2):
        self.pca_component = PCA(
            n_components=n_components, random_state=random_state
        ).fit_transform(self.embedding)

    def visualise(
        self,
        num_cluster,
        num_samples=10,
        thumb_scatter_size=(20, 20),
        figsize=(10, 10),
        random_state=42,
        random_pick=False,
    ):
        self._cluster_embedding(num_cluster=num_cluster, random_state=random_state)
        self._pca_embedding(random_state=random_state)

        dct_cls = {x: np.where(self.clusters_label == x)[0] for x in range(num_cluster)}
        colours = self.generate_rgb_colors(n=num_cluster)

        fig, ax = plt.subplots(figsize=figsize)
        for x in range(num_cluster):
            if random_pick:
                indices = np.random.choice(dct_cls[x], num_samples, replace=False)
            else:
                indices = dct_cls[x][:num_samples]
            ax.scatter(
                self.pca_component[indices][:, 0], self.pca_component[indices][:, 1]
            )
            for ix in indices:
                point = self.pca_component[ix]
                im, _, _ = self.get_img_patch(ix)
                im = im.resize(thumb_scatter_size)
                im_offset = OffsetImage(im)
                imagebox = AnnotationBbox(
                    im_offset,
                    (point[0], point[1]),
                    bboxprops=dict(edgecolor=colours[x]),
                    pad=0.1,
                )
                ax.add_artist(imagebox)


class WSIVisualGridPatch(WSIVisual):
    """
    visualise grid of patches
    """

    def __init__(self, fpath_wsi, fpath_embedding, fpath_patch):
        super().__init__(fpath_wsi)
        self._load_patch_info(fpath_patch)
        self.fpath_embedding = fpath_embedding

        self.data_embedding = load_pickle(self.fpath_embedding)
        self.embedding = self.data_embedding["embeddings"]
        self.coords = self.data_embedding["coords"]

    def visualise(
        self,
        downsample: int = 32,
        save_path: str = None,
        draw_index: bool = False,
        figsize: list[int] = (20, 20),
        dpi: int = 300,
        linewidth: int = 10,
    ):
        patch_size_x = self.patch_info["source_patch_size"][0] // downsample
        patch_size_y = self.patch_info["source_patch_size"][1] // downsample
        coords = [
            self.patch_info["patch"][x]["top_left"]
            for x in list(self.patch_info["patch"].keys())
        ]

        thumbnail = self.get_wsi_thumbnail(downsample=downsample)
        draw = ImageDraw.Draw(thumbnail)

        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except IOError:
            font = ImageFont.load_default()
        if linewidth > 0:
            for ixy, (x, y) in enumerate(coords):
                x = x // downsample
                y = y // downsample
                if draw_index:
                    draw.text([x, y], f"{ixy}", fill="black", font=font)

                if (x, y - patch_size_y) not in coords:
                    draw.line(
                        [(x, y), (x + patch_size_x, y)], fill="black", width=linewidth
                    )

                if (x - patch_size_x, y) not in coords:
                    draw.line(
                        [(x, y), (x, y + patch_size_y)], fill="black", width=linewidth
                    )

                if (x, y + patch_size_y) not in coords:
                    draw.line(
                        [(x, y + patch_size_y), (x + patch_size_x, y + patch_size_y)],
                        fill="black",
                        width=linewidth,
                    )

                if (x + patch_size_x, y) not in coords:
                    draw.line(
                        [(x + patch_size_x, y), (x + patch_size_x, y + patch_size_y)],
                        fill="black",
                        width=linewidth,
                    )
            # draw.rectangle(
            #     [(x, y), (x + patch_size_x, y + patch_size_y)], outline="black", width=2
            # )
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(thumbnail)
        ax.axis("off")
        if save_path != None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=False)
            plt.close()
        else:
            plt.show()
