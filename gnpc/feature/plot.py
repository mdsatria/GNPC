from skimage.exposure import equalize_hist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gnpc.default_modules import *


class VisualGraph:
    def __init__(self, wsi_path, embedding_path, downsampling=16):

        self.downsampling = downsampling
        self.wsi = ops.OpenSlide(wsi_path)

        self.graph_data = torch.load(embedding_path)

    def make_canvas(
        self,
        canvas: np.ndarray,
        nodes: np.ndarray,
        edges: np.ndarray,
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

        for idx, (src, dst) in enumerate(edges):
            src_ = to_int_tuple(nodes[src])
            dst_ = to_int_tuple(nodes[dst])
            color = to_int_tuple(edge_colors[idx])
            cv2.line(canvas, src_, dst_, color, thickness=edge_size)

        # draw the nodes
        for idx, node in enumerate(nodes):
            node_ = to_int_tuple(node)
            color = to_int_tuple(node_colors[idx])
            cv2.circle(canvas, node_, node_size, color, thickness=-1)
        return canvas

    def plot_graph(self, node_size=25, figsize=(20, 20), dpi=100, edge_size=3):

        self.graph_data.x = StandardScaler().fit_transform(self.graph_data.x)
        node_colors = PCA(n_components=3).fit_transform(self.graph_data.x)[:, [1, 0, 2]]
        for channel in range(node_colors.shape[-1]):
            node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
        node_colors = (node_colors * 255).astype(np.uint8)

        plot_dimension = [(x // self.downsampling) for x in self.wsi.dimensions]
        node_coordinates = np.array(self.graph_data.coordinates) / (self.downsampling)

        edges = self.graph_data.edge_index.T
        thumbnail = np.array(self.wsi.get_thumbnail(plot_dimension))
        thumb_overlaid = self.make_canvas(
            thumbnail.copy(),
            node_coordinates,
            edges,
            node_colors=node_colors,
            node_size=node_size,
            edge_size=edge_size,
        )

        fig, axes = plt.subplots(1, figsize=figsize, dpi=dpi)
        axes.imshow(thumb_overlaid)
        axes.axis("off")
        plt.show()
