import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GraphSurv(torch.nn.Module):
    def __init__(
        self,
        num_feature: int,
        num_class: int = 1,
        nhid: int = 512,
        is_weighted: bool = True,
        return_attention: bool = True,
        is_multimodal: bool = False,
        dropout: float = 0.4,
        conv: str = "GCN",
        activation_conv: str = "LeakyReLU",
        use_norm_fcn: bool = True,
        drop_fcn: bool = 0.5,
        activation_fcn: str = "SELU",
        use_last_activation_fcn: bool = True,
        pool_ratio: bool = 0.5,
        num_layer: int = 4,
        clinical_len: int = 7,
    ):
        super(GraphSurv, self).__init__()
        if is_weighted:
            assert conv == "GCN", f"weighted graph is only supported by {conv}!"
        self.num_feature = num_feature
        self.num_class = num_class
        self.nhid = nhid
        self.is_weighted = is_weighted
        self.is_multimodal = is_multimodal
        self.return_attention = return_attention
        self.clinical_len = clinical_len

        self.dropout = dropout
        self.conv = conv
        self.activation_conv = activation_conv
        self.use_norm_fcn = use_norm_fcn
        self.drop_fcn = drop_fcn
        self.activation_fcn = activation_fcn
        self.use_last_activation_fcn = use_last_activation_fcn
        self.pool_ratio = pool_ratio
        self.num_layer = num_layer

        self.gnn_layers = {}
        for i in range(self.num_layer):
            if i == 0:
                self.gnn_layers[f"conv_{self.conv}_{i}"] = self._build_conv(
                    self.num_feature, self.nhid
                )
            else:
                self.gnn_layers[f"conv_{self.conv}_{i}"] = self._build_conv(
                    self.nhid, self.nhid
                )
            self.gnn_layers[f"nonlinear_{i}"] = eval(f"nn.{self.activation_conv}()")
            self.gnn_layers[f"pool_{i}"] = self._build_pool()

        self.gnn_layers = nn.ModuleDict(self.gnn_layers)
        if self.is_multimodal:
            self.fcn_graph = self._build_fcn_modal1()
            self.fcn_clinical = self._build_fcn_modal2()
            self.fcn_final = self._build_last_fcn()
        else:
            self.fcn_final = self._build_fcn_modal1()

    def _build_conv(self, n_in, n_out):
        if self.conv == "GIN":
            conv_layer = pyg_nn.GINConv(nn.Linear(n_in, n_out))
        elif self.conv == "GAT":
            conv_layer = pyg_nn.GATv2Conv(n_in, n_out)
        elif self.conv == "GCN":
            conv_layer = pyg_nn.GCNConv(n_in, n_out)
        return conv_layer

    def _build_pool(self):
        return pyg_nn.SAGPooling(self.nhid, ratio=self.pool_ratio)

    def _build_fcn(self, dims, final_fcn):
        layers = []
        for i in range(len(dims) - 1):
            if i and self.drop_fcn is not None:
                layers.append(nn.Dropout(self.drop_fcn))
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # if self.use_norm_fcn and (not final_fcn and (i < (len(dims) - 2))):
            #     layers.append(nn.BatchNorm1d(dims[i + 1]))

            if (i < (len(dims) - 2)) or not final_fcn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(eval(f"nn.{self.activation_fcn}()"))

        return nn.Sequential(*layers)

    def _build_fcn_modal1(self):
        dims = [
            self.nhid * 2,
            self.nhid,
            self.nhid // 2,
            self.num_class,
        ]
        final_fcn = True
        if self.is_multimodal:
            dims = [
                self.nhid * 2,
                self.nhid,
            ]
            final_fcn = False
        return self._build_fcn(dims, final_fcn)

    def _build_fcn_modal2(self):
        dims = [
            self.clinical_len,
            self.nhid,
        ]
        return self._build_fcn(dims, False)

    def _build_last_fcn(self):
        dims = [
            self.nhid * 2,
            self.nhid // 2,
            self.num_class,
        ]
        return self._build_fcn(dims, True)

    def forward(self, graph_data, clinical_data=None):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        if self.is_weighted:
            edge_weight = graph_data.edge_weight

        x_total = None
        dct_attention = {}

        for i in range(self.num_layer):
            if self.is_weighted:
                x = self.gnn_layers[f"conv_{self.conv}_{i}"](
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

            else:
                x = self.gnn_layers[f"conv_{self.conv}_{i}"](x=x, edge_index=edge_index)
            x = self.gnn_layers[f"nonlinear_{i}"](x)
            x, edge_index, edge_weight, batch, perm, attention_score = self.gnn_layers[
                f"pool_{i}"
            ](x, edge_index, None, batch)
            x_readout = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            if x_total is None:
                x_total = x_readout
            else:
                x_total += x_readout

            if self.return_attention:
                dct_attention[f"layer{i}"] = [perm, attention_score]

        if self.is_multimodal:

            x_graph = self.fcn_graph(x_total)
            x_clinical = self.fcn_clinical(clinical_data)
            x_combined = torch.cat([x_graph, x_clinical], dim=1)
            logits = self.fcn_final(x_combined)
        else:
            logits = self.fcn_final(x_total)
        return logits, dct_attention
