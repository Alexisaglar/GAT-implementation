import torch
import torch.nn as nn
from torch.optim import Adam


class GAT(toch.nn.Module):
    def __init__(
        self,
        num_of_layers,
        num_heads_per_layer,
        num_features_per_layer,
        add_skip_connection=True,
        bias=True,
        dropout=0.6,
        log_attention_weights=False,
    ):
        super.__init__()
        if (
            not num_of_layers
            == len(num_heads_per_layer)
            == len(num_features_per_layer) - 1
        ):
            print(f"Enter valid arch params")

        num_heads_per_layer = [
            1
        ] + num_heads_per_layer  # trick to create GAT layers below

        gat_layers = []

        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + i],
                num_of_heads=num_heads_per_layer[i + i],
                concat=(
                    True if i < num_of_layers - 1 else False
                ),  # last layer does mean avg.
                activation=(
                    nn.ELU() if i < num_of_layers - 1 else None
                ),  # last layer outputs raw value
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
            )
            gat_layers.append(layer)
        self.gat_net = nn.Sequential(
            *gat_layers,
        )

        def forward(self, data):
            return self.gat_net(data)


class GATLayer(torch.nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(
        self,
        num_in_features,
        num_out_features,
        num_of_heads,
        concat=True,
        activation=nn.ELU(),
        dropout_prob=0.6,
        add_skip_connection=True,
        bias=True,
        log_attention_weights=False,
    ):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # concat or average attention heads
        self.add_skip_connection = add_skip_connection

        # Trainable weights: linear projection matrix denoted as "W" in paper
        self.linear_proj = nn.Linear(
            num_in_features, num_of_heads * num_out_features, bias=False
        )

        self.scoring_fn_target = nn.Parameter(
            torch.Tensor(num_of_heads * num_out_features)
        )
        self.scoring_fn_source = nn.Parameter(
            torch.Tensor(num_of_heads * num_out_features)
        )

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))

        if add_skip_connection:
            self.skip_proj = nn.Linear(
                num_in_features, num_of_heads * num_out_features, bias=False
            )
        else:
            self.register_parameter("skip_proj", None)

        # end of trainable weights

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None  # for later visualisation purposes

        self.init_params()

    def forward(self, data):
        # Linear projection + regularisation
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        if not edge_index.shape[0] == 2:
            print(f"Expected edge index with shape=(2,E) got {edge_index.shape}")

        # shape = (N, FIN) apply dropout
        in_nodes_features = self.dropout(in_nodes_features)

        # input features are projected into NH independent output features (one for each attention head)
        # shape = (N, FIN) * (FIN , NH*FOUT) converts it inot a tensor shape = (N , NH , FOUT)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(
            -1, self.num_of_heads, self.num_out_features
        )
        nodes_features_proj = self.dropout(nodes_features_proj)

        # Edge attention calculation
        # apply scoring function
        # shape = (N, NH, FOUT) * (1,NH, FOUT) -> (N, NH, 1) -> (N, NH)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=1)

        # lift the scores for source/target nodes based on the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = (
            self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        )
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neigborhood_aware_softmax(
            scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes
        )
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Neighborhood aggregation or message passing
        nodes_features_proj_lifted_weighted = (
            nodes_features_proj_lifted * attentions_per_edge
        )

        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(
            nodes_features_proj_lifted_weighted,
            edge_index,
            in_nodes_features,
            num_of_nodes,
        )

        # Residual/skip connections, concat and bias
        out_nodes_features = self.skip_concat_bias(
            attentions_per_edge, in_nodes_features, out_nodes_features
        )

        return (out_nodes_features, edge_index)

    # helper functions
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):

        # make logits <= 0 so that e^logit <= 1 to improve numerical stability
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # calculate denominator shape = (E, NH)
        neighborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, num_of_nodes
        )

        attentions_per_edge = exp_scores_per_edge / (
            neighborhood_aware_denominator + 1e-16
        )

        # shape = (E, NH) -> (E, NH, 1) for multiplication purposes
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(
        self, exp_scores_per_edge, trg_index, num_of_nodes
    ):
        # E -> (E, NH)
        trg_index_broadcasted = self.explicit_brodcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes

        neighborhood_sums = torch.zeros(
            size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device
        )

        neighborhood_sums.scatter_add_(
            self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge
        )

        # expand again so shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(
        self,
        nodes_features_proj_lifted_weighted,
        edge_index,
        in_nodes_features,
        num_of_nodes,
    ):
        # shape = (N, NH)
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(
            size, dtype=in_nodes_features.dtype, device=in_nodes_features.device
        )

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted
        )

        out_nodes_features.scatter_add_(
            self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted
        )

        return out_nodes_features


    def lift(
        self, scores_source, scores_target, nodes_features_matrix_proj, edge_index
    ):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj = nodes_features_matrix_proj.index_select(
            self.nodes_dim, src_nodes_index
        )

        return scores_source, scores_target, nodes_features_matrix_proj


    def explicit_broadcast(self, first_dimension, second_dimension):
        for _ in range(first_dimension.dim(), second_dimension.dim()):
            first_dimension = first_dimension.unsqueeze(-1)

        return first_dimension.expand_as(second_dimension)


    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(
        self, attention_coefficients, in_nodes_features, out_nodes_features
    ):
        if (
            self.log_attention_weights
        ):  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if (
                out_nodes_features.shape[-1] == in_nodes_features.shape[-1]
            ):  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(
                    -1, self.num_of_heads, self.num_out_features
                )

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(
                -1, self.num_of_heads * self.num_out_features
            )
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return (
            out_nodes_features
            if self.activation is None
            else self.activation(out_nodes_features)
        )
