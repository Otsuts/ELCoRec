import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HeteroGraphConv


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_feat,
        num_heads=8,
        embedding_size=32,
        user_fields=5,
        item_fields=3,
        feature_fields=1,
    ):
        super(GAT, self).__init__()
        self.embedding = nn.Embedding(num_feat, embedding_size, 0)
        # self.layer1 = MultiHeadGATLayer(in_dim, embedding_size, num_heads)
        # self.layer2 = MultiHeadGATLayer(embedding_size * num_heads, out_dim, 1)
        self.layer1 = HeteroGraphConv(
            {
                "belongto": GATConv(in_dim, embedding_size, num_heads),
                "hasinstance": GATConv(in_dim, embedding_size, num_heads),
                "interacted": GATConv(in_dim, embedding_size, num_heads),
                "clickby": GATConv(in_dim, embedding_size, num_heads),
            }
        )

        self.layer2 = HeteroGraphConv(
            {
                "belongto": GATConv(embedding_size * num_heads, out_dim, 1),
                "hasinstance": GATConv(embedding_size * num_heads, out_dim, 1),
                "interacted": GATConv(embedding_size * num_heads, embedding_size, 1),
                "clickby": GATConv(embedding_size * num_heads, embedding_size, 1),
            }
        )
        self.user_emb_map = nn.Linear(user_fields * embedding_size, embedding_size)
        self.item_emb_map = nn.Linear(item_fields * embedding_size, embedding_size)
        self.feat_emb_map = nn.Linear(feature_fields * embedding_size, embedding_size)

    def get_emb(self, user_feat, g):
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)
        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()
        target_user_emb = h["User"].squeeze()
        return user_emb, target_item_emb

    def forward(self, user_feat, g):
        # print(user_feat, g.nodes["Item"].data["id"])
        # assert 0
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)

        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()

        target_user_emb = h["User"].squeeze()

        # logits = torch.matmul(user_emb, target_item_emb.T).sum(dim=0)
        logits = F.cosine_similarity(target_user_emb, target_item_emb)
        return torch.sigmoid(logits)


class GAT_BookCrossing(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_feat,
        num_heads=8,
        embedding_size=32,
        user_fields=5,
        item_fields=3,
        feature_fields=1,
    ):
        super(GAT_BookCrossing, self).__init__()
        self.embedding = nn.Embedding(num_feat, embedding_size, 0)
        # self.layer1 = MultiHeadGATLayer(in_dim, embedding_size, num_heads)
        # self.layer2 = MultiHeadGATLayer(embedding_size * num_heads, out_dim, 1)
        self.layer1 = HeteroGraphConv(
            {
                "belongto": GATConv(in_dim, embedding_size, num_heads),
                "hasinstance": GATConv(in_dim, embedding_size, num_heads),
                "interacted": GATConv(in_dim, embedding_size, num_heads),
                "clickby": GATConv(in_dim, embedding_size, num_heads),
            }
        )

        self.layer2 = HeteroGraphConv(
            {
                "belongto": GATConv(embedding_size * num_heads, out_dim, 1),
                "hasinstance": GATConv(embedding_size * num_heads, out_dim, 1),
                "interacted": GATConv(embedding_size * num_heads, embedding_size, 1),
                "clickby": GATConv(embedding_size * num_heads, embedding_size, 1),
            }
        )
        self.user_emb_map = nn.Linear(user_fields * embedding_size, embedding_size)
        self.item_emb_map = nn.Linear(item_fields * embedding_size, embedding_size)
        self.feat_emb_map = nn.Linear(feature_fields * embedding_size, embedding_size)

    def get_emb(self, user_feat, g):
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    # g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)
        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()
        target_user_emb = h["User"].squeeze()
        return user_emb, target_item_emb

    def forward(self, user_feat, g):
        # print(user_feat, g.nodes["Item"].data["id"])
        # assert 0
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)

        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()

        target_user_emb = h["User"].squeeze()

        # logits = torch.matmul(user_emb, target_item_emb.T).sum(dim=0)
        logits = F.cosine_similarity(target_user_emb, target_item_emb)
        return torch.sigmoid(logits)


class GAT_ml_25m(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=8,
        embedding_size=32,
        user_fields=1,
        item_fields=3,
        feature_fields=1,
    ):
        super(GAT_ml_25m, self).__init__()
        self.embedding = nn.Embedding(281000, embedding_size, 0)
        # self.layer1 = MultiHeadGATLayer(in_dim, embedding_size, num_heads)
        # self.layer2 = MultiHeadGATLayer(embedding_size * num_heads, out_dim, 1)
        self.layer1 = HeteroGraphConv(
            {
                "belongto": GATConv(in_dim, embedding_size, num_heads),
                "hasinstance": GATConv(in_dim, embedding_size, num_heads),
                "interacted": GATConv(in_dim, embedding_size, num_heads),
                "clickby": GATConv(in_dim, embedding_size, num_heads),
            }
        )

        self.layer2 = HeteroGraphConv(
            {
                "belongto": GATConv(embedding_size * num_heads, out_dim, 1),
                "hasinstance": GATConv(embedding_size * num_heads, out_dim, 1),
                "interacted": GATConv(embedding_size * num_heads, embedding_size, 1),
                "clickby": GATConv(embedding_size * num_heads, embedding_size, 1),
            }
        )
        self.user_emb_map = nn.Linear(user_fields * embedding_size, embedding_size)
        self.item_emb_map = nn.Linear(item_fields * embedding_size, embedding_size)
        self.feat_emb_map = nn.Linear(feature_fields * embedding_size, embedding_size)

    def get_emb(self, user_feat, g):
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)
        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()
        return user_emb, target_item_emb

    def forward(self, user_feat, g):
        # print(user_feat, g.nodes["Item"].data["id"])
        # assert 0
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"User": user_emb, "Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)

        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h["User"] = h["User"].view(g.num_nodes("User"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()

        target_user_emb = h["User"].squeeze()

        # logits = torch.matmul(user_emb, target_item_emb.T).sum(dim=0)
        logits = F.cosine_similarity(target_user_emb, target_item_emb)
        return torch.sigmoid(logits)


class GAT_AZ_Toys(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=8,
        embedding_size=32,
        user_fields=1,
        item_fields=3,
        feature_fields=1,
    ):
        super(GAT_AZ_Toys, self).__init__()
        self.embedding = nn.Embedding(380000, embedding_size, 0)
        # self.layer1 = MultiHeadGATLayer(in_dim, embedding_size, num_heads)
        # self.layer2 = MultiHeadGATLayer(embedding_size * num_heads, out_dim, 1)
        self.layer1 = HeteroGraphConv(
            {
                "belongto": GATConv(in_dim, embedding_size, num_heads),
                "hasinstance": GATConv(in_dim, embedding_size, num_heads),
                "interacted": GATConv(in_dim, embedding_size, num_heads),
                "clickby": GATConv(in_dim, embedding_size, num_heads),
            }
        )

        self.layer2 = HeteroGraphConv(
            {
                "belongto": GATConv(embedding_size * num_heads, out_dim, 1),
                "hasinstance": GATConv(embedding_size * num_heads, out_dim, 1),
                # "interacted": GATConv(in_dim, embedding_size, num_heads),
                # "clickby": GATConv(in_dim, embedding_size, num_heads),
            }
        )
        self.user_emb_map = nn.Linear(user_fields * embedding_size, embedding_size)
        self.item_emb_map = nn.Linear(item_fields * embedding_size, embedding_size)
        self.feat_emb_map = nn.Linear(feature_fields * embedding_size, embedding_size)

    def get_emb(self, user_feat, g):
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)
        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h = self.layer2(g, h)

        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()
        return user_emb, target_item_emb

    def forward(self, user_feat, g):
        # print(user_feat, g.nodes["Item"].data["id"])
        # assert 0
        bs = user_feat.shape[0]
        user_emb = self.user_emb_map(self.embedding(user_feat).view(bs, -1))
        item_emb = self.item_emb_map(
            torch.cat(
                (
                    self.embedding(g.nodes["Item"].data["id"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    self.embedding(g.nodes["Item"].data["rating"]).view(
                        g.num_nodes("Item"), -1
                    ),
                    g.nodes["Item"].data["ts_emb"],
                ),
                dim=-1,
            )
        )
        feat_emb = self.feat_emb_map(
            self.embedding(g.nodes["Feature"].data["id"]).view(
                g.num_nodes("Feature"), -1
            )
        )
        x = {"Item": item_emb, "Feature": feat_emb}
        h = self.layer1(g, x)
        h["Feature"] = h["Feature"].view(g.num_nodes("Feature"), -1)
        h["Item"] = h["Item"].view(g.num_nodes("Item"), -1)
        h = self.layer2(g, h)
        target_item_emb = h["Item"][
            g.nodes["Item"].data["is_target"].nonzero().squeeze()
        ].squeeze()

        # logits = torch.matmul(user_emb, target_item_emb.T).sum(dim=0)
        logits = F.cosine_similarity(user_emb, target_item_emb)
        return torch.sigmoid(logits)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": a}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata["z"] = z

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
