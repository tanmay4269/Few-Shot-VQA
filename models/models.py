import logging
from copy import deepcopy

import torch
import torch.nn as nn
from models.functions import ReverseLayerF

from transformers import AutoModel, BertModel


class VLModel(nn.Module):
    def __init__(self, cfg, embed_dim=768, return_embeddings=False):
        super(VLModel, self).__init__()

        self.return_embeddings = return_embeddings
        self.fusion_mode = cfg["fusion_mode"]
        self.embed_dim = embed_dim

        # should suppress params warning
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "transformers" in logger.name.lower():
                logger.setLevel(logging.ERROR)

        self.i_encoder = AutoModel.from_pretrained(cfg["image_encoder"])
        self.q_encoder = BertModel.from_pretrained(cfg["text_encoder"])

        self.num_stacked_attn = cfg["num_stacked_attn"]
        output_dim = cfg["n_classes"]

        self.attn_heads = cfg["num_attn_heads"]
        attn_embed_dim = 768

        """< Criss-Cross Fusion >"""
        MHA_module = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    attn_embed_dim,
                    self.attn_heads,
                    dropout=cfg["criss_cross__drop_p"],
                    batch_first=True,
                )
                for _ in range(self.num_stacked_attn)
            ]
        )

        for attr_name in [
            "attn_q2i",
            "attn_i2q",
            "attn_i",
            "attn_q",
        ]:
            setattr(self, attr_name, deepcopy(MHA_module))

        """</ Criss-Cross Fusion >"""

        self.post_concat__dropout = nn.Dropout(p=cfg["post_concat__drop_p"])
        self.embed_attn__add_residual = cfg["embed_attn__add_residual"]
        self.embed_attn__dropout = nn.Dropout(p=cfg["embed_attn__drop_p"])

        if self.fusion_mode == "cat":
            attn_embed_dim *= 2

        self.attn_e = nn.ModuleList(
            [
                nn.MultiheadAttention(attn_embed_dim, self.attn_heads, batch_first=True)
                for _ in range(self.num_stacked_attn)
            ]
        )

        self.label_classifier = self.init_ffn(
            cfg["label_classifier__use_bn"],
            cfg["label_classifier__drop_p"],
            self.embed_dim,
            output_dim,
            repeat_layers=cfg['label_classifier__repeat_layers']
        )

    def init_ffn(self, use_bn, drop_p, embed_dim, output_dim, repeat_layers=[0,0]):
        if self.fusion_mode == "cat":
            embed_dim *= 2

        ffn = nn.Sequential()

        def get_layer(in_, out_=None):
            if out_ is None:
                out_ = in_

            layer = [
                nn.Linear(in_, out_),
                nn.ReLU(),
            ]

            if use_bn:
                layer.append(nn.BatchNorm1d(out_))

            layer.append(nn.Dropout(p=drop_p))

            return layer

        # Early Layers
        ffn.extend(get_layer(embed_dim))
        for _ in range(repeat_layers[0]):
            ffn.extend(get_layer(embed_dim))

        # Middle Layers
        ffn.extend(get_layer(embed_dim, embed_dim // 2))
        for _ in range(repeat_layers[1]):
            ffn.extend(get_layer(embed_dim // 2, embed_dim // 2))

        # Final Layers
        if self.fusion_mode == "cat":
            ffn.extend(get_layer(embed_dim // 2, embed_dim // 4))
            ffn.append(nn.Linear(embed_dim // 4, output_dim))
        else:
            ffn.append(nn.Linear(embed_dim // 2, output_dim))

        return ffn

    def cross_attn(self, i, q, attn_q2i, attn_i2q):
        i_attended, _ = attn_q2i(i, q, q)
        q_attended, _ = attn_i2q(q, i, i)

        return i_attended, q_attended

    def self_attn_1(self, embedding, attn_e, use_prev=False):
        prev_embedding = embedding
        if not use_prev:
            prev_embedding = torch.zeros(
                (embedding.shape[0], 1, embedding.shape[-1])
            ).cuda()

        embedding, _ = attn_e(prev_embedding, embedding, embedding)
        return embedding

    def self_attn_2(self, i, q, attn_i, attn_q, use_prev=False):
        prev_i = i
        if not use_prev:
            prev_i = torch.zeros((i.shape[0], 1, self.embed_dim)).cuda()
        i_embeddings, _ = attn_i(prev_i, i, i)

        prev_q = q
        if not use_prev:
            prev_q = torch.zeros((q.shape[0], 1, self.embed_dim)).cuda()
        q_embeddings, _ = attn_q(prev_q, q, q)

        return i_embeddings, q_embeddings

    def get_embedding(self, i_tokens, q_tokens):
        with torch.no_grad():
            i_embeddings = self.i_encoder(**i_tokens).last_hidden_state
            q_embeddings = self.q_encoder(**q_tokens).last_hidden_state

        for stack in range(self.num_stacked_attn):
            # Cross Attention
            i_attended, q_attended = self.cross_attn(
                i_embeddings, q_embeddings, self.attn_q2i[stack], self.attn_i2q[stack]
            )

            # Self Attention
            i_embeddings, q_embeddings = self.self_attn_2(
                i_attended,
                q_attended,
                self.attn_i[stack],
                self.attn_q[stack],
                use_prev=stack > 0,
            )

        # Concat
        if self.fusion_mode == "cat":
            embedding = torch.cat(
                (i_embeddings, q_embeddings), dim=-1
            )  # along channels
        elif self.fusion_mode == "cat_v2":
            i_embeddings = i_embeddings.reshape(
                i_embeddings.shape[0], -1, self.embed_dim // 2
            )
            q_embeddings = q_embeddings.reshape(
                q_embeddings.shape[0], -1, self.embed_dim // 2
            )
            embedding = torch.cat((i_embeddings, q_embeddings), dim=-1)
        elif self.fusion_mode == "mult":
            embedding = i_embeddings * q_embeddings
        elif self.fusion_mode == "add":
            embedding = i_embeddings + q_embeddings

        embedding = self.post_concat__dropout(embedding)

        # Self Attetntion
        for stack in range(self.num_stacked_attn):
            new_embedding = self.self_attn_1(
                embedding, self.attn_e[stack], use_prev=stack > 0
            )

            if self.embed_attn__add_residual:
                embedding = embedding + new_embedding
            else:
                embedding = new_embedding

            embedding = self.embed_attn__dropout(embedding)

        return embedding

    def forward(self, i_tokens, q_tokens):
        embedding = self.get_embedding(i_tokens, q_tokens)
        embedding = embedding.squeeze(1)

        logits = self.label_classifier(embedding)

        if self.return_embeddings:
            return logits, embedding.cpu().numpy()

        return logits


class VLModel_IS(VLModel):
    def __init__(self, cfg, embed_dim=768, return_embeddings=False):
        super().__init__(cfg, embed_dim, return_embeddings)

        self.grad_reversal = cfg['domain_adaptation_method'] == 'domain_adversarial'

        self.domain_classifier = self.init_ffn(
            cfg["domain_classifier__use_bn"],
            cfg["domain_classifier__drop_p"],
            self.embed_dim,
            output_dim=1,
            repeat_layers=cfg['domain_classifier__repeat_layers'],
        )

    def forward(self, i_tokens, q_tokens, alpha=None):
        embedding = self.get_embedding(i_tokens, q_tokens)
        embedding = embedding.squeeze(1)

        label_logits = self.label_classifier(embedding)

        domain_features = embedding if not alpha or not self.grad_reversal else ReverseLayerF.apply(embedding, alpha)
        domain_logits = self.domain_classifier(domain_features)

        if self.return_embeddings:
            return label_logits, domain_logits, embedding.cpu().numpy()

        return label_logits, domain_logits
