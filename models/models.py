import torch
import torch.nn as nn
from models.VLModel import VLModel
from models.functions import ReverseLayerF
from data.dataset import VQADataset

class DANN_VLModel(VLModel):
    def __init__(self, cfg, embed_dim=768, return_embeddings=False):
        super().__init__(cfg, embed_dim, return_embeddings)

        self.grad_reversal = cfg['domain_adaptation_method'] == 'domain_adversarial'
        
        self.num_types = cfg['n_types']
        self.use_label_type_classifier = cfg['use_label_type_classifier']
        self.append_label_type_logits = cfg['append_label_type_logits']
        self.give_location_of_labels_in_label_type = cfg['give_location_of_labels_in_label_type']

        self.ffn_domain_classifier = self.init_ffn(
            cfg["domain_classifier__use_bn"],
            cfg["domain_classifier__drop_p"],
            self.embed_dim,
            output_dim=1,
            repeat_layers=cfg['domain_classifier__repeat_layers'],
        )

        self.init_label_classifier(
            cfg["label_classifier__use_bn"],
            cfg["label_classifier__drop_p"],
            self.embed_dim,
            cfg['n_classes'],
            cfg['n_types'],
            repeat_layers=cfg['label_classifier__repeat_layers']
        )
        
    def init_label_classifier(
            self, 
            use_bn, drop_p, embed_dim, 
            num_labels, num_label_types,
            repeat_layers=[0,0]):
        if self.fusion_mode == "cat":
            embed_dim *= 2

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
        self.ffn_early_layers = nn.Sequential()
        self.ffn_early_layers.extend(get_layer(embed_dim))
        for _ in range(repeat_layers[0]):
            self.ffn_early_layers.extend(get_layer(embed_dim))

        # Middle Layers
        self.ffn_middle_layers = nn.Sequential()
        self.ffn_middle_layers.extend(get_layer(embed_dim, embed_dim // 2))
        for _ in range(repeat_layers[1]):
            self.ffn_middle_layers.extend(get_layer(embed_dim // 2, embed_dim // 2))

        # Label type classifier
        self.ffn_label_type_classifier = nn.Sequential()
        if self.fusion_mode == "cat":
            self.ffn_label_type_classifier.extend(get_layer(embed_dim // 2, embed_dim // 4))
            self.ffn_label_type_classifier.append(nn.Linear(embed_dim // 4, num_label_types))
        else:
            self.ffn_label_type_classifier.append(nn.Linear(embed_dim // 2, num_label_types))

        # append label type info into label_classifier's input
        self.ffn_append_label_type = nn.Linear(
            in_features = self.embed_dim + num_label_types,
            out_features = self.embed_dim
        )
        
        self.append_and_shrink_labels_in_predicted_label_type = nn.Linear(
            in_features = self.embed_dim + num_labels,
            out_features = self.embed_dim
        )        
        
        # Label classifier
        self.ffn_label_classifier = nn.Sequential()
        if self.fusion_mode == "cat":
            self.ffn_label_classifier.extend(get_layer(embed_dim // 2, embed_dim // 4))
            self.ffn_label_classifier.append(nn.Linear(embed_dim // 4, num_labels))
        else:
            self.ffn_label_classifier.append(nn.Linear(embed_dim // 2, num_labels))


    def label_classifier_forward(self, x):
        x = self.ffn_early_layers(x)
        x = self.ffn_middle_layers(x)

        if self.use_label_type_classifier:
            label_type_logits = self.ffn_label_type_classifier(x)
            
            if self.append_label_type_logits:
                x = self.ffn_append_label_type(torch.cat((x, label_type_logits), dim=1))
                
            if self.give_location_of_labels_in_label_type:
                label_type_indices = torch.argmax(label_type_logits, dim=1)
                labels_in_label_type = VQADataset.get_labels_in_label_type_tensor(label_type_indices)
                
                x = self.append_and_shrink_labels_in_predicted_label_type(
                    torch.cat((x, labels_in_label_type), dim=1)
                )
                
        else:
            label_type_logits = torch.zeros(x.shape[0], self.num_types).cuda()
            
        label_logits = self.ffn_label_classifier(x)

        return label_logits, label_type_logits
        

    def forward(self, i_tokens, q_tokens, alpha=None):
        embedding = self.get_embedding(i_tokens, q_tokens)
        embedding = embedding.squeeze(1)

        domain_features = embedding if not alpha or not self.grad_reversal else ReverseLayerF.apply(embedding, alpha)
        domain_logits = self.ffn_domain_classifier(domain_features)

        label_logits, label_type_logits = self.label_classifier_forward(embedding)

        if self.return_embeddings:
            return label_logits, domain_logits, label_type_logits, embedding.cpu().numpy()
            
        return label_logits, domain_logits, label_type_logits
