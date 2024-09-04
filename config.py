cfg = {
    "name": "DANN",
    
    ### DataLoader ###
    # "n_classes": 12,
    # "n_types": 4,
    "n_classes": 4,
    "n_types": 2,

    'label_type_to_labels' : {
        'yes_no': ['yes', 'no'],
        'colors': ['red', 'blue'],
    },
    
    "v2_samples_per_answer": 300,
    "abs_samples_per_answer": 300,
    
    "v2_samples_per_answer_train": 150,
    "abs_samples_per_answer_train": 150,
    
    "v2_samples_per_answer_val": 50,
    "abs_samples_per_answer_val": 50,
    "source_domain": "v2",
    
    ### VLModel ###
    "image_encoder": "facebook/dinov2-base",
    "text_encoder": "bert-base-uncased",
    
    ## Embedder
    "num_attn_heads": 8,
    "fusion_mode": "cat",
    "num_stacked_attn": 1,
    "criss_cross__drop_p": 0.0,
    "post_concat__drop_p": 0.0,
    "embed_attn__add_residual": False,
    "embed_attn__drop_p": 0.0,
    
    ## Label Type
    'use_label_type_classifier': False,
    # 'use_label_type_classifier': True,
    'append_label_type_logits': False,
    
    ## Label Classifier
    "label_classifier__use_bn": True,
    "label_classifier__drop_p": 0.0,
    "label_classifier__repeat_layers": [0, 0],
    
    ## Domain Classifier
    "domain_classifier__use_bn": True,
    "domain_classifier__drop_p": 0.5,
    "domain_classifier__repeat_layers": [2, 2],
    
    ### Objective ###
    "domain_adaptation_method": "domain_adversarial",  # 'naive', 'importance_sampling', 'domain_adversarial'
    
    ### Trainer ###
    "relaxation_period": -1,
    "epochs": 30,
    "batch_size": 150,
    "base_lr": 1e-4,
    "weight_decay": 5e-4,
    
    ### Logging ###
    # "print_logs": False,
    "print_logs": True,
    "show_plot": True,
    "weights_save_root": "./weights/raw",
}