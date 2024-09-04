label_type_50 = {
    'colors':        ['beige', 'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'red', 'red and white', 'tan', 'white', 'yellow'],
    'objects':       ['bench', 'chair', 'couch', 'floor', 'table', 'tv', 'blanket', 'book', 'frisbee', 'skateboard', 'soccer'],
    'living-things': ['baby', 'bird', 'boy', 'cat', 'dog', 'fish', 'flowers', 'girl', 'man', 'mouse', 'tree', 'woman'],
    'actions':       ['eating', 'playing', 'sitting', 'sleeping', 'standing', 'walking'],
    'locations':     ['park', 'sidewalk', 'living room', 'on table', 'sky'],
    'foods':         ['apple', 'pizza', 'sandwich', 'wine', 'food'],
    'numbers':       ['0', '1', '2', '3', '4', '5', '6'],
    'responses':     ['no', 'no one', 'nothing', 'yes'],
    'directions':    ['left', 'right'],
}

label_type_20 = {
    'colors':        ['beige', 'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'red', 'red and white', 'tan', 'white', 'yellow'],
    'objects':       ['bench', 'chair', 'couch', 'floor', 'table', 'tv', 'blanket', 'book', 'frisbee', 'skateboard', 'soccer', 'bike', 'car', 'bottle', 'cup', 'plate'],
    'living-things': ['baby', 'bird', 'boy', 'cat', 'dog', 'fish', 'flowers', 'girl', 'man', 'mouse', 'tree', 'woman', 'duck', 'eagle', 'mushrooms'],
    'actions':       ['eating', 'playing', 'sitting', 'sleeping', 'standing', 'walking', 'running', 'jumping', 'drinking'],
    'locations':     ['park', 'sidewalk', 'living room', 'on table', 'sky', 'on floor', 'on grass'],
    'foods':         ['apple', 'pizza', 'sandwich', 'wine', 'food', 'cheese', 'hot dog', 'bread', 'steak'],
    'numbers':       ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    'responses':     ['no', 'no one', 'nothing', 'yes'],
    'directions':    ['left', 'right'],
    'times':         ['morning', 'afternoon', 'evening', 'night'],
    'weather':       ['sunny', 'clouds', 'sunset', 'rainy'],
    'patterns':      ['checkered', 'floral'],
}


samples_per_answer = 50
label_type_to_labels = label_type_50
n_labels = 0
for k in label_type_to_labels:
    n_labels += len(label_type_to_labels[k])

cfg = {
    "name": "DANN",
    
    ### DataLoader ###
    "n_classes": n_labels,
    "n_types": len(label_type_to_labels),
    
    'label_type_to_labels': label_type_to_labels,
    
    "v2_samples_per_answer": samples_per_answer,
    "abs_samples_per_answer": samples_per_answer,
    
    "v2_samples_per_answer_train": samples_per_answer // 2,
    "abs_samples_per_answer_train": samples_per_answer // 2,
    
    "v2_samples_per_answer_val": samples_per_answer // 2,
    "abs_samples_per_answer_val": samples_per_answer // 2,
    
    "source_domain": "v2",
    
    ## Allow Min Samples
    "min_samples_mode": True,  # will use atleast samples_per_answer per label
    
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
    'use_label_type_classifier': True,
    # 'use_label_type_classifier': False,
    'append_label_type_logits': False,
    'give_location_of_labels_in_label_type': True,
    
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
    "train_modes": ['DANN', 'label_type'],  # ['DANN', 'label_type', 'label']
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