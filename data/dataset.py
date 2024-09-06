import json
from collections import defaultdict, deque

import numpy as np
from PIL import Image

import config

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from transformers import AutoImageProcessor, BertTokenizer


class VQADataset(Dataset):
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data

        self.i_processor = AutoImageProcessor.from_pretrained(cfg['image_encoder'])
        self.q_tokenizer = BertTokenizer.from_pretrained(cfg['text_encoder'], clean_up_tokenization_spaces=True)
        
        min_size = 100
        max_size = 200

        self.augmentation = A.Compose(
            [
                A.CoarseDropout(
                    max_holes=1,
                    min_holes=1,
                    
                    max_height=max_size,
                    max_width=max_size,
                    min_height=min_size,
                    min_width=min_size,
                    
                    fill_value=0,
                    mask_fill_value=None,
                    p=1.0
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        idx_tr = torch.tensor(data_item['answer_id'])
        label = F.one_hot(idx_tr, num_classes=self.cfg['n_classes']).float()

        idx_tr = torch.tensor(data_item['answer_type'])
        label_type = F.one_hot(idx_tr, num_classes=self.cfg['n_types']).float()

        image = Image.open(data_item['image_path']).convert('RGB')
        image = np.array(image)
        
        if self.cfg['mask_patches']:
            augmented = self.augmentation(image=image)
            image = augmented['image']
            
            
        i_tokens = self.i_processor(images=image, return_tensors='pt')
        q_tokens = self.q_tokenizer(
            data_item['question'], 
            padding="max_length", 
            max_length=self.cfg['max_q_len'], 
            truncation=True, 
            return_tensors='pt')

        # dirty way to fix dimention issue:
        i_tokens['pixel_values'] = i_tokens['pixel_values'].squeeze(0)

        for key, value in q_tokens.items():
            q_tokens[key] = value.squeeze(0)

        return i_tokens, q_tokens, label, label_type
    
    @classmethod
    def get_labels_in_label_type_tensor(self, label_type_indices):
        # assumes arg: label_type is a tensor of indices in 
        # self.cfg['label_type_to_labels'] a batch
        
        label_types_list = list(config.label_type_to_labels.keys())
        
        batch_size = len(label_type_indices)
        many_hot = torch.zeros(batch_size, config.n_labels, dtype=torch.float32)
        
        starts = torch.tensor([len(label_types_list[i]) for i in label_type_indices])
        ends = starts + torch.tensor([len(label_types_list[i]) for i in label_type_indices]) - 1
        
        for i in range(batch_size):
            many_hot[i, starts[i]:ends[i] + 1] = 1
        
        return many_hot.cuda()


class DA_DataLoader:
    def __init__(self, v2_loader, abs_loader):
        self.len = len(v2_loader) + len(abs_loader)
        return zip(v2_loader, abs_loader)

    def __len__(self):
        return self.len
    
    
def data_processing_v2(
        cfg, 
        vqa_v2, vqa_abs, 
        train_val_split=0.8):

    n_classes = cfg['n_classes']
    v2_samples_per_answer = cfg['v2_samples_per_answer']
    abs_samples_per_answer = cfg['abs_samples_per_answer']

    with open(vqa_v2['questions_path'], 'r') as file:
        v2_questions = json.load(file)

    with open(vqa_v2['annotations_path'], 'r') as file:
        v2_annotations = json.load(file)

    with open(vqa_abs['questions_path'], 'r') as file:
        abs_questions = json.load(file)

    with open(vqa_abs['annotations_path'], 'r') as file:
        abs_annotations = json.load(file)

    v2_questions = v2_questions['questions']
    v2_annotations = v2_annotations['annotations']

    abs_questions = abs_questions['questions']
    abs_annotations = abs_annotations['annotations']

    max_q_len = 0
    v2_question_id_map = {}  # question_id : question_text
    abs_question_id_map = {}

    for question in v2_questions:
        v2_question_id_map[question['question_id']] = question['question']
        max_q_len = max(max_q_len, len(question['question'].split()))
    
    for question in abs_questions:
        abs_question_id_map[question['question_id']] = question['question']
        max_q_len = max(max_q_len, len(question['question'].split()))

    cfg['max_q_len'] = max_q_len

    # Collecting answers
    v2_answers = defaultdict(list)  # answer : list of annotation indices
    abs_answers = defaultdict(list)

    for annotation in v2_annotations:
        v2_answers[annotation['multiple_choice_answer']] += [(annotation['image_id'], annotation['question_id'])]

    for annotation in abs_annotations:
        abs_answers[annotation['multiple_choice_answer']] += [(annotation['image_id'], annotation['question_id'])]

    # Filtering answers
    filtered_v2_answers = []
    filtered_v2_num_samples = []

    for answer, sample in v2_answers.items():
        if len(sample) >= v2_samples_per_answer:
            filtered_v2_answers.append(answer)
            filtered_v2_num_samples.append(len(sample))

    filtered_abs_answers = []
    filtered_abs_num_samples = []

    for answer, sample in abs_answers.items():
        if len(sample) >= abs_samples_per_answer:
            filtered_abs_answers.append(answer)
            filtered_abs_num_samples.append(len(sample))

    # filtered_answers = []
    # filtered_num_samples = []
    # filtered_v2_samples = []
    # filtered_abs_samples = []

    # for answer in filtered_abs_answers:
    #     if answer not in filtered_v2_answers:
    #         continue

    #     filtered_answers.append(answer)
    #     filtered_num_samples.append(len(v2_answers[answer]) + len(abs_answers[answer]))
    #     filtered_v2_samples.append(v2_answers[answer][:v2_samples_per_answer])
    #     filtered_abs_samples.append(abs_answers[answer][:abs_samples_per_answer])

    # samples_lowerbound = min(cfg['v2_samples_per_answer'], cfg['abs_samples_per_answer'])

    # if samples_lowerbound >= 300:  # 16 labels
    #     label_types = {
    #         'yes_no': ['yes', 'no'],
    #         'numbers': ['0', '1', '2', '3', '4', '5'],
    #         'colors': ['brown', 'red', 'yellow', 'blue', 'green', 'white'],
    #         'living_things': ['dog', 'cat']
    #     }
    # elif samples_lowerbound >= 250: # 18 labels
    #     label_types = {
    #         'yes_no': ['yes', 'no'],
    #         'numbers': ['0', '1', '2', '3', '4', '5'],
    #         'colors': ['brown', 'red', 'yellow', 'blue', 'green', 'white', 'gray', 'black'],
    #         'living_things': ['dog', 'cat']
    #     }
    # elif samples_lowerbound >= 200: # 25 labels
    #     label_types = {
    #         'yes_no': ['yes', 'no'],
    #         'numbers': ['0', '1', '2', '3', '4', '5'],
    #         'colors': ['brown', 'red', 'yellow', 'blue', 'green', 'white', 'gray', 'black'],
    #         'living_things': ['dog', 'cat', 'women'],
    #         'non_living_things': ['food', 'table'],
    #         'others': ['soccer', 'left', 'right', 'nothing'],
    #     }
    # elif samples_lowerbound >= 150: # 32 labels
    #     label_types = {
    #         'yes_no': ['yes', 'no'],
    #         'numbers': ['0', '1', '2', '3', '4', '5'],
    #         'colors': ['brown', 'red', 'yellow', 'blue', 'green', 'white', 'gray', 'black', 'orange'],
    #         'living_things': ['dog', 'cat', 'women', 'man'],
    #         'non_living_things': ['food', 'table', 'apple', 'wine'],
    #         'others': ['soccer', 'beach', 'left', 'right', 'nothing'],
    #     }
    
    
    label_type_to_labels = cfg['label_type_to_labels']

    label_types = list(label_type_to_labels.keys())

    # labels and label_type
    all_labels = []
    for key in label_type_to_labels:
        all_labels.extend(label_type_to_labels[key])

    label_to_type = {}
    for key, labels in label_type_to_labels.items():
        for label in labels:
            label_to_type[label] = key
    
    if cfg['print_logs']:
        print(f'Labels: {all_labels}')

    filtered_answers = []
    filtered_num_samples = []
    filtered_v2_samples = []
    filtered_abs_samples = []

    for answer in all_labels:
        filtered_answers.append(answer)
        filtered_num_samples.append(len(v2_answers[answer]) + len(abs_answers[answer]))
        
        if cfg['min_samples_mode']:
            filtered_v2_samples.append(v2_answers[answer])
            filtered_abs_samples.append(abs_answers[answer])
        else:
            filtered_v2_samples.append(v2_answers[answer][:v2_samples_per_answer])
            filtered_abs_samples.append(abs_answers[answer][:abs_samples_per_answer])

    if cfg['print_logs']:
        print(f'Number of Common Labels = {len(filtered_answers)} | n_classes = {n_classes}')

    if cfg['print_logs'] and len(filtered_answers) < n_classes:
        print(f'Updating n_classes from {n_classes} to {len(filtered_answers)}')
        cfg['n_classes'] = n_classes = len(filtered_answers)


    v2_train_data = []
    v2_val_data = []
    for answer, samples in zip(filtered_answers, filtered_v2_samples):
        for i, sample in enumerate(samples):
            image_id, question_id = sample
            image_id = str(image_id).zfill(6)
            ext = '.jpg'

            image_path = vqa_v2['image_root'] + str(image_id) + ext
            question = v2_question_id_map[question_id]

            if i < cfg['v2_samples_per_answer_val']:
                data = v2_val_data
            elif not cfg['min_samples_mode']:
                if i < cfg['v2_samples_per_answer_val'] + cfg['v2_samples_per_answer_train']:
                    data = v2_train_data
                else:
                    break
            else:
                data = v2_train_data

            data.append({
                'image_path': image_path,
                'question': question,
                'answer_id': filtered_answers.index(answer),
                'answer_type': label_types.index(label_to_type[answer]),
            })

    if cfg['print_logs']:
        print(f'V2: \tTrain size = {len(v2_train_data)} \
            | Val size = {len(v2_val_data)} | Total = {len(v2_train_data) + len(v2_val_data)}')

    abs_train_data = []
    abs_val_data = []
    for answer, samples in zip(filtered_answers[:n_classes], filtered_abs_samples[:n_classes]):
        for i, sample in enumerate(samples):
            image_id, question_id = sample
            image_id = str(image_id).zfill(5)
            ext = '.png'

            image_path = vqa_abs['image_root'] + str(image_id) + ext
            question = abs_question_id_map[question_id]

            if i < cfg['abs_samples_per_answer_val']:
                data = abs_val_data
            elif not cfg['min_samples_mode']:
                if i < cfg['abs_samples_per_answer_val'] + cfg['abs_samples_per_answer_train']:
                    data = abs_train_data
                else:
                    break
            else:
                data = abs_train_data

            data.append({
                'image_path': image_path,
                'question': question,
                'answer_id': filtered_answers.index(answer),
                'answer_type': label_types.index(label_to_type[answer]),
            })

    if cfg['print_logs']:
        print(f'Abs: \tTrain size = {len(abs_train_data)} \
            | Val size = {len(abs_val_data)} | Total = {len(abs_train_data) + len(abs_val_data)}')

        print('-' * 20)
        
    return (v2_train_data, v2_val_data), (abs_train_data, abs_val_data)
