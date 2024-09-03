import comet_ml
from comet_ml import Experiment

import torch
from trainers.DATrainer import DATrainer

import optuna
from optuna.trial import TrialState



class OptunaTrainer:
    def __init__(self, n_trials, n_jobs):
        self.vqa_v2 = {
            "type": "v2",
            "image_root": "data/vqa-v2/val2014/val2014/COCO_val2014_000000",
            "questions_path": "data/vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json",
            "annotations_path": "data/vqa-v2/v2_mscoco_val2014_annotations.json",
        }

        self.vqa_abs = {
            "type": "abs",
            "image_root": "data/vqa-abstract/img_train/abstract_v002_train2015_0000000",
            "questions_path": "data/vqa-abstract/questions_train/OpenEnded_abstract_v002_train2015_questions.json",
            "annotations_path": "data/vqa-abstract/annotations_train/abstract_v002_train2015_annotations.json",
        }

        self.optuna_cfg = {
            'n_trials': n_trials,
            'n_jobs': n_jobs,
            'comet_logging': True
            # 'comet_logging': False
        }


        self.default_cfg = {
            'name': 'DANN',

            ### DataLoader ###
            'n_classes': 10,
            'v2_samples_per_answer': 300,
            'abs_samples_per_answer': 300,

            'v2_samples_per_answer_train': 100,
            'abs_samples_per_answer_train': 100,

            'v2_samples_per_answer_val': 50,
            'abs_samples_per_answer_val': 50,
            
            'source_domain': 'abs',
            
            
            ### VLModel ###
            'image_encoder': 'facebook/dinov2-base',
            'text_encoder': 'bert-base-uncased',
            
            ## Embedder
            'num_attn_heads': 8,
            'fusion_mode': 'cat',
            'num_stacked_attn': 1, 
            
            'criss_cross__drop_p': 0.0,
            'post_concat__drop_p': 0.0, 
            'embed_attn__add_residual': False,
            'embed_attn__drop_p': 0.0,

            ## Label Classifier
            'label_classifier__use_bn': False,
            'label_classifier__drop_p': 0.0,
            'label_classifier__repeat_layers': [2, 2], 

            ## Domain Classifier
            'domain_classifier__use_bn': True,
            'domain_classifier__drop_p': 0.5,
            'domain_classifier__repeat_layers': [2, 2], 


            ### Objective ###
            'domain_adaptation_method': 'domain_adversarial',  # 'naive', 'importance_sampling', 'domain_adversarial'


            ### Trainer ###
            'relaxation_period': -1,

            'epochs': 30,
            'batch_size': 150,
            'base_lr': 0.001,
            'weight_decay': 0,
            
            ### Logging ###
            'print_logs': False,
            'show_plot': True,
            
            'weights_save_root': './weights/raw'
        }

    def init_trainer(self, trial):
        self.sampled_cfg = {
            # # DataLoader ###
            'v2_samples_per_answer_train': 50 * trial.suggest_int('v2_samples_per_answer', low=1, high=5),
            'abs_samples_per_answer_train': 50 * trial.suggest_int('abs_samples_per_answer', low=1, high=5),
            'source_domain': trial.suggest_categorical('source_domain', ['v2', 'abs']),

            # ### VLModel ###
            # ## Embedder
            # 'num_attn_heads': trial.suggest_categorical('num_attn_heads', [2, 4, 8, 16]),
            # 'fusion_mode': trial.suggest_categorical('fusion_mode', ['cat', 'cat_v2', 'mult', 'add']), 
            # 'num_stacked_attn': trial.suggest_categorical('num_stacked_attn', [1, 2, 4, 8]),
            
            # 'criss_cross__drop_p': trial.suggest_categorical('criss_cross__drop_p', [0.0, 0.25, 0.5]),
            # 'post_concat__drop_p': trial.suggest_categorical('post_concat_drop_p', [0.0, 0.25, 0.5]),
            # 'embed_attn__add_residual': trial.suggest_categorical('embed_attn__add_residual', [False, True]),
            # 'embed_attn__drop_p': trial.suggest_categorical('embed_attn__drop_p', [0.0, 0.25, 0.5]),

            # ## Label Classifier
            'label_classifier__use_bn': trial.suggest_categorical('label_classifier__use_bn', [False, True]),
            'label_classifier__drop_p': trial.suggest_categorical('label_classifier__drop_p', [0.0, 0.25, 0.5]),
            'label_classifier__repeat_layers': trial.suggest_categorical(
                'label_classifier__repeat_layers', [0, 1, 2]
            ),

            # ## Domain Classifier
            'domain_classifier__use_bn': trial.suggest_categorical('domain_classifier__use_bn', [False, True]),
            'domain_classifier__drop_p': trial.suggest_categorical('domain_classifier__drop_p', [0.0, 0.25, 0.5]),
            'domain_classifier__repeat_layers': trial.suggest_categorical(
                'domain_classifier__repeat_layers', [0, 1, 2]
            ),

            ### Trainer ###
            'base_lr': trial.suggest_float('base_lr', low=1e-5, high=1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', low=1e-6, high=1e-4),
        }


        self.cfg = self.default_cfg
        for k, v in self.sampled_cfg.items():
            self.cfg[k] = v

        for k in ['label_classifier__repeat_layers', 'domain_classifier__repeat_layers']:
            l = self.cfg[k]
            self.cfg[k] = [l, l]

        trainer = DATrainer(self.cfg, self.vqa_v2, self.vqa_abs)

        trial.params['n_classes'] = self.cfg['n_classes']  # Coz it could update in data_processing_v2

        return trainer

    def objective(self, trial):
        trainer = self.init_trainer(trial)

        if self.optuna_cfg['comet_logging']:
            config = trainer.cfg
            experiment = comet_ml.OfflineExperiment(
                api_key="vGoIrJjMbmcYScW8fWPCX5hU5",
                project_name="FS-VQA",
                workspace="tanmay4269"
            )
            experiment.set_name(config['title'])
            experiment.log_parameters(config)
            

        eval_loss = trainer.train(show_plot=True, optuna_trial=trial, comet_expt=experiment)

        if self.optuna_cfg['comet_logging']:
            experiment.end()
        
        return eval_loss
    
    def run(self):
        study = optuna.create_study(direction="minimize")
        print(f"Sampler is {study.sampler.__class__.__name__}")

        study.optimize(self.objective, n_trials=self.optuna_cfg['n_trials'], n_jobs=self.optuna_cfg['n_jobs'])

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))