import torch
from trainer import DA_Trainer

import optuna
from optuna.trial import TrialState


class OptunaTrainer:
    def __init__(self):
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
            'n_trials': 10,
        }

        self.default_cfg = {
            'name': 'NaiveSampling',

            ### DataLoader ###
            'n_classes': 10,
            'v2_samples_per_answer': 300,
            'abs_samples_per_answer': 150,
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

            ## Domain Classifier
            'domain_classifier__use_bn': False,
            'domain_classifier__drop_p': 0.0,
            'domain_classifier__repeat_layers': [0, 0], 


            ### Objective ###
            'domain_adaptation_method': 'naive',  # 'naive', 'importance_sampling', 'domain_adversarial'


            ### Trainer ###
            'epochs': 10,
            'batch_size': 100,
            'base_lr': 0.001,
            'weight_decay': 0,

            
            ### Logging ###
            'show_plot': True,
            'weights_save_root': './weights/raw'
            # comet_logging
        }

    def init_trainer(self, trial):
        self.sampled_cfg = {
            # # DataLoader ###
            # 'n_classes': trial.suggest_int('n_classes', low=1, high=25),
            # 'v2_samples_per_answer': 50 * trial.suggest_int('v2_samples_per_answer', low=1, high=6),
            # 'abs_samples_per_answer': 50 * trial.suggest_int('abs_samples_per_answer', low=1, high=6),
            # 'source_domain': trial.suggest_categorical('source_domain', ['v2', 'abs']),

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
            # 'label_classifier__use_bn': trial.suggest_categorical('label_classifier__use_bn', [False, True]),
            # 'label_classifier__drop_p': trial.suggest_categorical('label_classifier__drop_p', [0.0, 0.25, 0.5]),

            # ## Domain Classifier
            # 'domain_classifier__use_bn': trial.suggest_categorical('domain_classifier__use_bn', [False, True]),
            # 'domain_classifier__drop_p': trial.suggest_categorical('domain_classifier__drop_p', [0.0, 0.25, 0.5]),
            

            ### Trainer ###
            'base_lr': trial.suggest_float('base_lr', low=1e-5, high=1e-3, log=True),
            # 'weight_decay': trial.suggest_float('weight_decay', low=1e-6, high=1e-4),
        }

        self.cfg = self.default_cfg
        for k, v in self.sampled_cfg.items():
            self.cfg[k] = v

        trainer = DA_Trainer(self.cfg, self.vqa_v2, self.vqa_abs)

        trial.params['n_classes'] = self.cfg['n_classes']  # Coz it could update in data_processing_v2

        return trainer

    def objective(self, trial):
        trainer = self.init_trainer(trial)
        
        min_eval_loss = float("inf")
        
        for epoch in range(trainer.num_epochs):
            metrics = trainer.step(epoch)
            metrics['epoch'] = epoch
            eval_loss = metrics['eval_loss']
            
            # Plotting
            if self.cfg['show_plot'] and epoch > 0 and epoch % 10 == 0:
                self.plot(epoch + 1)
            
            # Saving weights
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                torch.save(trainer.model.state_dict(), self.cfg["weights_save_path"])

            # Logging
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)

            # Optuna stuff
            trial.report(eval_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return eval_loss
    
    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.optuna_cfg['n_trials'])

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