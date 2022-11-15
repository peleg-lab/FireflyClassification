import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from metrics import Metrics
from models.lightning_rnn import LITGRU
from modules.data_module import FireflyDataModule, RealFireflyData


class ModelRunner:
    def __init__(self, hparams):
        self.hparams = hparams
        self.hparams.output_size = self.hparams.n_classes
        self.data_dir = self.hparams.data_dir
        self.dataloaders_dir = self.hparams.dataloaders_dir
        self.metrics = Metrics()
        self.augmentations = {
            'generate': self.hparams.generate
        }

        self.rnn_checkpoint_path = "/ckpts/"
        self.rnn_log_path = "lightning_logs"

        self.top_2 = False
        self.run_cm = False
        self.plot_clusters = False

    def set_stat_bools(self, top_2=False, run_cm=False, plot_clusters=False):
        self.top_2 = top_2
        self.run_cm = run_cm
        self.plot_clusters = plot_clusters

    def run_rnn(self):
        # runs the main training/val loop, etc...
        if self.hparams.resume_from != "none":
            self.set_stat_bools(True, True, True)
            p = os.getcwd() + self.rnn_checkpoint_path + self.hparams.resume_from
            if "sweep" not in self.hparams.resume_from:
                pretrained_model = LITGRU.load_from_checkpoint(p, map_location=lambda storage, loc: storage)

                pretrained_model.eval()
                pretrained_model.freeze()
                data = RealFireflyData(data_dir='data',
                                       augmentations=self.augmentations,
                                       class_limit=pretrained_model.hparams.n_classes,
                                       batch_size=pretrained_model.hparams.batch_size,
                                       num_samples=pretrained_model.hparams.n_samples,
                                       augment_train=pretrained_model.hparams.augment_train,
                                       val_split=pretrained_model.hparams.val_split,
                                       gen_seed=pretrained_model.hparams.gen_seed,
                                       downsample=pretrained_model.hparams.downsample
                                       )
                pretrained_models = [pretrained_model]
                self.metrics.eval_metrics(pretrained_models, [data.test_dataloader()],
                                          top_2=self.top_2,
                                          run_cm=self.run_cm,
                                          plot_clusters=self.plot_clusters)
            else:
                pretrained_models = []
                data = []
                for i,ckpt in enumerate(os.listdir(p)):
                    ckpt_p = p + '/' + ckpt
                    pretrained_model = LITGRU.load_from_checkpoint(ckpt_p, map_location=lambda storage, loc: storage)

                    pretrained_model.eval()
                    pretrained_model.freeze()
                    if self.hparams.load:
                        _data = torch.load(self.dataloaders_dir + "dataloader_{}.pt".format(i))
                    else:
                        rf_data = RealFireflyData(data_dir='data',
                                                  augmentations=self.augmentations,
                                                  class_limit=pretrained_model.hparams.n_classes,
                                                  batch_size=pretrained_model.hparams.batch_size,
                                                  num_samples=pretrained_model.hparams.n_samples,
                                                  augment_train=pretrained_model.hparams.augment_train,
                                                  val_split=pretrained_model.hparams.val_split,
                                                  gen_seed=pretrained_model.hparams.gen_seed,
                                                  downsample=pretrained_model.hparams.downsample
                                                  )
                        _data = rf_data.test_dataloader()
                    pretrained_models.append(pretrained_model)
                    data.append(_data)
                self.metrics.eval_metrics(pretrained_models, data, top_2=self.top_2, run_cm=self.run_cm,
                                          plot_clusters=self.plot_clusters)
        else:
            data = FireflyDataModule(data_dir='data',
                                     augmentations=self.augmentations,
                                     class_limit=self.hparams.n_classes,
                                     batch_size=self.hparams.batch_size,
                                     num_samples=self.hparams.n_samples,
                                     augment_train=self.hparams.augment_train,
                                     val_split=self.hparams.val_split,
                                     gen_seed=self.hparams.gen_seed,
                                     downsample=self.hparams.downsample)

            model = LITGRU(self.hparams)
            logger = TensorBoardLogger(save_dir=os.getcwd(),
                                       version=self.hparams.version,
                                       name=self.rnn_log_path)
            trainer = Trainer(max_epochs=self.hparams.epochs,
                              gradient_clip_val=self.hparams.grad_clip,
                              enable_progress_bar=self.hparams.enable_progress_bar,
                              check_val_every_n_epoch=5,
                              fast_dev_run=self.hparams.test,
                              logger=logger,
                              gpus=self.hparams.gpus,
                              callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=50)]
                              )
            trainer.fit(model, data)
            trainer.test(model=model, dataloaders=data.test_dataloader())
