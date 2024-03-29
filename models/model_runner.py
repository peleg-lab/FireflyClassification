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
        self.augmentations = None
        self.hparams.output_size = self.hparams.n_classes
        self.data_dir = self.hparams.data_dir
        self.dataloaders_dir = self.hparams.dataloaders_dir
        self.data_file = self.hparams.data_file or 'flash_sequence_data.csv'
        self.metrics = Metrics(hparams)

        self.rnn_checkpoint_path = "/ckpts/"
        self.rnn_log_path = "lightning_logs"

    def run_rnn(self, return_stats=False):
        # runs the main training/val loop, etc...
        if self.hparams.resume_from != "none":
            p = os.getcwd() + self.rnn_checkpoint_path + self.hparams.resume_from

            # one model
            if "all" not in self.hparams.resume_from:
                pretrained_model = LITGRU.load_from_checkpoint(p, map_location=lambda storage, loc: storage)

                pretrained_model.eval()
                pretrained_model.freeze()
                data = RealFireflyData(data_dir=self.hparams.data_dir,
                                       augmentations=self.augmentations,
                                       class_limit=pretrained_model.hparams.n_classes,
                                       batch_size=pretrained_model.hparams.batch_size,
                                       val_split=pretrained_model.hparams.val_split,
                                       gen_seed=pretrained_model.hparams.gen_seed,
                                       downsample=pretrained_model.hparams.downsample,
                                       data_path=self.data_file,
                                       flip=False,
                                       dataset_date=self.hparams.dataset_date,
                                       dataset_ratio=self.hparams.dataset_ratio
                                       )
                pretrained_models = [pretrained_model]
                if return_stats:
                    cm = self.metrics.eval_metrics(pretrained_models, [data.test_dataloader()],
                                                   top_2=self.hparams.top_2,
                                                   run_cm=self.hparams.run_cm,
                                                   plot_clusters=self.hparams.plot_clusters,
                                                   write_indices=self.hparams.write_indices,
                                                   return_stats=return_stats,
                                                   track_indices=self.hparams.track_indices)
                    return cm, data.test_dataloader().dataset.indices
                else:
                    self.metrics.eval_metrics(pretrained_models, [data.test_dataloader()],
                                              top_2=self.hparams.top_2,
                                              run_cm=self.hparams.run_cm,
                                              plot_clusters=self.hparams.plot_clusters,
                                              write_indices=self.hparams.write_indices,
                                              return_stats=return_stats,
                                              track_indices=self.hparams.track_indices)

            # ensemble of models
            else:
                pretrained_models = []
                data = []
                for i,ckpt in enumerate(os.listdir(p)):
                    ckpt_p = p + '/' + ckpt
                    print(ckpt)
                    try:
                        pretrained_model = LITGRU.load_from_checkpoint(ckpt_p, map_location=lambda storage, loc: storage)
                    except Exception:
                        print('problem w {}'.format(ckpt))
                        continue
                    pretrained_model.eval()
                    pretrained_model.freeze()
                    if self.hparams.load:
                        _data = torch.load(self.dataloaders_dir + "dataloader_{}.pt".format(i))
                    else:
                        rf_data = RealFireflyData(data_dir=self.hparams.data_dir,
                                                  augmentations=self.augmentations,
                                                  class_limit=pretrained_model.hparams.n_classes,
                                                  batch_size=pretrained_model.hparams.batch_size,
                                                  val_split=pretrained_model.hparams.val_split,
                                                  gen_seed=pretrained_model.hparams.gen_seed,
                                                  downsample=pretrained_model.hparams.downsample,
                                                  data_path=self.data_file,
                                                  flip=self.hparams.flip,
                                                  # use hparams dates to eval on different data slices
                                                  dataset_date=self.hparams.dataset_date,
                                                  dataset_ratio=self.hparams.dataset_ratio
                                                  )
                        _data = rf_data.test_dataloader()
                    pretrained_models.append(pretrained_model)
                    data.append(_data)
                self.metrics.eval_metrics(pretrained_models, data,
                                          top_2=self.hparams.top_2,
                                          run_cm=self.hparams.run_cm,
                                          plot_clusters=self.hparams.plot_clusters,
                                          write_indices=self.hparams.write_indices,
                                          track_indices=self.hparams.track_indices)
        else:
            data = FireflyDataModule(data_dir='data',
                                     augmentations=self.augmentations,
                                     class_limit=self.hparams.n_classes,
                                     batch_size=self.hparams.batch_size,
                                     val_split=self.hparams.val_split,
                                     gen_seed=self.hparams.version,
                                     downsample=self.hparams.downsample,
                                     data_path=self.data_file,
                                     flip=self.hparams.flip,
                                     dataset_date=self.hparams.dataset_date,
                                     dataset_ratio=self.hparams.dataset_ratio
                                     )

            model = LITGRU(self.hparams)
            logger = TensorBoardLogger(save_dir=os.getcwd(),
                                       version=self.hparams.version,
                                       name=self.rnn_log_path)
            trainer = Trainer(max_epochs=self.hparams.epochs,
                              gradient_clip_val=self.hparams.grad_clip,
                              enable_progress_bar=self.hparams.enable_progress_bar,
                              check_val_every_n_epoch=5,
                              log_every_n_steps=25,
                              fast_dev_run=self.hparams.test,
                              logger=logger,
                              gpus=self.hparams.gpus,
                              callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=50)]
                              )
            trainer.fit(model, data)
            trainer.test(model=model, dataloaders=data.test_dataloader())
