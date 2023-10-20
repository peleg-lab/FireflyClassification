import torch
import numpy
import pickle
import datetime
import pytorch_lightning as pl
from os.path import exists

from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from sklearn.metrics import confusion_matrix, f1_score

from data.names import names

from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence


class LITGRU(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.vocab_size = len([0,1,'<pad>'])
        self.embedding_dim = self.hparams['embedding_dim']
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hparams['hidden_size'],
                          num_layers=self.hparams['n_layers'],
                          dropout=self.hparams['dropout'],
                          batch_first=True,
                          bias=True).to(self.device)
        # layers
        self.fc = nn.Linear(self.hparams['hidden_size'], self.hparams['n_classes'])
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.previous_hidden = self.init_hidden()

        # criterion
        self.criterion = F.cross_entropy

        # optimization
        self.automatic_optimization = True
        self.USE_LR_SCHEDULER = True
        self.LOSS_AT_FINAL_STEP_ONLY = False

        # metrics
        self.metrics = {}

        # date string
        self.dstr = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

    def init_hidden(self):
        return torch.zeros(self.hparams['n_layers'],
                           self.hparams['batch_size'],
                           self.hparams['hidden_size'],
                           requires_grad=True).to(self.device)

    def forward(self, x):
        outs, hidden_state = self.gru(x)
        outputs = self.fc(self.relu(outs))

        return outputs, hidden_state

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        seq_lens = []
        for _x_ in x:
            z = numpy.where(_x_.cpu() != 2)[0]
            seq_lens.append(len(z))
        seq_lens = torch.LongTensor(seq_lens)
        y = y.to(self.device)
        embed_x = self.embed(x.to(torch.int64))
        packed = pack_padded_sequence(embed_x, seq_lens.cpu().numpy(), batch_first=True,
                                      enforce_sorted=False)
        padded = pad_packed_sequence(packed, batch_first=True)
        batch_output, hidden = self.forward(padded[0].to(self.device))
        self.previous_hidden = hidden.detach()

        # compute loss
        output_logits = batch_output.permute(1, 0, 2)
        final_logit = output_logits[-1]
        loss = self.criterion(final_logit, y)

        # compute acc

        softmax_vals = nn.Softmax(dim=1)(final_logit)
        preds = softmax_vals.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(preds,
                                               y,
                                               task='multiclass',
                                               num_classes=self.hparams['n_classes'])

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc * 100,
            },
            prog_bar=True
        )
        return {"loss": loss, "acc": acc * 100}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        logs = {'loss': avg_loss.cpu().numpy(), 'acc': avg_acc.cpu().numpy()}
        self.metrics['train'] = logs
        if exists(self.logger.log_dir + '/full_metrics_' + self.dstr + '.pickle'):
            with open(self.logger.log_dir + '/full_metrics_' + self.dstr + '.pickle', 'rb') as f:
                data = pickle.load(f)
            with open(self.logger.log_dir + '/full_metrics_' + self.dstr + '.pickle', 'wb') as f:
                data['train']['loss'] = numpy.append(data['train']['loss'], self.metrics['train']['loss'])
                data['train']['acc'] = numpy.append(data['train']['acc'], self.metrics['train']['acc'])
                pickle.dump(data, f)
        else:
            with open(self.logger.log_dir + '/full_metrics_' + self.dstr + '.pickle', 'wb') as f:
                pickle.dump(self.metrics, f)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        seq_lens = []
        for _x_ in x:
            z = numpy.where(_x_.cpu() != 2)[0]
            seq_lens.append(len(z))
        seq_lens = torch.LongTensor(seq_lens)
        y = y.to(self.device)
        embed_x = self.embed(x.to(torch.int64))
        packed = pack_padded_sequence(embed_x, seq_lens.cpu().numpy(), batch_first=True,
                                      enforce_sorted=False)
        padded = pad_packed_sequence(packed, batch_first=True)
        batch_output, hidden = self.forward(padded[0].to(self.device))
        self.previous_hidden = hidden.detach()

        output_logits = batch_output.permute(1, 0, 2)
        final_logit = output_logits[-1]
        loss = self.criterion(final_logit, y)

        # compute acc
        softmax_vals = nn.Softmax(dim=1)(final_logit)
        preds = softmax_vals.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(preds, y,
                                               task='multiclass',
                                               num_classes=self.hparams['n_classes'])
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc * 100,
            },
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        seq_lens = []
        for _x_ in x:
            z = numpy.where(_x_.cpu() != 2)[0]
            seq_lens.append(len(z))
        seq_lens = torch.LongTensor(seq_lens)
        y = y.to(self.device)
        embed_x = self.embed(x.to(torch.int64))
        packed = pack_padded_sequence(embed_x, seq_lens.cpu().numpy(), batch_first=True,
                                      enforce_sorted=False)
        padded = pad_packed_sequence(packed, batch_first=True)
        batch_output, hidden = self.forward(padded[0].to(self.device))
        self.previous_hidden = hidden.detach()

        # compute loss
        output_logits = batch_output.permute(1, 0, 2)
        final_logit = output_logits[-1]
        loss = self.criterion(final_logit, y)
        y_pred = []
        y_true = []
        softmax_vals = nn.Softmax(dim=1)(final_logit)
        y_pred += list(softmax_vals.argmax(dim=1).cpu().detach().numpy())
        y_true += list(y.cpu().detach().numpy())
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['cm'] = cm
        test_acc = numpy.mean([yt == yp for yt, yp in zip(y_true, y_pred)])
        self.log_dict(
            {'test_loss': loss,
             'test_acc': test_acc,
             }
        )
        return {'test_loss': loss, 'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': avg_acc * 100}
        self.metrics['test'] = logs
        with open(self.logger.log_dir + '/full_metrics.pickle', 'wb') as f:
            pickle.dump(self.metrics, f)
        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc * 100, 'log': logs, 'progress_bar': logs}

    @property
    def optimizer(self):
        o = self.optimizers()
        return o

    @property
    def scheduler(self):
        s = self.lr_schedulers()
        return s

    def configure_optimizers(self):
        optimizer = optim.Adam(self.gru.parameters(), lr=self.hparams['learning_rate'])
        optimizer_config = {
            "optimizer": optimizer,
        }
        return optimizer_config

    @staticmethod
    def add_model_specific_args(subparser):
        """
        Specify the hyperparams for this GRU
        """
        # gpus
        subparser.add_argument('--gpus', default=0, type=int,
                               help='Whether to use GPUs to train')

        # runtime
        subparser.add_argument('--version', default=1.0, type=float,
                               help='Version string')
        subparser.add_argument('--resume_from', required=False, type=str, default='none',
                               help='Checkpoint of pretrained model')
        subparser.add_argument('--enable_progress_bar', action='store_true',
                               help='Whether to enable progress bar logging')

        # network and optimization params
        subparser.add_argument('--input_size', default=1, type=int,
                               help='Dimensionality of each value in sequence')
        subparser.add_argument('--n_layers', default=2, type=int,
                               help='GRU gates')
        subparser.add_argument('--learning_rate', default=0.00001, type=float,
                               help='Initial learning rate')
        subparser.add_argument('--batch_size', default=64, type=int,
                               help='Batch size')
        subparser.add_argument('--hidden_size', default=128, type=int,
                               help='Dimension of hidden layer')
        subparser.add_argument('--dropout', default=0.0, type=float,
                               help='Dropout rate')
        subparser.add_argument('--grad_clip', default=0.5, type=float,
                               help='Gradient clip norm value')
        subparser.add_argument('--epochs', default=1000, type=int,
                               help='Training epochs')
        subparser.add_argument('--embedding_dim', default=10, type=int,
                               help='Embedding dimension for packing/padding sequences')
        subparser.add_argument('--dataloaders_dir', default="ckpts/dataloaders", type=str,
                               help='Path to data loaders dir, if loading from test dataloaders')
        subparser.add_argument('--data_dir', default="data", type=str,
                               help='Path to data dir')
        subparser.add_argument('--data_file', type=str,
                               help='Data file name')

        subparser.add_argument('--load', action='store_true', help='Enable loading from saved dataloaders')

        # training specific (for this model)
        subparser.add_argument('--n_classes', default=4, type=int,
                               help='Number of classes in the training data')
        subparser.add_argument('--gen_seed', default=42.0, type=float,
                               help='Seed for the test train split')
        subparser.add_argument('--val_split', default=0.10, type=float,
                               help='Test data ratio (0.10 = 10% of data is in the test set')
        subparser.add_argument('--downsample', type=int, default=0,
                               help='Whether to downsample to the smallest class dataset size')
        subparser.add_argument('--model', type=str, default='gru',
                               help='Model type (should be first)')
        subparser.add_argument('--flip', type=int, default=0,
                               help='Use imbalanced set as training instead of test')
        subparser.add_argument('--dataset_date', type=str, default='',
                               help='Exclude a date from training (to be tested on downstream)')
        subparser.set_defaults(load=False)
        subparser.set_defaults(enable_progress_bar=False)
        return subparser

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def probs_at_thresholds(self, test_dataloader):
        y_pred = []
        y_true = []
        with torch.no_grad():  # Deactivate gradients for the following code
            for data_inputs, data_labels, data_names in test_dataloader:
                # Determine prediction of model on test set
                x, y, _ = data_inputs, data_labels, data_names
                seq_lens = []
                for _x_ in x:
                    z = numpy.where(_x_.cpu() != 2)[0]
                    seq_lens.append(len(z))
                seq_lens = torch.LongTensor(seq_lens)
                y = y.to(self.device)
                embed_x = self.embed(x.to(torch.int64))
                packed = pack_padded_sequence(embed_x, seq_lens.cpu().numpy(), batch_first=True,
                                              enforce_sorted=False)
                padded = pad_packed_sequence(packed, batch_first=True)
                batch_output, hidden = self.forward(padded[0].to(self.device))
                # compute loss
                output_logits = batch_output.permute(1, 0, 2)
                final_logit = output_logits[-1]
                softmax_vals = nn.Softmax(dim=1)(final_logit)
                y_pred.append(softmax_vals.cpu().detach().numpy())
                y_true.append(list(y.cpu().detach().numpy()))

        return y_pred, y_true

    def eval_real_data(self, test_dataloader, n_classes, save=True, return_cm=False, top_2=False, plot_clusters=False):
        y_pred = []
        y_true = []
        topx = 2
        y_pred_topx = []
        softmax_indices_for_each_class = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        if plot_clusters:
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig)
        with torch.no_grad():  # Deactivate gradients for the following code
            indexx = 0
            for data_inputs, data_labels, data_names, in test_dataloader:
                # Determine prediction of model on test set
                x, y, _ = data_inputs, data_labels, data_names
                indices = test_dataloader.dataset.indices[indexx:indexx+test_dataloader.batch_size]
                indexx += test_dataloader.batch_size
                seq_lens = []
                for _x_ in x:
                    z = numpy.where(_x_.cpu() != 2)[0]
                    seq_lens.append(len(z))
                seq_lens = torch.LongTensor(seq_lens)
                y = y.to(self.device)
                embed_x = self.embed(x.to(torch.int64))
                packed = pack_padded_sequence(embed_x, seq_lens.cpu().numpy(), batch_first=True,
                                              enforce_sorted=False)
                padded = pad_packed_sequence(packed, batch_first=True)
                batch_output, hidden = self.forward(padded[0].to(self.device))
                # compute loss
                if plot_clusters:
                    self.pca_cluster(hidden, y, ax)
                output_logits = batch_output.permute(1, 0, 2)
                final_logit = output_logits[-1]
                softmax_vals = nn.Softmax(dim=1)(final_logit)
                for i,j in enumerate(softmax_vals):
                    max_prob = j.max().item()
                    tru = y[i].item()
                    pre = j.argmax().item()
                    if tru == pre:
                        softmax_indices_for_each_class[tru].append((indices[i], max_prob))

                y_pred.extend(list(softmax_vals.argmax(dim=1).cpu().detach().numpy()))
                y_true.extend(list(y.cpu().detach().numpy()))
                y_pred_topx.extend(list([x[:topx] for x in (-softmax_vals).argsort()]))

        if plot_clusters:
            ax.set_facecolor('white')
            plt.savefig('figs/pca_3d{}.eps'.format(self.hparams['version']), dpi=600, facecolor='white')
            plt.close(fig)
        if return_cm:
            return self.accuracy_eval_via_confusion_matrix(y_true, y_pred, y_pred_topx, n_classes, topx, save, return_cm, top_2)
        else:
            self.accuracy_eval_via_confusion_matrix(y_true, y_pred, y_pred_topx, n_classes, topx, save, return_cm, top_2)

    @staticmethod
    def accuracy_eval_via_confusion_matrix(y_true, y_pred, y_pred_topx, n_classes_model, topx, save, return_cm, top_2):
        acc = numpy.mean([yt == yp for yt, yp in zip(y_true, y_pred)])
        topx_acc = numpy.mean([y_t in y_p for y_t, y_p in zip(y_true, y_pred_topx)])

        if n_classes_model == 3:
            nams = names.name_dict[3]
        elif n_classes_model == 14:
            nams = names.name_dict['shorter_names']
        elif n_classes_model == 6:
            nams = names.name_dict[6]
        elif n_classes_model == 5:
            nams = names.name_dict[5]
        elif n_classes_model == 4:
            nams = names.name_dict[4]
        else:
            nams = names.name_dict['all_names']
        print(f"Accuracy of the model: {100.0 * acc:4.2f}%")
        print(f"Accuracy of the top {topx} guesses: {100.0 * topx_acc:4.2f}%")

        all_predicted = set(y_pred)
        all_expected_predicted = range(len(nams))
        for x in list(all_expected_predicted):
            if x not in list(all_predicted):
                random_y = numpy.random.choice(y_true, 1)[0]
                y_true.append(random_y)
                y_pred.append(x)

        if top_2:
            new_y_pred = []
            for yt, (yp_1, yp_2) in zip(y_true, y_pred_topx):
                if yp_1 == yt:
                    new_y_pred.append(yp_1)
                elif yp_2 == yt:
                    new_y_pred.append(yp_2)
                else:
                    new_y_pred.append(yp_1)
            y_pred = new_y_pred
        cm = confusion_matrix(y_true, y_pred)
        cm = numpy.transpose(numpy.transpose(cm) / cm.astype(numpy.float).sum(axis=1))
        if not return_cm:
            fig, ax3 = plt.subplots(figsize=(12, 8))
            img = ax3.imshow(cm, cmap='Blues')
            for i in range(n_classes_model):
                for j in range(n_classes_model):
                    text = ax3.text(j, i, numpy.round_(cm[i, j], 2),
                                    ha="center", va="center", color="w")
            ax3.set_xlabel('Predicted Species')
            ax3.set_ylabel('True Species')
            ax3 = plt.gca()
            new_xticks = numpy.arange(0, cm.shape[0], 1)
            ax3.set_xticks(new_xticks)
            ax3.set_yticks(new_xticks)
            ax3.set_xticklabels(nams, rotation=90)
            ax3.set_yticklabels(nams, rotation=0)

            plt.tight_layout()
            cbar = fig.colorbar(img)
            if save:
                plt.savefig('figs/cm_{}classes_{}acc.png'.format(n_classes_model,
                                                                 f1_score(y_true, y_pred, average='weighted')))
        else:
            return cm
        print('done')

    @staticmethod
    def pca_cluster(hidden, y, ax):

        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(hidden[1])
        d = dict()
        d['pca-one'] = pca_results[:, 0]
        d['pca-two'] = pca_results[:, 1]
        d['pca-three'] = pca_results[:, 2]
        d['y'] = y
        df = pd.DataFrame(d)
        colors = {0: 'dodgerblue', 1: 'cyan', 2: 'mediumorchid', 3: 'maroon',
                    4: 'olivedrab', 5: 'orange', 6: 'midnightblue'}

        for s in df.y.unique():
            # if s == 0:
            #     mask = (df['y'] == s) & (df['pca-two'] > 5)
            # elif s == 1:
            #     mask = (df['y'] == s) & (df['pca-one'] < 0)
            # elif s == 2:
            #     mask = (df['y'] == s) & (df['pca-one'] > 0)
            # else:
            mask = (df['y'] == s)

            ax.scatter(df['pca-one'][mask], df['pca-two'][mask], df['pca-three'][mask], color=colors[s],
                       alpha=0.8)
