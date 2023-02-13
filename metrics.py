import pickle
import torch
import io
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from data.names import names

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc, \
    confusion_matrix, \
    classification_report, \
    roc_curve, \
    precision_recall_curve, \
    average_precision_score


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Metrics:
    def __init__(self):
        self.CPU_Unpickler = CPU_Unpickler
        self.rnn_logs = "./lightning_logs/"
        self.fig_path = './figs'

    @staticmethod
    def gather(arr, idx):
        ret_arr = []
        for x in range(len(arr)):
            for y in range(len(arr[x])):
                sublist = arr[x][y][:, idx]
                ret_arr = np.concatenate((ret_arr, sublist))
        return ret_arr

    def eval_metrics(self, pretrained_models, data, top_2=False, run_cm=False, plot_clusters=False):
        n_classes_model = pretrained_models[0].hparams.n_classes
        plot = True
        cm = None
        model_cms = []
        num_runs = len(pretrained_models)
        if run_cm:
            for model, d in zip(pretrained_models, data):
                model_cm = model.eval_real_data(d, model.hparams.n_classes,
                                                save=False, return_cm=True, top_2=top_2, plot_clusters=plot_clusters)
                if cm is None:
                    cm = model_cm
                    model_cms.append(model_cm)
                else:
                    for i in range(n_classes_model):
                        for j in range(n_classes_model):
                            cm[i, j] += model_cm[i, j]
                    model_cms.append(model_cm)
            cm = cm / len(pretrained_models)
            self.plot_cm(cm, model_cms, n_classes_model, num_runs, top_2=top_2)

        y_tests = []
        y_preds = []
        class_precisions = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
        class_f1s = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
        class_recalls = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
        all_reports = []
        for model, d in zip(pretrained_models, data):
            y_pred, y_true = model.probs_at_thresholds(d)
            cr = classification_report([x for y in y_true for x in y],
                                       [x for y in np.argmax(y_pred, axis=2) for x in y],
                                       labels=list(range(n_classes_model)),
                                       output_dict=True)
            all_reports.append(cr)
            for c_label in range(n_classes_model):
                class_precisions[str(c_label)].append(cr[str(c_label)]['precision'])
                class_recalls[str(c_label)].append(cr[str(c_label)]['recall'])
                class_f1s[str(c_label)].append(cr[str(c_label)]['f1-score'])
            y_true = np.array(y_true).flatten()
            lb = LabelBinarizer()
            lb.fit(y_true)
            y_test = lb.transform(y_true)
            y_tests.append(y_test.reshape(np.array(y_pred).shape))
            y_preds.append(np.array(y_pred))
        if plot:
            self.plot_roc_curve(n_classes_model, y_tests, y_preds)
            self.plot_pre_rec_curve(n_classes_model, y_tests, y_preds)
            self.report_auroc(n_classes_model, y_tests, y_preds)
            self.report_auprc(n_classes_model, y_tests, y_preds)
        else:
            self.report_auroc(n_classes_model, y_tests, y_preds)
            self.report_auprc(n_classes_model, y_tests, y_preds)

        for c_label in range(n_classes_model):
            c_label = str(c_label)
            print('Precision for class {}: {}'.format(c_label, np.mean(class_precisions[c_label])))
            print('Precision std for class {}: {}'.format(c_label, np.std(class_precisions[c_label])))
            print('Recall for class {}: {}'.format(c_label, np.mean(class_recalls[c_label])))
            print('Recall std for class {}: {}'.format(c_label, np.std(class_recalls[c_label])))
            print('f1 for class {}: {}'.format(c_label, np.mean(class_f1s[c_label])))
            print('f1 std for class {}: {}'.format(c_label, np.std(class_f1s[c_label])))

    def timeseries_and_cm(self, path, cnn=True):
        if cnn:
            with open(self.cnn_logs + path, 'rb') as f:
                data = CPU_Unpickler(f).load()
        else:
            with open(self.rnn_metrics + path, 'rb') as f:
                data = CPU_Unpickler(f).load()
        losses = data['train']['loss']  # .numpy()
        accs = data['train']['acc']  # .numpy()
        epochs = np.arange(len(losses))

        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))

        ax1.plot(epochs, losses)
        ax1.set_title('Loss over time')
        ax2.plot(epochs, accs)
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Acc')
        ax2.set_title('Accuracy over time')
        plt.tight_layout()
        plt.savefig(self.fig_path + '/metrics_over_time_{}'.format(path.split('full')[0].replace('.', '_')))
        plt.close(fig)

    def plot_roc_curve(self, n_classes_model, y_trues, y_preds):
        colormap = {0: 'dodgerblue', 1: 'cyan', 2: 'mediumorchid', 3: 'maroon',
                    4: 'olivedrab', 5: 'orange', 6: 'midnightblue'}
        fig, ax = plt.subplots()
        to_save = {}
        for (idx, c_label) in enumerate(range(n_classes_model)):
            y_true = self.gather(y_trues, idx)
            y_pred = self.gather(y_preds, idx)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            to_save[idx] = {'fpr': [f for f in fpr], 'tpr': [t for t in tpr], 'thresholds': [x for x in thresholds]}
            ax.plot(fpr, tpr,
                    label='{}'.format(names.name_dict[n_classes_model][idx]),
                    color=colormap[idx],
                    lw=2,
                    )
        ax.plot([0, 1], [0, 1], color="dimgray", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.fig_path + '/roc_curves.png')

    def report_auroc(self, n_classes_model, y_trues, y_preds):
        for (idx, c_label) in enumerate(range(n_classes_model)):
            y_true = self.gather(y_trues, idx)
            y_pred = self.gather(y_preds, idx)

            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            print('Class {}: AUROC {}'.format(c_label, auc(fpr, tpr)))

    def plot_pre_rec_curve(self, n_classes_model, y_trues, y_preds):
        fig, c_ax = plt.subplots(1, 5, figsize=(25, 3))
        to_save = {}
        for (idx, c_label) in enumerate(range(n_classes_model)):
            y_true = self.gather(y_trues, c_label)
            y_pred = self.gather(y_preds, c_label)
            len_pos_sample = len([q for q in y_true if q == 1])
            total = len(y_true)
            no_skill = len_pos_sample / total
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            to_save[idx] = {'precision': [p for p in precision], 'recall': [r for r in recall], 'thresholds': [t for t in thresholds]}
            c_ax[idx].plot(recall, precision,
                           label='Precision-recall curve (species %i)'.format(c_label),
                           color="maroon",
                           lw=2,
                           )
            c_ax[idx].plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline')
            c_ax[idx].set_xlim([0.0, 1.0])
            c_ax[idx].set_ylim([0.0, 1.05])
            c_ax[idx].set_xlabel("Recall")
            c_ax[idx].set_ylabel("Precision")
            c_ax[idx].set_title("{}".format(names.name_dict[n_classes_model][c_label]))
        plt.tight_layout()
        plt.savefig(self.fig_path + '/pre_rec_curves.png')

    def report_auprc(self, n_classes_model, y_trues, y_preds):
        for (idx, c_label) in enumerate(range(n_classes_model)):
            y_true = self.gather(y_trues, idx)
            y_pred = self.gather(y_preds, idx)

            len_pos_sample = len([q for q in y_true if q == 1])
            total = len(y_true)
            no_skill = len_pos_sample / total
            auprc = average_precision_score(y_true, y_pred)
            print('Class {}: AuPRC {} vs. baseline {}'.format(c_label, auprc, no_skill))

    def plot_cm(self, cm, model_cms, n_classes_model, num_runs, top_2=False):
        acc = round(np.trace(cm) / cm.sum(), 4)
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) - true_pos
        false_neg = np.sum(cm, axis=1) - true_pos

        precision = round(np.mean(true_pos / (true_pos + false_pos)), 4)
        recall = round(np.mean(true_pos / (true_pos + false_neg)), 4)
        accs = []
        precs = []
        recs = []
        for _cm in model_cms:
            acc = round(np.trace(_cm) / _cm.sum(), 4)
            true_pos = np.diag(_cm)
            false_pos = np.sum(_cm, axis=0) - true_pos
            false_neg = np.sum(_cm, axis=1) - true_pos

            precision = round(np.mean(true_pos / (true_pos + false_pos)), 4)
            recall = round(np.mean(true_pos / (true_pos + false_neg)), 4)
            accs.append(acc)
            precs.append(precision)
            recs.append(recall)
        print("Precision: {}, Recall: {}, Accuracy: {}".format(np.mean(precs), np.mean(recs), np.mean(accs)))

        fig, ax = plt.subplots()
        sns.heatmap(cm, linewidths=1, ax=ax, annot=True, fmt=".2g", cmap='viridis')
        ax.set_ylabel('True Species')
        ax.set_xlabel('Predicted Species')
        ax.set_xticklabels(names.name_dict['all_names'], rotation=90)
        ax.set_yticklabels(names.name_dict['all_names'], rotation=0)
        if not top_2:
            plt.savefig(self.fig_path + '/cm_{}classes_{}acc_{}pre_{}rec.png'.format(
                n_classes_model, acc, precision, recall, num_runs)
            )
        else:
            plt.savefig(self.fig_path + '/cm_{}classes_{}acc_{}pre_{}rec_top_2.png'.format(
                n_classes_model, acc, precision, recall, num_runs)
            )
