import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import os

def svmLit(df, params_lit, random_seed: int = None):
    """
    Performs SVM species classification on the 3-parameter space (# of flashes, flash length, gap length) as trained on literature data
    and tested on the entirety of collected samples

    Parameters:
    df: pandas dataframe containing species label, # flashes, flash length, and gap length for each sample.
        # flashes column must be named 'nf', 'num_flashes', 'num_flash', 'numflashes', or 'numflash'
        flash length column must be named 'fl', 'flash_length', 'flashlength', 'flash'
        gap length column must be named 'gl', 'gap_length', 'gap', 'ipi'
        numerical species label column must be named 'species_label'
        string species label column must be named 'species'
    params_lit: pandas dataframe containing species label, # flashes, flash length, and gap length for each literature sample
        # flashes column must be named 'nf', 'num_flashes', 'num_flash', 'numflashes', or 'numflash'
        flash length column must be named 'fl', 'flash_length', 'flashlength', 'flash'
        gap length column must be named 'gl', 'gap_length', 'gap', 'ipi'
        string species label column must be named 'species'
    random_seed: optional seed for random number generator

    Returns:
    acc: accuracy
    prec: overall precision
    rec: overall recall
    conf_mat: # species x # species confusion matrix
    y_test: labels of test data
    rbf_pred: predicted labels
    y_score: prediction scores
    metrics: dict of per-class classification metrics
    """

    nf_labels = ['nf', 'num_flashes', 'num_flash', 'numflashes', 'numflash']
    fl_labels = ['fl', 'flash_length', 'flashlength', 'flash']
    gap_labels = ['gl', 'gap_length', 'gap', 'ipi']

    if not any(label in nf_labels for label in list(df.columns)):
        raise Exception('No number of flashes column found in df')
    if not any(label in fl_labels for label in list(df.columns)):
        raise Exception('No flash length column found in df')
    if not any(label in gap_labels for label in list(df.columns)):
        raise Exception('No gap length column found in df')
    if 'species' not in list(df.columns):
        raise Exception('species column not found in df')
    if 'species_label' not in list(df.columns):
        raise Exception('species_label column not found in df')
    if not np.issubdtype(df.species_label.dtype, np.number):
        raise Exception('species_label is not numerical')
    if not any(label in nf_labels for label in list(params_lit.columns)):
        raise Exception('No number of flashes column found in params_lit')
    if not any(label in fl_labels for label in list(params_lit.columns)):
        raise Exception('No flash length column found in params_lit')
    if not any(label in gap_labels for label in list(params_lit.columns)):
        raise Exception('No gap length column found in params_lit')
    if 'species' not in list(params_lit.columns):
        raise Exception('species column not found in params_lit')

    num_species = len(params_lit)
    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed

    # if necessary, trim df to contain only species found in literature data
    df = df[df['species_label'].isin([df[df['species']==label].iloc[0].species_label for label in list(params_lit['species'])])]
    df = df.sample(frac=1,random_state=seed).reset_index(drop=True)

    X_test = df.iloc[:,[np.where(df.columns.isin(nf_labels))[0][0],
                np.where(df.columns.isin(fl_labels))[0][0],
                np.where(df.columns.isin(gap_labels))[0][0]]].to_numpy()
    y_test = df['species_label'].to_numpy()

    X_train = params_lit.iloc[:,[np.where(params_lit.columns.isin(nf_labels))[0][0],
            np.where(params_lit.columns.isin(fl_labels))[0][0],
            np.where(params_lit.columns.isin(gap_labels))[0][0]]].to_numpy()
    y_train = np.array([df[df['species']==label].iloc[0].species_label for label in list(params_lit['species'])])

    rbf = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=1, C=1)).fit(X_train, y_train)
    y_score = rbf.decision_function(X_test)

    rbf_pred = rbf.predict(X_test)
    metrics = classification_report(y_test,rbf_pred,output_dict=True)
    conf_mat = confusion_matrix(y_test,rbf_pred,normalize='true')
    acc = metrics['accuracy']
    prec = metrics['macro avg']['precision']
    rec = metrics['macro avg']['recall']

    y_score = np.stack(y_score,axis=0)
    return acc, prec, rec, conf_mat, y_test, rbf_pred, y_score, metrics

def svmPop(df, num_iter, train_split, random_seed: int = None):
    """
    Performs SVM species classification on the 3-parameter space (# of flashes, flash length, gap length) as trained on population reference data and tested on remaining samples
    The number of training samples for each species is equal, taken to be train_split times the number of samples of the species with the fewest samples

    Parameters:
    df: pandas dataframe containing species label, # flashes, flash length, and gap length for each sample.
        # flashes column must be named 'nf', 'num_flashes', 'num_flash', 'numflashes', or 'numflash'
        flash length column must be named 'fl', 'flash_length', 'flashlength', 'flash'
        gap length column must be named 'gl', 'gap_length', 'gap', 'ipi'
        numerical species label column must be named 'species_label'
        string species label column must be named 'species'
    num_iter: number of iterations to perform with reshuffled data
    train_split: percentage of data used for training
    random_seed: optional seed for random number generator

    Returns:
    - accuracy of each iteration
    - overall precision of each iteration
    - overall recall of each iteration
    - # species x # species confusion matrix
    - labels of test data
    - predicted labels
    - prediction scores
    - per-species precision
    - per-species recall
    """

    nf_labels = ['nf', 'num_flashes', 'num_flash', 'numflashes', 'numflash']
    fl_labels = ['fl', 'flash_length', 'flashlength', 'flash']
    gap_labels = ['gl', 'gap_length', 'gap', 'ipi']

    if not any(label in nf_labels for label in list(df.columns)):
        raise Exception('No number of flashes column found in df')
    if not any(label in fl_labels for label in list(df.columns)):
        raise Exception('No flash length column found in df')
    if not any(label in gap_labels for label in list(df.columns)):
        raise Exception('No gap length column found in df')
    if 'species' not in list(df.columns):
        raise Exception('species column not found in df')
    if 'species_label' not in list(df.columns):
        raise Exception('species_label column not found in df')
    if not np.issubdtype(df.species_label.dtype, np.number):
        raise Exception('species_label is not numerical')

    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    num_species = len(np.unique(df['species_label']))

    accs = []
    precs = []
    recs = []
    conf_mat = np.zeros((num_species,num_species))
    y_trues = []
    y_preds = []
    y_scores = []
    precs_sp = np.zeros((num_species, num_iter))
    recs_sp = np.zeros((num_species,num_iter))

    for iter in range(num_iter):
        np.random.seed(seed+iter)
        df = df.sample(frac=1,random_state=seed+iter).reset_index(drop=True)
        X = df.iloc[:,[np.where(df.columns.isin(nf_labels))[0][0],
                    np.where(df.columns.isin(fl_labels))[0][0],
                    np.where(df.columns.isin(gap_labels))[0][0]]].to_numpy()
        Y = df['species_label'].to_numpy()
        smallest_set = np.min(df['species_label'].value_counts())
        train_inds = np.hstack(([np.where(Y==sp)[0][:int(train_split*smallest_set)] for sp in np.unique(df['species_label'])]))
        X_train = X[train_inds]
        y_train = Y[train_inds]
        X_test = np.delete(X,train_inds,0)
        y_test = np.delete(Y,train_inds,0)

        rbf = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=1, C=1)).fit(X_train, y_train)
        y_score = rbf.decision_function(X_test)

        rbf_pred = rbf.predict(X_test)
        metrics = classification_report(y_test,rbf_pred,output_dict=True)
        conf_mat += confusion_matrix(y_test,rbf_pred,normalize='true')
        acc = metrics['accuracy']
        prec = metrics['macro avg']['precision']
        rec = metrics['macro avg']['recall']
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        y_trues.append(y_test)
        y_preds.append(rbf_pred)
        y_scores.append(y_score)
        for sp in range(num_species):
            precs_sp[sp,iter] = metrics[str(sp)]['precision']
            recs_sp[sp,iter] = metrics[str(sp)]['recall']

    return accs, precs, recs, conf_mat/num_iter, [i for j in y_trues for i in j], [i for j in y_preds for i in j], np.stack(y_scores,axis=0), precs_sp, recs_sp
