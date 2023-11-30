import numpy as np
from numpy.random import randint
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
import os

def dtwLit(df, literature_sequences, literature_labels, random_seed: int = None):
    """
    Computes distance using dynamic time warping between each sample and each literature sequence and predicts species based on the shortest distance

    Parameters:
    df: pandas dataframe containing species labels (string and numeric), 3 parameters, and flash sequence for each sample
        numerical species label column must be named 'species_label'
        string species label column must be named 'species'
        flash sequence column headers must be numeric
    literature_sequences: list of literature sequences
    literature_labels: list of species corresponding to each literature sequence
    random_seed: optional seed for random number generator

    Returns:
    acc: accuracy
    prec: overall precision
    rec: overall recall
    conf_mat: # species x # species confusion matrix
    y_true: labels of test data
    y_pred: predicted labels
    y_score: prediction scores
    metrics: dict of per-class classification metrics
    """

    if 'species' not in list(df.columns):
        raise Exception('species column not found in df')
    if 'species_label' not in list(df.columns):
        raise Exception('species_label column not found in df')

    species_with_seq = [df[df['species']==label].iloc[0].species_label for label in literature_labels] # species that have corresponding literature sequences
    num_species = len(species_with_seq)
    seq_cols = [col_ind for col_ind in df.columns.to_list() if isinstance(col_ind,int)] # column indices of flash sequence
    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle df
    np.random.seed(seed)

    # Compute predictions for each test set
    predicts = []
    scores = []

    for species in species_with_seq:
        curr_spec_test = df[df['species_label']==species][seq_cols].to_numpy()
        prediction = []
        spec_scores = []
        for i in range(curr_spec_test.shape[0]):
            if not any(np.isnan(curr_spec_test[i])):
                seq = curr_spec_test[i]
            else:
                seq = curr_spec_test[i][:np.where(np.isnan(curr_spec_test[i]))[0][0]]
            dtws = []
            for j in range(len(literature_sequences)):
                distance, path = fastdtw(seq, np.array(literature_sequences[j]), dist=euclidean)
                dtws.append(distance)
            score = softmax(np.multiply(dtws,-1))
            if np.all(np.isclose(score, score[0])):
                # if all scores are the same, randomly choose a species
                pred_class = species_with_seq[randint(0,high=num_species)]
            else:
                # argmax of softmax
                pred_class = species_with_seq[np.argmax(score)]
            prediction.append(pred_class)
            spec_scores.append(score)
        predicts.append(prediction)
        scores.append(spec_scores)

    y_true = []
    for species in species_with_seq:
        y_true.append(species*np.ones(df['species_label'].value_counts()[species],dtype=np.int8))
    y_true = [i for j in y_true for i in j]
    y_pred = [i for j in predicts for i in j]
    y_score = [i for j in scores for i in j]
    metrics = classification_report(y_true,y_pred,output_dict=True)
    conf_mat = confusion_matrix(y_true,y_pred,normalize='true')
    acc = metrics['accuracy']
    prec = metrics['macro avg']['precision']
    rec = metrics['macro avg']['recall']
    prec_weighted = metrics['weighted avg']['precision']
    rec_weighted = metrics['weighted avg']['recall']

    return acc, prec, rec, prec_weighted, rec_weighted, conf_mat, y_true, y_pred, np.stack(y_score,axis=0), metrics
