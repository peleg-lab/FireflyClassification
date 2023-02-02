import numpy as np
from numpy.random import randint
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
import os

def dotProductLit(df, literature_sequences, literature_labels, random_seed: int = None):
    """
    Computes dot product between each sample and each literature sequence and predicts species based on the largest value

    Parameters:
    df: pandas dataframe containing species labels (string and numeric) and flash sequence for each sample
        numerical species label column must be named 'species_label'
        string species label column must be named 'species'
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
    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    ref_seqs = np.zeros((num_species,df.shape[1]-2))
    for sp in range(len(species_with_seq)):
        ref_seqs[sp,:len(literature_sequences[sp])] = literature_sequences[sp]

    # Compute predictions for each test set
    predicts = []
    scores = []

    np.random.seed(seed)
    for species in species_with_seq:
        curr_spec_test = df[df['species_label'] == species]
        curr_spec_test = curr_spec_test.iloc[:,2:].to_numpy()
        prediction = []
        spec_scores = []
        for i in range(curr_spec_test.shape[0]):
            seq = np.nan_to_num(curr_spec_test[i],nan=0.0)
            areas = []
            for j in range(ref_seqs.shape[0]):
                areas.append(sum(seq*ref_seqs[j]))
            score = softmax(areas)
            if np.all(np.isclose(score, score[0])):
                pred_class = species_with_seq[randint(0,high=num_species)]
            else:
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

    return acc, prec, rec, conf_mat, y_true, y_pred, np.stack(y_score,axis=0), metrics


def dotProductPop(df, num_iter, train_split, random_seed: int = None):
    """
    Computes dot product between each sample and each population reference and predicts species based on the largest value
    The set of each species' samples is divided into a training subset where the number of training samples is taken to be train_split times the number of samples of the species with the fewest samples,
    such that each training subset is the same size
    Population references are then taken by averaging the sequences in each training subset

    Parameters:
    df: pandas dataframe containing species labels (string and numeric) and flash sequence for each sample
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

    if 'species' not in list(df.columns):
        raise Exception('species column not found in df')
    if 'species_label' not in list(df.columns):
        raise Exception('species_label column not found in df')

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
    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    for iter in range(num_iter):
        np.random.seed(seed+iter)
        df = df.sample(frac=1, random_state=seed+iter).reset_index(drop=True)
        test_df = df.copy()
        Y = df['species_label'].to_numpy()
        # Generate reference sequences from training sets
        ref_seqs = np.zeros((num_species,df.shape[1]-2))
        for species in np.unique(df['species_label']):
            #downsample
            inds = np.where(Y==species)[0][:np.min(df['species_label'].value_counts())]
            inds_train = inds[:int(train_split*len(inds))]
            curr_spec_train = df.iloc[inds_train,:]
            curr_spec_train = curr_spec_train.iloc[:,2:].to_numpy()
            test_df = test_df.drop(inds_train)
            ref_seqs[species,:] = np.nansum(curr_spec_train,axis=0)/curr_spec_train.shape[0]

        # Compute predictions for each test set
        predicts = []
        scores = []
        for species in np.unique(df['species_label']):
            curr_spec_test = test_df[test_df['species_label'] == species]
            curr_spec_test = curr_spec_test.iloc[:,2:].to_numpy()
            prediction = []
            spec_scores = []
            for i in range(curr_spec_test.shape[0]):
                seq = np.nan_to_num(curr_spec_test[i],nan=0.0)
                areas = []
                for j in range(ref_seqs.shape[0]):
                    areas.append(sum(seq*ref_seqs[j]))
                score = softmax(areas)
                if np.all(np.isclose(score, score[0])):
                    pred_class = randint(0,high=num_species)
                else:
                    pred_class = np.argmax(score)
                prediction.append(pred_class)
                spec_scores.append(score)
            predicts.append(prediction)
            scores.append(spec_scores)


        y_true = []
        for species in np.unique(df['species_label']):
            y_true.append(species*np.ones(test_df['species_label'].value_counts()[species],dtype=np.int8))
        y_true = [i for j in y_true for i in j]
        y_pred = [i for j in predicts for i in j]
        y_score = [i for j in scores for i in j]
        metrics = classification_report(y_true,y_pred,output_dict=True)
        conf_mat += confusion_matrix(y_true,y_pred,normalize='true')
        acc = metrics['accuracy']
        prec = metrics['macro avg']['precision']
        rec = metrics['macro avg']['recall']
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        y_trues.append(y_true)
        y_preds.append(y_pred)
        y_scores.append(y_score)
        for sp in range(num_species):
            precs_sp[sp,iter] = metrics[str(sp)]['precision']
            recs_sp[sp,iter] = metrics[str(sp)]['recall']

    return accs, precs, recs, conf_mat/num_iter, [i for j in y_trues for i in j], [i for j in y_preds for i in j], np.stack(y_scores,axis=0), precs_sp, recs_sp
