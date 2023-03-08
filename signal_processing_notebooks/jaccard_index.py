import numpy as np
from numpy.random import randint
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, KFold
import os

def get_intersection(A, B):
    """
    Computes intersection between two binary-valued sequences
    """
    count = 0
    for i in range(min(len(A), len(B))):
        if A[i] > 0 and B[i] > 0:
            count += 1
    return count

def get_union(A, B):
    """
    Computes union between two binary-valued sequences
    """
    count = 0
    for i in range(max(len(A), len(B))):
        if i > len(A) or i > len(B):
            break
        if i < len(A) and A[i] > 0:
            count += 1
        elif i < len(B) and B[i] > 0:
            count += 1
    return count

def jaccard(A, B):
    """
    Computes Jaccard index between two binary-valued sequences
    """
    i = get_intersection(A, B)
    u = get_union(A, B)
    return i/u

def get_intersection_cont(A, B):
    """
    Computes intersection between two real-valued sequences
    """
    count = 0
    for i in range(min(len(A), len(B))):
        if A[i] > 0 and B[i] > 0:
            count += A[i]*B[i]
    return count

def get_union_cont(A, B):
    """
    Computes union between two real-valued sequences
    """
    return np.sum(A) + np.sum(B)

def jaccard_cont(A, B):
    """
    Computes Jaccard index between two real-valued sequences
    """
    i = get_intersection_cont(A, B)
    u = get_union_cont(A, B)
    return i/(u-i)

def jaccardLit(df, literature_sequences, literature_labels, random_seed: int = None):
    """
    Computes Jaccard index between each sample and each literature sequence and predicts species based on the largest value

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
        curr_spec_test = df[df['species_label'] == species]
        curr_spec_test = curr_spec_test[seq_cols].to_numpy()
        prediction = []
        spec_scores = []
        for i in range(curr_spec_test.shape[0]):
            if not any(np.isnan(curr_spec_test[i])):
                seq = curr_spec_test[i]
            else:
                seq = curr_spec_test[i][:np.where(np.isnan(curr_spec_test[i]))[0][0]]
            jacc = []
            for litseq in literature_sequences:
                jacc.append(jaccard(seq,litseq))
            score = softmax(jacc)
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

    return acc, prec, rec, conf_mat, y_true, y_pred, np.stack(y_score,axis=0), metrics

def jaccardPop(df, k, train_split, random_seed: int = None):
    """
    Computes Jaccard index between each sample and each population reference and predicts species based on the largest value
    Performs a stratified k-fold cross-validation; then, for each fold, the data is further split into a training set
    where the number of training samples is taken to be train_split times the number of samples of the species with the fewest samples,
    such that each training set is the same size. Remaining data is used for testing.
    Population references are then taken by averaging the sequences in each training sets

    Parameters:
    df: pandas dataframe containing species labels (string and numeric), 3 parameters, and flash sequence for each sample
        numerical species label column must be named 'species_label'
        string species label column must be named 'species'
        flash sequence column headers must be numeric
    k: number of stratified folds
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
    seq_cols = [col_ind for col_ind in df.columns.to_list() if isinstance(col_ind,int)] # column indices of flash sequence

    accs = []
    precs = []
    recs = []
    conf_mat = np.zeros((num_species,num_species))
    y_trues = []
    y_preds = []
    y_scores = []
    precs_sp = np.zeros((num_species, k))
    recs_sp = np.zeros((num_species,k))

    class_sizes = df.species_label.value_counts().sort_index().to_numpy() # number of samples in each class, in ascending order of species_label

    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    np.random.seed(seed)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle df

    strat_cv = StratifiedKFold(n_splits=k)
    for iter, (train_index, test_indices) in enumerate(strat_cv.split(df, df.species_label.values)):
        train_indices = []
        test_indices = test_indices.tolist()
        for c in range(num_species):
            c_indices = train_index[np.where(df.iloc[train_index].species_label == c)] # get which train indices belong to species c
            mi = np.min(df.iloc[train_index].species_label.value_counts())
            k_inner = int(len(c_indices)/(mi*train_split))
            if k_inner > 1:
                inner_split = KFold(n_splits=k_inner)
                for j, (tr_i, te_i) in enumerate(inner_split.split(c_indices)):
                    if iter % k_inner == j:
                        train_indices.extend(c_indices[te_i])
                        test_indices.extend(c_indices[tr_i])
            else:
                k_inner = int(len(c_indices)/(len(c_indices) - mi*train_split))
                inner_split = KFold(n_splits=k_inner)
                for j, (tr_i, te_i) in enumerate(inner_split.split(c_indices)):
                    if iter % k_inner == j:
                        train_indices.extend(c_indices[tr_i])
                        test_indices.extend(c_indices[te_i])


        # all training data
        df_train = df.iloc[train_indices]
        # all test data
        df_test = df.iloc[test_indices]

        # Generate reference sequences from training sets
        ref_seqs = np.zeros((num_species,len(seq_cols)))
        for species in np.unique(df['species_label']):
            curr_spec_train = df_train[df_train['species_label'] == species]
            curr_spec_train = curr_spec_train[seq_cols].to_numpy()
            ref_seqs[species,:] = np.nansum(curr_spec_train,axis=0)/curr_spec_train.shape[0]

        # Compute predictions for each test sets
        predicts = []
        scores = []
        for species in np.unique(df['species_label']):
            curr_spec_test = df_test[df_test['species_label'] == species]
            curr_spec_test = curr_spec_test[seq_cols].to_numpy()
            prediction = []
            spec_scores = []
            for i in range(curr_spec_test.shape[0]):
                if not any(np.isnan(curr_spec_test[i])):
                    seq = curr_spec_test[i]
                else:
                    seq = curr_spec_test[i][:np.where(np.isnan(curr_spec_test[i]))[0][0]]
                jacc = []
                for j in range(ref_seqs.shape[0]):
                    jacc.append(jaccard_cont(seq,ref_seqs[j]))
                score = softmax(jacc)
                if np.all(np.isclose(score, score[0])):
                    pred_class = randint(0,high=num_species)
                else:
                    pred_class = np.argmax(score)
                prediction.append(pred_class)
                spec_scores.append(score)
            predicts.append(prediction)
            scores.append(spec_scores)

        y_true = []
        for species in np.unique(df_test['species_label']):
            y_true.append(species*np.ones(df_test['species_label'].value_counts()[species],dtype=np.int8))
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

    return accs, precs, recs, conf_mat/k, [i for j in y_trues for i in j], [i for j in y_preds for i in j], np.vstack(y_scores), precs_sp, recs_sp
