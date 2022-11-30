import numpy as np
from numpy.random import randint
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report

def get_intersection(A, B):
    count = 0
    for i in range(min(len(A), len(B))):
        if A[i] > 0 and B[i] > 0:
            count += 1
    return count

def get_union(A, B):
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
    i = get_intersection(A, B)
    u = get_union(A, B)
    return i/u

def get_intersection_cont(A, B):
    count = 0
    for i in range(min(len(A), len(B))):
        if A[i] > 0 and B[i] > 0:
            count += A[i]*B[i]
    return count

def get_union_cont(A, B):
    return np.sum(A) + np.sum(B)

def jaccard_cont(A, B):
    i = get_intersection_cont(A, B)
    u = get_union_cont(A, B)
    return i/(u-i)

def jaccardLit(df, literature_sequences):
    num_species = len(literature_sequences)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle df

    # Compute predictions for each test set
    predicts = []
    scores = []

    for species in range(num_species):
        curr_spec_test = df[df['species_label'] == species]
        curr_spec_test = curr_spec_test.iloc[:,2:].to_numpy()
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
                pred_class = randint(0,high=num_species)
            else:
                # argmax of softmax
                pred_class = np.argmax(score)
            prediction.append(pred_class)
            spec_scores.append(score)
        predicts.append(prediction)
        scores.append(spec_scores)

    y_true = []
    for species in range(num_species):
        y_true.append(species*np.ones(df['species_label'].value_counts()[species]))
    y_true = [i for j in y_true for i in j]
    y_pred = [i for j in predicts for i in j]
    y_score = [i for j in scores for i in j]
    metrics = classification_report(y_true,y_pred,output_dict=True)
    conf_mat = confusion_matrix(y_true,y_pred,normalize='true')
    acc = metrics['accuracy']
    prec = metrics['macro avg']['precision']
    rec = metrics['macro avg']['recall']

    return acc, prec, rec, conf_mat, y_true, y_pred, np.stack(y_score,axis=0), metrics

def jaccardPop(df, num_iter, train_split):
    # computes jaccard index with pop ref instead of literature sequences
    num_species = len(np.unique(df['species_label']))
    accs = []
    precs = []
    recs = []
    y_trues = []
    y_preds = []
    y_scores = []
    conf_mat = np.zeros((num_species,num_species))
    ref_seqs = np.zeros((num_species,df.shape[1]-2))
    precs_sp = np.zeros((num_species, num_iter))
    recs_sp = np.zeros((num_species,num_iter))

    for iter in range(num_iter):
        df = df.sample(frac=1).reset_index(drop=True)
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
