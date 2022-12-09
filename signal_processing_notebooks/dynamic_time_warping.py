import numpy as np
from numpy.random import randint
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report

def dtwLit(df, literature_sequences):
    num_species = len(literature_sequences)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle df
    Y = df['species_label'].to_numpy()

    # Compute predictions for each test set
    predicts = []
    scores = []

    for species in range(num_species):
        curr_spec_test = df[df['species_label']==species].iloc[:,2:].to_numpy()
        prediction = []
        spec_scores = []
        for i in range(curr_spec_test.shape[0]):
            if not any(np.isnan(curr_spec_test[i])):
                seq = curr_spec_test[i]
            else:
                seq = curr_spec_test[i][:np.where(np.isnan(curr_spec_test[i]))[0][0]]
            dtws = []
            for j in range(len(literature_sequences)):
                distance, path = fastdtw(seq, literature_sequences[j], dist=euclidean)
                dtws.append(distance)
            score = softmax(np.multiply(dtws,-1))
            if np.all(np.isclose(score, score[0])):
                pred_class = randint(0,high=num_species)
            else:
                pred_class = np.argmax(score)
            prediction.append(pred_class)
            spec_scores.append(score)
        predicts.append(prediction)
        scores.append(spec_scores)

    y_true = []
    for species in range(num_species):
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
