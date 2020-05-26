import os
import MFCC
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk

CLASS_LABELS = {"cho_biet", "khach", "khong", "toi", "nguoi"}
# Directory name of sound files must match labels

TEST_SIZE = 5
# Number of sound files reserved for testing (<100)

#np.set_printoptions(precision=3)

def get_label_data(label):
    files = os.listdir(label)
    test_mfcc = [MFCC.get_mfcc(os.path.join(label, f)) for f in files[:TEST_SIZE] if f.endswith("wav")]
    train_mfcc = [MFCC.get_mfcc(os.path.join(label, f)) for f in files[TEST_SIZE:] if f.endswith("wav")]
    return train_mfcc, test_mfcc

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    return kmeans  

if __name__ == "__main__":
    train_dataset = {}
    test_dataset = {}
    for label in CLASS_LABELS:
        train_dataset[label], test_dataset[label] = get_label_data(os.path.join("SoundFiles", label))

    # Get all vectors in the datasets
    all_vectors_x = np.concatenate([np.concatenate(v, axis=0) for k, v in train_dataset.items()], axis=0)
    all_vectors_y = np.concatenate([np.concatenate(v, axis=0) for k, v in test_dataset.items()], axis=0)
    #print("Vectors", all_vectors_x.shape)a
    # Run K-Means algorithm to get clusters
    kmeans_x = clustering(all_vectors_x)
    kmeans_y = clustering(all_vectors_y)
    #print("Centers", kmeans_x.cluster_centers_.shape)

    models = {}
    for labels in CLASS_LABELS:
        class_vectors_x = train_dataset[labels]
        class_vectors_y = test_dataset[labels]
        # convert all vectors to the cluster index
        # dataset['one'] = [O^1, ... O^R]
        # O^r = (c1, c2, ... ct, ... cT)
        # O^r size T x 1
        train_dataset[labels] = list([kmeans_x.predict(v).reshape(-1,1) for v in train_dataset[labels]])
        test_dataset[labels] = list([kmeans_y.predict(v).reshape(-1,1) for v in test_dataset[labels]])
        hmm = hmmlearn.hmm.MultinomialHMM(
            n_components=12, random_state=0, n_iter=1000, verbose=True,
            startprob_prior=np.array([0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1]),
            transmat_prior=np.array([
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.0,0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.5,0.1,0.1,],
                [0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,0.1,],
                [0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,0.1,],
                [0.0,0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.1,0.5,0.1,0.1,],
                [0.0,0.0,0.0,0.0,0.0,0.0,1.1,0.1,0.5,0.1,0.1,0.1,],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.5,0.1,0.1,0.1,0.1,],
            ]),
        )
        X = np.concatenate(train_dataset[labels])
        lengths = list([len(x) for x in train_dataset[labels]])
        print("Training class: ", labels)
        print(X.shape, lengths, len(lengths))
        hmm.fit(X, lengths=lengths)
        models[labels] = hmm
        
    print("Training done")

    print("Testing (Higher is better)")
    model_acc = {}
    for true_cname in CLASS_LABELS:
        hits = 0
        for O in train_dataset[true_cname]:
            evals = {cname : model.score(O, [len(O)]) for cname, model in models.items()}
            print(true_cname, evals)
            if max(evals.keys(), key=(lambda k: evals[k])) == true_cname:
                print("Hit")
                hits += 1
            else:
                print("Miss")
        model_acc[true_cname] = hits

    print(model_acc)

    print("Exporting models")
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "wb") as file: pk.dump(models[label], file)