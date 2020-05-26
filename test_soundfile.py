import os
import MFCC
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="Image input", required=True)
args = parser.parse_args()

CLASS_LABELS = {"cho_biet", "khach", "khong", "toi", "nguoi"}

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    return kmeans  

if __name__ == "__main__":
    models = {}
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "rb") as file: models[label] = pk.load(file)

    sound_mfcc = MFCC.get_mfcc(args.test)
    kmeans = clustering(sound_mfcc)
    sound_mfcc = kmeans.predict(sound_mfcc).reshape(-1,1)

    evals = {cname : model.score(sound_mfcc, [len(sound_mfcc)]) for cname, model in models.items()}
    cmax = max(evals.keys(), key=(lambda k: evals[k]))
    print(evals)
    print("Conclusion: " + cmax)