import random as rand
import numpy as np
import MFCC
import os

CLASS_LABELS = {"cho_biet", "khach", "khong", "toi", "nguoi"}
# Directory name of sound files must match labels

TEST_SIZE = 20
# Number of sound files reserved for testing (<100)

def get_label_data(label):
    files = os.listdir(label)
    test_mfcc = [MFCC.get_mfcc(os.path.join(label, f)) for f in files[:TEST_SIZE] if f.endswith("wav")]
    train_mfcc = [MFCC.get_mfcc(os.path.join(label, f)) for f in files[TEST_SIZE:] if f.endswith("wav")]
    return test_mfcc

if __name__ == "__main__":
    dataset = {}
    for label in CLASS_LABELS:
        dataset[label] = get_label_data(os.path.join("SoundFiles", label))

print(dataset[label][0])