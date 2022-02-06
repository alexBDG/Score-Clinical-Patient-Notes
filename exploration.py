"""
Data Exploration.
"""

# System imports.
import os
import time

# Data management imports.
import numpy as np
import pandas as pd

# Vizualisation imports.
import seaborn as sns
import matplotlib.pyplot as plt

# Mathematical imports.
from sklearn.feature_extraction.text import CountVectorizer



class ExploreFile():

    def __init__(self, verbose=0):
        self.df = pd.read_csv(self.file)
        self.verbose = verbose

    def show(self):
        fig, axs = plt.subplots(1, 1)
        self.count_cases(ax=axs)
        plt.show()

    def count_cases(self, ax):
        sns.countplot(data=self.df, x="case_num", ax=ax)



class ExploreTrain(ExploreFile):

    def __init__(self, *args, **kwargs):
        self.file = os.path.join(
            'nbme-score-clinical-patient-notes', 'train.csv'
        )
        # Initialize mother class
        super().__init__(*args, **kwargs)



class ExplorePatientNotes(ExploreFile):

    def __init__(self, *args, **kwargs):
        self.file = os.path.join(
            'nbme-score-clinical-patient-notes', 'patient_notes.csv'
        )
        # Initialize mother class
        super().__init__(*args, **kwargs)


    def count_words(self, ax):

        if self.verbose>0:
            print("[VECTORIZE] starting...")
            start = time.time()
        self.vectorize = CountVectorizer()
        X_train = self.vectorize.fit_transform(self.df["pn_history"])
        
        if self.verbose>0:
            print("[VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        if self.verbose>0:
            print("[VECTORIZE] counting & sorting...")
            start = time.time()
            X_train = X_train.toarray()
            X_c = X_train.sum(0)
            X_b = np.argsort(X_c)
            
        if self.verbose>0:
            print("[VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        if self.verbose>0:
            print("[VECTORIZE] get names...")
            self.words = np.array(self.vectorize.get_feature_names_out())
            self.all_words = np.hstack([
                np.array([w]*c, dtype='U100') for w, c 
                in zip(self.words[X_b][-50:], X_c[X_b][-50:])
                ])[::-1]
            
        if self.verbose>0:
            print("[VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        # Now plot
        sns.countplot(x=self.all_words, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    def show(self):
        fig, axs = plt.subplots(2, 1)
        self.count_cases(axs[0])
        self.count_words(axs[1])
        plt.show()



if __name__ == "__main__":

    ex_train = ExploreTrain(verbose=1)
    ex_train.show()
    ex_patientnotes = ExplorePatientNotes(verbose=1)
    ex_patientnotes.show()
