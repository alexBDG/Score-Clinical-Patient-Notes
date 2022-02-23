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
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ML imports.
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



class ExploreFile():

    def __init__(self, verbose=0):
        self.df = pd.read_csv(self.file)#.head(10000)
        print(self.df)
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
        self.compute_tfidf()


    def count_words(self, ax):

        if self.verbose>0:
            print("[COUNT VECTORIZE] starting...")
            start = time.time()
        self.vectorize = CountVectorizer()
        X = self.vectorize.fit_transform(self.df["pn_history"])
        if self.verbose>0:
            print("[COUNT VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        if self.verbose>0:
            print("[COUNT VECTORIZE] counting & sorting...")
            start = time.time()
        X = X.toarray()
        X_c = X.sum(0)
        X_b = np.argsort(X_c)
        if self.verbose>0:
            print("[COUNT VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        if self.verbose>0:
            print("[COUNT VECTORIZE] get names...")
        words = np.array(self.vectorize.get_feature_names_out())
        all_words = np.hstack([
            np.array([w]*c, dtype='U100') for w, c
            in zip(words[X_b][-50:], X_c[X_b][-50:])
        ])[::-1]
        if self.verbose>0:
            print("[COUNT VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        # Now plot
        sns.countplot(x=all_words, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    def compute_tfidf(self):

        if self.verbose>0:
            print("[TF-IDF VECTORIZE] starting...")
            start = time.time()
        vectorizer = TfidfVectorizer(
            stop_words='english', ngram_range=(1,1),
            max_df=.6, min_df=.01
        )
        X = vectorizer.fit_transform([
            '\n\n'.join(self.df[self.df['case_num']==case]["pn_history"].tolist())
            for case in self.df['case_num'].unique()
        ])
        if self.verbose>0:
            print("[TF-IDF VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))

        if self.verbose>0:
            print("[TF-IDF VECTORIZE] densifing...")
            start = time.time()
        feature_names = vectorizer.get_feature_names_out()
        dense = X.todense()
        denselist = dense.tolist()
        self.df_tfidf = pd.DataFrame(denselist, columns=feature_names).transpose()
        self.df_tfidf.columns = [
            'case_num: '+str(case) for case in self.df['case_num'].unique()
        ]
        if self.verbose>0:
            print("[TF-IDF VECTORIZE] elapsed time {:.2f}s".format(time.time()-start))


    def word_clood(self, ax, case):

        # Get the mask of this picture
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 1300 ** 2
        mask = 255 * mask.astype(int)

        # Generate word cloud picture
        wordcloud = WordCloud(
            width=3000, height=3000, background_color='white',
            stopwords=set(STOPWORDS), mask=mask, max_words=500,
        )
        wordcloud = wordcloud.generate_from_frequencies(
            self.df_tfidf['case_num: '+str(case)]
        )

        # Display it
        ax.set_title('case_num: '+str(case))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")


    def show(self):
        fig = plt.figure()
        gs = GridSpec(3, 5, figure=fig)
        self.count_cases(fig.add_subplot(gs[0, :2]))
        self.count_words(fig.add_subplot(gs[0, 2:]))
        for case in self.df['case_num'].unique():
            if case<5:
                self.word_clood(fig.add_subplot(gs[1, case]), case)
            else:
                self.word_clood(fig.add_subplot(gs[2, case-5]), case)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    ex_train = ExploreTrain(verbose=1)
    ex_train.show()
    ex_patientnotes = ExplorePatientNotes(verbose=1)
    ex_patientnotes.show()
