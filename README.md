![Python](https://img.shields.io/badge/python-3.9-blue.svg)



# Score-Clinical-Patient-Notes
From Kaggle competition: NBME - Score Clinical Patient Notes



## Installation
 1. Download data on the official competition page: [kaggle](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/data)

 2. Extract the compressed folder in this git repository under the name: *nbme-score-clinical-patient-notes*

Then you should have the correct working space, such as:
```
Score-Clinical-Patient-Notes
│   README.md
│   .gitignore
|   requirements.txt
|   exploration.py
│
└───nbme-score-clinical-patient-notes
    │   features.csv
    │   patient_notes.csv
    |   sample_submission.csv
    |   test.csv
    |   train.csv
```

 3. Install all python libraries using `requirements.txt`.

```console
usr@home:~$ pip install -r requirements.txt
```



## Data exploration
Start to explore the input data...
```console
usr@home:~$ python exploration.py
```
