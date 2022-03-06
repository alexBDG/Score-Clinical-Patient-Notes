import os
import ast
import pandas as pd



class DataLoader():
    """Manage all data loading and cleaning.

    References
    ----------
    `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
    nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
    """

    def __init__(self, *args, **kwargs):
        self.folder = 'nbme-score-clinical-patient-notes'
        self.features_path = os.path.join(self.folder, 'features.csv')
        self.patient_notes_path = os.path.join(self.folder, 'patient_notes.csv')


    def load(self):
        """Load all datasets.
        """
        self._load_features()
        self._apply_correction_on_features()
        self._load_patient_notes()
        self._load_data()
        self._apply_correction_on_data()


    def merge(self):
        """Merge the three DataFrame to one.
        """
        self.data = self.data.merge(
            self.features, on=['feature_num', 'case_num'], how='left'
        )
        self.data = self.data.merge(
            self.patient_notes, on=['pn_num', 'case_num'], how='left'
        )


    def _load_data(self):
        """Load train file.
        """
        self.data = pd.read_csv(self.data_path)


    def _load_features(self):
        """Load features file.
        """
        self.features = pd.read_csv(self.features_path)
        self._apply_correction_on_features()
        self.features_to_index = {
            v: k for k, v in self.features['feature_text'].to_dict().items()
        }


    def _load_patient_notes(self):
        """Load patient notes file.
        """
        self.patient_notes = pd.read_csv(self.patient_notes_path)


    def _apply_correction_on_features(self):
        """Correct some features.

        References
        ----------
        `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
        nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
        """
        self.features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"


    def _apply_correction_on_data(self):
        """Correct some annotations.
        """
        pass



class TrainLoader(DataLoader):
    """Manage all data loading and cleaning.

    References
    ----------
    `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
    nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = os.path.join(self.folder, 'train.csv')


    def merge(self):
        """Merge the three DataFrame to one.
        """
        super().merge()
        self.data['annotation_length'] = self.data['annotation'].apply(len)


    def _load_data(self):
        """Load train file.
        """
        super()._load_data()
        self.data = pd.read_csv(self.data_path)
        self.data['annotation'] = self.data['annotation'].apply(
            ast.literal_eval
        )
        self.data['location'] = self.data['location'].apply(
            ast.literal_eval
        )


    def _apply_correction_on_data(self):
        """Correct some annotations.

        References
        ----------
        `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
        nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
        """

        self.data.loc[338, 'annotation'] = ast.literal_eval(
            '[["father heart attack"]]')
        self.data.loc[338, 'location'] = ast.literal_eval(
            '[["764 783"]]')

        self.data.loc[621, 'annotation'] = ast.literal_eval(
            '[["for the last 2-3 months"]]')
        self.data.loc[621, 'location'] = ast.literal_eval(
            '[["77 100"]]')

        self.data.loc[655, 'annotation'] = ast.literal_eval(
            '[["no heat intolerance"], ' + \
            '["no cold intolerance"]]')
        self.data.loc[655, 'location'] = ast.literal_eval(
            '[["285 292;301 312"], ' + \
            '["285 287;296 312"]]')

        self.data.loc[1262, 'annotation'] = ast.literal_eval(
            '[["mother thyroid problem"]]')
        self.data.loc[1262, 'location'] = ast.literal_eval(
            '[["551 557;565 580"]]')

        self.data.loc[1265, 'annotation'] = ast.literal_eval(
            '[[\'felt like he was going to "pass out"\']]')
        self.data.loc[1265, 'location'] = ast.literal_eval(
            '[["131 135;181 212"]]')

        self.data.loc[1396, 'annotation'] = ast.literal_eval(
            '[["stool , with no blood"]]')
        self.data.loc[1396, 'location'] = ast.literal_eval(
            '[["259 280"]]')

        self.data.loc[1591, 'annotation'] = ast.literal_eval(
            '[["diarrhoe non blooody"]]')
        self.data.loc[1591, 'location'] = ast.literal_eval(
            '[["176 184;201 212"]]')

        self.data.loc[1615, 'annotation'] = ast.literal_eval(
            '[["diarrhea for last 2-3 days"]]')
        self.data.loc[1615, 'location'] = ast.literal_eval(
            '[["249 257;271 288"]]')

        self.data.loc[1664, 'annotation'] = ast.literal_eval(
            '[["no vaginal discharge"]]')
        self.data.loc[1664, 'location'] = ast.literal_eval(
            '[["822 824;907 924"]]')

        self.data.loc[1714, 'annotation'] = ast.literal_eval(
            '[["started about 8-10 hours ago"]]')
        self.data.loc[1714, 'location'] = ast.literal_eval(
            '[["101 129"]]')

        self.data.loc[1929, 'annotation'] = ast.literal_eval(
            '[["no blood in the stool"]]')
        self.data.loc[1929, 'location'] = ast.literal_eval(
            '[["531 539;549 561"]]')

        self.data.loc[2134, 'annotation'] = ast.literal_eval(
            '[["last sexually active 9 months ago"]]')
        self.data.loc[2134, 'location'] = ast.literal_eval(
            '[["540 560;581 593"]]')

        self.data.loc[2191, 'annotation'] = ast.literal_eval(
            '[["right lower quadrant pain"]]')
        self.data.loc[2191, 'location'] = ast.literal_eval(
            '[["32 57"]]')

        self.data.loc[2553, 'annotation'] = ast.literal_eval(
            '[["diarrhoea no blood"]]')
        self.data.loc[2553, 'location'] = ast.literal_eval(
            '[["308 317;376 384"]]')

        self.data.loc[3124, 'annotation'] = ast.literal_eval(
            '[["sweating"]]')
        self.data.loc[3124, 'location'] = ast.literal_eval(
            '[["549 557"]]')

        self.data.loc[3858, 'annotation'] = ast.literal_eval(
            '[["previously as regular"], ' + \
            '["previously eveyr 28-29 days"], ' + \
            '["previously lasting 5 days"], ' + \
            '["previously regular flow"]]')
        self.data.loc[3858, 'location'] = ast.literal_eval(
            '[["102 123"], ' + \
            '["102 112;125 141"], ' + \
            '["102 112;143 157"], ' + \
            '["102 112;159 171"]]')

        self.data.loc[4373, 'annotation'] = ast.literal_eval(
            '[["for 2 months"]]')
        self.data.loc[4373, 'location'] = ast.literal_eval(
            '[["33 45"]]')

        self.data.loc[4763, 'annotation'] = ast.literal_eval(
            '[["35 year old"]]')
        self.data.loc[4763, 'location'] = ast.literal_eval(
            '[["5 16"]]')

        self.data.loc[4782, 'annotation'] = ast.literal_eval(
            '[["darker brown stools"]]')
        self.data.loc[4782, 'location'] = ast.literal_eval(
            '[["175 194"]]')

        self.data.loc[4908, 'annotation'] = ast.literal_eval(
            '[["uncle with peptic ulcer"]]')
        self.data.loc[4908, 'location'] = ast.literal_eval(
            '[["700 723"]]')

        self.data.loc[6016, 'annotation'] = ast.literal_eval(
            '[["difficulty falling asleep"]]')
        self.data.loc[6016, 'location'] = ast.literal_eval(
            '[["225 250"]]')

        self.data.loc[6192, 'annotation'] = ast.literal_eval(
            '[["helps to take care of aging mother and in-laws"]]')
        self.data.loc[6192, 'location'] = ast.literal_eval(
            '[["197 218;236 260"]]')

        self.data.loc[6380, 'annotation'] = ast.literal_eval(
            '[["No hair changes"], ' + \
            '["No skin changes"], ' + \
            '["No GI changes"], ' + \
            '["No palpitations"], ' + \
            '["No excessive sweating"]]')
        self.data.loc[6380, 'location'] = ast.literal_eval(
            '[["480 482;507 519"], ' + \
            '["480 482;499 503;512 519"], ' + \
            '["480 482;521 531"], ' + \
            '["480 482;533 545"], ' + \
            '["480 482;564 582"]]')

        self.data.loc[6562, 'annotation'] = ast.literal_eval(
            '[["stressed due to taking care of her mother"], ' + \
            '["stressed due to taking care of husbands parents"]]')
        self.data.loc[6562, 'location'] = ast.literal_eval(
            '[["290 320;327 337"], ' + \
            '["290 320;342 358"]]')

        self.data.loc[6862, 'annotation'] = ast.literal_eval(
            '[["stressor taking care of many sick family members"]]')
        self.data.loc[6862, 'location'] = ast.literal_eval(
            '[["288 296;324 363"]]')

        self.data.loc[7022, 'annotation'] = ast.literal_eval(
            '[["heart started racing and felt numbness for the 1st time in ' + \
                'her finger tips"]]')
        self.data.loc[7022, 'location'] = ast.literal_eval(
            '[["108 182"]]')

        self.data.loc[7422, 'annotation'] = ast.literal_eval(
            '[["first started 5 yrs"]]')
        self.data.loc[7422, 'location'] = ast.literal_eval(
            '[["102 121"]]')

        self.data.loc[8876, 'annotation'] = ast.literal_eval(
            '[["No shortness of breath"]]')
        self.data.loc[8876, 'location'] = ast.literal_eval(
            '[["481 483;533 552"]]')

        self.data.loc[9027, 'annotation'] = ast.literal_eval(
            '[["recent URI"], ' + \
            '["nasal stuffines, rhinorrhea, for 3-4 days"]]')
        self.data.loc[9027, 'location'] = ast.literal_eval(
            '[["92 102"], ' + \
            '["123 164"]]')

        self.data.loc[9938, 'annotation'] = ast.literal_eval(
            '[["irregularity with her cycles"], ' + \
            '["heavier bleeding"], ' + \
            '["changes her pad every couple hours"]]')
        self.data.loc[9938, 'location'] = ast.literal_eval(
            '[["89 117"], ' + \
            '["122 138"], ' + \
            '["368 402"]]')

        self.data.loc[9973, 'annotation'] = ast.literal_eval(
            '[["gaining 10-15 lbs"]]')
        self.data.loc[9973, 'location'] = ast.literal_eval(
            '[["344 361"]]')

        self.data.loc[10513, 'annotation'] = ast.literal_eval(
            '[["weight gain"], ' + \
            '["gain of 10-16lbs"]]')
        self.data.loc[10513, 'location'] = ast.literal_eval(
            '[["600 611"], ' + \
            '["607 623"]]')

        self.data.loc[11551, 'annotation'] = ast.literal_eval(
            '[["seeing her son knows are not real"]]')
        self.data.loc[11551, 'location'] = ast.literal_eval(
            '[["386 400;443 461"]]')

        self.data.loc[11677, 'annotation'] = ast.literal_eval(
            '[["saw him once in the kitchen after he died"]]')
        self.data.loc[11677, 'location'] = ast.literal_eval(
            '[["160 201"]]')

        self.data.loc[12124, 'annotation'] = ast.literal_eval(
            '[["tried Ambien but it didnt work"]]')
        self.data.loc[12124, 'location'] = ast.literal_eval(
            '[["325 337;349 366"]]')

        self.data.loc[12279, 'annotation'] = ast.literal_eval(
            '[["heard what she described as a party later than evening ' + \
                'these things did not actually happen"]]')
        self.data.loc[12279, 'location'] = ast.literal_eval(
            '[["405 459;488 524"]]')

        self.data.loc[12289, 'annotation'] = ast.literal_eval(
            '[["experienced seeing her son at the kitchen table these ' + \
                'things did not actually happen"]]')
        self.data.loc[12289, 'location'] = ast.literal_eval(
            '[["353 400;488 524"]]')

        self.data.loc[13238, 'annotation'] = ast.literal_eval(
            '[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
        self.data.loc[13238, 'location'] = ast.literal_eval(
            '[["293 307"], ["321 331"]]')

        self.data.loc[13297, 'annotation'] = ast.literal_eval(
            '[["without improvement when taking tylenol"], ' + \
            '["without improvement when taking ibuprofen"]]')
        self.data.loc[13297, 'location'] = ast.literal_eval(
            '[["182 221"], ' + \
            '["182 213;225 234"]]')

        self.data.loc[13299, 'annotation'] = ast.literal_eval(
            '[["yesterday"], ["yesterday"]]')
        self.data.loc[13299, 'location'] = ast.literal_eval(
            '[["79 88"], ["409 418"]]')

        self.data.loc[13845, 'annotation'] = ast.literal_eval(
            '[["headache global"], ' + \
            '["headache throughout her head"]]')
        self.data.loc[13845, 'location'] = ast.literal_eval(
            '[["86 94;230 236"], ' + \
            '["86 94;237 256"]]')

        self.data.loc[14083, 'annotation'] = ast.literal_eval(
            '[["headache generalized in her head"]]')
        self.data.loc[14083, 'location'] = ast.literal_eval(
            '[["56 64;156 179"]]')



class TestLoader(DataLoader):
    """Manage all data loading and cleaning.

    References
    ----------
    `From Kaggle Notebook <https://www.kaggle.com/manojprabhaakr/
    nbme-deberta-large-baseline-inference?scriptVersionId=87542390&cellId=16>`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = os.path.join(self.folder, 'test.csv')
        self.submission_path = os.path.join(self.folder,'sample_submission.csv')


    def load(self):
        """Load all datasets.
        """
        super().load()
        self._load_submission()


    def _load_submission(self):
        """Load train file.
        """
        self.submission = pd.read_csv(self.submission_path)
