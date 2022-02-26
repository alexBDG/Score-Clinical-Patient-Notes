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

    def __init__(self):
        folder = 'nbme-score-clinical-patient-notes'
        self.features_path = os.path.join(folder, 'features.csv')
        self.patient_notes_path = os.path.join(folder, 'patient_notes.csv')
        self.train_path = os.path.join(folder, 'train.csv')


    def load(self):
        """Load all datasets.
        """
        self._load_features()
        self._load_patient_notes()
        self._load_train()


    def merge(self):
        """Merge the three DataFrame to one.
        """
        self.train = self.train.merge(
            self.features, on=['feature_num', 'case_num'], how='left'
        )
        self.train = self.train.merge(
            self.patient_notes, on=['pn_num', 'case_num'], how='left'
        )
        self.train['annotation_length'] = self.train['annotation'].apply(len)


    def _load_features(self):
        """Load features file.
        """
        self.features = pd.read_csv(self.features_path)
        self._apply_correction_on_features()


    def _load_patient_notes(self):
        """Load patient notes file.
        """
        self.patient_notes = pd.read_csv(self.patient_notes_path)


    def _load_train(self):
        """Load train file.
        """
        self.train = pd.read_csv(self.train_path)
        self.train['annotation'] = self.train['annotation'].apply(
            ast.literal_eval
        )
        self.train['location'] = self.train['location'].apply(
            ast.literal_eval
        )
        self._apply_correction_on_train()


    def _apply_correction_on_features(self):
        """Correct some features.

        References
        ----------
        `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
        nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
        """
        self.features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"


    def _apply_correction_on_train(self):
        """Correct some annotations.

        References
        ----------
        `From Kaggle Notebook <https://www.kaggle.com/yasufuminakama/
        nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17>`
        """

        self.train.loc[338, 'annotation'] = ast.literal_eval(
            '[["father heart attack"]]')
        self.train.loc[338, 'location'] = ast.literal_eval(
            '[["764 783"]]')

        self.train.loc[621, 'annotation'] = ast.literal_eval(
            '[["for the last 2-3 months"]]')
        self.train.loc[621, 'location'] = ast.literal_eval(
            '[["77 100"]]')

        self.train.loc[655, 'annotation'] = ast.literal_eval(
            '[["no heat intolerance"], ' + \
            '["no cold intolerance"]]')
        self.train.loc[655, 'location'] = ast.literal_eval(
            '[["285 292;301 312"], ' + \
            '["285 287;296 312"]]')

        self.train.loc[1262, 'annotation'] = ast.literal_eval(
            '[["mother thyroid problem"]]')
        self.train.loc[1262, 'location'] = ast.literal_eval(
            '[["551 557;565 580"]]')

        self.train.loc[1265, 'annotation'] = ast.literal_eval(
            '[[\'felt like he was going to "pass out"\']]')
        self.train.loc[1265, 'location'] = ast.literal_eval(
            '[["131 135;181 212"]]')

        self.train.loc[1396, 'annotation'] = ast.literal_eval(
            '[["stool , with no blood"]]')
        self.train.loc[1396, 'location'] = ast.literal_eval(
            '[["259 280"]]')

        self.train.loc[1591, 'annotation'] = ast.literal_eval(
            '[["diarrhoe non blooody"]]')
        self.train.loc[1591, 'location'] = ast.literal_eval(
            '[["176 184;201 212"]]')

        self.train.loc[1615, 'annotation'] = ast.literal_eval(
            '[["diarrhea for last 2-3 days"]]')
        self.train.loc[1615, 'location'] = ast.literal_eval(
            '[["249 257;271 288"]]')

        self.train.loc[1664, 'annotation'] = ast.literal_eval(
            '[["no vaginal discharge"]]')
        self.train.loc[1664, 'location'] = ast.literal_eval(
            '[["822 824;907 924"]]')

        self.train.loc[1714, 'annotation'] = ast.literal_eval(
            '[["started about 8-10 hours ago"]]')
        self.train.loc[1714, 'location'] = ast.literal_eval(
            '[["101 129"]]')

        self.train.loc[1929, 'annotation'] = ast.literal_eval(
            '[["no blood in the stool"]]')
        self.train.loc[1929, 'location'] = ast.literal_eval(
            '[["531 539;549 561"]]')

        self.train.loc[2134, 'annotation'] = ast.literal_eval(
            '[["last sexually active 9 months ago"]]')
        self.train.loc[2134, 'location'] = ast.literal_eval(
            '[["540 560;581 593"]]')

        self.train.loc[2191, 'annotation'] = ast.literal_eval(
            '[["right lower quadrant pain"]]')
        self.train.loc[2191, 'location'] = ast.literal_eval(
            '[["32 57"]]')

        self.train.loc[2553, 'annotation'] = ast.literal_eval(
            '[["diarrhoea no blood"]]')
        self.train.loc[2553, 'location'] = ast.literal_eval(
            '[["308 317;376 384"]]')

        self.train.loc[3124, 'annotation'] = ast.literal_eval(
            '[["sweating"]]')
        self.train.loc[3124, 'location'] = ast.literal_eval(
            '[["549 557"]]')

        self.train.loc[3858, 'annotation'] = ast.literal_eval(
            '[["previously as regular"], ' + \
            '["previously eveyr 28-29 days"], ' + \
            '["previously lasting 5 days"], ' + \
            '["previously regular flow"]]')
        self.train.loc[3858, 'location'] = ast.literal_eval(
            '[["102 123"], ' + \
            '["102 112;125 141"], ' + \
            '["102 112;143 157"], ' + \
            '["102 112;159 171"]]')

        self.train.loc[4373, 'annotation'] = ast.literal_eval(
            '[["for 2 months"]]')
        self.train.loc[4373, 'location'] = ast.literal_eval(
            '[["33 45"]]')

        self.train.loc[4763, 'annotation'] = ast.literal_eval(
            '[["35 year old"]]')
        self.train.loc[4763, 'location'] = ast.literal_eval(
            '[["5 16"]]')

        self.train.loc[4782, 'annotation'] = ast.literal_eval(
            '[["darker brown stools"]]')
        self.train.loc[4782, 'location'] = ast.literal_eval(
            '[["175 194"]]')

        self.train.loc[4908, 'annotation'] = ast.literal_eval(
            '[["uncle with peptic ulcer"]]')
        self.train.loc[4908, 'location'] = ast.literal_eval(
            '[["700 723"]]')

        self.train.loc[6016, 'annotation'] = ast.literal_eval(
            '[["difficulty falling asleep"]]')
        self.train.loc[6016, 'location'] = ast.literal_eval(
            '[["225 250"]]')

        self.train.loc[6192, 'annotation'] = ast.literal_eval(
            '[["helps to take care of aging mother and in-laws"]]')
        self.train.loc[6192, 'location'] = ast.literal_eval(
            '[["197 218;236 260"]]')

        self.train.loc[6380, 'annotation'] = ast.literal_eval(
            '[["No hair changes"], ' + \
            '["No skin changes"], ' + \
            '["No GI changes"], ' + \
            '["No palpitations"], ' + \
            '["No excessive sweating"]]')
        self.train.loc[6380, 'location'] = ast.literal_eval(
            '[["480 482;507 519"], ' + \
            '["480 482;499 503;512 519"], ' + \
            '["480 482;521 531"], ' + \
            '["480 482;533 545"], ' + \
            '["480 482;564 582"]]')

        self.train.loc[6562, 'annotation'] = ast.literal_eval(
            '[["stressed due to taking care of her mother"], ' + \
            '["stressed due to taking care of husbands parents"]]')
        self.train.loc[6562, 'location'] = ast.literal_eval(
            '[["290 320;327 337"], ' + \
            '["290 320;342 358"]]')

        self.train.loc[6862, 'annotation'] = ast.literal_eval(
            '[["stressor taking care of many sick family members"]]')
        self.train.loc[6862, 'location'] = ast.literal_eval(
            '[["288 296;324 363"]]')

        self.train.loc[7022, 'annotation'] = ast.literal_eval(
            '[["heart started racing and felt numbness for the 1st time in ' + \
                'her finger tips"]]')
        self.train.loc[7022, 'location'] = ast.literal_eval(
            '[["108 182"]]')

        self.train.loc[7422, 'annotation'] = ast.literal_eval(
            '[["first started 5 yrs"]]')
        self.train.loc[7422, 'location'] = ast.literal_eval(
            '[["102 121"]]')

        self.train.loc[8876, 'annotation'] = ast.literal_eval(
            '[["No shortness of breath"]]')
        self.train.loc[8876, 'location'] = ast.literal_eval(
            '[["481 483;533 552"]]')

        self.train.loc[9027, 'annotation'] = ast.literal_eval(
            '[["recent URI"], ' + \
            '["nasal stuffines, rhinorrhea, for 3-4 days"]]')
        self.train.loc[9027, 'location'] = ast.literal_eval(
            '[["92 102"], ' + \
            '["123 164"]]')

        self.train.loc[9938, 'annotation'] = ast.literal_eval(
            '[["irregularity with her cycles"], ' + \
            '["heavier bleeding"], ' + \
            '["changes her pad every couple hours"]]')
        self.train.loc[9938, 'location'] = ast.literal_eval(
            '[["89 117"], ' + \
            '["122 138"], ' + \
            '["368 402"]]')

        self.train.loc[9973, 'annotation'] = ast.literal_eval(
            '[["gaining 10-15 lbs"]]')
        self.train.loc[9973, 'location'] = ast.literal_eval(
            '[["344 361"]]')

        self.train.loc[10513, 'annotation'] = ast.literal_eval(
            '[["weight gain"], ' + \
            '["gain of 10-16lbs"]]')
        self.train.loc[10513, 'location'] = ast.literal_eval(
            '[["600 611"], ' + \
            '["607 623"]]')

        self.train.loc[11551, 'annotation'] = ast.literal_eval(
            '[["seeing her son knows are not real"]]')
        self.train.loc[11551, 'location'] = ast.literal_eval(
            '[["386 400;443 461"]]')

        self.train.loc[11677, 'annotation'] = ast.literal_eval(
            '[["saw him once in the kitchen after he died"]]')
        self.train.loc[11677, 'location'] = ast.literal_eval(
            '[["160 201"]]')

        self.train.loc[12124, 'annotation'] = ast.literal_eval(
            '[["tried Ambien but it didnt work"]]')
        self.train.loc[12124, 'location'] = ast.literal_eval(
            '[["325 337;349 366"]]')

        self.train.loc[12279, 'annotation'] = ast.literal_eval(
            '[["heard what she described as a party later than evening ' + \
                'these things did not actually happen"]]')
        self.train.loc[12279, 'location'] = ast.literal_eval(
            '[["405 459;488 524"]]')

        self.train.loc[12289, 'annotation'] = ast.literal_eval(
            '[["experienced seeing her son at the kitchen table these ' + \
                'things did not actually happen"]]')
        self.train.loc[12289, 'location'] = ast.literal_eval(
            '[["353 400;488 524"]]')

        self.train.loc[13238, 'annotation'] = ast.literal_eval(
            '[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
        self.train.loc[13238, 'location'] = ast.literal_eval(
            '[["293 307"], ["321 331"]]')

        self.train.loc[13297, 'annotation'] = ast.literal_eval(
            '[["without improvement when taking tylenol"], ' + \
            '["without improvement when taking ibuprofen"]]')
        self.train.loc[13297, 'location'] = ast.literal_eval(
            '[["182 221"], ' + \
            '["182 213;225 234"]]')

        self.train.loc[13299, 'annotation'] = ast.literal_eval(
            '[["yesterday"], ["yesterday"]]')
        self.train.loc[13299, 'location'] = ast.literal_eval(
            '[["79 88"], ["409 418"]]')

        self.train.loc[13845, 'annotation'] = ast.literal_eval(
            '[["headache global"], ' + \
            '["headache throughout her head"]]')
        self.train.loc[13845, 'location'] = ast.literal_eval(
            '[["86 94;230 236"], ' + \
            '["86 94;237 256"]]')

        self.train.loc[14083, 'annotation'] = ast.literal_eval(
            '[["headache generalized in her head"]]')
        self.train.loc[14083, 'location'] = ast.literal_eval(
            '[["56 64;156 179"]]')
