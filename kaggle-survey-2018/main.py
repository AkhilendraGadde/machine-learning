"""
@author: Akhilendra Gadde
"""

import numpy as np
import pandas as pd


def tweak_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    
    """
    Preprocess pipeline for kaggle-survey-2018
    :param raw: Input DataFrame
    :return data: Outputs DataFrame
    """
    
    columns = {
        'Q1': 'gender',
        'Q2': 'age',
        'Q3': 'country',
        'Q4': 'education',
        'Q5': 'major',
        'Q8': 'years_exp',
        'Q9': 'compensation',
        'Q10': 'ml_used',
        'Q16_Part_1': 'python',
        'Q16_Part_2': 'r',
        'Q16_Part_3': 'sql',
        # 'Q17': 'pref_lang',    
        'Q6': 'target'
    }
    raw = raw.iloc[1:].filter(columns).rename(columns=columns).reset_index(drop=True)
    
    raw.gender = raw.gender.apply(str.lower)
    raw.age = raw.age.apply(lambda x: x.replace("+", '').split('-')[0]).astype(int)
    raw.years_exp = raw.years_exp.fillna('0').apply(lambda x: x.replace("+", '').split('-')[0]).astype(float)
    raw.python = raw.python.fillna(0).replace("Python", 1).astype(int)
    raw.r = raw.r.fillna(0).replace("R", 1).astype(int)
    raw.sql = raw.sql.fillna(0).replace("SQL", 1).astype(int)
    
    raw.compensation = raw \
        .compensation \
        .fillna('0') \
        .apply(lambda x: x.replace('+', '').replace(',', '').replace("500000", "500").split('-')[0]) \
        .replace("I do not wish to disclose my approximate yearly compensation", 0) \
        .astype(int) \
        .mul(1_000)

    # Country
    country = [
        raw.country == 'United States of America',
        raw.country == 'India'
    ]
    c_values = ["usa", "india"]
    raw.country = np.select(country, c_values)

    # Major
    raw.major = raw.major.where(raw.major.isin(raw.major.value_counts().index[:3]), 'Other')
    major = [
        raw.major == 'Computer science (software engineering, etc.)',
        raw.major == 'Engineering (non-computer focused)',
        raw.major == 'Mathematics or statistics'
    ]
    m_values = ["cs", "eng", "stat"]
    raw.major = np.select(major, m_values, None)
    raw.major = raw.major.fillna("other").astype(object)

    # Education
    edu_conditions = [
        raw.education == 'Master’s degree',
        raw.education == 'Bachelor’s degree',
        raw.education == 'Doctoral degree',
        raw.education == 'Some college/university study without earning a bachelor’s degree',
        raw.education == 'Professional degree',
        raw.education == 'I prefer not to answer',
        raw.education == 'No formal education past high school'
    ]
    edu_values = [
        '30',
        '20',
        '40',
        '10',
        '35',
        None,
        '5'
    ]
    raw.education = np.select(edu_conditions, edu_values, None)
    raw.education = raw.education.fillna(0).astype(int)

    # ML Used
    ml_conditions = [
        raw.ml_used == 'I do not know',
        raw.ml_used == 'No (we do not use ML methods)',
        raw.ml_used == 'We are exploring ML methods (and may one day put a model into production)',
        raw.ml_used == 'We have well established ML methods (i.e., models in production for more than 2 years)',
        raw.ml_used == 'We recently started using ML methods (i.e., models in production for less than 2 years)',
        raw.ml_used == 'We use ML methods for generating insights (but do not put working models into production)',
    ]
    ml_values = [
        'No',
        'No',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
    ]
    raw.ml_used = np.select(ml_conditions, ml_values, None)
    raw.ml_used = raw.ml_used.fillna('No').astype(object)

    data = raw \
        .query("country.isin(['usa', 'india']) and target.isin(['Data Scientist', 'Software Engineer'])") \
        .reset_index(drop=True).copy(deep=True)

    # print(data.head())
    return data


if __name__ == '__main__':

    file = r'datasets/multipleChoiceResponses.csv'
    df = pd.read_csv(file, low_memory=False)
    tweak_dataset(df).to_pickle(r'datasets/output.p.gz', compression='gzip')
