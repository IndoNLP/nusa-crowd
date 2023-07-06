import pandas as pd
from pathlib import Path

def create_saintek_and_soshum_dataset(saintek_question: Path, soshum_question: Path, saintek_score: Path, soshum_score: Path):
    # creating dataframes
    saintek_question = pd.read_csv(saintek_question, sep=",", header="infer")
    saintek_score = pd.read_csv(saintek_score, sep=",", header="infer")
    soshum_question = pd.read_csv(soshum_question, sep=",", header="infer")
    soshum_score = pd.read_csv(soshum_score, sep=",", header="infer")

    # renaming
    saintek_question.rename(columns={'jawaban': 'kunci_jawaban'}, inplace='true')
    soshum_question.rename(columns={'jawaban': 'kunci_jawaban'}, inplace='true')

    # joining
    saintek_joined = pd.merge(saintek_question, saintek_score, how='left', on = 'type-problem')
    soshum_joined = pd.merge(soshum_question, soshum_score, how='left', on = 'type-problem')

    # adding answer key
    saintek_joined = pd.concat([saintek_joined, build_answer_key_score_dataframe(saintek_question)], ignore_index=True)
    soshum_joined = pd.concat([soshum_joined, build_answer_key_score_dataframe(soshum_question)], ignore_index=True)

    # re-case-ing (kebab to snake)
    saintek_joined.rename(columns={'type-problem': 'type_problem'}, inplace=True)
    soshum_joined.rename(columns={'type-problem': 'type_problem'}, inplace=True)

    # concatenate
    saintek_and_soshum = pd.concat([saintek_joined, soshum_joined], ignore_index=True).reset_index()

    return saintek_and_soshum


def build_answer_key_score_dataframe(question: pd.DataFrame):
    return pd.DataFrame({
        'type-problem' : question['type-problem'],
        'pertanyaan' : question['pertanyaan'],
        'kunci_jawaban' : question['kunci_jawaban'],
        'jawaban' : question['kunci_jawaban'],
        'score' : [5]*question.shape[0],
    })
