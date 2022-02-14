import pandas as pd
from nltk.tokenize import word_tokenize

def process_dataset(self, dataset_path):
    df = pd.read_json(dataset_path)
    df = df.drop_duplicates(subset=["passage", "poem"])
    df["tokens"] = df["passage"].apply(lambda x: word_tokenize(x))
    ner_tags = list()
    for i in range(df.shape[0]):
        indices = df["indices"][i]
        length = len(df["tokens"][i])
        ner_tag = ['O' for _ in range(length)]
        for idx in indices:
            ner_tag[idx] = 'W'
        ner_tags.append(ner_tag)
    df["ner_tags"] = ner_tags
    return df