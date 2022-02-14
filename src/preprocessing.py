import pandas as pd

def filterDataFrame(df, perplexity_limit=1500, full=True):
  df.drop_duplicates(inplace=True, ignore_index=True, subset=['poem', 'haiku', "ppl-gpt2"])
  filtered_df = df[df["ppl-gpt2"] <= perplexity_limit]
  filtered_df.reset_index(drop=True, inplace=True)
  filtered_df.rename(columns={"poem": "passage", "haiku": "poem"}, inplace=True)
  if not full:
    return filtered_df

  filtered_df = filtered_df.loc[filtered_df.groupby('passage')['ppl-gpt2'].idxmin()]
  filtered_df.reset_index(drop=True, inplace=True)
  return filtered_df

df = pd.read_json('data/data.json')
#df.rename(columns={"poem": "passage", "haiku": "poem"}, inplace=True)
# df = filterDataFrame(df, full=True)
print(df.shape)
print(df.head())
#df.to_json('data/data.json')
