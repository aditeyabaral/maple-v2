import itertools
import pandas as pd

token_selector_versions = ["v1", "v2"]

selector_models = [
    "bert-base-cased",
    "roberta-base",
    "distilbert-base-cased",
    # "xlm-roberta-base",
]

selector_mode = [
    "whole-word",
    "token",
]

use_perplexity = [True, False]
use_grammar_checker = [True, False]

# find all combinations of the above

combinations = list(itertools.product(
    token_selector_versions,
    selector_models,
    selector_mode,
    use_perplexity,
    use_grammar_checker
))

df = pd.DataFrame(combinations, columns=[
    "selector_model",
    "token_selector_version",
    "selector_mode",
    "use_perplexity",
    "use_grammar_checker"
])

# if token selector version is v1, then set selector mode to None
df_v1 = df[df["token_selector_version"] == "v1"]
df_v1["selector_mode"] = None

df_v1.to_excel("combinations_v1.xlsx", index=False)
