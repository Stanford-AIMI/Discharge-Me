import os
import json
import numpy as np
import pandas as pd

import evaluate

from bleu import Bleu
from rouge import Rouge
from bertscore import BertScore  #


def calculate_scores(generated, reference, metrics):
    if not metrics:
        raise ValueError("No metrics specified for scoring.")
    print("Beginning scoring...")

    scores = {}
    for metric in metrics:
        scores[metric] = {"discharge_instructions": [], "brief_hospital_course": []}

    # initialize scorers
    if "bleu" in metrics:
        bleuScorer = Bleu()
        print("bleuScorer initialized")
    if "rouge" in metrics:
        scores["rouge"]["discharge_instructions"] = [[], [], []]
        scores["rouge"]["brief_hospital_course"] = [[], [], []]
        rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
        print("rougeScorer initialized")
    if "bertscore" in metrics:
        bertScorer = BertScore()
        print("bertScorer initialized")
    if "meteor" in metrics:
        meteorScorer = evaluate.load("meteor")
        print("meteorScorer initialized")

    def calculate_scores(rows_ref, rows_gen):
        if "bleu" in metrics:
            temp = bleuScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["bleu"]["discharge_instructions"].extend(temp)
            temp = bleuScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["bleu"]["brief_hospital_course"].extend(temp)
        if "rouge" in metrics:
            temp = rougeScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["rouge"]["discharge_instructions"][0].extend(
                    temp['rouge1'],
            )
            scores["rouge"]["discharge_instructions"][1].extend(
                    temp['rouge2'],
            )
            scores["rouge"]["discharge_instructions"][2].extend(
                    temp['rougeL'],
            )
            temp = rougeScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["rouge"]["brief_hospital_course"][0].extend(
                    temp['rouge1'],
            )
            scores["rouge"]["brief_hospital_course"][1].extend(
                    temp['rouge2'],
            )
            scores["rouge"]["brief_hospital_course"][2].extend(
                    temp['rougeL'],
            )
        if "bertscore" in metrics:
            temp = bertScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["bertscore"]["discharge_instructions"].extend(temp)
            temp = bertScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["bertscore"]["brief_hospital_course"].extend(temp)
        if "meteor" in metrics:
            temp = meteorScorer.compute(
                references=rows_ref["discharge_instructions"].tolist(),
                predictions=rows_gen["discharge_instructions"].tolist(),
            )
            scores["meteor"]["discharge_instructions"].append(temp["meteor"])
            temp = meteorScorer.compute(
                references=rows_ref["discharge_instructions"].tolist(),
                predictions=rows_gen["discharge_instructions"].tolist(),
            )
            scores["meteor"]["brief_hospital_course"].append(temp["meteor"])

        # print progress
        current_row = i + 128
        if current_row % 128 == 0:
            print(f"Processed {current_row}/{len(generated)} samples.", flush=True)

    reference.set_index("hadm_id", drop=False, inplace=True)
    generated.set_index("hadm_id", drop=False, inplace=True)

    batch_size = 128
    for i in range(0, len(generated), batch_size):
        rows_ref = reference[i : i + batch_size]
        rows_gen = generated[i : i + batch_size]
        calculate_scores(rows_ref=rows_ref, rows_gen=rows_gen)

    print(f"Processed {len(generated)}/{len(generated)} samples.", flush=True)
    print("Done.")
    return scores


def compute_overall_score(scores):
    print("Computing overall score...")
    leaderboard = {}

    metrics = list(scores.keys())

    if "bleu" in metrics:
        bleu_discharge_instructions = np.mean(scores["bleu"]["discharge_instructions"])
        bleu_brief_hospital_course = np.mean(scores["bleu"]["brief_hospital_course"])
        leaderboard["bleu"] = np.mean(
            [bleu_discharge_instructions, bleu_brief_hospital_course]
        )
    if "rouge" in metrics:
        rouge_1_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][0]
        )
        rouge_2_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][1]
        )
        rouge_l_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][2]
        )
        rouge_1_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][0]
        )
        rouge_2_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][1]
        )
        rouge_l_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][2]
        )

        leaderboard["rouge1"] = np.mean(
            [rouge_1_discharge_instructions, rouge_1_brief_hospital_course]
        )
        leaderboard["rouge2"] = np.mean(
            [rouge_2_discharge_instructions, rouge_2_brief_hospital_course]
        )
        leaderboard["rougel"] = np.mean(
            [rouge_l_discharge_instructions, rouge_l_brief_hospital_course]
        )
    if "bertscore" in metrics:
        bertscore_discharge_instructions = np.mean(
            scores["bertscore"]["discharge_instructions"]
        )
        bertscore_brief_hospital_course = np.mean(
            scores["bertscore"]["brief_hospital_course"]
        )
        leaderboard["bertscore"] = np.mean(
            [bertscore_discharge_instructions, bertscore_brief_hospital_course]
        )
    if "meteor" in metrics:
        meteor_discharge_instructions = np.mean(
            scores["meteor"]["discharge_instructions"]
        )
        meteor_brief_hospital_course = np.mean(
            scores["meteor"]["brief_hospital_course"]
        )
        leaderboard["meteor"] = np.mean(
            [meteor_discharge_instructions, meteor_brief_hospital_course]
        )

    # normalize sacrebleu to be between 0 and 1
    for key in leaderboard.keys():
        if key == "sacrebleu":
            leaderboard[key] = leaderboard[key] / 100

    overall_score = np.mean(list(leaderboard.values()))
    leaderboard["overall"] = overall_score

    print("Done.")
    return leaderboard


reference_dir = os.path.join("/app/input/", "ref")
generated_dir = os.path.join("/app/input/", "res")
score_dir = "/app/output/"

print("Reading generated texts...")
generated = pd.read_csv(
    os.path.join(generated_dir, "submission.csv"), keep_default_na=False
)
reference = pd.read_csv(
    os.path.join(reference_dir, "discharge_target.csv"), keep_default_na=False
)

# covert all elements to string
generated["discharge_instructions"] = generated["discharge_instructions"].astype(str)
reference["discharge_instructions"] = reference["discharge_instructions"].astype(str)

generated["brief_hospital_course"] = generated["brief_hospital_course"].astype(str)
reference["brief_hospital_course"] = reference["brief_hospital_course"].astype(str)

# convert to single-line strings by removing newline characters
generated["discharge_instructions"] = generated["discharge_instructions"].str.replace(
    "\n", " "
)
reference["discharge_instructions"] = reference["discharge_instructions"].str.replace(
    "\n", " "
)

generated["brief_hospital_course"] = generated["brief_hospital_course"].str.replace(
    "\n", " "
)
reference["brief_hospital_course"] = reference["brief_hospital_course"].str.replace(
    "\n", " "
)

# convert all hadm_id to int
generated["hadm_id"] = generated["hadm_id"].astype(int)
reference["hadm_id"] = reference["hadm_id"].astype(int)

# check for invalid submissions
if not generated.shape[0] == reference.shape[0]:
    raise ValueError(
        "Submission does not contain the correct number of rows. Please check your submission file."
    )

if list(generated["hadm_id"].unique()) != list(reference["hadm_id"].unique()):
    missing_hadm_ids = set(reference["hadm_id"].unique()) - set(
        generated["hadm_id"].unique()
    )
    extra_hadm_ids = set(generated["hadm_id"].unique()) - set(
        reference["hadm_id"].unique()
    )
    print(f"Missing hadm_ids: {missing_hadm_ids}")
    print(f"Extra hadm_ids: {extra_hadm_ids}")
    raise ValueError(
        "Submission does not contain all hadm_ids from the test set. Please check your submission file."
    )

if not generated["hadm_id"].nunique() == len(generated):
    raise ValueError(
        "Submission contains duplicate hadm_ids. Please check your submission file."
    )

generated = generated.sort_values(by="hadm_id")
reference = reference.sort_values(by="hadm_id")
print("Done.")

scores = calculate_scores(
    generated,
    reference,
    metrics=["bleu", "rouge", "bertscore", "meteor"],
)

leaderboard = compute_overall_score(scores)

with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
    score_file.write(json.dumps(leaderboard))
