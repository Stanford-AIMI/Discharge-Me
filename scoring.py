import os
import json
import numpy as np
import pandas as pd

import evaluate
from rouge_score import rouge_scorer


def calculate_scores(generated, reference, metrics):
    if not metrics:
        raise ValueError("No metrics specified for scoring.")
    print("Beginning scoring...")

    scores = {}
    for metric in metrics:
        scores[metric] = {"discharge_instructions": [], "brief_hospital_course": []}

    # initialize scorers
    if "bleu" in metrics:
        bleuScorer = evaluate.load("bleu")
    if "rouge" in metrics:
        rougeScorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    if "bertscore" in metrics:
        bertScorer = evaluate.load("bertscore")
    if "gbleu" in metrics:
        gbleuScorer = evaluate.load("google_bleu")
    if "sacrebleu" in metrics:
        sacrebleuScorer = evaluate.load("sacrebleu")
    if "meteor" in metrics:
        meteorScorer = evaluate.load("meteor")

    def calculate_score(row_A, row_B):
        if "bleu" in metrics:
            temp = bleuScorer.compute(
                references=[row_A["discharge_instructions"]],
                predictions=[row_B["discharge_instructions"]],
            )
            scores["bleu"]["discharge_instructions"].append(temp["bleu"])
            temp = bleuScorer.compute(
                references=[row_A["brief_hospital_course"]],
                predictions=[row_B["brief_hospital_course"]],
            )
            scores["bleu"]["brief_hospital_course"].append(temp["bleu"])
        if "rouge" in metrics:
            temp = rougeScorer.score(
                target=row_A["discharge_instructions"],
                prediction=row_B["discharge_instructions"],
            )
            scores["rouge"]["discharge_instructions"].append(
                [
                    temp["rouge1"].fmeasure,
                    temp["rouge2"].fmeasure,
                    temp["rougeL"].fmeasure,
                ]
            )
            temp = rougeScorer.score(
                row_A["brief_hospital_course"], row_B["brief_hospital_course"]
            )
            scores["rouge"]["brief_hospital_course"].append(
                [
                    temp["rouge1"].fmeasure,
                    temp["rouge2"].fmeasure,
                    temp["rougeL"].fmeasure,
                ]
            )
        if "bertscore" in metrics:
            temp = bertScorer.compute(
                references=[row_A["discharge_instructions"]],
                predictions=[row_B["discharge_instructions"]],
                lang="en",
            )
            scores["bertscore"]["discharge_instructions"].append(temp["f1"][0])
            temp = bertScorer.compute(
                references=[row_A["brief_hospital_course"]],
                predictions=[row_B["brief_hospital_course"]],
                lang="en",
            )
            scores["bertscore"]["brief_hospital_course"].append(temp["f1"][0])
        if "gbleu" in metrics:
            temp = gbleuScorer.compute(
                references=[row_A["discharge_instructions"]],
                predictions=[row_B["discharge_instructions"]],
            )
            scores["gbleu"]["discharge_instructions"].append(temp["google_bleu"])
            temp = gbleuScorer.compute(
                references=[row_A["brief_hospital_course"]],
                predictions=[row_B["brief_hospital_course"]],
            )
            scores["gbleu"]["brief_hospital_course"].append(temp["google_bleu"])
        if "sacrebleu" in metrics:
            temp = sacrebleuScorer.compute(
                references=[row_A["discharge_instructions"]],
                predictions=[row_B["discharge_instructions"]],
            )
            scores["sacrebleu"]["discharge_instructions"].append(temp["score"])
            temp = sacrebleuScorer.compute(
                references=[row_A["brief_hospital_course"]],
                predictions=[row_B["brief_hospital_course"]],
            )
            scores["sacrebleu"]["brief_hospital_course"].append(temp["score"])
        if "meteor" in metrics:
            temp = meteorScorer.compute(
                references=[row_A["discharge_instructions"]],
                predictions=[row_B["discharge_instructions"]],
            )
            scores["meteor"]["discharge_instructions"].append(temp["meteor"])
            temp = meteorScorer.compute(
                references=[row_A["brief_hospital_course"]],
                predictions=[row_B["brief_hospital_course"]],
            )
            scores["meteor"]["brief_hospital_course"].append(temp["meteor"])

        # print progress
        current_row = i + 1
        if current_row % 100 == 0:
            print(f"Processed {current_row}/{len(generated)} samples.", flush=True)

    generated.set_index("hadm_id", drop=False, inplace=True)
    reference.set_index("hadm_id", drop=False, inplace=True)

    for i, row_gen in enumerate(generated.to_dict("records")):
        row_ref = reference.loc[row_gen["hadm_id"]].to_dict()
        calculate_score(row_ref, row_gen)
        # print(i)

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
            [sample[0] for sample in scores["rouge"]["discharge_instructions"]]
        )
        rouge_2_discharge_instructions = np.mean(
            [sample[1] for sample in scores["rouge"]["discharge_instructions"]]
        )
        rouge_l_discharge_instructions = np.mean(
            [sample[2] for sample in scores["rouge"]["discharge_instructions"]]
        )
        rouge_1_brief_hospital_course = np.mean(
            [sample[0] for sample in scores["rouge"]["brief_hospital_course"]]
        )
        rouge_2_brief_hospital_course = np.mean(
            [sample[1] for sample in scores["rouge"]["brief_hospital_course"]]
        )
        rouge_l_brief_hospital_course = np.mean(
            [sample[2] for sample in scores["rouge"]["brief_hospital_course"]]
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
    if "gbleu" in metrics:
        gbleu_discharge_instructions = np.mean(
            scores["gbleu"]["discharge_instructions"]
        )
        gbleu_brief_hospital_course = np.mean(scores["gbleu"]["brief_hospital_course"])
        leaderboard["gbleu"] = np.mean(
            [gbleu_discharge_instructions, gbleu_brief_hospital_course]
        )
    if "sacrebleu" in metrics:
        sacrebleu_discharge_instructions = np.mean(
            scores["sacrebleu"]["discharge_instructions"]
        )
        sacrebleu_brief_hospital_course = np.mean(
            scores["sacrebleu"]["brief_hospital_course"]
        )
        leaderboard["sacrebleu"] = np.mean(
            [sacrebleu_discharge_instructions, sacrebleu_brief_hospital_course]
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
print("Done.")

scores = calculate_scores(
    generated, reference, metrics=["bleu", "rouge", "gbleu", "sacrebleu", "meteor"]
)
leaderboard = compute_overall_score(scores)

with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
    score_file.write(json.dumps(leaderboard))
