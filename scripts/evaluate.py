#!/usr/bin/env python3
"""
    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into
    the structured DST output.

    We then run the main DST Evaluation script to get results.
"""
import argparse
import json
from scripts.convert import parse_flattened_results_from_file
from scripts.evaluate_dst import evaluate_from_flat_list
from absl import app, flags
import nltk
import numpy as np

def open_txt(input_path):
    with open(input_path, "r") as F:
        all_responses = F.readlines()
        responses=[]
        has_EOB=True
        for i in all_responses:
            if i.find("<EOB>")<0:
                print("no response\n",i)
                responses.append("")
            else:
                responses.append(i.split("<EOB>")[1].split(" \n")[0].split("<EOS>")[0])
        F.close()
    return responses

def open_json(input_path):
    with open(input_path, "r") as F:
        for i in F:
            retrieval = json.loads(i)
        F.close()
    return retrieval

def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())

def evaluate_response_generation(gt_responses, model_responses):
    """Evaluates response generation using the raw data and model predictions.
    """
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for gt_response, response in zip(gt_responses, model_responses):
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [normalize_sentence(gt_response)],
            normalize_sentence(response),
            smoothing_function=chencherry.method1
        )
        bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)

def evaluate_response_retrieval(gt_responses, model_scores, use_bleu = False):
    """Evaluates response retrieval using the raw data and model predictions.
    """
    # NOTE: Update this later to include gt_index for candidates.
    gt_ranks = []
    for model_datum in model_scores:
        gt_score = model_datum[0]
        if use_bleu:
            gt_ranks.append(np.sum(np.array(model_datum) > gt_score) + 1)
        else:
            gt_ranks.append(np.sum(np.array(model_datum) < gt_score) + 1)
    gt_ranks = np.array(gt_ranks)
    return {
        "r1": np.mean(gt_ranks <= 1),
        "r5": np.mean(gt_ranks <= 5),
        "r10": np.mean(gt_ranks <= 10),
        "mean": np.mean(gt_ranks),
        "mrr": np.mean(1 / gt_ranks)
    }

if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target, line-separated format (.txt)')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--input_path_retrieval',
                        help='path for model retrieval output, line-separated format (.json)')
    parser.add_argument('--output_path_report',
                        help='path for saving evaluation summary (.json)')
    parser.add_argument('--bleu_retr',
                        action="store_true",
                        default=False,
                        help='if using bleu for retrieval evaluation, use it')
    parser.add_argument('--skip_bleu',
                        action="store_true",
                        default=False,
                        help='if skip_bleu')
    parser.add_argument('--skip_retr',
                        action="store_true",
                        default=False,
                        help='if skip retrieval eval')
    parser.add_argument('--skip_belief',
                        action="store_true",
                        default=False,
                        help='if skip_belief')

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    if args.bleu_retr:
        input_path_retrieval = args.input_path_retrieval.replace("scores", "scores_bleu")
    else:
        input_path_retrieval = args.input_path_retrieval
    output_path_report = args.output_path_report
    
    target_responses = open_txt(input_path_target)
    predict_responses = open_txt(input_path_predicted)

    if input_path_retrieval is not None:
        predict_retrieval = open_json(input_path_retrieval)
    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(input_path_target)
    list_predicted = parse_flattened_results_from_file(input_path_predicted)

    # Evaluate
    report = {}
    if not args.skip_belief:
        report["Belief"] = evaluate_from_flat_list(list_target, list_predicted)
    if not args.skip_bleu:
        report["Response"] = {"BLEU":evaluate_response_generation(target_responses, predict_responses)}
    if input_path_retrieval is not None and not args.skip_retr:
        report["Response"]["Retrieval"] = evaluate_response_retrieval(None, predict_retrieval, args.bleu_retr)
    print(report)
    
    
    # Save report
    with open(output_path_report, 'w') as f_out:
        json.dump(report, f_out)
