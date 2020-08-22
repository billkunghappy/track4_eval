import nltk
import numpy as np
import argparse
import json
import torch.nn as nn
from tqdm import trange

def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True, help="input predict(by model) file")
    parser.add_argument("--output_file_path", required=True, help="output path")
    parser.add_argument("--model_type", required=True, type = int, help="model type, 0:furniture_to, 1:furniture, 2:fashion_to, 3:fashion")
    parser.add_argument("--file_type", required=True, help="model type , furniture_to, furniture, fashion_to, fashion")
    parser.add_argument("--cand_num", type = int, default = 100, help="candidates num, should be 100")
    args = parser.parse_args()
    # Load candidate file and predict file
    all_model_types = ["furniture_to", "furniture", "fashion_to", "fashion"]
    model_type = all_model_types[args.model_type]
    cand_file = "gt_data/cand_{}/{}_{}_dials_retrieval_candidates.json".format(model_type, model_type, args.file_type)
    print("Using " + cand_file +" as candidate file...")
    with open(cand_file,"r") as F:
        for i in F:
            cand_data = json.loads(i)
        F.close()
    #pred_file = "gpt2_dst/data/{}/{}_{}_dials_predict.txt".format(model_type, model_type, args.file_type)
    predict_file = args.input_file_path
    input_data = []
    with open(predict_file,"r") as F:
        for i in F:
            target_response = ""
            if i.find("<EOB>")>=0:
                target_response = i.split("<EOB>")[-1].split("<EOS>")[0]
            else:
                print("------------- no <EOB> ----------------")
            input_data.append(target_response)
    #First deal with the pool, tokenize everything
    print(cand_data.keys())
    cand_pool = cand_data['system_transcript_pool']
    # deal with cand data
    input_cand = []
    for i in cand_data['retrieval_candidates']:
        for j in i['retrieval_candidates']:
            input_cand.append([ cand_pool[k] for k in j['retrieval_candidates']])
    #input cand should be shape = (total_turn, 100)
    if (len(input_cand) != len(input_data)):
        print("ERROR: INPUT DATA AND CANDIDATES HAVE DIFFERRENT LENGTH")
        print(len(input_cand), len(input_data))
        exit(0)

    #specify loss
    #print([tokenizer.decode(i[0]) for i in input_cand[0:1]])
    all_scores = []
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for i in trange(len(input_data)):
        scores=[]
        for j in input_cand[i]:
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(input_data[i])],
                normalize_sentence(j),
                smoothing_function=chencherry.method1
            )
            scores.append(bleu_score)

        all_scores.append(scores)
    
    #dump file
    with open(args.output_file_path, "w") as F:
        json.dump(all_scores, F)
        F.close()


if __name__ == "__main__":
    main()

