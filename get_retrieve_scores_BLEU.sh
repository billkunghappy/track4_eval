#!/bin/bash
TYPE=$1
POSTFIX=$2
CUDA=$3
PATH_DIR=$(realpath .)

# Evaluate (furniture, non-multimodal)
if [ "$TYPE" = "0" ]
then
	CUDA_VISIBLE_DEVICES=$CUDA scripts/eval_retrieve_bleu.py \
		--input_file_path="${PATH_DIR}"/results/furniture_to/furniture_devtest_dials_predicted${POSTFIX}.txt \
		--output_file_path="${PATH_DIR}"/results/furniture_to/furniture_devtest_retrieve_scores_bleu${POSTFIX}.json \
		--model_type $TYPE \
		--file_type devtest
		#--use_tokenidx
fi

# Evaluate (furniture, multi-modal)
if [ "$TYPE" = "1" ]
then
	CUDA_VISIBLE_DEVICES=$CUDA python3 scripts/eval_retrieve_bleu.py \
		--input_file_path="${PATH_DIR}"/results/furniture/furniture_devtest_dials_predicted${POSTFIX}.txt \
		--output_file_path="${PATH_DIR}"/results/furniture/furniture_devtest_retrieve_scores_bleu${POSTFIX}.json \
		--model_type $TYPE \
		--file_type devtest
		#--use_tokenidx
fi

# Evaluate (Fashion, non-multimodal)
if [ "$TYPE" = "2" ]
then
	CUDA_VISIBLE_DEVICES=$CUDA python3 scripts/eval_retrieve_bleu.py \
		--input_file_path="${PATH_DIR}"/results/fashion_to/fashion_devtest_dials_predicted${POSTFIX}.txt \
		--output_file_path="${PATH_DIR}"/results/fashion_to/fashion_devtest_retrieve_scores_bleu${POSTFIX}.json \
		--model_type $TYPE \
		--file_type devtest
		#--use_tokenidx
fi

# Evaluate (Fashion, multi-modal)
if [ "$TYPE" = "3" ]
then
	CUDA_VISIBLE_DEVICES=$CUDA python3 scripts/eval_retrieve_bleu.py \
		--input_file_path="${PATH_DIR}"/results/fashion/fashion_devtest_dials_predicted${POSTFIX}.txt \
		--output_file_path="${PATH_DIR}"/results/fashion/fashion_devtest_retrieve_scores_bleu${POSTFIX}.json \
		--model_type $TYPE \
		--file_type devtest
		#--use_tokenidx
fi
