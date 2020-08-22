#!/bin/bash
TYPE=$1
POSTFIX=$2
PATH_DIR=$(realpath .)

# Evaluate (furniture, non-multimodal)
if [ "$TYPE" = "0" ]
then
	python -m scripts.evaluate \
	    --input_path_target="${PATH_DIR}"/gt_data/furniture_to/furniture_devtest_dials_target.txt \
	    --input_path_predicted="${PATH_DIR}"/results/furniture_to/furniture_devtest_dials_predicted${POSTFIX}.txt \
	    --input_path_retrieval="${PATH_DIR}"/results/furniture/furniture_devtest_retrieve_scores${POSTFIX}.json \
	    --output_path_report="${PATH_DIR}"/results/furniture_to/furniture_devtest_dials_report${POSTFIX}.json \
	    --bleu_retr \
	    --skip_belief
fi

# Evaluate (furniture, multi-modal)
if [ "$TYPE" = "1" ]
then
	python -m scripts.evaluate \
	    --input_path_target="${PATH_DIR}"/gt_data/furniture/furniture_devtest_dials_target.txt \
	    --input_path_predicted="${PATH_DIR}"/results/furniture/furniture_devtest_dials_predicted${POSTFIX}.txt \
	    --output_path_report="${PATH_DIR}"/results/furniture/furniture_devtest_dials_report${POSTFIX}.json \
	    --input_path_retrieval="${PATH_DIR}"/results/furniture/furniture_devtest_retrieve_scores${POSTFIX}.json \
	    --bleu_retr \
	    --skip_belief
fi

# Evaluate (Fashion, non-multimodal)
if [ "$TYPE" = "2" ]
then
	python -m scripts.evaluate \
	    --input_path_target="${PATH_DIR}"/gt_data/fashion_to/fashion_devtest_dials_target.txt \
	    --input_path_predicted="${PATH_DIR}"/results/fashion_to/fashion_devtest_dials_predicted${POSTFIX}.txt \
	    --input_path_retrieval="${PATH_DIR}"/results/furniture/furniture_devtest_retrieve_scores${POSTFIX}.json \
	    --output_path_report="${PATH_DIR}"/results/fashion_to/fashion_devtest_dials_report${POSTFIX}.json \
	    --bleu_retr \
	    --skip_belief
fi

# Evaluate (Fashion, multi-modal)
if [ "$TYPE" = "3" ]
then
	python -m scripts.evaluate \
	    --input_path_target="${PATH_DIR}"/gt_data/fashion/fashion_devtest_dials_target.txt \
	    --input_path_predicted="${PATH_DIR}"/results/fashion/fashion_devtest_dials_predicted${POSTFIX}.txt \
	    --input_path_retrieval="${PATH_DIR}"/results/furniture/furniture_devtest_retrieve_scores${POSTFIX}.json \
	    --output_path_report="${PATH_DIR}"/results/fashion/fashion_devtest_dials_report${POSTFIX}.json \
	    --bleu_retr \
	    --skip_belief
fi
