python -m pdb pipeline.py --model_name_or_path google/mt5-base --data_dir ~/PrefixTuning_data/cnn_dm/lowdata/ --output_dir ~/PrefixTuning_data/xlsum/try/ref_mt5  --rouge_lang "english"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_eval --per_device_eval_batch_size 6

python pipeline.py --model_name_or_path google/mt5-base --data_dir ~/PrefixTuning_data/cnn_dm/lowdata/ --output_dir ~/PrefixTuning_data/xlsum/try/ref_mt5  --rouge_lang "english"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_predict --per_device_eval_batch_size 6


python pipeline.py --model_name_or_path ~/ckp/mT5_multilingual_XLSum --data_dir ~/data/XLSum_input/individual/amharic/ --output_dir ~/ckp/from_xlsum_ckp/amharic/  --rouge_lang "amharic"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_predict --per_device_eval_batch_size 6
