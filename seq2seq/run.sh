prefix-tuning train from orig-ckpt:
python   pipeline_prefix_tuning.py  --model_name_or_path google/mt5-base    --data_dir ~/data/XLSum_input/individual/amharic/ --output_dir ~/PrefixTuning_data/xlsum/hp_search/mt5-base/amharic/default  --rouge_lang "amharic"    --predict_with_generate   --per_device_train_batch_size 3 --per_device_eval_batch_size 3  --use_encoder_prefix --use_self_prefix --use_cross_prefix --overwrite_output_dir --n_val 200 --num_train_epochs 20 --logging_steps 3000 --save_steps 3000 --eval_steps 3000 --evaluation_strategy steps --do_train --metric_for_best_model rouge2 --greater_is_better True --learning_rate 5e-5 --preseqlen 200



prefix-tuning test from orig-ckpt(no-prefix):
python  pipeline_prefix_tuning.py  --model_name_or_path google/mt5-base    --data_dir ~/data/XLSum_input/individual/amharic/ --output_dir ~/PrefixTuning_data/xlsum/hp_search/mt5-base/amharic/orig_ckpt_ref  --rouge_lang "amharic"    --predict_with_generate    --do_predict  --per_device_train_batch_size 3 --per_device_eval_batch_size 3   --overwrite_output_dir 


prefix-tuning test from xlsum-ckpt(no-prefix):
python  pipeline_prefix_tuning.py  --model_name_or_path ~/ckp/mT5_multilingual_XLSum/    --data_dir ~/data/XLSum_input/individual/amharic/ --output_dir ~/PrefixTuning_data/xlsum/hp_search/mt5-base/amharic/orig_ckpt_ref  --rouge_lang "amharic"    --predict_with_generate   --do_predict  --per_device_train_batch_size 3 --per_device_eval_batch_size 3   --overwrite_output_dir 


python -m pdb pipeline.py --model_name_or_path google/mt5-base --data_dir ~/PrefixTuning_data/cnn_dm/lowdata/ --output_dir ~/PrefixTuning_data/xlsum/try/ref_mt5  --rouge_lang "english"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_eval --per_device_eval_batch_size 6

python pipeline.py --model_name_or_path google/mt5-base --data_dir ~/PrefixTuning_data/cnn_dm/lowdata/ --output_dir ~/PrefixTuning_data/xlsum/try/ref_mt5  --rouge_lang "english"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_predict --per_device_eval_batch_size 6


python pipeline.py --model_name_or_path ~/ckp/mT5_multilingual_XLSum --data_dir ~/data/XLSum_input/individual/amharic/ --output_dir ~/ckp/from_xlsum_ckp/amharic/  --rouge_lang "amharic"    --predict_with_generate     --length_penalty 0.6     --no_repeat_ngram_size 2     --max_source_length 512     --test_max_target_length 84     --do_predict --per_device_eval_batch_size 6
