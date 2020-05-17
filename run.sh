python3 pool_feats.py --input_file data/feat_all.pkl --output_file data/feat_pooled.pkl --feat_dim 45 --step 5 --max_step 2500
python3 script_train.py --seq_dim 512 --in_dim 45 --atten_size 16 --batch_size 64 --keep_proba 1 --model_dir model/iaan/ --record_file outputs/iaan.json --feat_dir data/feat_pooled.pkl
python3 script_test.py  --result_file outputs/iaan.json --feat_dir data/feat_pooled.pkl

