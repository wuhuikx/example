pip install -r requirements.txt
cd data
python vocab_downloader.py --type=bert-base-uncased
export VOCAB_FILE=bert-base-uncased-vocab.txt
bash parallel_create_pretraining_data.sh ../miniwiki
