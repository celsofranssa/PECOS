# activate venv and set Python path
source venv/bin/activate
export PYTHONPATH=PYTHONPATH:$pwd

model=XLinear
data=AmazonCat-13K
fold_idx=0

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[preprocess] \
  model=$model \
  data=$data \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > preprocess_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[fit] \
  model=$model \
  data=$data \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > fit_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[predict] \
  model=$model \
  data=$data \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > predict_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[eval] \
  model=$model \
  data=$data \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > eval_time.txt

