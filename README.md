# 实验命令参数

# --- IMDb 组 ---
python main.py --dataset imdb --model_type rnn --epochs 20
python main.py --dataset imdb --model_type lstm --epochs 20
python main.py --dataset imdb --model_type gru --epochs 20
python main.py --dataset imdb --model_type transformer --epochs 20
python main.py --dataset imdb --model_type bert --epochs 20 --lr 2e-5 --batch_size 16

# --- AG News 组 ---
python main.py --dataset ag_news --model_type rnn --epochs 20
python main.py --dataset ag_news --model_type lstm --epochs 20
python main.py --dataset ag_news --model_type gru --epochs 20
python main.py --dataset ag_news --model_type transformer --epochs 20
python main.py --dataset ag_news --model_type bert --epochs 20 --lr 2e-5 --batch_size 16

# --- SST-2 组 ---
python main.py --dataset sst2 --model_type rnn --epochs 20
python main.py --dataset sst2 --model_type lstm --epochs 20
python main.py --dataset sst2 --model_type gru --epochs 20
python main.py --dataset sst2 --model_type transformer --epochs 20
python main.py --dataset sst2 --model_type bert --epochs 20 --lr 2e-5 --batch_size 16