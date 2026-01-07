# 安装环境

项目所用环境为根目录下的 `requirements.txt`

执行以下命令安装

```bash
pip install -r requirements.txt
```

# 实验命令参数

> 在根目录运行

## --- IMDb 组 ---

```python
python main.py --dataset imdb --model_type rnn --epochs 20
python main.py --dataset imdb --model_type lstm --epochs 20
python main.py --dataset imdb --model_type gru --epochs 20
python main.py --dataset imdb --model_type transformer --epochs 20
python main.py --dataset imdb --model_type bert --epochs 20 --lr 2e-5 --batch_size 16
```

## --- AG News 组 ---

```python
python main.py --dataset ag_news --model_type rnn --epochs 20
python main.py --dataset ag_news --model_type lstm --epochs 20
python main.py --dataset ag_news --model_type gru --epochs 20
python main.py --dataset ag_news --model_type transformer --epochs 20
python main.py --dataset ag_news --model_type bert --epochs 20 --lr 2e-5 --batch_size 16
```

## --- SST-2 组 ---

```python
python main.py --dataset sst2 --model_type rnn --epochs 20
python main.py --dataset sst2 --model_type lstm --epochs 20
python main.py --dataset sst2 --model_type gru --epochs 20
python main.py --dataset sst2 --model_type transformer --epochs 20
python main.py --dataset sst2 --model_type bert --epochs 20 --lr 2e-5 --batch_size 16
```