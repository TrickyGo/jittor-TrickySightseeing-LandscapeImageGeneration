# jittor-TrickySightseeing-LandscapeImageGeneration
# Jittor 草图生成风景比赛

## 简介

本项目包含了TrickySightseeing小队的草图生成风景比赛的代码实现。

## 安装 

本项目可在 1 张 A100 上运行，batch_size=10。

## 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

## Part1: Random Sampling Model

### 训练（包含推理）
单卡训练可运行以下命令：
```
bash scripts/train.sh
```
在每一个epoch训完后会进行测试集推理。

### 推理
单卡推理可运行以下命令：
```
bash scripts/test.sh
```
即可load位于'results/saved_models/'中的预训练模型进行测试，并将测试结果保存至'results/'

## Part2: Part-level Style Transfer Model

### 为风格迁移寻找合适的参考图片（具有相同的标签组）
```
bash scripts/find_match.sh
```

### 训练（包含推理）
单卡训练可运行以下命令：
```
bash scripts/train_model_for_transfer.sh
```
在每一个epoch训完后会进行测试集推理。

### 推理（风格迁移）
单卡推理可运行以下命令：
```
bash scripts/transfer.sh
```
即可load位于'results/saved_models/'中的预训练模型进行测试，并将测试结果保存至'results/'
