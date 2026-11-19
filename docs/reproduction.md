# 复现说明

本仓库保存最终实验的实现与汇总结果。完整数据集、分类裁剪、FishDet-M预训练权重和最终权重不随仓库发布，因此仅克隆仓库不能重新训练或推理。这里没有将一次路径整理声称为完整复现。

## 无需模型即可检查

```text
python scripts/summarize_results.py
```

这会重新读取归档JSON，生成8组总体指标表，并核对简历采用的1280端到端基线与两阶段P2方案差值。

## 完整实验的目录

使用Python3.10及适配设备的PyTorch环境，安装`requirements.txt`。默认数据根目录为仓库的`data/`；如使用其他位置，设置环境变量`AQUARIUM_ROOT`。

```text
data/
  classes.txt
  fishdetm_pretrained/
    fishdetm_yolov8s_baseline_best.pt
    fishdetm_yolov8s_p2_best.pt
  ablation/
    yolo7/data.yaml
    yolo7/images/{train,val,test}/
    yolo7/labels/{train,val,test}/
    yolo_fish/data.yaml
    yolo_fish/images/{train,val,test}/
    yolo_fish/labels/{train,val,test}/
    crops/{train,val}/{class_name}/
    runs/classifier/best.pt
outputs/
  1280/runs/{run_name}/weights/best.pt
  640/runs/{run_name}/weights/best.pt
```

`classes.txt`顺序与示例七类YAML一致。先将`configs/`中的两个数据集YAML复制到对应位置，并修改其中的绝对`path`。一类定位器使用统一fish标签，不直接把七类标签用于一类训练。

## 训练与评估命令

```text
python scripts/train_classifier.py
python scripts/train_aquarium_fishdetm_transfer.py --imgsz 1280 --workers 0 --output outputs/1280
python scripts/train_aquarium_fishdetm_transfer.py --imgsz 640 --workers 0 --output outputs/640
python scripts/evaluate_aquarium_two_stage.py --runs outputs/1280/runs --output outputs/1280/evaluation --imgsz 1280
python scripts/evaluate_aquarium_two_stage.py --runs outputs/640/runs --output outputs/640/evaluation --imgsz 640
python scripts/evaluate_aquarium_scale_ap.py
```

训练器将同结构FishDet-M检查点直接交给Ultralytics，避免早期“手动拷权重、训练重建时丢失”的问题。默认100轮、batch2、patience30、seed20260807；分类器采用EfficientNet-B0、ImageNet预训练、类别加权损失及验证macro-F1选模。

训练脚本发现`best.pt`会跳过对应运行，请使用新的输出目录重跑；部分训练中断也可能留下`best.pt`，不能只凭该文件存在判定训练完整。归档结果采用已核验完成的运行。

原评估器中的`small_recall50`是历史辅助口径，阈值覆盖95.37%测试实例，不能用于主要小目标结论。最终尺度分析使用训练集面积四分位阈值和COCOeval分层AP，见`results/scale_stratified_ap.json`。

## 发布版的改动

只对副本中的硬编码目录改为相对目录或`AQUARIUM_ROOT`，抽出原分类器训练函数，并加入结果摘要脚本。未重训模型、重算全套预测或改变归档指标。
