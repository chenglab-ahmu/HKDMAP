# 1. 数据预处理 + holdout + 5-fold
python src/prepare_dataset.py --config configs/ablation_hgt.yaml

# 2. 外部节点特征构建
python src/external_node_features.py --config configs/ablation_hgt.yaml

# 3. 主训练：5-fold CV + final holdout
python scripts/main_pipeline.py --config configs/ablation_hgt.yaml