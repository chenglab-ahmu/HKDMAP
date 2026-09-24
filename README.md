# HKDMAP

HKDMAP 是一个异构知识驱动的双通道多任务关联预测模型，联合利用关联网络结构与外部生物知识，对以下三类关系进行二分类预测。

## 目录与脚本

```text
HKDMAP/
├── configs/
│   └── ablation_hgt.yaml        # 模型配置
├── scripts/
│   └── main_pipeline.py        # 主程序
├── src/
│   ├── prepare_dataset.py    
│   ├── data_utils.py           
│   ├── external_node_features.py # 外部特征
│   ├── metapath2vec.py          
│   ├── similarity_features.py  # 相似图
│   ├── hgt.py                  
│   ├── gat.py                
│   ├── dual_view_model.py      # 双通道编码与特征融合
│   ├── multi_task_decoder.py   # 三任务预测
│   ├── config_utils.py         
│   └── __init__.py             
└── requirements.txt            
```

## 数据

数据统一放在项目根目录的 `data/` 下，默认结构如下：

```text
data/
├── raw/                               #原始数据                            
├── processed/                        
│   ├── microbe_disease.txt            # 处理后的MDA
│   ├── drug_microbe.txt               # 处理后的DMA
│   ├── drug_disease_filt.txt          # 处理后的DDA
│   └── cv5_folds/                     
└── external/                        
    ├── taxonomy_features.pkl         # 微生物特征
    ├── mesh_features.pkl             # 疾病特征
    ├── drug_fingerprints.pkl          # 药物特征
    ├── feature_dims.json             
    ├── DS_pymeshsim_wang.npy          # 疾病语义相似度矩阵
    └── DS_pymeshsim_wang_ids.txt      
```

## 运行


```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 数据预处理与划分
python src/prepare_dataset.py --config configs/ablation_hgt.yaml --cv

# 3. 构建外部节点特征
python src/external_node_features.py --config configs/ablation_hgt.yaml

# 4. 训练与评估
python scripts/main_pipeline.py --config configs/ablation_hgt.yaml
```

