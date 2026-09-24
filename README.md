# HKDMAP

HKDMAP 是一个异构知识驱动的双通道多任务关联预测模型，联合利用关联网络结构与外部生物知识，对以下三类关系进行二分类预测。

## 目录与脚本

```text
HKDMAP/
├── configs/
│   └── ablation_hgt.yaml        # 数据路径、模型参数及训练配置
├── scripts/
│   └── main_pipeline.py        # 主程序：五折交叉验证、最终训练与测试
├── src/
│   ├── prepare_dataset.py      # 数据清洗、划分、负采样及训练图构建
│   ├── data_utils.py           # 数据读取、节点编号及关系边处理
│   ├── external_node_features.py # 构建微生物、疾病和药物的外部特征
│   ├── metapath2vec.py          # 元路径随机游走与节点嵌入预训练
│   ├── similarity_features.py  # 构建三类节点的相似度矩阵与相似图
│   ├── hgt.py                  # 结构通道的异构图 Transformer
│   ├── gat.py                  # 生物知识通道的图注意力网络
│   ├── dual_view_model.py      # 双通道编码与特征融合
│   ├── multi_task_decoder.py   # 三任务预测头与联合损失计算
│   ├── config_utils.py         # 配置读取、目录创建与结果保存
│   └── __init__.py             
└── requirements.txt            
```

## 数据

数据统一放在项目根目录的 `data/` 下，默认结构如下：

```text
data/
├── raw/                              
│   ├── microbe-disease.txt            # MD
│   ├── drug-microbe.txt               # DM
│   └── drug-disease.txt               # DD
├── processed/                        
│   ├── microbe_disease.txt            # 清洗后的微生物—疾病关联
│   ├── drug_microbe.txt               # 清洗后的药物—微生物关联
│   ├── drug_disease_filt.txt           # 过滤后的药物—疾病关联
│   └── cv5_folds/                     
└── external/                        
    ├── taxonomy_features.pkl         # 微生物分类学特征
    ├── mesh_features.pkl             # 疾病 MeSH 层级特征
    ├── drug_fingerprints.pkl          # 药物 Morgan 指纹
    ├── feature_dims.json             # 各类特征维度
    ├── DS_pymeshsim_wang.npy          # 疾病语义相似度矩阵
    ├── DS_pymeshsim_wang_ids.txt      # 矩阵对应的疾病 ID 列表
    └── DS_pymeshsim_wang_meta.json    # 相似度计算方法信息
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

