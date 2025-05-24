# Kaggle 练习项目

## 目录结构说明

📁 notebooks/       - Jupyter Notebook练习文件（按竞赛主题分类）
📁 data/           - 数据集（遵循Kaggle输入规范）
│   ├── raw/       - 原始数据（只读）
│   └── processed/ - 预处理后的数据
📁 src/            - 可复用工具函数
📁 config/         - 配置文件（数据库连接、路径参数等）
📁 docs/           - 实验文档与报告

## 使用规范
1. 每个notebook文件名需包含日期和主题（例：20240520_Titanic_EDA.ipynb）
2. 数据集版本通过git-lfs管理
3. 工具函数需带单元测试