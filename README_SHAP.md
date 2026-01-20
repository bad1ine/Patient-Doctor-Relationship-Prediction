# SHAP Explainability for Patient Trust Models

本项目在完成多分类模型训练后，对最优模型进行 SHAP 可解释性分析，并自动导出每个模型在三分类（Low/Medium/High）上的：
- SHAP Beeswarm（summary plot）
- Feature Importance Bar（plot_type="bar"）
> 代码主体位于：`shap.ipynb`


## 1. 输入数据与模型来源

### 1.1 数据文件
默认读取：
- `患者筛选数据0110.xlsx`（header=2）

并使用与训练流水线一致的索引配置：
- `cat_indices / cont_indices / item_indices / target_index`
- `model_feature_config`

### 1.2 模型训练
Notebook 前半段会执行与主流程一致的：
- Train/Test split
- 训练集分位数阈值切分三分类标签
- 5 折 GridSearchCV
最终得到：
- `best_models`（dict：模型名 → 最佳 Pipeline）


## 2. SHAP 分析策略

对每个模型：
1) 从 Pipeline 拆解出：
- `preprocessor = model_pipeline.named_steps['preprocessor']`
- `classifier = model_pipeline.named_steps['classifier']`
2) 对输入特征做相同预处理变换（OneHot + Scaler 等）
3) 根据模型类型选择 SHAP explainer：

- 树模型（XGBoost / RandomForest / LightGBM）  
  使用 `shap.TreeExplainer(classifier)`
- 线性模型（Logistic / LASSO_LR）  
  使用 `shap.LinearExplainer(classifier, masker=Independent(X_train_trans))`
- 核方法（SVM 等）  
  使用 `shap.KernelExplainer`  
  为控制计算量，默认对测试集采样 `X_test_trans[:50]`


## 3. 输出图像

对每个模型 × 每个类别（0/1/2）生成两类图：
- `SHAP_Beeswarm_{Model}_{Class}.png/.pdf`
- `SHAP_Bar_{Model}_{Class}.png/.pdf`

类别映射：
- 0: Low Trust
- 1: Medium Trust
- 2: High Trust


## 4. 运行方式

### 4.1 推荐：Jupyter Notebook
1) 安装依赖  
2) 将 `患者筛选数据0110.xlsx` 放到项目根目录（或修改路径）  
3) 从上到下顺序运行

### 4.2 依赖环境
建议 Python >= 3.9

主要依赖：
- pandas, numpy
- scikit-learn
- xgboost
- lightgbm
- shap
- matplotlib
- openpyxl

示例安装：
```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib openpyxl
