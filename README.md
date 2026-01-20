# Patient Trust Prediction Pipeline (Modeling + Validation + Calibration + DCA)
本项目基于问卷与人口学特征，构建多分类模型预测“信任水平”（Low / Medium / High），并提供：
- 5 折交叉验证网格搜索（GridSearchCV）
- Bootstrap 内部验证（含 95% CI）
- 外部验证（External Validation）
- 校准曲线（Calibration plot）
- 决策曲线分析（DCA, Decision Curve Analysis）
- DCA 净获益曲线“面积”（AUNBC 类似思想）的统计比较与显著性检验
- 校准指标表（Calibration slope + MSE，内外部对比）
> 代码主体位于：`patient_boruta_feature_selection_calibration_slope.ipynb.ipynb`


## 1. 数据与标签定义

### 1.1 数据文件
默认读取以下 Excel（需放在同目录或修改路径）：
- `内部验证数据集.xlsx`（内部验证数据集）
- `外部验证数据集.xlsx`（外部验证数据集）

读取方式：
- `pd.read_excel(file_path, header=2)`  
表示 Excel 前两行是说明/标题，第三行开始才是列名。

### 1.2 特征列索引（按 Excel 列位置）
代码用“列索引”而非列名来指定特征：
- 分类变量：`cat_indices = [1,3,4,5,6,7,8,9]`
- 连续变量：`cont_indices = [2]`
- 问卷题项：`item_indices = list(range(10, 80))`
- 目标变量（连续得分）：`target_index = 91`
> 注意：这些索引依赖 Excel 的列顺序。若 Excel 结构变化，必须同步更新索引。

### 1.3 多分类标签生成
目标列原本是连续分数，代码会把它分成三类（Low/Medium/High）：
1) 先 Train/Test 划分（80/20）  
2) 只在训练集上计算 1/3 与 2/3 分位数阈值（避免数据泄漏）
3) 用同一组阈值同时切训练集与测试集：
- 0: Low Trust
- 1: Medium Trust
- 2: High Trust


## 2. 模型与特征配置

### 2.1 模型列表
包含 5 类模型，并统一封装为 `Pipeline(preprocess + classifier)`：
- LASSO_LR（elasticnet LogisticRegression）
- SVM（RBF kernel, probability=True）
- Random Forest
- XGBoost（XGBClassifier）
- LightGBM（LGBMClassifier）

### 2.2 每个模型使用的特征（model_feature_config）
每个模型使用一组“列索引特征”，例如：
- LASSO_LR: `[2,71,80,72,...]`
- SVM: `[16,43,37,...]`
- ...
这些索引会被转换成实际列名，用于构建 `X_train_curr / X_test_curr`。


## 3. 预处理策略（Pipeline）

对每个模型的特征列自动拆分为：
- 数值列（continuous + item 等）：`SimpleImputer(mean) + StandardScaler`
- 类别列（demographics 中的分类变量）：`SimpleImputer(most_frequent) + OneHotEncoder`
最终通过 `ColumnTransformer` 合并进入模型。


## 4. 训练：5 折 GridSearchCV
- 交叉验证：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 优化目标：`roc_auc_ovr`（One-vs-Rest 多分类 AUC，macro 口径）
- 产物：`best_models` 字典（每个模型一份最优 estimator）


## 5. 内部验证（Bootstrap 95% CI + 排序表）
对测试集做 Bootstrap（默认 1000 次）计算指标，并给出 95% CI：
- AUC（macro OVR）
- Accuracy
- Macro-F1
- Cohen’s Kappa
- Brier Score（多分类 one-hot 形式）
- Sensitivity / Specificity / PPV / NPV（对每一类做 one-vs-rest，再汇总）
输出：
- 控制台显示结果表
- 导出 Excel：`Table_Internal_Validation_Sorted.xlsx`（按 AUC 降序）


## 6. 内部可视化：Calibration + DCA（按类别绘图）

对 3 个类别分别做 one-vs-rest：
- 校准曲线：`sklearn.calibration.calibration_curve(n_bins=10)`
- DCA：自定义 `calculate_net_benefit()`，计算不同阈值下净获益

输出（每个类别一张图，包含多个模型曲线）：
- `Internal_Fig_Low_Trust.png/.pdf`
- `Internal_Fig_Medium_Trust.png/.pdf`
- `Internal_Fig_High_Trust.png/.pdf`


## 7. 外部验证（External Validation）
读取 `外部验证集.xlsx`，沿用**训练集阈值 bins**切分标签，然后：
- 对每个模型计算同一套 Bootstrap 指标 + 95% CI
- 导出 Excel：`Table_External_Validation_Sorted.xlsx`


## 8. 外部可视化：Calibration + DCA

与内部一致，但数据换成外部验证集：
- `External_Fig_Low_Trust.png/.pdf`
- `External_Fig_Medium_Trust.png/.pdf`
- `External_Fig_High_Trust.png/.pdf`


## 9. DCA 净获益“面积”与统计比较

此部分将 DCA 曲线离散为 0~1（步长 0.01），并计算：
- 正净获益面积（pos area）
- 负净获益面积（neg area）
并以 “Treat-all” 作为 baseline 参照，形成：
- 每模型在每类别的面积柱状图
- 模型两两比较：Bootstrap 生成差值分布 → raw p-value → Holm-Bonferroni 校正
- 校正后 p-value 热力图
- 导出统计表 CSV（每类别一份）


## 10. 校准指标表（Calibration slope + MSE）

对每个类别 one-vs-rest：
- Calibration slope：拟合 `logit(Y) ~ alpha + beta*logit(p_hat)`，返回 beta
- MSE：`mean_squared_error(y_true, y_prob)`

并给出：
- 类别级（Low/Medium/High）内外部 slope/MSE
- Macro 平均（Total Average）

输出：
- 图片表格（学术风格）：`Table_Calibration_*.png/.pdf`
- 原始数据：`Calibration_Metrics_Academic.csv`


## 11. 运行方式

### 11.1 推荐：Jupyter Notebook
1. 安装依赖
2. 把 Excel 数据放到项目根目录（或修改 `file_path` / `EXTERNAL_FILE_PATH`）
3. 从上到下顺序运行

### 11.2 依赖环境
建议 Python >= 3.9

主要依赖：
- pandas, numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn（仅用于 DCA 面积统计热力图）
- openpyxl（读取/写入 Excel）

示例安装：
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl
