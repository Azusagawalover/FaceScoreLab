数据审计与独立评价工具

本目录提供来源审计、数据清单、可见属性和19类分割评价。工具不训练模型，不将预测当作人工真值。依赖Python 3.10+、NumPy和Pillow；使用自行配置的虚拟环境运行。

工具
- audit_sources.py：读取模型说明，核对资产、扫描明确限定的缓存目录，记录来源、许可与训练重叠known/unknown。可下载公开元数据，大小上限20MiB，不下载图像或压缩包。
- manifest.py：登记图像/像素哈希，导入已有匿名身份组，生成并验证保持身份及重复图隔离的划分。
- evaluate.py：读取独立标签和预测，计算属性或分割指标，作配对group bootstrap并保存JSON/CSV。
- annotation_guide.txt：真实标注、双审、模糊标签、类别和锁定测试规则。
- run_validation.py与test_*.py：合成输入的正确性和接口检查，不代表模型精度。

以下命令在本目录执行，所有输入与输出路径均为示例：
python audit_sources.py --output ../../work/data-audit --fetch-public-metadata
python manifest.py inventory --root ../../study/images --source phone_cohort_v1 --role unlabeled_candidate --output images.csv
python manifest.py import-identities --input images.csv --identity-map identity_import_private.csv --key-file ../../private/cohort.key --namespace phone_cohort_v1 --origin participant_code --output grouped.csv
python manifest.py split --input grouped.csv --ratios 0.7 0.15 0.15 --seed 20261113 --output split_v1.csv
python manifest.py validate --input split_v1.csv --output split_check.json
python evaluate.py attributes --labels attribute_annotations.csv --predictions candidate_a.csv --thresholds thresholds.json --split validation --bootstrap 2000 --output attribute_metrics.json
python evaluate.py segmentation --labels segmentation_annotations.json --predictions segmentation_predictions.csv --split validation --bootstrap 2000 --output segmentation_metrics.json
python run_validation.py --output ../../work/data-audit/tool-validation

输入约定
清单仅登记指定目录，身份组来源为官方元数据或自愿参与者编号，不通过人脸识别猜测。映射密钥需稳定保存且不附入公开源码；原始身份映射不写入输出。无身份元数据时可显式--allow-image-level，但只能说明较弱的图片/重复图隔离。追加样本使用--previous和--additions-train-only保持旧划分。
CSV为UTF-8；JSON可为对象数组或{"records":[...]}。属性采用image_id+attribute长表，target为0/1/unknown；CelebA -1在导入时显式改为0。分割每图一行，掩膜需同尺寸、同坐标、同mask_schema，不自动调整真值。
真实标签dataset_role=accuracy_set，label_source为official_annotation或human_double_reviewed；双审保留不同reviewer ID及一致/仲裁状态。现有运行照片为functional_fixture，合成检查需显式--allow-functional-fixtures，结果保留functional_fixture_not_accuracy标记。
thresholds.json记录version、selection_split=validation及逐属性阈值；比较模型可分别指定--compare-thresholds，不能在测试集重调阈值。

输出解释
指标带输入SHA256、样本/组数量、覆盖率、零分母处理、配对差值与区间。independence_verified=false表示声明字段不能替代来源核查。--split test记录locked_test_accessed=true，不意味着测试结果可以用于调参。
F1、准确率、覆盖率和mIoU分别解释；工具检查通过不证明评分、属性或分割在真实照片上的精度。独立标签尚未取得时，保留未评测状态。
