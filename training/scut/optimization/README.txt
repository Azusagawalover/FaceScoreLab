SCUT回归优化与结果复核

实验记录
2026-11-13的固定方案完成R0～R5六配置初筛，并对R0与R4作seed17/42/2026确认，共10次完整训练。部署候选依据验证集选择R4 seed2026，当前App使用同一权重。结果见results/r4-seed2026-v2；run.json、metrics.json、training_log.json、逐图CSV和three_seed_comparison.json保留原始记录。

R4验证MAE=0.271313；实际共74轮，最佳为后部微调第27轮。官方fold1历史1100图的未裁剪MAE=0.276170、RMSE=0.363175、Pearson=0.847771。旧部署固定模型MAE=0.298498，两者MAE差为0.022329。旧固定模型与本实验R0的种子均值不是同一对照，不可混用。

依赖与输入
Python 3.12、torch 2.6.0、torchvision 0.21.0、ONNX及onnxruntime 1.20.1；CPU可运行。Spearman使用平均秩实现，不依赖scipy。
运行需显式提供registered_plan.json、原数据目录和development_manifest.csv。开发清单只含3960张train与440张validation，测试样本不会进入训练或导出。所有新输出应放在独立目录。

固定配置
共同设置：MobileNetV3 Small + LayerNorm + 128维回归头，ImageNet V1初始化，前9个features模块固定，BN统计固定，batch64、AdamW、weight_decay=0.01。头学习率0.0008；tail头/骨干学习率0.00012/0.00003。
R0：基线，head最多100轮/早停16；tail最多16轮/早停6。
R1：tail上限60轮、早停8。
R2：R1增加ReduceLROnPlateau，factor=0.5、patience=3。
R3：R2的MSE替换为Huber(delta=0.3)。
R4：R2增加原图及两种固定轻度增强视图，仅tail使用；每轮每张原图抽取一个视图。增强为水平翻转、±7度旋转、亮度与对比度0.9～1.1；种子由实验seed、文件名和视图号决定，验证图不增强。
R5：从原图/关键点构建未夹边的方形框，越界填RGB128，再缩到224；属于独立预处理对照，未部署为当前默认。
head最小改善1e-5；R0 tail为1e-5，R1～R5 tail为1e-4。PyTorch patience=3表示容忍3个连续bad epoch，第4个降低学习率。

命令入口
以下占位符需换为对应文件或新输出目录，项目根目录执行：
python training/scut/optimization/runner.py prepare-manifest --source <完整划分CSV> --output <开发清单CSV>
python training/scut/optimization/runner.py train --plan <计划JSON> --config R4 --seed 2026 --data-dir <数据目录> --dev-manifest <开发清单CSV> --output <新运行目录> --cache-dir <缓存目录> --max-hours 3 --threads 4
python training/scut/optimization/runner.py export --run <运行目录> --data-dir <数据目录> --dev-manifest <开发清单CSV> --threads 4

恢复与缓存
--resume要求配置、代码、输入清单及输出身份一致；--stop-after-batches/--stop-after-epochs在完整边界暂停。last.pt保存模型、优化器、scheduler、各RNG、阶段、游标和最佳状态；best.pt保存验证选中模型。原子提交不保留半步优化器状态。--smoke为小规模实现检查，不能进入发布锁。
缓存身份包括开发清单、实际输入内容、预处理版本、增强规则、前段权重、dtype和依赖版本，缓存文件另有SHA256。历史channels-last与连续NCHW的浮点路径差异已单独记录，不能把这类数值差异计作优化收益。
--max-hours是单run累计上限；实际完整阶段成本还包含准备、导出与中断工作。

候选锁定与评价
export仅接受完成任务，以最多20张验证图检查PyTorch/ONNX差值<1e-4，不读取测试图。
lock绑定候选checkpoint、ONNX、验证结果与清单哈希；evaluate-test先校验锁，再读取1100张历史测试标签。smoke、未完成任务或哈希不一致时拒绝评价。同一结果不覆盖不同实验。
summary工具summarize_results.py的screen/confirm仅使用开发结果；final读取已冻结评价。配方与部署seed按验证结果确定，不能用测试结果重选。
三种子报告分别列均值、样本标准差及逐种子结果。配对bootstrap按同一图片同时重采样，默认5000次、seed20261113、95%百分位区间；区间只针对固定模型与图片，不覆盖训练随机性或未知同人相关。各seed绝对误差均值不是集成预测。

实现检查与范围
保留test_runner.py、test_summarize_results.py等工具检查代码。合成输入、恢复一致性和ONNX数值核对只能验证实现，不能作为SCUT准确率、属性F1、分割mIoU或真机验收。完整计划、数据和历史依赖不随Android构建自动执行，历史JSON及其哈希不作重写。
