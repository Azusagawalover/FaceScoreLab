SCUT-FBP5500 评分回归

当前默认模型
当前App使用optimization目录的R4 seed2026模型。MobileNetV3 Small仅以torchvision ImageNet1K V1权重初始化，再进行本项目任务训练；未使用第三方SCUT颜值检查点初始化。网络接LayerNorm、128维回归头，冻结前9个features模块及BN统计，先训练头，再微调features9～12。
训练/验证/历史测试为3960/440/1100；按验证原始MAE选模。R4实际完成74轮，选择后部微调第27轮，验证MAE=0.271313。未裁剪历史测试MAE=0.276170、RMSE=0.363175、Pearson=0.847771。详细配置见optimization/README.txt和optimization/results/r4-seed2026-v2。

数据来源与预处理
SCUT官方第1折训练部分4400张，其中440张按评分分箱和seed42划为验证集，其余3960张训练。官方1100张测试图像在冻结选模后评价；它已经被历史实验观察，不能称为全新盲测。
official保留官方划分标签。fty/mty/ftw/mtw与镜像AF/AM/CF/CM有确定性名称映射，5500个名称和评分核对后再使用；不采用镜像另给的划分。
图像按SCUT非商业研究条件取得，源码不包含完整图像集。
官方来源：https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
镜像：https://huggingface.co/datasets/MnLgt/scut-fbp5500

MediaPipe使用与App相同的face_landmarker.task及最大脸规则；faceCrop v1框最大边乘1.45、中心y上移size×0.06，并夹到图像边界。输入为224×224 RGB、ImageNet标准化的连续float32 NCHW。预测回归值采用1～5目标尺度，网络本身不强制限制输出范围；显示时裁剪并映射为1～10。

基线复现
Python 3.12与本目录requirements.txt；可使用CPU版PyTorch。Android构建不需要Python。
在项目根目录执行以下命令会复现早期基线，不会自动得到当前R4方案：
python training/scut/download_inputs.py --output training/scut/work
python training/scut/prepare_dataset.py --output training/scut/work
python training/scut/train_regressor.py --output training/scut/work

早期基线使用MSE、AdamW、冻结BN、batch64，头最多100轮，后部微调最多16轮。results保留其训练日志、官方划分、逐图预测及评价；旧部署测试MAE=0.298498。更新Android资产由integrate_model.py执行，使用前应先核对候选身份与评测。

评价边界
当前与旧模型比较使用同一历史测试集；文件去重不等于人物身份互斥。指标衡量对数据集主观评分标注的拟合，不是客观审美、Pearson准确率或真机端到端准确率。BiSeNet、FaceAttribNet和FPEM为独立公开预训练扩展，不能沿用本回归模型成绩。
