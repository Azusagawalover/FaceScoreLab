公开深度模型的来源与转换

Android运行所需模型已包含于app/src/main/assets，构建App不需要Python或下载模型。本目录用于检查公开权重来源、历史转换实现和导出数值。

当前来源
人脸解析：BiSeNet ResNet18，CelebAMask-HQ，19类。
https://github.com/yakhyo/face-parsing
分发镜像：https://huggingface.co/PayamFard123/dermaintel-face-parsing
FPEM：官方公开检查点本地推理。
https://github.com/Estella-LH/FPEM
CLIP：https://github.com/openai/CLIP
FaceAttribNet：https://github.com/yakhyo/face-attribute ，上游Qualcomm模型说明随third_party保存。

历史转换
prepare_models.py和download_models.py仍包含旧ResNet50评分器的下载/转换流程，来源为https://huggingface.co/evanlyhf/scut-fbp5500-beauty 。该评分器已经由本项目SCUT MobileNetV3 Small替换，不是当前默认模型。beauty_model_card.txt保留原模型卡，不将其成绩计为本项目训练结果。
重新导出可能更新Android资产和元数据，应在独立目录复核后再选择部署；不要用历史转换流程覆盖当前SCUT模型。

环境与处理
使用本目录requirements.txt配置独立Python环境；PyTorch 2.6.0的CPU执行即可。download_models.py按哈希核对原检查点；prepare_models.py检查旧评分器及分割导出数值，prepare_fpem.py严格加载官方FPEM权重并比较可移植实现与ONNX。
FPEM输入为RGB 0～1，按上游public_fpem比例缩放至224/112/160并在右/下补零。五条固定文本特征由同一检查点预计算，导出移除未消费训练头，保留最终评分运算。
Parser与FPEM使用FP16权重存储，并经Cast交给FP32 CPU算子，不代表全流程FP16计算。

评价与许可
转换一致性只说明数值移植，不代替真实评分或分割真值评价。FPEM未在本项目重训，也未复现LiveBeauty官方基准；检查点文件名的0.9243不是本地测得指标。
上游来源、许可和修改范围保留于third_party及App模型元数据。公开代码许可与数据集/检查点使用条件应分别遵守。
