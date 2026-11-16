FaceScoreLab
Android 人脸评分与可见属性分析
版本：3.4.1（versionCode 16）

项目功能
FaceScoreLab 支持相册选图、拍照和实时相机预览，使用 MediaPipe 提取478个人脸关键点，并在本地运行CNN评分、几何测量、可见属性和人脸分割。分析结果经用户点击保存后写入本地历史。

评分提供通用、男性实验、女性实验三个档位，分别绑定实际不同的模型权重。通用模型为推荐默认；两组专用模型的验证改善均未达到0.005门槛，不能据此宣称更准确。可选性别参考学习的是CelebA Male二元标注，结果不会自动切换评分档位，需用户主动确认。

构建与运行
1. 使用 Android Studio 打开本目录，在本机配置 Android SDK 路径及 Gradle JDK。
2. 需要 Android SDK 34、JDK 17或21。工程使用 AGP 8.5.1、Gradle 8.7，Java源码级别为8。
3. 首次同步需取得Gradle和Android依赖；依赖已缓存时可离线构建。
4. Windows项目根目录构建命令：
   .\gradlew.bat :app:assembleDebug :app:lintDebug
   设备测试包：.\gradlew.bat :app:assembleDebugAndroidTest

源码包含全部9个运行模型（8个ONNX和1个TASK），构建不依赖配套APK，也不需要安装Python或重新训练。最低Android版本为7.0（API 24），compileSdk/targetSdk为34；支持arm64-v8a、armeabi-v7a和x86。

目录结构
app/src/main       Android界面、推理、历史存储及运行资产。
app/src/androidTest 设备契约和输入缩放一致性检查。
training           数据准备、训练、模型转换与评价工具；各子目录README说明输入和结果范围。
docs               文献依据、测量定义、照片署名及按版本记录的历史验证证据。
third_party        上游代码、来源说明和许可。
gradle             依赖版本目录与Gradle Wrapper。

模型与实验
- 通用CNN：ImageNet初始化的MobileNetV3 Small，在SCUT-FBP5500上训练；当前为R4 seed2026。原始目标量表1～5分，显示时裁剪并映射到1～10分，原始回归值单独保留。
- 分组CNN：冻结通用R4骨干，分别适配男性/女性数据组的回归头，由用户手动选择。
- 几何参考：MEBeauty 68点标注训练的15维岭回归；运行时使用MediaPipe语义点映射，结果与主CNN分开保存。
- 三项可见属性与可选性别参考：本项目在ImageNet权重上迁移训练的独立模型。
- MediaPipe、FaceAttribNet、BiSeNet和FPEM：集成公开预训练模型；来源与许可见第三方说明。

SCUT官方fold1历史测试MAE为0.276170、RMSE为0.363175、Pearson为0.847771。CelebA三标签固定测试样本1992/2000张有输出，Macro F1为0.897380。相关系数、F1和检测覆盖率均不是同一个指标，也不能代表真机拍照准确率。

验证与适用范围
构建检查：使用JDK 21完成assembleDebug、assembleDebugAndroidTest和lintDebug；lint为0错误、44条提示。构建检查不等于设备运行或精度验收。
原有JSON评测、训练协议和验证文件保留其原始版本与内容。docs中的code8/code11检查属于历史记录，不能直接作为code16全部功能已重测的证明。模拟器功能、数值一致性与模型精度需要分别评价。

照片在本地处理，App未声明互联网权限；论文链接由外部浏览器打开。分割独立mIoU/Dice、FaceAttribNet五属性独立精度以及代表性真机端到端评价尚未完成。研究页展示可见二维测量、动作系数和图像条件对照，不输出个人疾病、认知健康或性取向结论。

数据与授权
SCUT、MEBeauty、CelebA及公开模型遵循各自的数据使用与许可条件。示例照片保留作者、来源及CC BY-SA许可。详细说明见“第三方来源与实验边界.txt”、docs/visual-portrait-credits.txt及third_party各目录。
