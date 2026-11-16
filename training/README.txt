FaceScoreLab 训练与评价

子目录
scut：SCUT-FBP5500基线训练，optimization为当前R4方法与比较工具，profiles为分组回归头适配。
visible_attributes：CelebA微笑、眼镜、嘴微张三标签训练、导出和固定测试。
gender：CelebA Male二元标注的可选参考模型与验证集校准。
deep：公开模型的下载、转换和数值核对，包含历史转换流程。
evaluation：来源审计、数据清单、属性/分割评价及工具验证。
visible_actions、extensions：动作系数与可见测量的检查材料。

Android源码已包含运行权重。训练工具仅用于实验复核或另行训练，不是编译App的前置条件。原始图像数据与大型外部训练环境未包含在工程中；数据须按各自许可取得。

MEBeauty几何回归
入口：train_geometry.py。使用NumPy求解标准化15维几何特征的岭回归，不需要scikit-learn。
输入位于training/data：官方landmarks.csv及train_2022/test_2022/val_2022.txt。
来源：https://github.com/fbplab/MEBeauty-database ，限非商业研究并遵守原始图像权利。

项目根目录示例：
python training/train_geometry.py --out training/results --assets training/exported-assets
python training/check_feature_parity.py --results training/results --java-home <JDK目录> --android-jar <Android-34的android.jar>
命令中的占位符需替换为实际路径。显式指定输出目录可避免更改App当前资产。

几何特征包括面部高宽、眼间距、眼宽、眼裂开合、眉眼距、鼻宽/鼻长投影、嘴宽、外唇高、嘴部开合、下半脸、左右差异、下颌宽/夹角和鼻宽与嘴宽比。双眼轴先作二维旋转校正，再归一化；定义与GeometryAnalyzer.featureVector68对应。发际线无法从原标注可靠取得，不进入回归。

2459行原始68点标注按basename分为2445组，排除3组未匹配划分的记录。重复文件组按test、val、train优先级保留于单一集合，首条坐标作代表，重复评分取均值。该处理是文件级去重，不证明身份互斥。实际训练/验证/测试为1712/218/512；训练集拟合截尾与标准化参数，验证集选择alpha=10，测试不参与调参。

模型原生尺度1～10，测试MAE=0.925474、RMSE=1.160382、Pearson=0.551067。数据集标注使用dlib风格68点，App使用MediaPipe478点语义对应，存在检测器差异，不能把离线标注误差当作手机照片准确率。

参照组统计由用户选择，与CNN档位和可选CelebA标签参考分别记录。模型不用于疾病、认知健康或性取向判断。各训练目录说明具体的锁定、抽样、阈值与评测范围；合成测试只验证实现。
