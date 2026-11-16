CelebA三项可见属性

任务与结果
标签顺序为Smiling、Eyeglasses、Mouth_Slightly_Open，对应微笑、眼镜、嘴微张；保留CelebA标注含义，不另分普通眼镜与太阳镜。
当前模型以ImageNet1K V1初始化MobileNetV3 Small，不使用CelebA属性分类检查点。共享576维池化特征、LayerNorm/128维共享头和三个独立线性输出头，输出三个logit，分别经sigmoid判定，不使用互斥softmax。

计划训练/验证/固定测试为20000/2000/2000张，实际有输出19883/1988/1992张。实际完成头训练22轮及tail8轮，选择tail第8轮。验证Macro F1=0.888694；测试三类F1=0.886735/0.918919/0.886486，Macro F1=0.897380，覆盖率99.6%。每类冻结阈值为0.425/0.850/0.475。
head-only模型在同一固定测试队列上的Macro F1=0.832308；最终模型按验证集选择。测试队列复用，身份独立未证明，也不代表额外独立真机测试。结果与逐图CSV见results/final-tail8-seed42。

数据接口
开发CSV包含name,split,image_path,Smiling,Eyeglasses,Mouth_Slightly_Open。split仅train/validation；image_path为图像实际绝对路径，basename须等于name。
标签已正规化为0/1，0是已知阴性。空、NA或?表示unknown；可选<label>_mask仅0/1，mask=0不进入监督。CelebA原始-1/+1须先显式转换，不能直接传入。
subset_protocol.json须记录seed42和实际selected_names，prepare核对精确名单，而不只核对数量。没有取得身份表，仅使用官方image-level划分。

训练设置
seed42、batch64，先冻结特征训练头，最多50轮、早停8轮、最小F1改善1e-4。AdamW头学习率8e-4、weight_decay=0.01。可显式设置--tail-epochs 1～8，解冻features9～12，前0～8保持固定；tail头/骨干学习率1e-4/2.5e-5，早停3轮，BN统计保持冻结。
损失为masked BCEWithLogitsLoss，正类权重只从有效train计数计算；unknown标签不贡献梯度。每轮仅在validation逐类搜索0.05～0.95、步长0.025的阈值，以最高F1为优先，平局选最接近0.5、再选较高值。模型及阈值一起冻结，测试不调阈值。

统一输入
使用App同一face_landmarker.task、最大脸规则、num_faces=3及三项0.5门槛。faceCrop v1框最大边×1.45，中心y上移size×0.06，float32计算并夹到图像边界。224×224双线性缩放、RGB/255、ImageNet mean/std、连续float32 NCHW。
Android三属性缩放由VisibleAttributeResize对齐训练Pillow13滤波与舍入，区别于其他模型的Android缩放入口。来源许可见项目根THIRD_PARTY_VISIBLE_RESIZE_LICENSE.txt。
检测失败行仍保留真实标签及原因，不能把无输出记作阴性；同时报告总样本、输出数、覆盖率、每类支持数和unknown数。

复现入口
项目根目录使用python training/visible_attributes/runner.py <子命令> --help查看参数。
prepare：指定开发manifest、抽样protocol、face-task及准备输出。
train：指定prepared、独立output、cache-dir、seed42，可选--tail-epochs 8。
export：指定run与prepared，用最多20张验证图核对PyTorch/ONNX logits差<1e-4及判定一致，生成visible_attributes.onnx、android_metadata.json和export_metadata.json，不自动更新App资产。
lock：指定run、heldout-count及heldout-seed，在读取测试标签前冻结模型/阈值/预处理。历史固定测试2000张，抽样seed20261113。
prepare-heldout、evaluate-heldout：先验证锁，再读取官方test池固定名单与标签，保留逐图结果，阈值不再重选。

缓存与恢复
源清单、抽样协议或裁剪契约变化时使用新输出目录。缓存绑定输入字节、准备清单、完整骨干权重、预处理、dtype及依赖。--resume保存并恢复模型、AdamW、RNG、阶段、batch游标、最佳模型和阈值。代码或输入身份变化时不能按同一run续训。--smoke为实现检查，不可发布；--max-hours约束准备与训练的累计时长，跨实验成本另计。

Android接口
input float32 [N,3,224,224]；logits float32 [N,3]。逐项sigmoid后与该类阈值比较，使用>=。阈值、标签顺序、模型哈希、face-task哈希和预处理须同步；无脸时返回无结果。输出是未校准sigmoid分数，不能把0.8解释为80%正确概率。
自训练三标签、FaceAttribNet五标签和MediaPipe动作系数保持独立来源。test_visible_attributes.py检查标签掩码、名单、缓存、恢复、导出及锁定等实现；这些合成/小样本工具检查不是模型精度结论。数据使用遵守CelebA非商业研究条款，见third_party/CelebA。
