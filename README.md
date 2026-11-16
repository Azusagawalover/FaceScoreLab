# FaceScoreLab

Android 人脸评分与可见属性分析课程项目，版本 3.4.1（versionCode 16）。

支持相册选图、拍照、实时相机预览、CNN 评分、人脸关键点、可见属性、分割及本地历史记录。项目功能、评价结果与实验范围见 [完整说明](README.txt)。

## 下载源码与模型

仓库包含全部 9 个 App 运行模型，使用 **Git LFS** 管理。请先安装 Git 和 Git LFS，再克隆：

```sh
git lfs install
git clone https://github.com/Azusagawalover/FaceScoreLab.git
cd FaceScoreLab
git lfs pull
```

直接使用 GitHub 的 Download ZIP 时，模型可能只有 LFS 指针；推荐按以上方式克隆。正常构建不需要 Python 或重新训练模型。

## Android Studio 运行

1. 用 Android Studio 打开项目根目录。
2. 配置 Android SDK 34 和 Gradle JDK 17 或 21。
3. 同步依赖，将 App 运行到 Android 7.0（API 24）或以上设备。

工程使用 Gradle 8.7、AGP 8.5.1、Java 8 源码级别。首次同步需要下载依赖。

Windows 构建命令：

```powershell
.\gradlew.bat :app:assembleDebug :app:lintDebug
```

可选模型完整性检查（Python 3.11 或以上）：

```sh
python restore_bundled_models.py --check
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `app` | Android 界面、推理、历史存储、模型和设备契约 |
| `training` | 数据准备、训练、导出、评价程序与原始实验记录 |
| `docs` | 方法、论文依据、示例照片署名与历史验证记录 |
| `third_party` | 上游代码、来源和许可 |

## 验证与来源

已通过 `assembleDebug`、`assembleDebugAndroidTest` 构建和 `lintDebug`；静态检查为 0 错误、44 条提示。编译通过不代表完成真机或独立精度验收。

数据集、示例照片、第三方代码和预训练模型遵循各自授权与使用条件；公开仓库不改变原有许可。详见 [第三方来源与实验边界](第三方来源与实验边界.txt)、[图标许可](LUCIDE_ICONS_LICENSE.txt) 和 `third_party`。原始训练数据与 Python 训练检查点未包含。
