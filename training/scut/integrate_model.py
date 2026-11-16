"""Install only a completed, evaluated training artifact into Android assets."""
from pathlib import Path
import argparse,hashlib,json,shutil

PROJECT=Path(__file__).resolve().parents[2]
arguments=argparse.ArgumentParser();arguments.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'work')
ROOT=arguments.parse_args().output.resolve()
ASSETS=PROJECT/'app/src/main/assets'
evaluation=json.loads((ROOT/'evaluation.json').read_text(encoding='utf8'))
audit=json.loads((ROOT/'dataset_audit.json').read_text(encoding='utf8'))
model=ROOT/'beauty_mobilenetv3_small.onnx'
assert hashlib.sha256(model.read_bytes()).hexdigest()==evaluation['model_sha256']
assert evaluation['test_raw']['mae']<evaluation['constant_train_mean_baseline']['mae']
assert evaluation['counts']['test']>1000
shutil.copyfile(model,ASSETS/model.name)
shutil.copyfile(ROOT/'evaluation.json',ASSETS/'scut_evaluation.json')
shutil.copyfile(ROOT/'dataset_audit.json',ASSETS/'scut_dataset_audit.json')
metrics=evaluation['test_raw'];counts=evaluation['counts']
summary=(f"本项目训练 MobileNetV3 Small；ImageNet 初始化。\n"
         f"训练 {counts['train']} 张 · 验证 {counts['validation']} 张 · 独立测试 {counts['test']} 张。\n"
         f"SCUT 官方第1折测试：MAE {metrics['mae']:.4f}，RMSE {metrics['rmse']:.4f}，Pearson {metrics['pearson']:.4f}。\n"
         f"误差按原始 1～5 分计算；常数均值基线 MAE {evaluation['constant_train_mean_baseline']['mae']:.4f}。\n"
         f"仅为该基准裁剪图像的测试结果，不能等同于真机拍照准确率。")
(ASSETS/'scut_experiment_summary.txt').write_text(summary,encoding='utf8')
manifest=json.loads((ASSETS/'deep_models.json').read_text(encoding='utf8'))
manifest['cnn']={'architecture':evaluation['architecture'],'training_by':'This project',
    'model_version':evaluation['model_version'],'initialization':evaluation['initialization'],
    'training':evaluation['training'],'dataset':'SCUT-FBP5500, official fold 1',
    'dataset_mirror':audit['source'],'native_scale':[1,5],'display_scale':[1,10],
    'display_mapping':'1 + 9 * (clip(raw,1,5)-1) / 4',
    'held_out_benchmark_evaluated':True,'real_phone_end_to_end_evaluated':False,
    'evaluation':evaluation,'asset':model.name}
manifest['attributes']={'source':'https://github.com/yakhyo/face-attribute',
    'original_source':'https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net',
    'model_version':'faceattribnet-qualcomm-2025-v1','license':'BSD-3-Clause',
    'labels':['left_eye_open','right_eye_open','eyeglasses','mask','sunglasses'],
    'preprocessing':'RGB 0-1, 128x128 aspect-preserving centered black padding. Android uses bilinear resize.',
    'independent_accuracy_evaluated':False}
manifest['explanations']={'method':'Five semantic-region occlusion experiments, RGB128 replacement',
    'display':'Original 1-10 score minus occluded 1-10 score',
    'caveat':'Model sensitivity under synthetic occlusion, not causal inference or an objective beauty standard'}
manifest['storage']='Self-trained CNN and attribute model use FP32. Parser and FPEM use FP16 weight storage with casts to FP32 CPU operators.'
manifest.get('converted_initializers',{}).pop('cnn',None)
manifest['assets']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in ASSETS.glob('*.onnx') if p.name!='beauty_resnet50.onnx'}
manifest['previous_model_checks']=manifest.pop('checks',[])
(ASSETS/'deep_models.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
# The v2 model is backed up in the v2 source archive and raw model folder.
old=ASSETS/'beauty_resnet50.onnx'
if old.exists():
    saved=ROOT/'before-v3/beauty_resnet50.onnx'
    saved.parent.mkdir(parents=True,exist_ok=True)
    if not saved.exists():shutil.copyfile(old,saved)
    assert hashlib.sha256(saved.read_bytes()).hexdigest()==hashlib.sha256(old.read_bytes()).hexdigest()
    old.unlink()
evidence=PROJECT/'training/scut/results';evidence.mkdir(exist_ok=True)
for name in ['evaluation.json','dataset_audit.json','training_log.json','test_predictions.csv','training_and_test.png','beauty_mobilenetv3_small.pt']:
    shutil.copyfile(ROOT/name,evidence/name)
shutil.copyfile(ROOT/'dataset/split_manifest.csv',evidence/'split_manifest.csv')
print(summary,flush=True)
