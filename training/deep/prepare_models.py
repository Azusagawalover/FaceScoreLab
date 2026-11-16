from pathlib import Path
import hashlib,json,time,copy,argparse
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
import onnxruntime as ort
import torch,timm
from PIL import Image

PROJECT=next((p for p in Path(__file__).resolve().parents if (p/'app/src/main').is_dir()),Path(r'D:\AndroidStudioProjects\FaceScoreLab'))
arguments=argparse.ArgumentParser()
arguments.add_argument('--models',type=Path,default=Path(__file__).parent/'downloads')
arguments.add_argument('--assets',type=Path,default=PROJECT/'app/src/main/assets')
args=arguments.parse_args();MODELS=args.models;ASSETS=args.assets
EXPECTED={'beauty_regressor.pt':'54f55b2d03fe68b9bdb992d4992ea7d245b4dd25e0d376dca8f599463ff4a312',
          'face_parsing_hf.onnx':'0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'}
for name,digest in EXPECTED.items():
    got=hashlib.sha256((MODELS/name).read_bytes()).hexdigest()
    assert got==digest,(name,got)
torch.set_num_threads(2)
state=torch.load(MODELS/'beauty_regressor.pt',map_location='cpu',weights_only=True)
print('checkpoint',type(state).__name__,list(state)[:6],flush=True)
model=timm.create_model('resnet50',pretrained=False,num_classes=1)
model.load_state_dict(state,strict=True);model.eval()
example=torch.zeros(1,3,224,224)
torch.onnx.export(model,example,str(MODELS/'beauty_resnet50_full.onnx'),input_names=['input'],output_names=['score'],opset_version=17,dynamo=False)

def compact(graph,path):
    casts=[];converted=0
    for initializer in list(graph.graph.initializer):
        if initializer.data_type!=TensorProto.FLOAT:continue
        values=numpy_helper.to_array(initializer)
        if values.size<64 or np.max(np.abs(values))>60000:continue
        name=initializer.name;half_name=name+'_stored_fp16'
        graph.graph.initializer.remove(initializer)
        graph.graph.initializer.append(numpy_helper.from_array(values.astype(np.float16),half_name))
        casts.append(helper.make_node('Cast',[half_name],[name],to=TensorProto.FLOAT,name=name+'_restore_fp32'))
        converted+=1
    old=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(casts+old)
    onnx.checker.check_model(graph);onnx.save(graph,path)
    return converted

cnn=onnx.load(MODELS/'beauty_resnet50_full.onnx')
cnn_count=compact(cnn,ASSETS/'beauty_resnet50.onnx')
parser_full=onnx.load(MODELS/'face_parsing_hf.onnx')
first_output=parser_full.graph.output[0].name
parser=copy.deepcopy(parser_full)
parser.graph.node.extend([
    helper.make_node('ArgMax',[first_output],['class_ids_i64'],axis=1,keepdims=0,select_last_index=0),
    helper.make_node('Cast',['class_ids_i64'],['mask'],to=TensorProto.UINT8),
])
del parser.graph.output[:]
parser.graph.output.append(helper.make_tensor_value_info('mask',TensorProto.UINT8,[1,512,512]))
seg_count=compact(parser,ASSETS/'face_parsing_mask.onnx')

def normalize(path,size):
    im=Image.open(path).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
    values=np.asarray(im,dtype=np.float32)/255
    values=(values-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return values.transpose(2,0,1)[None].copy()
options=ort.SessionOptions();options.intra_op_num_threads=2
sessions=[ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider']) for p in [MODELS/'beauty_resnet50_full.onnx',ASSETS/'beauty_resnet50.onnx',MODELS/'face_parsing_hf.onnx',ASSETS/'face_parsing_mask.onnx']]
checks=[]
for file in sorted((ASSETS/'demo').glob('*.jpg')):
    x=normalize(file,224)
    with torch.inference_mode():reference=float(model(torch.from_numpy(x)).item())
    full=float(sessions[0].run(None,{'input':x})[0].reshape(-1)[0])
    compact_score=float(sessions[1].run(None,{'input':x})[0].reshape(-1)[0])
    y=normalize(file,512)
    original=sessions[2].run([sessions[2].get_outputs()[0].name],{sessions[2].get_inputs()[0].name:y})[0].argmax(1).astype(np.uint8)
    mask=sessions[3].run(None,{sessions[3].get_inputs()[0].name:y})[0]
    agreement=float((original==mask).mean())
    check={'image':file.name,'torch_native_score':reference,'onnx_full_score':full,'android_model_native_score':compact_score,
           'torch_onnx_absolute_difference':abs(reference-full),'compact_score_difference':abs(full-compact_score),
           'segmentation_pixel_agreement':agreement,'predicted_classes':np.unique(mask).tolist()}
    assert abs(reference-full)<1e-4 and abs(full-compact_score)<.03 and agreement>.97,check
    checks.append(check);print(json.dumps(check),flush=True)

manifest={'date':'2026-11-06','cnn':{'source':'https://huggingface.co/evanlyhf/scut-fbp5500-beauty',
          'source_sha256':EXPECTED['beauty_regressor.pt'],'architecture':'timm ResNet50, image regression',
          'training_by':'third-party author; SCUT-FBP5500','native_scale':[1,5],
          'display_scale':[1,10],'display_mapping':'1 + 9 * (clip(raw,1,5)-1) / 4',
          'independent_end_to_end_accuracy_evaluated':False},
          'segmentation':{'source':'https://github.com/yakhyo/face-parsing',
          'download_mirror':'https://huggingface.co/PayamFard123/dermaintel-face-parsing',
          'source_sha256':EXPECTED['face_parsing_hf.onnx'],'architecture':'BiSeNet ResNet18, CelebAMask-HQ, 19 labels',
          'ground_truth_segmentation_accuracy_evaluated':False},
          'storage':'Large FP32 initializers stored in FP16 and restored with Cast for CPU FP32 operators. This is not a claim of FP16 computation.',
          'converted_initializers':{'cnn':cnn_count,'segmentation':seg_count},'checks':checks,
          'assets':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in ASSETS.glob('*.onnx')}}
(MODELS/'model_conversion_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(ASSETS/'deep_models.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('MODELS_READY',flush=True)
