from pathlib import Path
import sys,json,hashlib,copy,time,argparse
import torch,numpy as np
from PIL import Image
import onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
PROJECT=next((p for p in Path(__file__).resolve().parents if (p/'app/src/main').is_dir()),Path(r'D:\AndroidStudioProjects\FaceScoreLab'))
arguments=argparse.ArgumentParser()
arguments.add_argument('--models',type=Path,default=Path(__file__).parent/'downloads')
arguments.add_argument('--assets',type=Path,default=PROJECT/'app/src/main/assets')
arguments.add_argument('--fpem-source',type=Path,default=PROJECT/'third_party/FPEM')
arguments.add_argument('--clip-source',type=Path,default=PROJECT/'third_party/CLIP')
args=arguments.parse_args();MODELS=args.models;ASSETS=args.assets
sys.path[:0]=[str(args.clip_source),str(args.fpem_source)]
torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
source=args.fpem_source/'nets/Clips.py'
text=source.read_text(encoding='utf8')
source.write_text(text.replace('torch.cuda.current_device()',"torch.device('cpu')"),encoding='utf8')
import clip
from clip.model import build_model

checkpoint=torch.load(MODELS/'fpem-official.pth',map_location='cpu',weights_only=True)
state=checkpoint['state_dict']
state={k.removeprefix('FPEM_add.'):v for k,v in state.items()}
del checkpoint
clip_state={k.removeprefix('FACLIP.clip.'):v for k,v in state.items() if k.startswith('FACLIP.clip.')}
def offline_clip_load(*args,**kwargs):
    return build_model(clip_state).float(),None
clip.load=offline_clip_load
from nets.Clips import FPEM
model=FPEM().float().eval().requires_grad_(False)
model.load_state_dict(state,strict=True)
del state,clip_state
print('FPEM_STRICT_LOAD_OK',sum(p.numel() for p in model.parameters()),flush=True)

class PortableFPEM(torch.nn.Module):
    def __init__(self,original):
        super().__init__();self.visual=original.FACLIP.clip.visual
        self.face=original.SwinGFVQA_ca;self.fuse=original.fuse
        with torch.inference_mode():
            text=original.FACLIP.clip.encode_text(original.texts).float()
            text=text/text.norm(dim=-1,keepdim=True)
            scale=original.FACLIP.clip.logit_scale.exp().detach().clone()
        self.register_buffer('text_prior',text.t().clone())
        self.register_buffer('logit_scale',scale.clone())
        self.register_buffer('ratings',torch.arange(1,6,dtype=torch.float32).reshape(1,5))
    def forward(self,image224,image112,image160):
        _,features,_=self.face([image112,image160])
        visual=self.visual(image224).float()
        fused=self.fuse(visual.unsqueeze(1),features.unsqueeze(1))
        fused=fused/fused.norm(dim=-1,keepdim=True)
        logits=self.logit_scale*torch.matmul(fused,self.text_prior)
        probability=torch.softmax(logits,dim=-1).squeeze(1)
        score=(probability*self.ratings).sum(dim=1,keepdim=True)
        return score,probability

portable=PortableFPEM(model).eval()
def image_tensor(path,size):
    image=Image.open(path).convert('RGB')
    data=torch.from_numpy(np.asarray(image,dtype=np.float32).copy().transpose(2,0,1))/255
    h,w=data.shape[-2:];factor=size/max(h,w);nh,nw=int(h*factor),int(w*factor)
    data=torch.nn.functional.interpolate(data[None],size=(nh,nw),mode='bilinear',align_corners=False,antialias=True)
    return torch.nn.functional.pad(data,(0,size-nw,0,size-nh))
checks=[]
examples=[]
for file in [ASSETS/'demo/girl.jpg',ASSETS/'demo/guy.jpg']:
    inputs=tuple(image_tensor(file,size) for size in [224,112,160]);examples.append(inputs)
    with torch.inference_mode():
        # Upstream squeezes batch dimension for B=1. A duplicated batch preserves
        # its exact unmodified scoring code while letting us check a single photo.
        original=model(tuple(x.repeat(2,1,1,1) for x in inputs))[0][0].item()
        actual=portable(*inputs)[0].item()
    assert abs(actual-original)<1e-4,(actual,original)
    checks.append({'image':file.name,'official_cpu_score':original,'portable_torch_score':actual,'difference':abs(actual-original)})
    print(json.dumps(checks[-1]),flush=True)

full_path=MODELS/'fpem_portable_full.onnx'
torch.onnx.export(portable,examples[0],str(full_path),input_names=['image224','image112','image160'],output_names=['score','distribution'],opset_version=17,dynamo=False)
print('FPEM_EXPORTED',full_path.stat().st_size,flush=True)
graph=onnx.load(full_path);casts=[]
for tensor in list(graph.graph.initializer):
    if tensor.data_type!=TensorProto.FLOAT:continue
    value=numpy_helper.to_array(tensor)
    if value.size<64 or np.max(np.abs(value))>60000:continue
    name=tensor.name;half=name+'_stored_fp16';graph.graph.initializer.remove(tensor)
    graph.graph.initializer.append(numpy_helper.from_array(value.astype(np.float16),half))
    casts.append(helper.make_node('Cast',[half],[name],to=TensorProto.FLOAT,name=name+'_restore_fp32'))
old=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(casts+old)
onnx.checker.check_model(graph)
asset=ASSETS/'fpem_mobile.onnx'
onnx.save(graph,asset);del graph,model,portable
opt=ort.SessionOptions();opt.intra_op_num_threads=2
session=ort.InferenceSession(str(asset),sess_options=opt,providers=['CPUExecutionProvider'])
for check,inputs in zip(checks,examples):
    prediction=session.run(None,{n:x.numpy() for n,x in zip(['image224','image112','image160'],inputs)})
    raw=float(prediction[0].reshape(-1)[0]);difference=abs(raw-check['official_cpu_score'])
    check.update(android_onnx_score=raw,onnx_absolute_difference=difference,distribution=prediction[1].reshape(-1).tolist())
    assert difference<.03,check
manifest={'source':'https://github.com/Estella-LH/FPEM','checkpoint':'pretrained/fpem_srcc_0.9243.pth',
    'checkpoint_sha256':'5e18fb2a807366bd1f4bf8498493437059c60d4862a2f5b54ab664cc92af9615',
    'strict_state_dict_load':True,'source_changes':['CPU device selection in Clips.py; checkpoint-supplied CLIP initialization avoids downloading duplicate weights'],
    'inference_optimization':['Fixed five prompt text features precomputed from the same checkpoint','Unconsumed training heads removed by ONNX export','FP16 weight storage with FP32 CPU operators'],
    'input_preprocessing':'RGB 0-1, aspect-preserving bilinear resize and right/bottom zero padding at 224/112/160, matching upstream public_fpem dataloader',
    'native_scale':[1,5],'checks':checks,'asset_bytes':asset.stat().st_size,'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
    'benchmark_reproduced':False,'accuracy_note':'Official checkpoint inference port and numerical checks only; no reproduction of LiveBeauty benchmark metrics or training is claimed.'}
(MODELS/'fpem_conversion_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(asset.parent/'fpem_model.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('FPEM_MOBILE_READY',manifest['asset_bytes'],flush=True)
