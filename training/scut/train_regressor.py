"""Reproducible CPU transfer learning, held-out evaluation, and mobile ONNX export.

Only ImageNet weights initialize the model. No SCUT-trained weights are used.
The early CNN is frozen; the regression head and late convolution blocks are trained.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,json,random,time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
import onnx,onnxruntime as ort

PROJECT=Path(__file__).resolve().parents[2]
arguments=argparse.ArgumentParser();arguments.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'work')
ROOT=arguments.parse_args().output.resolve()
DATA=ROOT/'dataset';MODEL_VERSION='scut-mobilenetv3-small-selftrained-fold1-v1'
SEED=42

class BeautyNet(nn.Module):
    def __init__(self,features,mean):
        super().__init__();self.features=features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(nn.LayerNorm(576),nn.Linear(576,128),nn.Hardswish(),nn.Dropout(.2),nn.Linear(128,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('target_mean',torch.tensor(float(mean),dtype=torch.float32))
    def forward(self,x):return self.head(self.pool(self.features(x)).flatten(1))+self.target_mean

def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    return {'n':len(y),'mae':float(np.abs(y-p).mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'pearson':float(np.corrcoef(y,p)[0,1]) if p.std()>1e-10 and y.std()>1e-10 else None}

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    rows=list(csv.DictReader((DATA/'split_manifest.csv').open(encoding='utf8')))
    rows=[r for r in rows if r['split'] in ['train','validation','test']]
    names=[r['name'] for r in rows];y=torch.tensor([float(r['score']) for r in rows],dtype=torch.float32).reshape(-1,1)
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    train_idx,val_idx,test_idx=(indices[s] for s in ['train','validation','test'])
    pretrained=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    initial_tail={k:v.detach().clone() for k,v in pretrained.features.state_dict().items() if int(k.split('.')[0])>=9}
    mean=y[train_idx].mean().item();model=BeautyNet(pretrained.features,mean)
    # Cache the frozen early layers once. This is identical to evaluating them in every epoch.
    split=9;early=nn.Sequential(*list(model.features.children())[:split]).eval()
    tail=nn.Sequential(*list(model.features.children())[split:])
    for p in early.parameters():p.requires_grad=False
    cache=DATA/'frozen_early_maps.npy';cache_names=DATA/'frozen_early_names.json'
    valid_cache=cache.exists() and cache_names.exists() and json.loads(cache_names.read_text())==names
    if not valid_cache:
        bank=None;start=time.monotonic()
        for start_i in range(0,len(rows),32):
            images=[]
            for name in names[start_i:start_i+32]:
                array=np.asarray(Image.open(DATA/'crops'/(name+'.png')).convert('RGB'),dtype=np.float32)/255
                array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                images.append(array.transpose(2,0,1))
            with torch.inference_mode():maps=early(torch.from_numpy(np.stack(images))).numpy()
            if bank is None:bank=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(rows),*maps.shape[1:]))
            bank[start_i:start_i+len(maps)]=maps
            if start_i%320==0:print('early features',start_i+len(maps),'/',len(rows),round(time.monotonic()-start,1),'s',flush=True)
        bank.flush();del bank;cache_names.write_text(json.dumps(names))
    maps=torch.from_numpy(np.load(cache).copy())
    frozen_vectors=[]
    tail.eval()
    with torch.inference_mode():
        for i in range(0,len(rows),64):frozen_vectors.append(model.pool(tail(maps[i:i+64])).flatten(1))
    vectors=torch.cat(frozen_vectors)
    logs=[];best_mae=float('inf');best=None;best_stage=None;best_epoch=0
    loss_fn=nn.MSELoss();t0=time.monotonic()
    def evaluate(idx,stage):
        model.eval();parts=[]
        with torch.inference_mode():
            for chunk in range(0,len(idx),128):
                batch=idx[chunk:chunk+128]
                feat=vectors[batch] if stage=='head' else model.pool(tail(maps[batch])).flatten(1)
                parts.append(model.head(feat)+model.target_mean)
        return torch.cat(parts).numpy().ravel()
    for stage,epochs,lr,patience in [('head',100,.0008,16),('late_finetune',16,.00012,6)]:
        if stage=='late_finetune':model.load_state_dict(best);tail=nn.Sequential(*list(model.features.children())[split:])
        params=list(model.head.parameters()) if stage=='head' else [
            {'params':tail.parameters(),'lr':lr*.25},{'params':model.head.parameters(),'lr':lr}]
        optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
        stale=0
        for epoch in range(1,epochs+1):
            model.head.train();tail.eval()  # Frozen BatchNorm running stats on this small dataset.
            order=train_idx[np.random.permutation(len(train_idx))];total=0
            for start_i in range(0,len(order),64):
                batch=order[start_i:start_i+64];optimizer.zero_grad(set_to_none=True)
                feat=vectors[batch] if stage=='head' else model.pool(tail(maps[batch])).flatten(1)
                pred=model.head(feat)+model.target_mean;loss=loss_fn(pred,y[batch]);loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.head.parameters())+(list(tail.parameters()) if stage!='head' else []),5)
                optimizer.step();total+=loss.item()*len(batch)
            pred=evaluate(val_idx,stage);valid=metrics(y[val_idx].numpy().ravel(),pred)
            log={'stage':stage,'epoch':epoch,'train_mse':total/len(order),'validation':valid,'elapsed_seconds':time.monotonic()-t0}
            logs.append(log)
            (ROOT/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf8')
            print(json.dumps(log),flush=True)
            if valid['mae']<best_mae-1e-5:
                best_mae=valid['mae'];best=copy.deepcopy(model.state_dict());best_stage=stage;best_epoch=epoch;stale=0
                torch.save({'model_state':best,'version':MODEL_VERSION,'target_mean':mean,'best_stage':stage,'best_epoch':epoch},ROOT/'beauty_mobilenetv3_small.pt')
            else:stale+=1
            if stale>=patience:break
    model.load_state_dict(best);model.eval();tail=nn.Sequential(*list(model.features.children())[split:])
    # Test labels are first consulted here, after model selection has finished.
    raw=evaluate(test_idx,'late_finetune');target=y[test_idx].numpy().ravel();clipped=np.clip(raw,1,5)
    baseline=metrics(target,np.full(len(target),mean))
    with (ROOT/'test_predictions.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['name','target_1_5','prediction_raw','prediction_clipped','absolute_error','dataset_group'])
        for i,t,p,c in zip(test_idx,target,raw,clipped):writer.writerow([names[i],float(t),float(p),float(c),float(abs(t-p)),rows[i]['dataset_group']])
    evaluation={'model_version':MODEL_VERSION,'architecture':'MobileNetV3 Small + LayerNorm + 128-unit regression head',
        'initialization':'torchvision ImageNet1K V1 only; never initialized with an SCUT-trained beauty checkpoint',
        'training':'Freeze features 0-8; train regression head, then fine-tune features 9-12 and head; freeze all BatchNorm running statistics',
        'selected_stage':best_stage,'selected_epoch':best_epoch,'seed':SEED,
        'counts':{s:len(i) for s,i in indices.items()},'target_scale':[1,5],
        'selection_metric':'validation MAE on raw predictions','validation_mae':best_mae,
        'test_raw':metrics(target,raw),'test_clipped':metrics(target,clipped),'constant_train_mean_baseline':baseline,
        'groups':{},'limitation':'SCUT official fold 1, successful detector crops only. Not real-phone end-to-end accuracy. Dataset labels reflect subjective raters. Exact duplicates removed; identity overlap cannot be excluded.',
        'duration_seconds':time.monotonic()-t0}
    initial_file=Path(torch.hub.get_dir())/'checkpoints/mobilenet_v3_small-047dcff4.pth'
    evaluation['imagenet_checkpoint_sha256']=hashlib.sha256(initial_file.read_bytes()).hexdigest()
    evaluation['imagenet_checkpoint_url']=MobileNet_V3_Small_Weights.IMAGENET1K_V1.url
    evaluation['late_block_weight_change_l2']=float(np.sqrt(sum(float(torch.sum((model.features.state_dict()[k]-value).float()**2)) for k,value in initial_tail.items())))
    for group in sorted({r['dataset_group'] for r in rows}):
        subset=np.array([rows[i]['dataset_group']==group for i in test_idx]);evaluation['groups'][group]=metrics(target[subset],raw[subset])
    sample=torch.randn(1,3,224,224)
    path=ROOT/'beauty_mobilenetv3_small.onnx'
    torch.onnx.export(model,sample,str(path),input_names=['input'],output_names=['score'],opset_version=17,dynamic_axes={'input':{0:'batch'},'score':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    parity=[]
    for i in test_idx[:20]:
        array=np.asarray(Image.open(DATA/'crops'/(names[i]+'.png')),dtype=np.float32)/255
        array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=array.transpose(2,0,1)[None].copy()
        with torch.inference_mode():native=float(model(torch.from_numpy(x)).item())
        exported=float(session.run(None,{'input':x})[0].item())
        assert abs(native-exported)<1e-4
        parity.append({'name':names[i],'pytorch':native,'onnx':exported,'absolute_difference':abs(native-exported)})
    evaluation['conversion_parity']=parity
    evaluation['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    evaluation['checkpoint_sha256']=hashlib.sha256((ROOT/'beauty_mobilenetv3_small.pt').read_bytes()).hexdigest()
    (ROOT/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf8')
    print('TRAINING_COMPLETE',json.dumps(evaluation,ensure_ascii=True),flush=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.titlesize':12})
    fig,axes=plt.subplots(1,2,figsize=(6.2,3.2))
    axes[0].plot(np.arange(1,len(logs)+1),[r['validation']['mae'] for r in logs])
    axes[0].set(xlabel='Epoch',ylabel='MAE (1-5)',title='Validation MAE')
    axes[1].scatter(target,raw,s=6,alpha=.4);axes[1].plot([1,5],[1,5],'k--')
    axes[1].set(xlabel='Human score',ylabel='Prediction',title='Held-out test')
    fig.tight_layout();fig.savefig(ROOT/'training_and_test.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
