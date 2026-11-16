from pathlib import Path
import argparse,urllib.request,hashlib,time
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'downloads')
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
files=[
 ('beauty_regressor.pt','https://huggingface.co/evanlyhf/scut-fbp5500-beauty/resolve/main/beauty_regressor.pt',94363765,'54f55b2d03fe68b9bdb992d4992ea7d245b4dd25e0d376dca8f599463ff4a312'),
 ('face_parsing_hf.onnx','https://huggingface.co/PayamFard123/dermaintel-face-parsing/resolve/main/resnet18.onnx',53205364,'0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'),
 ('fpem-official.pth','https://media.githubusercontent.com/media/Estella-LH/FPEM/main/pretrained/fpem_srcc_0.9243.pth',582528073,'5e18fb2a807366bd1f4bf8498493437059c60d4862a2f5b54ab664cc92af9615')]
for name,url,size,digest in files:
    target=args.output_dir/name
    if target.is_file() and target.stat().st_size==size and hashlib.sha256(target.read_bytes()).hexdigest()==digest:
        print(name,'already verified',flush=True);continue
    for attempt in range(6):
        offset=target.stat().st_size if target.exists() else 0
        if offset>=size:break
        try:
            request=urllib.request.Request(url+f'?download=true&offset={offset}&attempt={attempt}',headers={'Range':f'bytes={offset}-{size-1}'})
            with urllib.request.urlopen(request,timeout=40) as response:
                if offset and response.status!=206:raise RuntimeError('Server did not honor resume')
                if response.status==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Unexpected range')
                with target.open('ab' if offset else 'wb') as output:
                    while True:
                        data=response.read(262144)
                        if not data:break
                        output.write(data)
            print(name,target.stat().st_size,'/',size,flush=True)
        except Exception as error:
            print(name,'retry',attempt+1,type(error).__name__,flush=True);time.sleep(1)
    if not target.is_file() or target.stat().st_size!=size or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        raise RuntimeError(name+' did not pass full SHA256 validation; partial data kept for resume')
print('ALL_MODELS_VERIFIED',flush=True)
