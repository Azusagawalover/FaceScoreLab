"""Download public research inputs with byte length and SHA256 verification."""
from pathlib import Path
import concurrent.futures as cf
import argparse,hashlib, json, time, urllib.request

arguments=argparse.ArgumentParser();arguments.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'work')
ROOT=arguments.parse_args().output.resolve()
DATA = ROOT / 'dataset'
DATA.mkdir(parents=True, exist_ok=True)

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'FaceScoreLab-course-research/3'}), timeout=80)

def download(url, target, size, digest=None):
    if target.exists() and target.stat().st_size == size:
        if not digest or hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print('verified existing', target.name, flush=True)
            return
    chunks = (size + 8*1024*1024-1)//(8*1024*1024)
    parts = target.parent / (target.name+'.parts')
    parts.mkdir(exist_ok=True)
    def part(index):
        start = index*8*1024*1024
        end = min(size, start+8*1024*1024)-1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end-start+1: return path
        for attempt in range(5):
            try:
                request = urllib.request.Request(url+('&' if '?' in url else '?')+f'part={index}&retry={attempt}', headers={'Range':f'bytes={start}-{end}', 'User-Agent':'FaceScoreLab-course-research/3'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206: raise ValueError('range ignored: '+str(response.status))
                    raw = response.read()
                if len(raw)!=end-start+1: raise ValueError('length mismatch')
                path.write_bytes(raw)
                print(target.name, index+1, '/', chunks, flush=True)
                return path
            except Exception as error:
                print('retry',target.name,index,attempt,str(error),flush=True)
                time.sleep(1+attempt)
        raise RuntimeError(f'Failed {target.name} part {index}')
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(part,range(chunks)))
    temporary=target.with_suffix(target.suffix+'.partial')
    with temporary.open('wb') as output:
        for i in range(chunks): output.write((parts/str(i)).read_bytes())
    actual=hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest and actual!=digest: raise ValueError('SHA256 mismatch '+actual)
    temporary.replace(target)
    print('completed',target.name,size,actual,flush=True)

if __name__=='__main__':
    source='https://huggingface.co/api/datasets/MnLgt/scut-fbp5500/tree/main?recursive=true'
    metadata=json.load(get(source))
    (DATA/'mirror_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for item in metadata:
        if item['path'].endswith('.parquet'):
            download('https://huggingface.co/datasets/MnLgt/scut-fbp5500/resolve/main/'+item['path'],DATA/Path(item['path']).name,item['size'],item['lfs']['oid'])
    release=json.load(get('https://api.github.com/repos/yakhyo/face-attribute/releases/tags/weights'))
    asset=next(x for x in release['assets'] if x['name']=='face_attrib_net.onnx')
    (ROOT/'attribute_release.json').write_text(json.dumps(asset,indent=2),encoding='utf-8')
    digest=asset.get('digest','').replace('sha256:','') or None
    download(asset['browser_download_url'],ROOT/'face_attrib_net.onnx',asset['size'],digest)
