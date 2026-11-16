"""Exact Pillow-13 RGB/tensor parity for the pure-Java optional attribute preprocessor.

Run with the project's training Python environment and --java-home pointing to a JDK.
No model, training data, app assets, or connected device is read or changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, __version__ as pillow_version

PROBE = r'''package com.example.facescorelab;
import java.nio.*;
import java.nio.file.*;
public class ResizeProbe {
 public static void main(String[] args)throws Exception {
  int width=Integer.parseInt(args[0]),height=Integer.parseInt(args[1]);
  byte[] raw=Files.readAllBytes(Paths.get(args[2]));int[] input=new int[width*height];
  for(int i=0;i<input.length;i++)input[i]=0xff000000|((raw[i*3]&255)<<16)|((raw[i*3+1]&255)<<8)|(raw[i*3+2]&255);
  int[] result=VisibleAttributeResize.resize(input,width,height,224,224);byte[] output=new byte[result.length*3];
  for(int i=0;i<result.length;i++){output[i*3]=(byte)(result[i]>>16);output[i*3+1]=(byte)(result[i]>>8);output[i*3+2]=(byte)result[i];}
  Files.write(Paths.get(args[3]),output);
  float[] values=VisibleAttributeResize.imageNetTensor(result);
  ByteBuffer tensor=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);tensor.asFloatBuffer().put(values);
  Files.write(Paths.get(args[4]),tensor.array());
 }
}'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", required=not bool(os.environ.get("JAVA_HOME")), default=os.environ.get("JAVA_HOME"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if pillow_version != "13.0.0":
        raise ValueError(f"The exported preprocessing uses Pillow 13.0.0; found {pillow_version}")
    project = args.project.resolve()
    work = project / "app/build/visible-resize-parity"
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "ResizeProbe.java"
    probe.write_text(PROBE, encoding="utf-8")
    source = project / "app/src/main/java/com/example/facescorelab/VisibleAttributeResize.java"
    suffix = ".exe" if os.name == "nt" else ""
    java_bin = Path(args.java_home) / "bin"
    subprocess.run([str(java_bin / ("javac" + suffix)), "-d", str(work), str(source), str(probe)], check=True)
    rng = np.random.default_rng(2026)
    sizes = [(640,480),(73,109),(224,224),(448,224),(1,1),(225,257),(1920,1080),
             (1,19),(19,1),(223,225),(225,223),(373,729),(224,447),(447,224)]
    cases = [(width,height,kind) for width,height in sizes for kind in ("pattern","checkerboard","gradient")]
    cases += [(int(rng.integers(1,1500)),int(rng.integers(1,1000)),"seeded-random") for _ in range(12)]
    rows = []
    for width,height,kind in cases:
        y,x = np.indices((height,width))
        if kind == "pattern":
            pixels = np.stack([(x*17+y*29+(x*y)%251)%256,(x*7+y*31+13)%256,(x*19+y*3+47)%256],axis=-1).astype("uint8")
        elif kind == "checkerboard":
            pixels = np.stack([((x+y)%2)*255,(x%2)*255,(y%2)*255],axis=-1).astype("uint8")
        elif kind == "gradient":
            pixels = np.stack([x*255//max(1,width-1),y*255//max(1,height-1),(x+y)*255//max(1,width+height-2)],axis=-1).astype("uint8")
        else:
            pixels = rng.integers(0,256,(height,width,3),dtype=np.uint8)
        expected = np.asarray(Image.fromarray(pixels).resize((224,224),Image.Resampling.BILINEAR))
        mean = np.asarray([.485,.456,.406],dtype=np.float32)[:,None,None]
        std = np.asarray([.229,.224,.225],dtype=np.float32)[:,None,None]
        expected_tensor = np.ascontiguousarray((expected.astype(np.float32).transpose(2,0,1)/np.float32(255)-mean)/std)
        (work / "input.rgb").write_bytes(pixels.tobytes())
        subprocess.run([str(java_bin / ("java" + suffix)),"-cp",str(work),"com.example.facescorelab.ResizeProbe",str(width),str(height),
                        str(work/"input.rgb"),str(work/"output.rgb"),str(work/"output.f32")],check=True)
        rgb = (work / "output.rgb").read_bytes()
        actual = np.frombuffer(rgb,dtype=np.uint8).reshape(224,224,3)
        tensor_bytes = (work / "output.f32").read_bytes()
        actual_tensor = np.frombuffer(tensor_bytes,dtype="<f4").reshape(3,224,224)
        difference = np.abs(actual.astype("int16")-expected.astype("int16"))
        row = {"width":width,"height":height,"fixture":kind,"max_pixel_difference":int(difference.max()),
               "different_channels":int(np.count_nonzero(difference)),"max_tensor_difference":float(np.max(np.abs(actual_tensor-expected_tensor))),
               "expected_rgb_sha256":sha(expected.tobytes()),"actual_rgb_sha256":sha(rgb),
               "expected_tensor_sha256":sha(expected_tensor.astype("<f4").tobytes()),"actual_tensor_sha256":sha(tensor_bytes)}
        rows.append(row)
        if row["different_channels"] or row["max_tensor_difference"]:
            print(json.dumps(row),flush=True)
    report = {"pillow_version":pillow_version,"java_resizer_sha256":sha(source.read_bytes()),"python_test_sha256":sha(Path(__file__).read_bytes()),
              "passed":all(row["expected_rgb_sha256"]==row["actual_rgb_sha256"] and row["expected_tensor_sha256"]==row["actual_tensor_sha256"] for row in rows),
              "case_count":len(rows),"cases":rows}
    report_path = project / "app/build/reports/visible-resize-parity.json"
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"report":str(report_path),"passed":report["passed"],"case_count":len(rows)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
