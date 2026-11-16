"""Compare the production Java feature extractor with held-out Python fixtures."""
from pathlib import Path
import json, subprocess, math, os, argparse, shutil

PROJECT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=PROJECT/'training/results')
parser.add_argument('--java-home',type=Path,default=os.environ.get('JAVA_HOME') or None,
                    help='JDK directory; defaults to JAVA_HOME, then javac on PATH')
sdk_home=os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT')
parser.add_argument('--android-jar',type=Path,default=Path(sdk_home)/'platforms/android-34/android.jar' if sdk_home else None,
                    help='Android 34 android.jar; defaults under ANDROID_HOME or ANDROID_SDK_ROOT')
args=parser.parse_args()
OUT=args.results.expanduser().resolve()
java_name='java.exe' if os.name=='nt' else 'java'
javac_name='javac.exe' if os.name=='nt' else 'javac'
if args.java_home is not None:
    JAVA=args.java_home.expanduser().resolve()/'bin'
else:
    javac=shutil.which(javac_name)
    if javac is None:
        parser.error('Provide --java-home, set JAVA_HOME, or put a JDK on PATH.')
    JAVA=Path(javac).resolve().parent
if not (JAVA/javac_name).is_file() or not (JAVA/java_name).is_file():
    parser.error('The selected JDK must contain both java and javac executables.')
if args.android_jar is None:
    parser.error('Provide --android-jar or set ANDROID_HOME/ANDROID_SDK_ROOT.')
ANDROID=args.android_jar.expanduser().resolve()
if not ANDROID.is_file():
    parser.error('The selected Android SDK android.jar does not exist: '+str(ANDROID))
BUILD=OUT/'parity-build'
BUILD.mkdir(exist_ok=True)
fixtures=json.loads((OUT/'geometry_parity_fixtures.json').read_text(encoding='utf-8'))
cases=[]
for fi,fixture in enumerate(fixtures):
    coords=fixture['points']
    values=','.join('{'+str(coords[i])+','+str(coords[i+1])+'}' for i in range(0,len(coords),2))
    cases.append('double[] v'+str(fi)+'=GeometryAnalyzer.featureVector68(new double[][]{'+values+'}); System.out.println(java.util.Arrays.toString(v'+str(fi)+'));')
source='package com.example.facescorelab; public class FeatureParity { public static void main(String[] args) {'+''.join(cases)+'} }'
file=BUILD/'FeatureParity.java';file.write_text(source,encoding='utf-8')
sources=[str(PROJECT/'app/src/main/java/com/example/facescorelab'/f) for f in ['AnalysisResult.java','ScoringModel.java','GeometryAnalyzer.java']]
compile_result=subprocess.run([str(JAVA/javac_name),'-encoding','UTF-8','-cp',str(ANDROID),'-d',str(BUILD),*sources,str(file)],capture_output=True,text=True,encoding='utf-8',errors='replace')
if compile_result.returncode:
    raise RuntimeError(compile_result.stdout+compile_result.stderr)
run=subprocess.run([str(JAVA/java_name),'-cp',str(BUILD)+os.pathsep+str(ANDROID),'com.example.facescorelab.FeatureParity'],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
actual=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('[')]
errors=[abs(a-b) for fixture,got in zip(fixtures,actual) for a,b in zip(fixture['features'],got)]
result={'cases':len(actual),'features_per_case':15,'maximum_absolute_difference':max(errors),'passed':len(actual)==len(fixtures) and max(errors)<1e-10}
(OUT/'java_python_feature_parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if not result['passed']:raise AssertionError(result)
print(json.dumps(result))
