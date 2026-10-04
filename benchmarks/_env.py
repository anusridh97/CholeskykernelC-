"""Non-identifying benchmark metadata and shared timing/output utilities."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import torch
import doublesided_cholesky

def command_output(args):
    """Return version information without exposing command paths or hostnames."""
    try:return subprocess.check_output(args,stderr=subprocess.DEVNULL,text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):return None

def metadata() -> dict:
    """Record versions, GPU properties and installed source digests; no absolute paths."""
    root=Path(doublesided_cholesky.__file__).parent
    versions={}
    for name in ['torch','triton','numpy']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result={'python':platform.python_version(),'platform':platform.system()+' '+platform.release()+' '+platform.machine(),'package_version':doublesided_cholesky.__version__,'versions':versions,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'git_commit':command_output(['git','-C',str(root),'rev-parse','HEAD']),'nvcc':command_output(['nvcc','--version']),'kernel_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.suffix in ('.py','.cu','.cpp','.cuh')}}
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        result['gpu']={'name':props.name,'total_memory_bytes':props.total_memory,'compute_capability':[props.major,props.minor],'driver_version':command_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'])}
    return result

def configure_cuda() -> None:
    """Disable reduced precision before any benchmark computation."""
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def measure(function,prepare,repeats: int,warmups: int=2) -> dict:
    """Median CUDA-event milliseconds with state restoration outside timing."""
    if repeats<1 or warmups<0:raise ValueError('invalid sample counts')
    samples=[]
    for index in range(warmups+repeats):
        prepare();torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();function();end.record();end.synchronize()
        if index>=warmups:samples.append(start.elapsed_time(end))
    return {'median_ms':statistics.median(samples),'samples_ms':samples}

def write_results(payload: dict, output: Path) -> None:
    """Write environment/samples JSON and parallel flat median/memory CSV."""
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    rows=[]
    for entry in payload['rows']:
        row={key:value for key,value in entry.items() if not isinstance(value,dict)}
        for method,values in entry.get('methods',{}).items():row[method+'_ms']=values['median_ms']
        rows.append(row)
    with output.with_suffix('.csv').open('w',newline='') as handle:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
