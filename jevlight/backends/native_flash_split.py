"""Keep the pinned Torch Flash split-K arithmetic across native batch rows."""
from functools import lru_cache
from pathlib import Path


def native_splits(length,heads,sm_count,head_dim=256):
    """PyTorch 2.8 Flash's scalar-prefix occupancy heuristic, unchanged."""
    if min(length,heads,sm_count,head_dim)<=0:
        raise ValueError('Native Flash dimensions and SM count must be positive')
    block_n=256 if head_dim<=64 else 128 if head_dim<=128 else 64
    mblocks=(length+63)//64
    nblocks=(length+block_n-1)//block_n
    occupancy=heads*mblocks
    sms=sm_count*2
    if occupancy>=0.8*sms:return 1
    scores=[]
    previous=None
    for splits in range(1,min(128,sms,nblocks)+1):
        blocks=(nblocks+splits-1)//splits
        eligible=splits==1 or blocks!=previous
        waves=occupancy*splits/sms
        score=waves/(-(-occupancy*splits//sms)) if eligible else 0.
        scores.append(score);previous=blocks
    limit=0.85*max(scores)
    return next(i+1 for i,score in enumerate(scores) if score>=limit)


@lru_cache(maxsize=1)
def load_kernel():
    import hashlib
    import fcntl
    import os
    import shlex
    import subprocess
    import torch
    from torch.utils.cpp_extension import include_paths
    if torch.__version__.split('+')[0]!='2.8.0' or torch.version.cuda!='12.8':
        raise RuntimeError('Native Flash split ABI requires the pinned Torch 2.8.0 CUDA 12.8 environment')
    root=Path(__file__).resolve().parent
    cpp=root/'native_flash_split.cpp';header=root/'native_flash_split_params.h'
    abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)
    tag=hashlib.sha256(cpp.read_bytes()+header.read_bytes()+f'{torch.__version__}:{abi}'.encode()).hexdigest()[:16]
    build=root.parent.parent/'.local'/'kernels'/f'flash_split_{tag}'
    build.mkdir(parents=True,exist_ok=True)
    lib=Path(torch.__file__).resolve().parent/'lib'
    binary=build/'native_flash_split.so'
    with (build/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not binary.exists():
            pending=build/'native_flash_split.pending.so'
            command=shlex.split(os.environ.get('CXX','c++'))+['-shared','-fPIC','-std=c++17','-O2',
                f'-D_GLIBCXX_USE_CXX11_ABI={abi}',*(f'-I{path}' for path in include_paths()),
                str(cpp),f'-L{lib}',f'-Wl,-rpath,{lib}','-lc10','-ltorch','-ltorch_cpu','-ltorch_cuda','-o',str(pending)]
            compiled=subprocess.run(command,capture_output=True,text=True)
            if compiled.returncode:
                raise RuntimeError(f'Native Flash bridge compilation failed: {compiled.stderr[-8000:]}')
            pending.replace(binary)
    torch.ops.load_library(str(binary))
    return torch.ops.jevlight_native.flash_split256


def forward(query,key,value,splits,scale):
    import torch
    kernel=load_kernel()
    # Kernel layout is [batch,tokens,heads,dim]; the rows remain independent.
    rows=[tensor.transpose(1,2).contiguous() for tensor in (query,key,value)]
    stream=torch.cuda.current_stream(query.device).cuda_stream
    return kernel(*rows,splits,scale,stream)
