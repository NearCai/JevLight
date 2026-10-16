"""Reuse complete recent SUMO step inputs with matching native identities."""
import json
from pathlib import Path


def recorded_batch_cases(root, key, mode, expected):
    smoke=Path(root)/'results/experiments'
    # Short paired executions deliberately keep each record in its attempt,
    # leaving the formal records directory untouched.
    candidates=list((smoke/'records').glob(f'{key}_{mode}_*.json'))
    candidates.extend((smoke/'attempts').glob(f'{key}_{mode}_*/record.json'))
    paths=sorted(candidates,key=lambda path:path.stat().st_mtime,reverse=True)
    for path in paths:
        record=json.loads(path.read_text())
        if (record.get('model_key')!=key or record.get('mode')!=mode or record.get('stage')!='smoke'
                or record.get('status')!='completed' or
                any(record.get('hashes',{}).get(name)!=value for name,value in expected.items())):
            continue
        source=Path(record.get('attempt_path',''))/'native_requests.jsonl'
        if not source.is_file():continue
        rows=[]
        for line in source.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):break
            rows.append(json.loads(line))
        groups=[];cursor=0
        while cursor<len(rows):
            count=rows[cursor].get('batch_requests',1)
            if type(count) is not int or count<1:break
            chunk=rows[cursor:cursor+count]
            if len(chunk)!=count or any(row.get('batch_requests',1)!=count or 'request' not in row for row in chunk):break
            if count>1:groups.append([row['request'] for row in chunk])
            cursor+=count
        if groups:return str(source),groups
    return None,[]
