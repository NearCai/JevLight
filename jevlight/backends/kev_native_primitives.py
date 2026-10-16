"""Load unchanged pure Kev encoding/head code without its HF5 cache classes."""
import ast
from pathlib import Path
from types import ModuleType

ROOT=Path(__file__).resolve().parents[2]


def load_primitives():
    path=ROOT/'.local/model_sources/kev/kev/model.py'
    tree=ast.parse(path.read_text())
    names={'ContextOverflow','PointerHead','user_tokens','encode','rows_of'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            module=getattr(node,'module','')or''
            if not module.startswith('transformers'):nodes.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            # The frozen encoder's delimiter/context constants and regex only.
            if node.lineno<87:nodes.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in names:
            nodes.append(node)
    module=ModuleType('jevlight_frozen_kev_primitives')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
