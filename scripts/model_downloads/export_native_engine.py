"""Export frozen native decision heads for the dedicated pooling engine."""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]

def export_checkpoint(key):
    config=tomllib.loads((ROOT/f'config/{key}/serving.toml').read_text())
    source=ROOT/config['weights_path']
    output=ROOT/f'.local/models/{key}/native_vllm_checkpoint'
    manifest=json.loads((source.parent/'manifest.json').read_text())
    if config['family']=='laya':
        from vllm_jev.laya_export import export_laya
        if not (output/'laya_manifest.json').exists():export_laya(source,output,config['model_id'])
    elif config['family']=='decider':
        from vllm_jev.decision_export import export_decider
        if not (output/'decision_manifest.json').exists():
            export_decider(source,output,config['model_id'],manifest['revision'])
    elif config['family']=='kev':
        from vllm_jev.decision_export import export_kev
        from jevlight.backends.decision_model_service import local_base
        meta=json.loads((source/'adapter_config.json').read_text())
        import torch
        head=torch.load(source/'head.pt',map_location='cpu',weights_only=True)
        base=local_base(config,meta['base_model_name_or_path'],head['base_revision'])
        if not (output/'decision_manifest.json').exists():
            export_kev(Path(base),source,output,config['model_id'],manifest['revision'])
    elif config['family']=='clef':
        from vllm_jev.clef_export import export_clef
        if not (output/'clef_manifest.json').exists():export_clef(source,output,config['model_id'])
    elif config['family']=='agent-jev':
        from safetensors.torch import load_file,save_file
        import shutil
        if not (output/'agent_head.safetensors').exists():
            output.mkdir(parents=True,exist_ok=True)
            weights=load_file(str(source/'model.safetensors'))
            encoder={'model.'+k.removeprefix('path_encoder.backbone.'):v for k,v in weights.items()
                if k.startswith('path_encoder.backbone.')}
            head={k:v for k,v in weights.items() if not k.startswith('path_encoder.backbone.')}
            if not encoder or not head:raise ValueError('AgentJev published head/backbone split is incomplete')
            save_file(encoder,output/'model.safetensors');save_file(head,output/'agent_head.safetensors')
            (output/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'model.safetensors' for name in encoder}})+'\n')
            cfg=json.loads((source/'config.json').read_text());cfg['architectures']=['Qwen3ForCausalLM']
            (output/'config.json').write_text(json.dumps(cfg)+'\n')
            for name in ('tokenizer.json','tokenizer_config.json','temperatures.json'):
                if (source/name).exists():shutil.copy2(source/name,output/name)
        if not (output/'model.safetensors.index.json').exists():
            from safetensors import safe_open
            with safe_open(output/'model.safetensors',framework='pt',device='cpu') as file:
                names=list(file.keys())
            (output/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'model.safetensors' for name in names}})+'\n')
    elif config['family']=='nimble':
        from jevlight.backends.decision_model_service import local_base
        from transformers import Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        from vllm_jev.decision_export import _text_backbone
        if not (output/'nimble_export.json').exists():
            contract=json.loads((source/'schema_config.json').read_text())
            base=local_base(config,contract['model'],contract['revision'])
            model=Qwen3_5ForConditionalGeneration.from_pretrained(base,dtype=__import__('torch').bfloat16,local_files_only=True)
            model=PeftModel.from_pretrained(model,source,local_files_only=True).merge_and_unload()
            output.mkdir(parents=True,exist_ok=True);model.save_pretrained(output,safe_serialization=True)
            _text_backbone(output,json.loads((output/'config.json').read_text()))
            import shutil
            for name in ('tokenizer.json','tokenizer_config.json'):
                shutil.copy2(source/name,output/name)
            (output/'nimble_export.json').write_text(json.dumps(dict(source_revision=manifest['revision'],contract=contract))+'\n')
    else:raise ValueError('No released dedicated vLLM adapter for this frozen head')
    return config,output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    _, output = export_checkpoint(args.model)
    print(output)
