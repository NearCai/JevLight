"""Classify measured preflight context refusals without treating OOM as context."""
import re


def is_context_limit(reason):
    text = str(reason).lower()
    return any(marker in text for marker in (
        'would truncate', 'would be truncated', 'would truncate state',
        'shortened the state', 'exceeds serving limit', 'exceed encoder limit',
        'exceeds encoder limit', 'exceeds maximum', 'context length',
        'native encoder would truncate', 'input exceeds', 'max_input_tokens',
        'native option cap', 'native candidate embedding cap',
    )) or bool(re.search(r'(token|sequence|path).*(too long|exceed|maximum|limit)', text))


def context_details(reason):
    text = str(reason)
    actual = re.search(r'(?:request\s+)?(\d+)\s+tokens', text)
    limit = re.search(r'(?:state=|encoder limit[= ]|limit[= ]|max_length=)(\d+)', text)
    details = dict(reason_code='context_limit', reason_label='上下文限制',
                actual_request_tokens=int(actual.group(1)) if actual else None,
                native_limit_tokens=int(limit.group(1)) if limit else None)
    option = re.search(r'native (?:option|candidate embedding) cap (\d+) exceeded: \[([\d, ]+)\]', text)
    if option:
        details.update(context_component='candidate_option', native_option_limit_tokens=int(option.group(1)),
                       actual_option_max_tokens=max(int(value) for value in option.group(2).split(',')))
    return details
