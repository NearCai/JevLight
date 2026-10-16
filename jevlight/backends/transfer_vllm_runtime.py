"""Reuse the author-head engine with this single GPU's qualified capacity."""
import threading

from jevlight.backends.native_vllm_runtime import VllmDecisionRuntime, validate_choice


class TransferVllmRuntime(VllmDecisionRuntime):
    def native_batch(self, requests, batch_size):
        requests = list(requests)
        results, evidence = [], []
        for offset in range(0, len(requests), self.capacity):
            group = requests[offset:offset + self.capacity]
            results.extend(super().native_batch(group, batch_size))
            evidence.extend(self.last_step_batch['gpu_blocks'])
        self.last_step_batch = dict(
            request_count=len(requests), requested_capacity=batch_size,
            gpu_forward_rows=[row['gpu_rows'] for row in evidence],
            engine='vllm', engine_capacity=self.capacity, gpu_blocks=evidence,
            whole_step_one_forward=len(evidence) == 1,
            capacity_downgrade=self.selection.get('capacity_attempts', []),
        )
        for result in results:
            result['usage']['step_batch'] = self.last_step_batch
        return results


def backend(config, selection):
    from jevlight.backends import decision_model_service
    from jevlight.backends.decision_model_service import NativeBackend

    result = NativeBackend.__new__(NativeBackend)
    # Use the dedicated author-head engine with its validated batch capacity.
    result.config = dict(config, batch_size=int(config.get('batch_size',196)), inference_engine='vllm_native_heads')
    result.lock = threading.Lock()
    result.reported_model = config['model_id']
    result.log_path = None
    result.answer_validator = validate_choice
    result.runtime = TransferVllmRuntime(result.config, selection)
    return result
