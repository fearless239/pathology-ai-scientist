"""Trusted launcher instrumentation, executed only inside the experiment sandbox."""


def install_model_guard(requirements):
    import json
    from pathlib import Path
    import threading
    import weakref
    import torch

    allowed = {tuple(size) for size in requirements.get('input_sizes', [])}
    inference = requirements.get('inference_only', False)
    checkpoint = Path('/workspace/model_checkpoint.pt')
    if inference and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        raise RuntimeError('MODEL_CONTRACT_FAILED: inference requires /workspace/model_checkpoint.pt')
    original_load = torch.load
    original_state = torch.nn.Module.load_state_dict
    original_call = torch.nn.Module._call_impl
    loaded_mappings = []
    restored = weakref.WeakKeyDictionary()
    local = threading.local()
    observed = set()
    violations = []
    calls = 0
    custom_losses = set(requirements.get('custom_smoothing_classes', []))
    checked_losses = weakref.WeakSet()
    loss_backward_calls = [0]
    standard_required = requirements.get('standard_smoothing_required', False)
    standard_backward_calls = [0]
    original_cross_entropy = torch.nn.functional.cross_entropy

    def cross_entropy(input, target, *args, **kwargs):
        result = original_cross_entropy(input, target, *args, **kwargs)
        if standard_required and not inference and input.requires_grad:
            # Remaining positional arguments: weight, size_average, ignore_index,
            # reduce, reduction, label_smoothing (PyTorch's public signature).
            amount = kwargs.get('label_smoothing', args[5] if len(args) > 5 else 0.0)
            if type(amount) not in (int, float) or not 0 < amount < 1:
                fail('standard label smoothing requires an active coefficient between zero and one')
            def used(gradient):
                standard_backward_calls[0] += 1
                return gradient
            result.register_hook(used)
        return result

    def fail(reason):
        violations.append(reason)
        raise RuntimeError('MODEL_CONTRACT_FAILED: ' + reason)

    def remember(value):
        if isinstance(value, dict):
            if value and all(isinstance(v, torch.Tensor) for v in value.values()):
                loaded_mappings.append({k: v.detach().clone() for k, v in value.items()})
            else:
                for child in value.values():
                    remember(child)

    def load(file, *args, **kwargs):
        value = original_load(file, *args, **kwargs)
        if inference:
            name = getattr(file, 'name', file)
            if isinstance(name, (str, Path)) and Path(name).resolve() == checkpoint.resolve():
                remember(value)
        return value

    def load_state(module, state, *args, **kwargs):
        if inference:
            matches = any(set(state) == set(saved) and all(
                isinstance(state[k], torch.Tensor) and state[k].shape == saved[k].shape
                and torch.equal(state[k].detach().cpu(), saved[k].detach().cpu()) for k in saved
            ) for saved in loaded_mappings)
            if not matches:
                fail('load_state_dict must use weights from the required checkpoint')
        result = original_state(module, state, *args, **kwargs)
        if inference:
            if result.missing_keys or result.unexpected_keys:
                fail('checkpoint must restore the complete model; partial loading is forbidden')
            for child in module.modules():
                restored[child] = tuple((id(p), p._version) for p in child.parameters())
        return result

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from tensors(child)

    def call(module, *args, **kwargs):
        nonlocal calls
        depth = getattr(local, 'depth', 0)
        images = [t for t in tensors((args, kwargs)) if t.ndim == 4]
        if depth == 0 and images and any(True for _ in module.parameters()):
            for tensor in images:
                size = tuple(tensor.shape[-2:])
                if allowed and size not in allowed:
                    fail(f'actual model input {size} disagrees with approved sizes {sorted(allowed)}')
                observed.add(size)
            if inference:
                if module not in restored:
                    fail('model forward before successful checkpoint restoration')
                if restored[module] != tuple((id(p), p._version) for p in module.parameters()):
                    fail('model parameters changed after checkpoint restoration')
            calls += 1
        local.depth = depth + 1
        try:
            result = original_call(module, *args, **kwargs)
            if (not inference and type(module).__name__ in custom_losses
                    and module not in checked_losses and len(args) >= 2 and args[0].requires_grad):
                amount = getattr(module, 'smoothing', None)
                if type(amount) not in (int, float) or not 0 < amount < 1:
                    fail('custom label smoothing requires an active coefficient between zero and one')
                reference = torch.nn.functional.cross_entropy(args[0], args[1], label_smoothing=amount)
                if result.shape != reference.shape or not torch.allclose(result, reference, rtol=1e-5, atol=1e-6):
                    fail('custom smoothing loss differs from standard cross entropy with label smoothing')
                actual_grad = torch.autograd.grad(result, args[0], retain_graph=True)[0]
                expected_grad = torch.autograd.grad(reference, args[0], retain_graph=True)[0]
                if not torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6):
                    fail('custom smoothing gradient mismatch')
                def used(gradient):
                    loss_backward_calls[0] += 1
                    return gradient
                result.register_hook(used)
                checked_losses.add(module)
            return result
        finally:
            local.depth = depth

    torch.load = load
    torch.nn.Module.load_state_dict = load_state
    torch.nn.Module._call_impl = call
    torch.nn.functional.cross_entropy = cross_entropy

    def finish():
        if violations:
            raise RuntimeError('MODEL_CONTRACT_FAILED: ' + '; '.join(violations))
        if not calls:
            fail('no instrumented image-model forward was observed')
        if custom_losses and not inference and not loss_backward_calls[0]:
            fail('custom smoothing loss was not verified on a backward path')
        if standard_required and not inference and not standard_backward_calls[0]:
            fail('standard smoothing loss was not verified on a backward path')
        print('MODEL_CONTRACT_VERIFIED ' + json.dumps({
            'input_sizes': sorted(observed), 'model_calls': calls,
            'inference_only': inference, 'checkpoint_restored': bool(restored) if inference else None,
            'verified_custom_losses': len(checked_losses), 'loss_backward_calls': loss_backward_calls[0],
            'standard_smoothing_backward_calls': standard_backward_calls[0],
        }), flush=True)
    return finish
