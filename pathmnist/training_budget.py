"""Conservative static accounting for ordinary generated training helpers.

This preflight is not a security sandbox. Docker still enforces wall-clock and
resource limits. Unknown loop bounds around training are rejected.
"""
import ast

from .scientific_integrity import IntegrityError


EPOCH_NAMES = ('max_ep', 'epochs', 'num_epochs', 'max_epochs', 'epochs_per_candidate')


def training_workload(code):
    tree = ast.parse(code)
    values = {}
    helpers = {}
    def statements(node):
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            yield from statements(child)

    for node in statements(tree):
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {'backward', 'step'} for n in ast.walk(node)
        ):
            helpers[node.name] = node

    def literal(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = literal(node.left), literal(node.right)
            if type(left) is int and type(right) is int:
                return left + right if isinstance(node.op, ast.Add) else left - right if isinstance(node.op, ast.Sub) else left * right
        if isinstance(node, ast.Name):
            return values.get(node.id)
        try:
            return ast.literal_eval(node)
        except (TypeError, ValueError):
            return None

    def bound(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'enumerate' and node.args:
                return bound(node.args[0])
            if node.func.id == 'range':
                args = [literal(arg) for arg in node.args]
                if args and all(type(a) is int for a in args):
                    return len(range(*args))
        value = literal(node)
        return len(value) if isinstance(value, (list, tuple)) else None

    count, epochs = 0, 0
    unknown_epochs = False

    def walk(node, multiplier=1, epoch_loop=False):
        nonlocal count, epochs, unknown_epochs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, (ast.For, ast.While)):
            has_training = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id in helpers for n in ast.walk(node))
            if has_training:
                size = bound(node.iter) if isinstance(node, ast.For) else None
                if size is None or size < 1:
                    raise IntegrityError('Training helper loop requires a positive static bound')
                for child in node.body:
                    is_epoch = any(isinstance(ref, ast.Name) and ref.id in EPOCH_NAMES for ref in ast.walk(node.iter)) or isinstance(node.target, ast.Name) and node.target.id in ('epoch', 'ep')
                    walk(child, multiplier * size, epoch_loop or is_epoch)
                for child in node.orelse:
                    walk(child, multiplier)
                return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
            definition = helpers[node.func.id]
            names = [arg.arg for arg in definition.args.args]
            assigned = dict(zip(names, [literal(arg) for arg in node.args]))
            assigned.update({kw.arg: literal(kw.value) for kw in node.keywords})
            defaults = dict(zip(names[len(names)-len(definition.args.defaults):],
                                [literal(d) for d in definition.args.defaults]))
            amount = next((assigned.get(k, defaults.get(k)) for k in EPOCH_NAMES
                           if assigned.get(k, defaults.get(k)) is not None), None)
            if amount is None:
                has_epoch_loop = any(isinstance(n, ast.For) and any(
                    isinstance(ref, ast.Name) and ref.id in EPOCH_NAMES
                    for ref in ast.walk(n.iter)) for n in ast.walk(definition))
                if has_epoch_loop:
                    amount = next((values[k] for k in EPOCH_NAMES if type(values.get(k)) is int), None)
                elif epoch_loop:
                    # A helper called once per epoch is not an entire training launch.
                    count += 1
                    epochs += multiplier
                    return
            count += multiplier
            if type(amount) is int and amount > 0:
                epochs += multiplier * amount
            else:
                unknown_epochs = True
        for child in ast.iter_child_nodes(node):
            walk(child, multiplier, epoch_loop)

    walk(tree)
    return count, None if unknown_epochs else epochs
