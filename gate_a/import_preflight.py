"""Build a trusted sandbox launcher; never import generated dependencies on the host."""
from __future__ import annotations

import ast


def sandbox_launcher(code: str, script_path: str, requirements=None) -> str:
    """Check unconditional module-level imports before running any experiment code.

    Optional/conditional and function-local imports keep their original semantics.
    The check runs inside the same isolated container as the experiment, using its
    installed packages. It neither instantiates models nor downloads weights.
    """
    tree = ast.parse(code)
    imports = [
        (ast.unparse(node), node.module if isinstance(node, ast.ImportFrom) else None,
         [alias.name for alias in node.names], node.lineno)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    # This is trusted host-authored code, not part of the generated-code policy.
    # Passing it as an argument avoids a mutable launcher file in /workspace.
    guard = ''
    if requirements:
        import inspect
        from .model_runtime import install_model_guard
        guard = inspect.getsource(install_model_guard) + f'\n_finish_model_guard = install_model_guard({requirements!r})\n'
    execution = f'runpy.run_path({script_path!r}, run_name="__main__")'
    if requirements:
        execution = f'''try:
    runpy.run_path({script_path!r}, run_name="__main__")
except SystemExit as error:
    if error.code in (None, 0):
        _finish_model_guard()
    raise
else:
    _finish_model_guard()'''
    return f'''import difflib
import importlib
import importlib.metadata
import runpy
import sys

for statement, module_name, names, line in {imports!r}:
    try:
        exec(statement, {{}})
    except (ImportError, AttributeError) as error:
        details = []
        if module_name:
            try:
                module = importlib.import_module(module_name)
                for name in names:
                    if name != "*" and not hasattr(module, name):
                        candidates = difflib.get_close_matches(name, dir(module), n=3, cutoff=0.6)
                        details.append(f"{{name}}: available similar names={{candidates}}")
            except ImportError:
                pass
        root = (module_name or names[0]).split(".")[0]
        try:
            details.append(f"{{root}} version={{importlib.metadata.version(root)}}")
        except importlib.metadata.PackageNotFoundError:
            pass
        print(f"IMPORT_PREFLIGHT_FAILED line {{line}}: {{statement}}; {{error}}; "
              + "; ".join(details), file=sys.stderr, flush=True)
        print("Use the installed API. Do not change the approved model or disable pretrained weights to repair an import.",
              file=sys.stderr, flush=True)
        sys.exit(86)

sys.argv = [{script_path!r}]
{guard}
{execution}
'''



def validate_dataset_access(code: str, dataset_path) -> None:
    """Reject provably invalid unconditional NPZ reads; leave dynamic branches alone."""
    if dataset_path is None or not (dataset_path / "dataset.npz").is_file():
        return
    import numpy as np
    from .policy import CodePolicyError

    with np.load(dataset_path / "dataset.npz", allow_pickle=False) as data:
        keys = set(data.files)
    numpy_names, loaders, datasets = set(), set(), set()
    unknown = object()
    constants = {}

    def constant(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id, unknown)
        if isinstance(node, ast.Attribute) and node.attr == "files" and isinstance(node.value, ast.Name) and node.value.id in datasets:
            return keys
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = constant(node.left), constant(node.comparators[0])
            if left is unknown or right is unknown:
                return unknown
            if isinstance(node.ops[0], (ast.In, ast.NotIn)) and isinstance(right, (set, tuple, list, str)):
                try:
                    present = left in right
                except TypeError:
                    return unknown
                return present if isinstance(node.ops[0], ast.In) else not present
        if isinstance(node, ast.IfExp):
            test = constant(node.test)
            if isinstance(test, bool):
                return constant(node.body if test else node.orelse)
        return unknown

    def is_load(value):
        if not isinstance(value, ast.Call) or not value.args:
            return False
        func = value.func
        loader = (isinstance(func, ast.Name) and func.id in loaders) or (
            isinstance(func, ast.Attribute) and func.attr == "load"
            and isinstance(func.value, ast.Name) and func.value.id in numpy_names
        )
        path = value.args[0]
        return loader and constant(path) == "/dataset/dataset.npz"

    def unconditional_nodes(node):
        # Lazy and short-circuit expressions may never evaluate their subscripts.
        if isinstance(node, (ast.Lambda, ast.IfExp, ast.BoolOp, ast.ListComp,
                             ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from unconditional_nodes(child)

    def visit_statements(statements):
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy":
                        numpy_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
                loaders.update(a.asname or a.name for a in node.names if a.name == "load")
            elif isinstance(node, ast.With):
                for item in node.items:
                    if is_load(item.context_expr) and isinstance(item.optional_vars, ast.Name):
                        datasets.add(item.optional_vars.id)
                visit_statements(node.body)
            elif isinstance(node, ast.If):
                test = constant(node.test)
                if isinstance(test, bool):
                    visit_statements(node.body if test else node.orelse)
                else:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                            constants.pop(child.id, None)
                            datasets.discard(child.id)
            elif isinstance(node, (ast.For, ast.While, ast.Try)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        constants.pop(child.id, None)
                        datasets.discard(child.id)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
                for access in unconditional_nodes(node):
                    if (isinstance(access, ast.Subscript) and isinstance(access.value, ast.Name)
                            and access.value.id in datasets
                            and constant(access.slice) is not unknown
                            and (not isinstance(constant(access.slice), str) or constant(access.slice) not in keys)):
                        raise CodePolicyError(
                            f"DATASET_INTERFACE_FAILED line {access.lineno}: "
                            f"Resolved key {constant(access.slice)!r} is absent from the mounted NPZ. "
                            f"Available keys: {sorted(keys)}. Use the exported-view interface."
                        )
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    loaded = is_load(node.value)
                    resolved = constant(node.value)
                    for target in targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = resolved
                            datasets.discard(target.id)
                            if loaded:
                                datasets.add(target.id)
            # No hoisting across if/try/function/class scopes: inference-only
            # execution may intentionally lack train arrays used in a branch.

    visit_statements(ast.parse(code).body)
