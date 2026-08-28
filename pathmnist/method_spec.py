from __future__ import annotations

import ast
import json
import re
from typing import Any


METHOD_SPEC_PREFIX = "# PATH_AI_METHOD_SPEC: "


CONCEPT_ALIASES: dict[str, set[str]] = {
    "label_smoothing": {"labelsmoothing", "smoothedtargets", "smoothingalpha", "smoothingfactor"},
    "color_perturbation": {
        "colorjitter",
        "colourjitter",
        "stainjitter",
        "colorperturbation",
        "colourperturbation",
    },
    "rotation": {"rotation", "randomrotation", "rotate"},
    "flip": {
        "flip",
        "randomhorizontalflip",
        "randomverticalflip",
        "horizontalflip",
        "verticalflip",
    },
    "hard_example_mining": {"hardexample", "hardmining", "confidence"},
    "contrastive_learning": {
        "contrastive",
        "supervisedcontrastive",
        "supcon",
        "temperature",
    },
}


SIGNAL_CONCEPTS = {
    "color_jitter": "color_perturbation",
    "stain_jitter": "color_perturbation",
    "color_perturbation": "color_perturbation",
    "rotation": "rotation",
    "flip": "flip",
    "hard": "hard_example_mining",
    "confidence": "hard_example_mining",
    "contrastive": "contrastive_learning",
    "temperature": "contrastive_learning",
}

GENERIC_COMPONENT_CATEGORIES = {
    "architecture", "model", "modelarchitecture", "loss", "lossfunction",
    "optimizer", "optimization", "regularization",
    "augmentation",
    "dataaugmentation",
    "imageaugmentation",
    "geometricaugmentation",
    "trainingtransform",
    "cnnarchitecture", "dataloading", "classificationmetrics", "sgd",
    "supervisedtraining", "smoothingfactortuning", "evaluation", "training",
}

REQUIREMENT_OWNERS = {'convlayers': 'architecture', 'fromscratch': 'initialization',
                      'numclasses': 'architecture', 'trainsubsetfraction': 'data',
                      'testaccuracy': 'evaluation', 'crossentropy': 'loss'}


def known_component_category(category: str) -> bool:
    """Classify metadata, never use it as evidence of an implemented method."""
    normalized = normalize_symbol(category)
    if (normalized in GENERIC_COMPONENT_CATEGORIES or normalized in REQUIREMENT_OWNERS
            or canonical_concept(category) in CONCEPT_ALIASES):
        return True
    # Only compose tuning with a recognized concept/parameter, not arbitrary names.
    if normalized.endswith('tuning'):
        base = normalized[:-len('tuning')]
        return (canonical_concept(base) in CONCEPT_ALIASES
                or base in {'learningrate', 'batchsize', 'weightdecay', 'momentum'})
    return False


def dynamic_smoothing_lines(tree):
    """Function parameters need runtime verification, not literal-only rejection."""
    lines = []
    called = {_call_name(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in called:
            continue
        parameters = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and normalize_symbol(_call_name(node.func) or '').endswith(('crossentropy', 'crossentropyloss'))
                    and any(k.arg == 'label_smoothing' and isinstance(k.value, ast.Name)
                            and k.value.id in parameters for k in node.keywords)):
                lines.append(node.lineno)
    return lines


def classify_requirements(signals):
    """Read-only projection: legacy mixed signals retain their separate owners."""
    result = {'intervention': [], 'architecture': [], 'initialization': [], 'data': [], 'evaluation': [], 'loss': []}
    for signal in signals:
        owner = REQUIREMENT_OWNERS.get(normalize_symbol(signal), 'intervention')
        result[owner].append(signal)
    return result


def custom_smoothing_classes(code):
    """Candidates for sandbox numerical verification, not proof by class name."""
    tree = ast.parse(code)
    names = []
    calls = { _call_name(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call) }
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in calls:
            continue
        forward = next((f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == 'forward'), None)
        if forward is None:
            continue
        symbols = {_call_name(n.func) for n in ast.walk(forward) if isinstance(n, ast.Call)}
        if (any(s and s.endswith('log_softmax') for s in symbols)
                and any(s and s.endswith('gather') for s in symbols)
                and any(s and s.endswith('mean') for s in symbols)
                and any(isinstance(n, ast.Attribute) and n.attr == 'smoothing' for n in ast.walk(forward))):
            names.append(node.name)
    return names


def normalize_symbol(value: str) -> str:
    """Normalize snake_case, kebab-case, and CamelCase to one symbol form."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def canonical_concept(signal: str) -> str | None:
    normalized = normalize_symbol(signal)
    if normalized in {"transform", "transforms", "augmentation", "augment"}:
        return None
    direct = SIGNAL_CONCEPTS.get(signal.casefold())
    if direct:
        return direct
    for concept, aliases in CONCEPT_ALIASES.items():
        if normalized == normalize_symbol(concept) or normalized in aliases:
            return concept
    return signal.casefold()


def called_symbols(code: str) -> tuple[ast.AST | None, list[tuple[str, int]]]:
    """Collect functions/classes that are actually called, not merely imported."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, []
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            parts = [func.attr]
            value = func.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            name = ".".join(reversed(parts))
        else:
            continue
        calls.append((name, int(getattr(node, "lineno", 0))))
    return tree, calls


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return None
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _standard_training_call_lines(tree: ast.AST) -> tuple[set[int], bool]:
    """Trace standard Compose variables into dataset calls fed train arrays."""
    compose_lines: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        if normalize_symbol(_call_name(node.value.func) or "").endswith("compose"):
            compose_lines[target.id] = {
                int(getattr(child, "lineno", 0))
                for child in ast.walk(node.value)
                if isinstance(child, ast.Call)
            }
    used: set[int] = set()
    forwarded_parameters: set[str] = set()
    found_training_dataset = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = normalize_symbol(_call_name(node.func) or "")
        if "dataset" not in call_name:
            continue
        argument_names = {
            arg.id.casefold() for arg in node.args if isinstance(arg, ast.Name)
        }
        if not any("train" in name for name in argument_names):
            continue
        found_training_dataset = True
        transform = next(
            (item.value for item in node.keywords if item.arg == "transform"), None
        )
        if isinstance(transform, ast.Name):
            if transform.id in compose_lines:
                used.update(compose_lines[transform.id])
            else:
                forwarded_parameters.add(transform.id)
        elif isinstance(transform, ast.Call):
            used.update(
                int(getattr(child, "lineno", 0))
                for child in ast.walk(transform)
                if isinstance(child, ast.Call)
            )
    if forwarded_parameters:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg in forwarded_parameters
                    and isinstance(keyword.value, ast.Name)
                ):
                    used.update(compose_lines.get(keyword.value.id, set()))
    return used, found_training_dataset


def semantic_report(
    code: str, signals: list[str], method_spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    tree, calls = called_symbols(code)
    if tree is None:
        return {
            "status": "rejected",
            "passed": False,
            "required": [],
            "detected": [],
            "missing": [],
            "unknown": [],
            "calls": [],
            "signals": {},
            "has_training_operations": False,
            "reason": "invalid Python syntax",
        }
    required = list(
        dict.fromkeys(
            concept
            for signal in classify_requirements(signals)['intervention']
            if (concept := canonical_concept(signal)) is not None
        )
    )
    normalized_calls = [(normalize_symbol(name), name, line) for name, line in calls]
    training_call_lines, has_scoped_training_dataset = _standard_training_call_lines(tree)
    evidence: dict[str, list[int]] = {}
    for concept in required:
        aliases = {
            normalize_symbol(concept),
            *(normalize_symbol(alias) for alias in CONCEPT_ALIASES.get(concept, set())),
        }
        evidence[concept] = sorted(
            {
                line
                for normalized, _, line in normalized_calls
                if line > 0
                and any(alias in normalized for alias in aliases)
                and (concept not in {"color_perturbation", "rotation", "flip"}
                     or not has_scoped_training_dataset or line in training_call_lines)
            }
        )
    if "label_smoothing" in required:
        # A name or a MethodSpec claim alone does not establish active smoothing.
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        lines = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = normalize_symbol(_call_name(node.func) or "")
            if not name.endswith(("crossentropyloss", "crossentropy")):
                continue
            for keyword in node.keywords:
                if keyword.arg != "label_smoothing":
                    continue
                value = keyword.value
                amount = value.value if isinstance(value, ast.Constant) else constants.get(value.id) if isinstance(value, ast.Name) else None
                if type(amount) in (int, float) and 0 < amount < 1:
                    lines.append(node.lineno)
        lines.extend(dynamic_smoothing_lines(tree))
        custom = custom_smoothing_classes(code)
        lines.extend(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n.func) in custom)
        evidence["label_smoothing"] = lines
    # Structural contract fields are not necessarily function names.
    if "num_classes" in required:
        evidence["num_classes"] = sorted({node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "num_classes"})
    detected = [concept for concept in required if evidence.get(concept)]
    missing = [concept for concept in required if not evidence.get(concept)]
    declared = [] if method_spec is None else method_spec.get("components", [])
    unknown = sorted(
        {
            str(item.get("category"))
            for item in declared
            if isinstance(item, dict)
            and item.get("category")
            and not known_component_category(str(item['category']))
        }
    )
    has_training = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"backward", "step"}
        for node in ast.walk(tree)
    )
    status = "needs_review" if unknown else ("passed" if has_training and not missing else "rejected")
    return {
        "status": status,
        "passed": status == "passed",
        "required": required,
        "detected": detected,
        "missing": missing,
        "unknown": unknown,
        "calls": [{"symbol": name, "line": line} for _, name, line in normalized_calls],
        "signals": evidence,
        "has_training_operations": has_training,
        "requirement_groups": classify_requirements(signals),
        "runtime_checks": {'custom_smoothing_classes': custom_smoothing_classes(code),
                           'standard_smoothing_required': bool(dynamic_smoothing_lines(tree))},
    }


def validate_method_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("MethodSpec must be a JSON object")
    components = value.get("components")
    if not isinstance(components, list):
        raise ValueError("MethodSpec.components must be an array")
    normalized_components = []
    for item in components:
        if not isinstance(item, dict) or not isinstance(item.get("category"), str):
            raise ValueError("Each MethodSpec component needs a string category")
        symbols = item.get("implementation_symbols", [])
        if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError("implementation_symbols must be an array of strings")
        normalized_components.append(
            {
                "id": str(item.get("id") or item["category"]),
                "category": item["category"],
                "implementation_symbols": symbols,
            }
        )
    return {
        "schema_version": 1,
        "hypothesis": str(value.get("hypothesis", "")),
        "components": normalized_components,
        "changes": [str(item) for item in value.get("changes", [])],
        "preserved": [str(item) for item in value.get("preserved", [])],
    }


def parse_method_spec(text: str) -> dict[str, Any] | None:
    match = re.search(r"METHOD_SPEC\s*:\s*", text)
    if not match:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[match.end() :].lstrip())
        return validate_method_spec(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def infer_method_spec(code: str, signals: list[str]) -> dict[str, Any]:
    report = semantic_report(code, signals)
    calls = [item["symbol"] for item in report["calls"]]
    components = []
    for concept in report["detected"]:
        aliases = {normalize_symbol(alias) for alias in CONCEPT_ALIASES.get(concept, set())}
        symbols = [symbol for symbol in calls if any(alias in normalize_symbol(symbol) for alias in aliases)]
        components.append(
            {"id": concept, "category": concept, "implementation_symbols": symbols}
        )
    return validate_method_spec(
        {
            "hypothesis": "Legacy free-form code with host-inferred semantic evidence",
            "components": components,
            "changes": [],
            "preserved": [],
        }
    )


def attach_method_spec(code: str, spec: dict[str, Any]) -> str:
    payload = json.dumps(validate_method_spec(spec), ensure_ascii=False, separators=(",", ":"))
    lines = code.splitlines(keepends=True)
    insertion = 0
    if lines and lines[0].startswith("# Set random seed"):
        insertion = 1
    lines.insert(insertion, METHOD_SPEC_PREFIX + payload + "\n")
    return "".join(lines)


def extract_method_spec(code: str) -> dict[str, Any] | None:
    for line in code.splitlines()[:20]:
        if line.startswith(METHOD_SPEC_PREFIX):
            try:
                return validate_method_spec(json.loads(line[len(METHOD_SPEC_PREFIX) :]))
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
    return None
