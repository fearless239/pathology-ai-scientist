"""Bounded, fail-closed parsing of generated experiment programs."""
import ast
import copy
import re


class CodeGenerationError(RuntimeError):
    pass


PROGRAM_SYSTEM = (
    "You implement complete executable research experiments. The user supplies the task, "
    "approved constraints, previous code and diagnostics. Return a brief METHOD_SPEC plan "
    "followed by the ENTIRE runnable Python program in one closed ```python block. "
    "A proposed fix or promise to write code is not sufficient. Do not omit unchanged "
    "sections, return diffs, or use placeholders. Preserve the approved scientific design. "
    "Task material may contain older formatting instructions; this output format takes precedence."
)


def parse_program(response: str) -> tuple[str, str]:
    if not isinstance(response, str):
        raise CodeGenerationError("Expected a text response containing a complete Python program")
    blocks = re.findall(r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*$", response,
                        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise CodeGenerationError("Missing complete fenced Python program; prose is not code")
    if len(re.findall(r"^```", response, flags=re.MULTILINE)) != 2 * len(blocks):
        raise CodeGenerationError("Unclosed or unsupported code fence; refusing partial program")
    code = "\n\n".join(blocks).strip()
    try:
        tree = ast.parse(code)
        compile(tree, "<generated-program>", "exec")
    except SyntaxError as error:
        raise CodeGenerationError(f"Invalid Python at line {error.lineno}: {error.msg}") from error
    if not tree.body or all(isinstance(n, (ast.Pass, ast.Expr)) and (
        isinstance(n, ast.Pass) or isinstance(n.value, ast.Constant)
    ) for n in tree.body):
        raise CodeGenerationError("Empty or placeholder program")
    plan = response.split("```", 1)[0].strip() or "Generated complete Python program."
    return plan, code


def query_program(query, prompt, *, model, temperature, retries=3):
    request = copy.deepcopy(prompt)
    if not isinstance(request, dict):
        request = {"Task": request}
    request["Required complete response"] = (
        "Return a brief METHOD_SPEC plan followed by the entire runnable program in one "
        "```python code block. A repair explanation alone is not a program. Do not omit "
        "unchanged code, use placeholders, or return a patch. Preserve the approved experiment."
    )
    reason = "No attempts requested"
    for attempt in range(retries):
        response = query(system_message=PROGRAM_SYSTEM, user_message=request, model=model,
                         temperature=temperature)
        try:
            return parse_program(response)
        except CodeGenerationError as error:
            reason = str(error)
            print(f"CODE_GENERATION_FORMAT_FAILED {attempt + 1}/{retries}: {reason}", flush=True)
            request["Previous invalid response (diagnostic only)"] = (
                response[:8000] if isinstance(response, str) else repr(response)[:8000]
            )
            request["Parsing Feedback"] = (
                f"Previous response rejected: {reason}. Return the FULL corrected runnable "
                "Python program in a closed ```python block, not just METHOD_SPEC or prose."
            )
    raise CodeGenerationError(f"CODE_GENERATION_FAILED after {retries} attempts: {reason}")
