#!/usr/bin/env python3
import ast
import io
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("external/",)
C_LIKE = {".h", ".hpp", ".cpp", ".inc", ".cs", ".fbs", ".js"}
CSS = {".rcss", ".css"}
MARKUP = {".rml", ".html"}
PYTHON = {".py"}


@dataclass
class Comment:
    start: int
    end: int
    marked: bool
    docstring: bool = False


def is_cmake(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix == ".cmake"


def covers(path: Path) -> bool:
    relative = path.as_posix()
    if relative.startswith(EXCLUDED):
        return False
    return path.suffix in C_LIKE | CSS | MARKUP | PYTHON or is_cmake(path)


def skip_quoted(text: str, i: int, quote: str) -> int:
    i += 1
    while i < len(text) and text[i] != quote and text[i] != "\n":
        i += 2 if text[i] == "\\" else 1
    return i + 1


def is_digit_separator(text: str, i: int) -> bool:
    j = i
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_.'"):
        j -= 1
    return j < i and text[j].isdigit()


def raw_string_end(text: str, i: int) -> int | None:
    if i == 0 or text[i - 1] != "R":
        return None
    before = i - 2
    while before >= 0 and text[before] in "u8LU":
        before -= 1
    if before >= 0 and (text[before].isalnum() or text[before] == "_"):
        return None
    open_paren = text.find("(", i)
    delimiter = text[i + 1:open_paren]
    close = text.find(")" + delimiter + '"', open_paren)
    return len(text) if close < 0 else close + len(delimiter) + 2


def c_like_comments(text: str, offset: int = 0, end: int | None = None, template_strings: bool = False) -> list[Comment]:
    comments = []
    i, end = offset, len(text) if end is None else end
    while i < end:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i, end)
            j = end if j < 0 else j
            comments.append(Comment(i, j, text.startswith("//!", i)))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            j = end if j < 0 else j + 2
            comments.append(Comment(i, j, False))
            i = j
        elif c == '"':
            i = raw_string_end(text, i) or skip_quoted(text, i, '"')
        elif c == "'":
            i = i + 1 if is_digit_separator(text, i) else skip_quoted(text, i, "'")
        elif c == "`" and template_strings:
            j = i + 1
            while j < end and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return comments


def css_comments(text: str) -> list[Comment]:
    comments = []
    i = 0
    while i < len(text):
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = len(text) if j < 0 else j + 2
            comments.append(Comment(i, j, text.startswith("/*!", i)))
            i = j
        elif text[i] in "\"'":
            i = skip_quoted(text, i, text[i])
        else:
            i += 1
    return comments


def markup_comments(text: str) -> list[Comment]:
    comments = []
    i = 0
    while (i := text.find("<", i)) >= 0:
        if text.startswith("<!--", i):
            j = text.find("-->", i + 4)
            j = len(text) if j < 0 else j + 3
            comments.append(Comment(i, j, text.startswith("<!--!", i)))
            i = j
        elif text.startswith("<script", i):
            body = text.find(">", i) + 1
            close = text.find("</script>", body)
            close = len(text) if close < 0 else close
            comments += c_like_comments(text, body, close, template_strings=True)
            i = close
        else:
            i += 1
    return comments


def bracket_end(text: str, i: int) -> int | None:
    j = i + 1
    while j < len(text) and text[j] == "=":
        j += 1
    if j >= len(text) or text[j] != "[":
        return None
    close = text.find("]" + "=" * (j - i - 1) + "]", j)
    return len(text) if close < 0 else close + j - i + 1


def cmake_comments(text: str) -> list[Comment]:
    comments = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "#":
            j = bracket_end(text, i + 1) if i + 1 < len(text) and text[i + 1] == "[" else None
            if j is None:
                j = text.find("\n", i)
                j = len(text) if j < 0 else j
            comments.append(Comment(i, j, text.startswith("#!", i)))
            i = j
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        elif c == "[":
            i = bracket_end(text, i) or i + 1
        else:
            i += 1
    return comments


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def python_comments(text: str) -> list[Comment]:
    offsets = line_offsets(text)
    comments = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        start = offsets[token.start[0] - 1] + token.start[1]
        comments.append(Comment(start, start + len(token.string), token.string.startswith("#!")))
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        first = node.body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
            continue
        start = offsets[first.lineno - 1] + first.col_offset
        comments.append(Comment(start, offsets[first.end_lineno - 1] + first.end_col_offset, False, docstring=True))
    return sorted(comments, key=lambda comment: comment.start)


def comments_of(path: Path, text: str) -> list[Comment]:
    if path.suffix in PYTHON:
        return python_comments(text)
    if is_cmake(path):
        return cmake_comments(text)
    if path.suffix in CSS:
        return css_comments(text)
    if path.suffix in MARKUP:
        return markup_comments(text)
    return c_like_comments(text, template_strings=path.suffix == ".js")


MARKS = {"c": "//!", "css": "/*! */", "markup": "<!--! -->", "hash": "#!"}


def mark_of(path: Path) -> str:
    if path.suffix in PYTHON or is_cmake(path):
        return MARKS["hash"]
    if path.suffix in CSS:
        return MARKS["css"]
    if path.suffix in MARKUP:
        return MARKS["markup"]
    return MARKS["c"]


def violations(path: Path, text: str) -> list[tuple[int, str]]:
    found = []
    previous_line = None
    for comment in comments_of(path, text):
        line = text.count("\n", 0, comment.start) + 1
        last_line = text.count("\n", 0, comment.end) + 1
        is_shebang = comment.start == 0 and text.startswith("#!/")
        if is_shebang:
            continue
        if comment.docstring:
            found.append((line, "docstring: name the code instead"))
        elif not comment.marked:
            found.append((line, f"comment without the {mark_of(path)} mark: name the code instead"))
        elif last_line != line:
            found.append((line, "comment on more than one line"))
        elif previous_line == line - 1:
            found.append((line, "comment directly below another comment: keep one line"))
        previous_line = last_line
    return found


def tracked_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [Path(name) for name in listed.split("\0") if name]


def main(arguments: list[str]) -> int:
    files = [Path(argument) for argument in arguments] or tracked_files()
    failed = False
    for path in files:
        if not covers(path) or not (ROOT / path).is_file():
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        try:
            found = violations(path, text)
        except SyntaxError as error:
            found = [(error.lineno or 0, f"cannot parse: {error.msg}")]
        for line, message in found:
            print(f"{path.as_posix()}:{line}: {message}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
