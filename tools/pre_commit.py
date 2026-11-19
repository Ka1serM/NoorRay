import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP_EXTENSIONS = ("h", "hpp", "cpp", "inc")
PLATFORM_SUFFIXES = ("", "Desktop", "Web")
BUILD_DIRECTORY = ROOT / f"build-{platform.system().lower()}-relwithdebinfo"
WEB_BUILD_DIRECTORY = ROOT / "build-web-relwithdebinfo"


def git(*arguments: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arguments], cwd=ROOT, input=stdin, capture_output=True, text=True)


def staged_files() -> list[str]:
    listed = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").stdout
    return [name for name in listed.split("\0") if name and not name.startswith("external/")]


def format_staged(files: list[str]) -> bool:
    cpp = [name for name in files if name.rsplit(".", 1)[-1] in CPP_EXTENSIONS]
    if not cpp:
        return True
    extensions = ",".join(CPP_EXTENSIONS)
    patch = git("clang-format", "--staged", "--diff", "--whole-file", f"--extensions={extensions}", "--", *cpp).stdout
    if not patch.startswith("diff"):
        return True
    if git("apply", "--check", stdin=patch).returncode != 0:
        print("clang-format: a file has unstaged changes that conflict with its formatting. Stage or stash them, then commit again.")
        return False
    git("apply", "--cached", stdin=patch)
    git("apply", stdin=patch)
    formatted = [line.split(" b/", 1)[1] for line in patch.splitlines() if line.startswith("diff --git")]
    print("clang-format formatted: " + ", ".join(formatted))
    return True


def check_comments(files: list[str]) -> bool:
    return subprocess.run([sys.executable, str(Path(__file__).with_name("check_comments.py")), *files], cwd=ROOT).returncode == 0


def compile_database(build_directory: Path) -> dict[Path, dict] | None:
    database = build_directory / "compile_commands.json"
    if not database.is_file():
        print(f"clang-tidy: {database} does not exist. Configure the {build_directory.name.removeprefix('build-')} preset first.")
        return None
    return {Path(entry["file"]).resolve(): entry for entry in json.loads(database.read_text(encoding="utf-8"))}

def compiler_of(entry: dict) -> Path:
    command = entry.get("arguments") or entry["command"].split()
    return Path(command[0])


def translation_units(name: str, database: dict[Path, dict]) -> list[Path]:
    path = (ROOT / name).resolve()
    if path.suffix == ".cpp":
        return [path] if path in database else []
    candidates = (path.with_name(path.stem + suffix + ".cpp") for suffix in PLATFORM_SUFFIXES)
    return [candidate for candidate in candidates if candidate in database]


def run_tidy(binary: Path, directory: Path, unit: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), "-p", str(directory), "--quiet", str(unit)], cwd=ROOT, capture_output=True, text=True)


def web_entry_arguments(entry: dict) -> list[str]:
    arguments = shlex.split(entry["command"])
    parsed = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-Xclang" and index + 3 < len(arguments) and arguments[index + 1] in ("-include-pch", "-include"):
            index += 4
        elif argument == "-Xclang" and index + 1 < len(arguments) and arguments[index + 1] == "-fno-pch-timestamp":
            index += 2
        elif argument.startswith("-s") and argument[2:3].isupper() or argument == "-msimd128":
            index += 1
        else:
            parsed.append(argument)
            index += 1
    return parsed[1:]


def check_tidy(files: list[str]) -> bool:
    cpp = [name for name in files if name.rsplit(".", 1)[-1] in ("h", "hpp", "cpp")]
    if not cpp:
        return True
    database = compile_database(BUILD_DIRECTORY)
    if database is None:
        return False
    units = sorted({unit for name in cpp for unit in translation_units(name, database)})
    unchecked = [name for name in cpp if not translation_units(name, database)]
    web_database = compile_database(WEB_BUILD_DIRECTORY) if unchecked else {}
    web_units = sorted({unit for name in unchecked for unit in translation_units(name, web_database)}) if web_database else []
    unverified = [name for name in unchecked if not translation_units(name, web_database)] if web_database else unchecked
    if unverified:
        print(f"clang-tidy: not in the {BUILD_DIRECTORY.name} compile database, so not checked: " + ", ".join(unverified))
    if not units and not web_units:
        return True
    binary = compiler_of(next(iter(database.values()))).with_name("clang-tidy.exe" if os.name == "nt" else "clang-tidy")
    if not binary.is_file():
        print(f"clang-tidy: {binary} does not exist. Install the clang-tidy of the compiler that the build uses.")
        return False
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        results = list(pool.map(lambda unit: run_tidy(binary, BUILD_DIRECTORY, unit), units))
        if web_units:
            commands = [{"directory": web_database[unit]["directory"],
                         "file": web_database[unit]["file"],
                         "arguments": web_entry_arguments(web_database[unit])} for unit in web_units]
            with tempfile.TemporaryDirectory(prefix=f"{ROOT.name.lower()}-clang-tidy-") as directory:
                Path(directory, "compile_commands.json").write_text(json.dumps(commands), encoding="utf-8")
                results += list(pool.map(lambda unit: run_tidy(binary, Path(directory), unit), web_units))
    for result in results:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
    return all(result.returncode == 0 for result in results)


def main() -> int:
    files = staged_files()
    if not format_staged(files):
        return 1
    results = [check_comments(files), check_tidy(files)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
