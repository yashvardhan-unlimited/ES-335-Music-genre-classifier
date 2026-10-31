"""Check that line references in learn/*.md still point at the functions they name.

Run from repo root: python scripts/check_lesson_refs.py [--fix]
A reference looks like 'L94-101 `clip_handcrafted`' or 'Lines 21-45, `aggregate_clips`'. Bare
references ('line 61') name nothing, so they cannot be checked: prefer the named form.
Exits 1 if any reference is wrong; --fix rewrites them. A tool, not a test: moving code is not a bug.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

REF = re.compile(r"(\b[Ll](?:ines?\s+)?)(\d+)(?:([-–])(\d+))?(,?\s*(?:\*\*\s*)?`)(\w+)")
PATH = re.compile(r"`((?:src|scripts)/\w+\.py|app\.py)`")
Span = tuple[str, int, int]


def definitions() -> dict[str, list[Span]]:
    """Function or class name -> (file, first line including decorators, last line), tracked files only."""
    found: dict[str, list[Span]] = {}
    for f in subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split():
        for node in ast.walk(ast.parse(Path(f).read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                found.setdefault(node.name, []).append((f, start, node.end_lineno))
    return found


def fix_line(line: str, defs: dict[str, list[Span]], current_file: str | None) -> tuple[str, list[str]]:
    """The line with its references corrected, and one message per wrong reference.

    A name defined in several files is looked up in current_file, the file last named in the
    lesson. A name that stays ambiguous, or is not a definition, is left alone.
    """
    problems = []

    def repair(m: re.Match) -> str:
        hits = defs.get(m[6], [])
        hits = [h for h in hits if h[0] == current_file] or hits
        if len(hits) != 1:
            return m[0]
        file, start, end = hits[0]
        if int(m[2]) == start and (m[4] is None or int(m[4]) == end):
            return m[0]
        problems.append(f"{m[6]}: says {m[2]}-{m[4] or m[2]}, {file} has {start}-{end}")
        return f"{m[1]}{start}{m[3] or ''}{end if m[4] else ''}{m[5]}{m[6]}"

    return REF.sub(repair, line), problems


def main() -> int:
    fix, defs, wrong = "--fix" in sys.argv, definitions(), 0
    for lesson in sorted(Path("learn").glob("[0-9]*.md")):
        current, lines = None, []
        for number, line in enumerate(lesson.read_text(encoding="utf-8").split("\n"), 1):
            named = PATH.search(line)
            current = named[1] if named else current
            new, problems = fix_line(line, defs, current)
            lines.append(new)
            wrong += len(problems)
            for p in problems:
                print(f"{lesson.name}:{number} {p}")
        if fix:
            lesson.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{wrong} wrong references{', fixed' if fix and wrong else ''}")
    return 0 if fix or not wrong else 1


if __name__ == "__main__":
    sys.exit(main())
