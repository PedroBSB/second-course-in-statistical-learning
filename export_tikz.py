"""
export_tikz.py — Convert Python source files to LaTeX (tcolorbox) snippets.

Outputs ONLY the \\begin{tcolorbox}...\\end{tcolorbox} block so you can
paste it directly into any chapter of your Overleaf document.

The generated file starts with a comment block listing every package and
color definition you must add to your preamble once.

Usage:
    python export_tikz.py source/linear_regression.py        # XeLaTeX/LuaLaTeX (default)
    python export_tikz.py source/linear_regression.py --pdflatex  # pdfLaTeX fallback font
    python export_tikz.py                                     # all .py in source/

Output: images/<stem>.tex

Overleaf engine:
    XeLaTeX / LuaLaTeX  -> Menu > Compiler > XeLaTeX  (supports JetBrains Mono)
    pdfLaTeX            -> use --pdflatex flag          (uses Inconsolata instead)
"""

from __future__ import annotations

import io
import sys
import token as _token
import tokenize
from pathlib import Path

# ---------------------------------------------------------------------------
# VSCode Dark token colors (extracted from live CodeImage CSS via Playwright)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "background":  "#1E1E1E",  # editor background   (ͼoy)
    "frame_outer": "#151516",  # outer window frame
    "default":     "#9AD6FE",  # default text         (ͼou)
    "keyword":     "#529DDA",  # def, return, import… (ͼp6)
    "function":    "#DCDCA8",  # function names        (ͼpg)
    "number":      "#B4CDA7",  # numeric literals      (ͼpb)
    "string":      "#CE9178",  # string literals
    "comment":     "#8DA1B9",  # # comments            (ͼ4m)
    "operator":    "#FAFAFA",  # operators/punctuation (ͼpi)
    "bracket":     "#DBD700",  # () {} []              (ͼpj / ͼpk)
    "type":        "#4EC9B0",  # class / type names
    "decorator":   "#DCDCA8",  # decorators
    "lineno":      "#7C8083",  # line numbers          (ͼow)
    "selection":   "#264F78",  # selection background  (ͼox)
    "btn_red":     "#FF5F57",  # macOS close button
    "btn_yellow":  "#FEBC2E",  # macOS minimise button
    "btn_green":   "#28C840",  # macOS maximise button
}

# ---------------------------------------------------------------------------
# Python token classifier
# ---------------------------------------------------------------------------
PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

BUILTIN_TYPES = frozenset({
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "bytes", "bytearray", "complex", "type", "object",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "OSError",
    "IOError", "FileNotFoundError", "NotImplementedError",
})

TokenKind = str  # key into COLORS


def tokenize_python(source: str) -> list[tuple[int, int, int, int, TokenKind]]:
    """Return (start_row, start_col, end_row, end_col, colour_key) for every token."""
    results: list[tuple[int, int, int, int, TokenKind]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return results

    for tok in tokens:
        ttype = tok.type
        tstr  = tok.string
        srow, scol = tok.start
        erow, ecol = tok.end

        if ttype == _token.COMMENT:
            kind = "comment"
        elif ttype == _token.STRING:
            kind = "string"
        elif ttype == _token.NUMBER:
            kind = "number"
        elif ttype == _token.NAME:
            if tstr in PYTHON_KEYWORDS:
                kind = "keyword"
            elif tstr in BUILTIN_TYPES:
                kind = "type"
            else:
                kind = "default"
        elif ttype == _token.OP:
            kind = "bracket" if tstr in "()[]{}" else "operator"
        elif ttype in (_token.NEWLINE, _token.NL, _token.INDENT,
                       _token.DEDENT, _token.ENDMARKER):
            continue
        else:
            kind = "default"

        results.append((srow, scol, erow, ecol, kind))

    return results


def _annotate_function_names(
    source_lines: list[str],
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> list[tuple[int, int, int, int, TokenKind]]:
    """Upgrade the NAME token immediately after 'def'/'class' to 'function'."""
    out = []
    prev_kind = ""
    prev_str  = ""
    for span in spans:
        srow, scol, erow, ecol, kind = span
        token_str = source_lines[srow - 1][scol:ecol] if srow <= len(source_lines) else ""
        if prev_kind == "keyword" and prev_str in ("def", "class") and kind == "default":
            kind = "function"
        out.append((srow, scol, erow, ecol, kind))
        prev_kind = kind
        prev_str  = token_str
    return out


# ---------------------------------------------------------------------------
# Line colouring
# ---------------------------------------------------------------------------
def _colourise_line(
    line_text: str,
    line_no: int,
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> list[tuple[str, TokenKind]]:
    """Split one source line into (text_fragment, colour_key) segments."""
    relevant = sorted(
        (s for s in spans if s[0] == line_no and s[2] == line_no),
        key=lambda s: s[1],
    )
    segments: list[tuple[str, TokenKind]] = []
    cursor = 0
    for _, scol, _, ecol, kind in relevant:
        if scol > cursor:
            segments.append((line_text[cursor:scol], "default"))
        segments.append((line_text[scol:ecol], kind))
        cursor = ecol
    if cursor < len(line_text):
        segments.append((line_text[cursor:], "default"))
    return segments


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def _latex_escape(text: str) -> str:
    """Escape all LaTeX-special characters for use inside \\textcolor{}{}.

    Spaces are replaced with \\ (backslash-space) so that multiple
    consecutive spaces (indentation) are preserved in \\ttfamily context
    and are not collapsed by TeX's inter-word space rules.
    """
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{",  r"\{")
    text = text.replace("}",  r"\}")
    text = text.replace("$",  r"\$")
    text = text.replace("&",  r"\&")
    text = text.replace("#",  r"\#")
    text = text.replace("^",  r"\^{}")
    text = text.replace("_",  r"\_")
    text = text.replace("~",  r"\textasciitilde{}")
    text = text.replace("%",  r"\%")
    text = text.replace("<",  r"\textless{}")
    text = text.replace(">",  r"\textgreater{}")
    text = text.replace("|",  r"\textbar{}")
    # Replace every space with an explicit inter-word space so that
    # consecutive spaces (Python indentation) are not collapsed by TeX.
    text = text.replace(" ",  r"\ ")
    return text


def _color_name(key: str) -> str:
    """Map a COLORS key to a LaTeX color name, e.g. 'keyword' -> 'vscKeyword'."""
    # Handle keys with underscores: frame_outer -> vscFrameOuter
    parts = key.split("_")
    camel = "".join(p.capitalize() for p in parts)
    return f"vsc{camel}"


def _define_colors_block(colors: dict[str, str]) -> str:
    """Return one \\definecolor line per entry in *colors*."""
    lines = []
    for key, hexval in colors.items():
        r = int(hexval[1:3], 16)
        g = int(hexval[3:5], 16)
        b = int(hexval[5:7], 16)
        lines.append(f"\\definecolor{{{_color_name(key)}}}{{RGB}}{{{r},{g},{b}}}")
    return "\n".join(lines)


def _render_line(
    line_text: str,
    line_no: int,
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> str:
    """Render one source line as a sequence of \\textcolor{}{} commands."""
    segments = _colourise_line(line_text.rstrip("\n"), line_no, spans)
    parts = []
    for text, kind in segments:
        if not text:
            continue
        parts.append(f"\\textcolor{{{_color_name(kind)}}}{{{_latex_escape(text)}}}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Preamble comment block (engine-aware)
# ---------------------------------------------------------------------------
def _preamble_comment(use_pdflatex: bool) -> str:
    color_block = _define_colors_block(COLORS)
    if use_pdflatex:
        font_block = (
            "% pdfLaTeX font (JetBrains Mono not available; using Inconsolata)\n"
            "\\usepackage[scaled=0.85]{inconsolata}"
        )
        engine_note = "pdfLaTeX"
    else:
        font_block = (
            "% XeLaTeX / LuaLaTeX — Overleaf: Menu > Compiler > XeLaTeX\n"
            "\\usepackage{fontspec}\n"
            "\\setmonofont[Scale=0.85]{JetBrains Mono}"
        )
        engine_note = "XeLaTeX or LuaLaTeX"

    return f"""%% ============================================================
%% PASTE THE FOLLOWING INTO YOUR PREAMBLE (once per document)
%% Engine required: {engine_note}
%% ============================================================
%%
%% % ---- Font -------------------------------------------------
%% {font_block.replace(chr(10), chr(10) + "%% ")}
%%
%% % ---- Colors -----------------------------------------------
%% \\usepackage{{xcolor}}
%% {color_block.replace(chr(10), chr(10) + "%% ")}
%%
%% % ---- Packages (load BEFORE \\tcbset) ----------------------
%% \\usepackage{{alltt}}
%% \\usepackage{{tikz}}
%% \\usetikzlibrary{{calc}}
%% \\usepackage[most]{{tcolorbox}}
%% \\tcbuselibrary{{skins,breakable}}
%%
%% % ---- tcolorbox style (paste AFTER the libraries above) ----
%% \\tcbset{{
%%   codewindow/.style={{
%%     enhanced, breakable,
%%     arc=8pt, outer arc=8pt,
%%     boxrule=12pt,
%%     colframe=vscFrameOuter,
%%     colback=vscBackground,
%%     colupper=vscDefault,
%%     left=4pt, right=4pt, top=8pt, bottom=8pt,
%%   }},
%% }}
%% ============================================================"""


# ---------------------------------------------------------------------------
# Line wrapping
# ---------------------------------------------------------------------------
# Maximum visible characters per line before wrapping.
# At \tiny in a 4.5in text block minus ~2em line-number column ≈ 62 chars.
_WRAP_WIDTH = 62
# Indent added to continuation lines (matches common Python indent feel).
_CONT_INDENT = "    "
# Continuation marker colour (dim, like a line-continuation glyph).
_CONT_MARKER = f"\\textcolor{{vscLineno}}{{\\textbackslash{{}}}}"


def _visible_len(rendered: str) -> int:
    """Return the number of visible characters in a rendered LaTeX line.

    Strips all \\textcolor{name}{...} wrappers and other LaTeX commands,
    counting only the content characters.
    """
    import re
    # Remove \textcolor{name}{ ... } — handle up to 4 levels of nesting
    s = rendered
    for _ in range(6):
        s = re.sub(r"\\textcolor\{[^{}]+\}\{([^{}]*)\}", r"\1", s)
    # Remove remaining simple commands like \mbox{}, \textbackslash{}, etc.
    s = re.sub(r"\\[a-zA-Z]+\{[^{}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    return len(s)


def _split_segments_at(
    segments: list[tuple[str, TokenKind]],
    max_chars: int,
) -> tuple[list[tuple[str, TokenKind]], list[tuple[str, TokenKind]]]:
    """Split segment list so the first part has at most *max_chars* visible chars.

    Splits at a token boundary (never mid-token).
    Returns (first_part, rest).
    """
    acc = 0
    for i, (text, kind) in enumerate(segments):
        if acc + len(text) > max_chars:
            # Split this token's text at a space if possible
            for j in range(len(text) - 1, 0, -1):
                if text[j] in " ,(" and acc + j <= max_chars:
                    before = text[: j + 1]
                    after  = text[j + 1 :]
                    first  = segments[:i] + [(before, kind)]
                    rest   = [(after, kind)] + segments[i + 1 :]
                    return first, rest
            # No space found — split at token boundary
            return segments[:i], segments[i:]
        acc += len(text)
    return segments, []


def _render_line_wrapped(
    line_text: str,
    line_no: int,
    spans: list[tuple[int, int, int, int, TokenKind]],
    wrap_width: int = _WRAP_WIDTH,
    cont_indent: str = _CONT_INDENT,
) -> tuple[str, int]:
    """Render one source line, wrapping if longer than *wrap_width*.

    Returns (latex_string, number_of_physical_lines_produced).
    Continuation lines are indented and prefixed with a dim backslash marker.
    """
    segments = _colourise_line(line_text.rstrip("\n"), line_no, spans)
    physical_lines: list[str] = []

    first_pass = True
    while segments:
        first, segments = _split_segments_at(segments, wrap_width)
        parts = []
        for text, kind in first:
            if not text:
                continue
            parts.append(
                f"\\textcolor{{{_color_name(kind)}}}{{{_latex_escape(text)}}}"
            )
        rendered = "".join(parts)
        if not first_pass:
            # Continuation line: dim backslash + indent
            indent_escaped = _latex_escape(cont_indent)
            rendered = (
                f"{_CONT_MARKER}"
                f"\\textcolor{{vscDefault}}{{{indent_escaped}}}"
                + rendered
            )
        physical_lines.append(rendered)
        first_pass = False

    if not physical_lines:
        return "", 1

    return "\n".join(physical_lines), len(physical_lines)


# ---------------------------------------------------------------------------
# Main snippet builder
# ---------------------------------------------------------------------------
def build_snippet(source_path: Path, *, use_pdflatex: bool = False) -> str:
    """Return a LaTeX snippet (tcolorbox block only, no preamble/document).

    Each source line is rendered as an inline paragraph:
        \\noindent\\makebox[2em][r]{lineno}\\hspace{4pt}{code}\\par
    so the breakable tcolorbox can split between any two lines.
    No minipage, no alltt — spaces are preserved via explicit \\ (backslash-space)
    inserted by _latex_escape for every space character.
    Long lines are wrapped at _WRAP_WIDTH visible chars with a continuation marker.
    """
    source = source_path.read_text(encoding="utf-8")
    lines  = source.splitlines()

    raw_spans = tokenize_python(source)
    spans     = _annotate_function_names(lines, raw_spans)

    # Build one LaTeX paragraph per physical line.
    # Each paragraph: \noindent\makebox[2em][r]{lineno}\hspace{4pt}{code}\par
    body_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        lineno_tex = f"\\textcolor{{vscLineno}}{{{i}}}"
        if not line.strip():
            # Blank source line — emit an empty strut so vertical spacing is preserved
            body_lines.append(
                f"  \\noindent\\makebox[2.0em][r]{{{lineno_tex}}}\\hspace{{4pt}}"
                f"\\strut\\par"
            )
        else:
            rendered, n_physical = _render_line_wrapped(line, i, spans)
            physical = rendered.split("\n")
            # First physical line carries the real line number
            body_lines.append(
                f"  \\noindent\\makebox[2.0em][r]{{{lineno_tex}}}\\hspace{{4pt}}"
                f"{{\\tiny\\ttfamily {physical[0]}}}\\par"
            )
            # Continuation lines (wrapped) carry a dim ~ placeholder
            for cont in physical[1:]:
                cont_no = "\\textcolor{vscLineno}{~}"
                body_lines.append(
                    f"  \\noindent\\makebox[2.0em][r]{{{cont_no}}}\\hspace{{4pt}}"
                    f"{{\\tiny\\ttfamily {cont}}}\\par"
                )

    code_body = "\n".join(body_lines)
    filename = _latex_escape(source_path.name)
    preamble_hint = _preamble_comment(use_pdflatex)

    return f"""{preamble_hint}
%% Auto-generated by export_tikz.py — source: {source_path.name}
%%
\\begin{{tcolorbox}}[codewindow]

  % macOS chrome bar: traffic lights + filename
  {{\\tiny\\ttfamily%
  \\tikz[baseline=-0.6ex]{{%
    \\fill[vscBtnRed]    (0,0)     circle (4pt);
    \\fill[vscBtnYellow] (12pt,0)  circle (4pt);
    \\fill[vscBtnGreen]  (24pt,0)  circle (4pt);
  }}\\hspace{{10pt}}%
  \\textcolor{{vscDefault}}{{{filename}}}}}\\par
  \\vspace{{4pt}}%
  {{\\color{{vscLineno}}\\hrule height 0.3pt}}%
  \\vspace{{4pt}}%

  % Code lines — each is an independent paragraph so tcolorbox can break here
{code_body}

\\end{{tcolorbox}}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def export_file(source_file: Path, *, use_pdflatex: bool = False) -> Path:
    out_dir = Path("images")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (source_file.stem + ".tex")
    snippet = build_snippet(source_file, use_pdflatex=use_pdflatex)
    out_path.write_text(snippet, encoding="utf-8")
    print(f"  Written: {out_path}")
    return out_path


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_pdflatex = "--pdflatex" in sys.argv

    files = [Path(a) for a in args] if args else sorted(Path("source").glob("*.py"))

    if not files:
        print("No .py files found in source/")
        sys.exit(1)

    engine = "pdfLaTeX" if use_pdflatex else "XeLaTeX/LuaLaTeX"
    print(f"Engine: {engine}")

    for f in files:
        if not f.exists():
            print(f"  File not found: {f}")
            continue
        print(f"Processing: {f}")
        export_file(f, use_pdflatex=use_pdflatex)

    print("Done.")


if __name__ == "__main__":
    main()
