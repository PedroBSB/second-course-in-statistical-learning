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
    """Escape all LaTeX-special characters for use inside \\textcolor{}{}."""
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
%% % ---- Packages ---------------------------------------------
%% \\usepackage{{alltt}}
%% \\usepackage{{tikz}}
%% \\usepackage[most]{{tcolorbox}}
%% \\tcbuselibrary{{breakable}}
%% \\usetikzlibrary{{calc, positioning}}
%%
%% % ---- tcolorbox styles (paste once, before first use) ------
%% \\tcbset{{
%%   codewindow/.style={{
%%     enhanced, arc=10pt, outer arc=10pt, boxrule=0pt,
%%     colback=vscFrameOuter, colframe=vscFrameOuter,
%%     left=20pt, right=20pt, top=10pt, bottom=18pt,
%%   }},
%%   codepanel/.style={{
%%     enhanced, breakable,
%%     arc=6pt, outer arc=6pt, boxrule=0pt,
%%     colback=vscBackground, colframe=vscBackground,
%%     left=4pt, right=4pt, top=8pt, bottom=8pt,
%%   }},
%% }}
%% ============================================================"""


# ---------------------------------------------------------------------------
# Main snippet builder
# ---------------------------------------------------------------------------
def build_snippet(source_path: Path, *, use_pdflatex: bool = False) -> str:
    """Return a LaTeX snippet (tcolorbox block only, no preamble/document).

    Uses alltt for the code body so that:
    - spaces are preserved verbatim (indentation works)
    - line breaks are natural (no \\ needed, no "no line to end" errors)
    - \textcolor{}{} commands still work inside alltt
    """
    source = source_path.read_text(encoding="utf-8")
    lines  = source.splitlines()
    n_lines = len(lines)

    raw_spans = tokenize_python(source)
    spans     = _annotate_function_names(lines, raw_spans)

    # Inside alltt, only \, {, } are special.  Spaces and newlines are literal.
    # We already escape those three in _latex_escape, so _render_line is safe.
    # We just need to NOT use \\ for line breaks — alltt uses real newlines.
    code_lines = [
        _render_line(line, i, spans)
        for i, line in enumerate(lines, start=1)
    ]
    # alltt treats a blank line as a paragraph break (adds extra vertical space).
    # Replace empty rendered lines with a \mbox{} to suppress that.
    code_alltt = "\n".join(
        (line if line.strip() else "\\mbox{}")
        for line in code_lines
    )

    # Line-number column: one number per line, right-aligned, same font size.
    # Use a simple tabular with a fixed-width column.
    lineno_lines = "\n".join(
        f"    \\textcolor{{vscLineno}}{{{i}}} \\\\"
        for i in range(1, n_lines + 1)
    )

    filename = _latex_escape(source_path.name)
    preamble_hint = _preamble_comment(use_pdflatex)

    return f"""{preamble_hint}
%% Auto-generated by export_tikz.py — source: {source_path.name}
%%
\\begin{{tcolorbox}}[codewindow]

  % macOS traffic-light buttons
  \\noindent\\hspace{{4pt}}%
  \\tikz[baseline=-0.6ex]{{%
    \\fill[vscBtnRed]    (0,0)     circle (4pt);
    \\fill[vscBtnYellow] (12pt,0)  circle (4pt);
    \\fill[vscBtnGreen]  (24pt,0)  circle (4pt);
  }}\\par\\vspace{{6pt}}

  \\begin{{tcolorbox}}[codepanel]

    % Filename bar
    {{\\ttfamily\\scriptsize\\textcolor{{vscDefault}}{{{filename}}}}}\\par
    \\vspace{{2pt}}\\hrule height 0.3pt\\vspace{{4pt}}

    % Layout: line numbers (right-aligned) | code (alltt, preserves spaces)
    \\noindent
    \\begin{{minipage}}[t]{{2.0em}}%
      {{\\scriptsize\\ttfamily\\color{{vscLineno}}%
       \\raggedleft\\setlength{{\\baselineskip}}{{1.45em}}%
{lineno_lines}
      }}%
    \\end{{minipage}}%
    \\hspace{{4pt}}%
    \\begin{{minipage}}[t]{{\\dimexpr\\linewidth-2.0em-4pt\\relax}}%
      \\begin{{alltt}}%
\\textcolor{{vscDefault}}{{\\scriptsize\\ttfamily%
{code_alltt}%
}}%
      \\end{{alltt}}%
    \\end{{minipage}}

  \\end{{tcolorbox}}

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
