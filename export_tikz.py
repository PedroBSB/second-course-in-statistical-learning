"""
export_tikz.py — Convert Python source files to LaTeX (tcolorbox) code images.

Mimics the visual style of CodeImage with VSCode Dark theme:
- Dark background #1E1E1E
- JetBrains Mono font
- macOS-style window chrome (traffic-light buttons)
- VSCode Dark syntax colors for Python tokens

Usage:
    python export_tikz.py source/linear_regression.py   # single file
    python export_tikz.py                                # all .py in source/

Output: images/<stem>.tex  (compile with XeLaTeX or LuaLaTeX)
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# VSCode Dark token colors (extracted from live CodeImage CSS via Playwright)
# ---------------------------------------------------------------------------
COLORS = {
    "background":   "#1E1E1E",   # editor background  (ͼoy)
    "frame_outer":  "#151516",   # outer window frame
    "default":      "#9AD6FE",   # default text        (ͼou)
    "keyword":      "#529DDA",   # def, return, import… (ͼp6)
    "function":     "#DCDCA8",   # function names       (ͼpg)
    "number":       "#B4CDA7",   # numeric literals     (ͼpb)
    "string":       "#CE9178",   # string literals
    "comment":      "#8DA1B9",   # # comments           (ͼ4m)
    "operator":     "#FAFAFA",   # operators / punctuation (ͼpi)
    "bracket":      "#DBD700",   # () {} []             (ͼpj / ͼpk)
    "type":         "#4EC9B0",   # class / type names
    "decorator":    "#DCDCA8",   # decorators
    "lineno":       "#7C8083",   # line numbers         (ͼow)
    "selection":    "#264F78",   # selection background (ͼox)
    # macOS traffic-light buttons
    "btn_red":      "#FF5F57",
    "btn_yellow":   "#FEBC2E",
    "btn_green":    "#28C840",
}

# ---------------------------------------------------------------------------
# Minimal Python tokeniser using the standard `tokenize` module
# ---------------------------------------------------------------------------
import tokenize
import io
import token as _token

# Keywords we want to colour like VSCode Dark
PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

# Built-in type / exception names coloured like types
BUILTIN_TYPES = frozenset({
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "bytes", "bytearray", "complex", "type", "object",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "OSError",
    "IOError", "FileNotFoundError", "NotImplementedError",
})

TokenKind = str  # colour key from COLORS

def tokenize_python(source: str) -> list[tuple[int, int, int, int, TokenKind]]:
    """
    Tokenise *source* and return a list of
    (start_row, start_col, end_row, end_col, colour_key).
    Rows are 1-based; columns 0-based (same as tokenize module).
    """
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
            if tstr in "()[]{}":
                kind = "bracket"
            elif tstr in ("=", "==", "!=", "<", ">", "<=", ">=",
                          "+", "-", "*", "/", "//", "%", "**",
                          "&", "|", "^", "~", "<<", ">>",
                          "->", ":", ",", "."):
                kind = "operator"
            else:
                kind = "operator"
        elif ttype in (_token.NEWLINE, _token.NL, _token.INDENT,
                       _token.DEDENT, _token.ENDMARKER):
            continue
        else:
            kind = "default"

        results.append((srow, scol, erow, ecol, kind))

    return results


# Post-process: detect function names (NAME token immediately after 'def' or 'class')
def _annotate_function_names(
    source_lines: list[str],
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> list[tuple[int, int, int, int, TokenKind]]:
    """Upgrade the token *after* `def` / `class` to colour 'function'."""
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
# Colour a single source line into a list of (text, colour_key) segments
# ---------------------------------------------------------------------------
def colourise_line(
    line_text: str,
    line_no: int,                            # 1-based
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> list[tuple[str, TokenKind]]:
    """Return list of (text_fragment, colour_key) for one source line."""
    relevant = [s for s in spans if s[0] == line_no and s[2] == line_no]
    # Sort by start column
    relevant.sort(key=lambda s: s[1])

    segments: list[tuple[str, TokenKind]] = []
    cursor = 0
    for srow, scol, erow, ecol, kind in relevant:
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
    """Escape characters that are special in LaTeX listings/verbatim context.

    We use lstlisting with mathescape=false and escapeinside=||, so the only
    characters we need to handle specially are the ones used for our escape
    sequences and standard LaTeX specials.
    """
    # Replace backslash first to avoid double-escaping
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


def _color_cmd(key: str) -> str:
    """Return a LaTeX color name for the given token key."""
    return f"vsc{key.capitalize()}"


def _define_colors(colors: dict[str, str]) -> str:
    """Return \\definecolor commands for all COLORS entries."""
    lines = []
    for key, hexval in colors.items():
        name = _color_cmd(key)
        r = int(hexval[1:3], 16)
        g = int(hexval[3:5], 16)
        b = int(hexval[5:7], 16)
        lines.append(
            f"\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}"
        )
    return "\n".join(lines)


def render_line_latex(
    line_text: str,
    line_no: int,
    spans: list[tuple[int, int, int, int, TokenKind]],
) -> str:
    """Render one source line as a LaTeX fragment with inline colour commands."""
    segments = colourise_line(line_text.rstrip("\n"), line_no, spans)
    parts = []
    for text, kind in segments:
        if not text:
            continue
        escaped = _latex_escape(text)
        color   = _color_cmd(kind)
        parts.append(f"\\textcolor{{{color}}}{{{escaped}}}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Build the full .tex document
# ---------------------------------------------------------------------------
def build_tex(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    lines  = source.splitlines()

    # Tokenise
    raw_spans = tokenize_python(source)
    spans     = _annotate_function_names(lines, raw_spans)

    # Rendered code lines
    code_lines = []
    for i, line in enumerate(lines, start=1):
        rendered = render_line_latex(line, i, spans)
        code_lines.append(rendered)

    code_body = "\\\\\n".join(code_lines)
    n_lines   = len(lines)

    # Line-number column content
    lineno_col = "\\\\\n".join(
        f"\\textcolor{{{_color_cmd('lineno')}}}{{{i}}}"
        for i in range(1, n_lines + 1)
    )

    color_defs = _define_colors(COLORS)

    filename = source_path.name

    doc = rf"""% Auto-generated by export_tikz.py
% Source: {filename}
% Compile with: xelatex or lualatex
%
\documentclass{{standalone}}

% ---- Font & encoding -------------------------------------------------------
\usepackage{{fontspec}}
\setmonofont[
  Scale       = 0.85,
  Ligatures   = TeX,
]{{"JetBrains Mono"}}

% ---- Color -----------------------------------------------------------------
\usepackage{{xcolor}}
{color_defs}

% ---- TikZ / tcolorbox ------------------------------------------------------
\usepackage{{tikz}}
\usepackage[most]{{tcolorbox}}
\usetikzlibrary{{calc}}

% ---- Misc ------------------------------------------------------------------
\usepackage{{setspace}}

% ============================================================================
\begin{{document}}

% Outer frame (matches CodeImage outer background)
\begin{{tcolorbox}}[
  enhanced,
  arc          = 12pt,
  outer arc    = 12pt,
  boxrule      = 0pt,
  colback      = vscFrame_outer,
  colframe     = vscFrame_outer,
  left         = 28pt,
  right        = 28pt,
  top          = 22pt,
  bottom       = 22pt,
]

  % ---- Window chrome (macOS traffic-light buttons) -------------------------
  \begin{{tikzpicture}}[remember picture, overlay]
    \node[circle, fill=vscBtn_red,    minimum size=12pt, inner sep=0pt]
      at ($(current bounding box.north west)+(20pt,-16pt)$) {{}};
    \node[circle, fill=vscBtn_yellow, minimum size=12pt, inner sep=0pt]
      at ($(current bounding box.north west)+(36pt,-16pt)$) {{}};
    \node[circle, fill=vscBtn_green,  minimum size=12pt, inner sep=0pt]
      at ($(current bounding box.north west)+(52pt,-16pt)$) {{}};
  \end{{tikzpicture}}
  \vspace{{16pt}}

  % ---- Code window ---------------------------------------------------------
  \begin{{tcolorbox}}[
    enhanced,
    arc          = 8pt,
    outer arc    = 8pt,
    boxrule      = 0pt,
    colback      = vscBackground,
    colframe     = vscBackground,
    left         = 6pt,
    right        = 6pt,
    top          = 14pt,
    bottom       = 14pt,
    fontupper    = \ttfamily\small,
  ]
    % ---- Filename tab at top -----------------------------------------------
    \noindent
    \colorbox{{vscBackground!80!black}}{{%
      \textcolor{{vscDefault}}{{\ttfamily\scriptsize {filename}}}%
    }}
    \vspace{{6pt}}

    % ---- Line numbers + code side-by-side ----------------------------------
    \noindent
    \begin{{tabular}}{{@{{}}r@{{\hspace{{10pt}}}}l@{{}}}}
{lineno_col.replace(chr(10), chr(10) + "      ")} &
{code_body.replace(chr(10), chr(10) + "      ")} \\
    \end{{tabular}}
  \end{{tcolorbox}}

\end{{tcolorbox}}

\end{{document}}
"""
    return doc


# ---------------------------------------------------------------------------
# Helpers: table approach doesn't work well with multi-line code rendering.
# Use a minipage / alltt approach instead.
# ---------------------------------------------------------------------------
def build_tex_v2(source_path: Path) -> str:
    """
    Better layout: line numbers and code side-by-side using two minipages
    inside a tcolorbox.
    """
    source = source_path.read_text(encoding="utf-8")
    lines  = source.splitlines()

    # Tokenise
    raw_spans = tokenize_python(source)
    spans     = _annotate_function_names(lines, raw_spans)
    n_lines   = len(lines)

    # Rendered code lines
    code_rendered = []
    for i, line in enumerate(lines, start=1):
        rendered = render_line_latex(line, i, spans)
        # Preserve leading spaces (convert to \phantom or \hspace)
        leading = len(line) - len(line.lstrip())
        if leading > 0:
            space_str = r"\hspace{" + f"{leading * 0.52}em" + "}"
            # The leading spaces are already part of rendered; we keep them
        code_rendered.append(rendered)

    color_defs = _define_colors(COLORS)
    filename   = source_path.name

    # Build lineno block (one per line, right-aligned)
    lineno_lines = "\n".join(
        r"\textcolor{" + _color_cmd("lineno") + "}{" + str(i) + r"}\\"
        for i in range(1, n_lines + 1)
    )

    # Build code block (one rendered line per \\ )
    code_lines_block = "\n".join(
        rendered + r"\\"
        for rendered in code_rendered
    )

    doc = rf"""% Auto-generated by export_tikz.py
% Source: {filename}
% Compile with: xelatex or lualatex
%
\documentclass{{standalone}}

% ---- Font & encoding -------------------------------------------------------
\usepackage{{fontspec}}
\setmonofont[
  Scale = 0.85,
]{{"JetBrains Mono"}}

% ---- Color -----------------------------------------------------------------
\usepackage{{xcolor}}
{color_defs}

% ---- TikZ / tcolorbox ------------------------------------------------------
\usepackage{{tikz}}
\usepackage[most]{{tcolorbox}}
\usetikzlibrary{{calc, positioning}}

% ---- Misc ------------------------------------------------------------------
\usepackage{{microtype}}
\usepackage{{ragged2e}}

% ============================================================================
\begin{{document}}

\tcbset{{
  codewindow/.style={{
    enhanced,
    arc=12pt, outer arc=12pt,
    boxrule=0pt,
    colback=vscFrame_outer,
    colframe=vscFrame_outer,
    left=28pt, right=28pt,
    top=12pt, bottom=24pt,
  }},
  codepanel/.style={{
    enhanced,
    arc=8pt, outer arc=8pt,
    boxrule=0pt,
    colback=vscBackground,
    colframe=vscBackground,
    left=8pt, right=8pt,
    top=10pt, bottom=10pt,
    fontupper=\ttfamily\footnotesize,
  }},
}}

\begin{{tcolorbox}}[codewindow]

  % macOS traffic-light buttons
  \noindent\hspace{{4pt}}%
  \tikz{{
    \fill[vscBtn_red]    (0,0) circle (5pt);
    \fill[vscBtn_yellow] (14pt,0) circle (5pt);
    \fill[vscBtn_green]  (28pt,0) circle (5pt);
  }}
  \vspace{{10pt}}

  \begin{{tcolorbox}}[codepanel]

    % Filename tab
    \noindent{{\ttfamily\scriptsize\textcolor{{vscDefault}}{{{filename}}}}}
    \vspace{{4pt}}\hrule height 0.4pt \vspace{{6pt}}

    % Line numbers (right-aligned) | Code
    \noindent
    \begin{{minipage}}[t]{{1.6em}}
      \setlength{{\baselineskip}}{{1.45em}}
      \raggedleft
{lineno_lines}
    \end{{minipage}}%
    \hspace{{8pt}}%
    \begin{{minipage}}[t]{{\dimexpr\linewidth-1.6em-8pt\relax}}
      \setlength{{\baselineskip}}{{1.45em}}
      \raggedright
{code_lines_block}
    \end{{minipage}}

  \end{{tcolorbox}}

\end{{tcolorbox}}

\end{{document}}
"""
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def export_file(source_file: Path) -> Path:
    out_dir = Path("images")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (source_file.stem + ".tex")
    tex = build_tex_v2(source_file)
    out_path.write_text(tex, encoding="utf-8")
    print(f"  Written: {out_path}")
    return out_path


def main() -> None:
    args = sys.argv[1:]
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(Path("source").glob("*.py"))

    if not files:
        print("No .py files found in source/")
        sys.exit(1)

    for f in files:
        if not f.exists():
            print(f"  File not found: {f}")
            continue
        print(f"Processing: {f}")
        export_file(f)

    print("Done.")


if __name__ == "__main__":
    main()
