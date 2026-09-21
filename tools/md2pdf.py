"""Render a project markdown document as a typeset PDF.

    python tools/md2pdf.py BLUEPRINT.md SBCI_Blueprint.pdf "optional subtitle"

Handles the subset these documents use: headings, paragraphs with inline
bold/italic/code/links, pipe tables, fenced code blocks, bullet and numbered
lists, block quotes, horizontal rules, and an explicit ``[[pagebreak]]``.

Font note: the dependency graphs use box-drawing characters, which reportlab's
built-in Type 1 fonts do not contain and would render as solid black boxes. A
system monospace face covering them is registered where one exists -- DejaVu on
Linux, Menlo on macOS -- and failing that the characters are transliterated to
ASCII, which keeps the diagrams readable rather than broken.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

PAGE = LETTER
MARGIN = 0.9 * inch
AVAILABLE = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b5b5b")
RULE = colors.HexColor("#d4d4d4")
CODE_BG = colors.HexColor("#f5f5f4")
CODE_INK = colors.HexColor("#1f4e5f")
HEAD_BG = colors.HexColor("#ecebe8")
ACCENT = colors.HexColor("#8a5a2b")

BOX_DRAWING = {
    "─": "-",
    "│": "|",
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",
    "├": "+",
    "┤": "+",
    "┬": "+",
    "┴": "+",
    "┼": "+",
    "►": ">",
    "▶": ">",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "•": "*",
}

MONO_CANDIDATES = (
    ("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", None, "DejaVuMono"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", None, "DejaVuMono"),
    ("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf", None, "DejaVuMono"),
    ("/System/Library/Fonts/Menlo.ttc", 0, "Menlo"),
    ("/Library/Fonts/Andale Mono.ttf", None, "AndaleMono"),
)


def register_mono() -> str:
    """A monospace face with box-drawing coverage, or the built-in fallback."""
    for path, index, name in MONO_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            if index is None:
                pdfmetrics.registerFont(TTFont(name, path))
            else:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            return name
        except Exception:  # noqa: BLE001 - any font failure falls through
            continue
    return "Courier"


MONO = register_mono()
MONO_HAS_BOX = MONO != "Courier"


def ascii_fallback(text: str) -> str:
    """Transliterate glyphs the chosen font cannot draw."""
    if MONO_HAS_BOX:
        return text
    for source, target in BOX_DRAWING.items():
        text = text.replace(source, target)
    return text


def styles():
    """The paragraph styles the renderer draws with, keyed by role."""
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "DocTitle",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=body,
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            spaceBefore=18,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            spaceBefore=13,
            spaceAfter=5,
            textColor=ACCENT,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13.5,
            spaceBefore=12,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=15,
            bulletIndent=4,
            spaceAfter=3.5,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            leftIndent=12,
            textColor=MUTED,
            fontName="Helvetica-Oblique",
        ),
        "code": ParagraphStyle(
            "Code",
            parent=body,
            fontName=MONO,
            fontSize=7.8,
            leading=10.4,
            textColor=CODE_INK,
            backColor=CODE_BG,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=9,
            leftIndent=2,
        ),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.6, leading=11.6, spaceAfter=0),
        "cellhead": ParagraphStyle(
            "CellHead",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=11.6,
            spaceAfter=0,
        ),
    }


S = styles()


def inline(text: str) -> str:
    """Markdown inline markup to reportlab's mini-HTML."""
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#2563a8">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", text)
    text = text.replace("--", "–")

    def restore(match: re.Match) -> str:
        code = html.escape(spans[int(match.group(1))], quote=False)
        return f'<font face="{MONO}" size="8.4" color="#1f4e5f">{code}</font>'

    return re.sub("\x00(\\d+)\x00", restore, text)


def column_widths(rows: list[list[str]]) -> list[float]:
    """Share the page width in proportion to each column's longest cell."""
    count = len(rows[0])
    longest = [max(len(row[i]) for row in rows) for i in range(count)]
    # Compressed so one long column cannot starve the others, but not so hard
    # that a column of code tokens is forced to break words.
    weights = [max(w, 6) ** 0.78 for w in longest]
    total = sum(weights)
    return [AVAILABLE * w / total for w in weights]


def build_table(rows: list[list[str]]):
    """Turn parsed pipe-table rows into a styled reportlab Table."""
    header, *body = rows
    data = [[Paragraph(inline(c), S["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), S["cell"]) for c in row] for row in body]

    table = Table(data, colWidths=column_widths(rows), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#b9b7b2")),
                ("INNERGRID", (0, 1), (-1, -1), 0.3, RULE),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9b7b2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafaf9")]),
            ]
        )
    )
    return table


def split_row(line: str) -> list[str]:
    """Split one pipe-table line into its trimmed cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(markdown: str) -> list:
    """Parse the markdown into a flowable story for reportlab to lay out."""
    lines = markdown.split("\n")
    story: list = []
    i = 0
    first_heading = True

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        if stripped == "[[pagebreak]]":
            story.append(PageBreak())
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            text = ascii_fallback("\n".join(block))
            longest = max((len(row) for row in text.split("\n")), default=0)
            style = S["code"]
            if longest > 92:
                style = ParagraphStyle("CodeNarrow", parent=style, fontSize=6.6, leading=8.8)
            story.append(Preformatted(text, style))
            continue

        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip())
        ):
            rows = [split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            story.append(Spacer(1, 3))
            story.append(build_table(rows))
            story.append(Spacer(1, 9))
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            story.append(Spacer(1, 5))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            story.append(Spacer(1, 7))
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            if level == 1 and first_heading:
                story.append(Paragraph(inline(text), S["title"]))
                first_heading = False
            elif level == 1:
                story.append(Paragraph(inline(text), S["h1"]))
            elif level == 2:
                story.append(Paragraph(inline(text), S["h2"]))
            else:
                story.append(Paragraph(inline(text), S["h3"]))
            i += 1
            continue

        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(block)), S["quote"]))
            continue

        if re.match(r"^[-*]\s+(.*)$", stripped) or re.match(r"^(\d+)[.)]\s+(.*)$", stripped):
            items = []
            while i < len(lines):
                current = lines[i].strip()
                b = re.match(r"^[-*]\s+(.*)$", current)
                n = re.match(r"^(\d+)[.)]\s+(.*)$", current)
                if b:
                    items.append(("•", b.group(1)))
                elif n:
                    items.append((f"{n.group(1)}.", n.group(2)))
                elif current and items and not re.match(r"^(#{1,6}\s|\||```|\[\[)", current):
                    marker, text = items[-1]
                    items[-1] = (marker, f"{text} {current}")
                else:
                    break
                i += 1
            for marker, text in items:
                story.append(Paragraph(inline(text), S["bullet"], bulletText=marker))
            story.append(Spacer(1, 5))
            continue

        block = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or re.match(r"^(#{1,6}\s|[-*]\s|\d+[.)]\s|\||```|>|\[\[)", nxt)
                or re.match(r"^(-{3,}|\*{3,}|_{3,})$", nxt)
            ):
                break
            block.append(nxt)
            i += 1
        story.append(Paragraph(inline(" ".join(block)), S["body"]))

    return story


def decorate(canvas, doc):
    """Draw the running footer: document title on the left, page number right."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.55 * inch, doc.title)
    canvas.drawRightString(PAGE[0] - MARGIN, 0.55 * inch, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 0.72 * inch, PAGE[0] - MARGIN, 0.72 * inch)
    canvas.restoreState()


def main(source: str, destination: str, subtitle: str | None = None) -> None:
    """Render ``source`` markdown to ``destination`` PDF."""
    markdown = Path(source).read_text()
    story = convert(markdown)
    if subtitle:
        story.insert(1, Paragraph(inline(subtitle), S["subtitle"]))

    heading = re.search(r"^#\s+(.*)$", markdown, re.MULTILINE)
    title = heading.group(1) if heading else Path(source).stem

    doc = BaseDocTemplate(
        destination,
        pagesize=PAGE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=title,
        author="Work package 2",
    )
    frame = Frame(MARGIN, MARGIN, AVAILABLE, PAGE[1] - 2 * MARGIN, id="body")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    doc.build(story)
    print(f"wrote {destination}  (monospace face: {MONO})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
