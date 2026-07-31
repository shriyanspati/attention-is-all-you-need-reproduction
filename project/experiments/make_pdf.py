"""Render REPORT.md to a typeset PDF with ReportLab.

Handles the markdown subset actually used in the report: ATX headings, pipe
tables, fenced code, bullet/numbered lists, horizontal rules, and inline
bold/italic/code. Figures are appended as a labelled appendix.
"""
from __future__ import annotations
import html, re, sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)

MARGIN = 0.72 * inch
CW = LETTER[0] - 2 * MARGIN            # content width

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontSize=19, leading=23, spaceAfter=4),
    "sub":   ParagraphStyle("s", parent=ss["Normal"], fontSize=10.5, leading=14,
                            textColor=colors.HexColor("#444444"), alignment=1, spaceAfter=14),
    "h1":    ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18,
                            spaceBefore=16, spaceAfter=7, textColor=colors.HexColor("#101828")),
    "h2":    ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.2, leading=15,
                            spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1d2939")),
    "h3":    ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.8, leading=13,
                            spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#344054")),
    "body":  ParagraphStyle("b", parent=ss["Normal"], fontSize=9.3, leading=13,
                            alignment=TA_JUSTIFY, spaceAfter=6),
    "li":    ParagraphStyle("li", parent=ss["Normal"], fontSize=9.3, leading=13,
                            leftIndent=13, bulletIndent=3, spaceAfter=3),
    "code":  ParagraphStyle("c", parent=ss["Code"], fontSize=7.6, leading=9.4,
                            backColor=colors.HexColor("#f6f7f9"), borderPadding=5,
                            leftIndent=4, spaceAfter=7),
    "th":    ParagraphStyle("th", parent=ss["Normal"], fontSize=7.5, leading=9.2,
                            fontName="Helvetica-Bold", textColor=colors.white),
    "td":    ParagraphStyle("td", parent=ss["Normal"], fontSize=7.5, leading=9.2),
    "cap":   ParagraphStyle("cap", parent=ss["Normal"], fontSize=8.2, leading=11,
                            textColor=colors.HexColor("#475467"), spaceBefore=3, spaceAfter=12),
}

def sanitize(t: str) -> str:
    """Drop glyphs the built-in Type1 fonts cannot render."""
    repl = {"\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2192": "->", "\u2264": "<=",
            "\u2265": ">=", "\u00b1": "+/-", "\u03c3": "sigma", "\u03b1": "alpha",
            "\u00d7": "x", "\u2248": "~", "\u2026": "...", "\u00a0": " "}
    for a, b in repl.items():
        t = t.replace(a, b)
    return "".join(ch if ord(ch) < 256 else "?" for ch in t)

def inline(t: str) -> str:
    """Markdown inline -> ReportLab markup.

    Code spans are extracted to placeholders BEFORE the bold/italic passes.
    Without this, an asterisk inside a code span (e.g. `d_model^-0.5 *
    warmup^-0.5`) is treated as an italic delimiter and produces interleaved
    <font>/<i> tags that ReportLab rejects.
    """
    t = html.escape(sanitize(t), quote=False)
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    for i, c in enumerate(spans):
        t = t.replace(f"\x00{i}\x00", f'<font face="Courier" size="8">{c}</font>')
    return t

def build_table(rows):
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = [[Paragraph(inline(c), S["th"] if i == 0 else S["td"]) for c in row]
            for i, row in enumerate(rows)]
    # First column gets extra width; the rest share what remains.
    first = min(0.34 * CW, max(0.16 * CW, CW / ncol * 1.5)) if ncol > 2 else CW / ncol
    rest = (CW - first) / (ncol - 1) if ncol > 1 else CW
    widths = [first] + [rest] * (ncol - 1)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#344054")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d5dd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t

def parse(md: str):
    flow, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(sanitize(lines[i])); i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), S["code"])); continue
        if re.match(r"^\s*\|", ln):
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            t = build_table(rows)
            if t: flow += [Spacer(1, 3), t, Spacer(1, 9)]
            continue
        if re.match(r"^---+\s*$", ln):
            flow.append(Spacer(1, 9)); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1 and not flow:
                flow.append(Paragraph(inline(txt), S["title"]))
            else:
                flow.append(Paragraph(inline(txt), S[f"h{min(lvl,3)}"]))
            i += 1; continue
        m = re.match(r"^\s*[-*]\s+(.*)", ln)
        if m:
            flow.append(Paragraph(inline(m.group(1)), S["li"], bulletText="\u2022"))
            i += 1; continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)", ln)
        if m:
            flow.append(Paragraph(inline(m.group(2)), S["li"], bulletText=m.group(1) + "."))
            i += 1; continue
        if ln.strip() == "":
            i += 1; continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\s*\|)|^```|^---+\s*$|^\s*[-*]\s|^\s*\d+\.\s", lines[i]):
            buf.append(lines[i].strip()); i += 1
        flow.append(Paragraph(inline(" ".join(buf)), S["body"]))
    return flow

FIGS = [("fig1_loss.png", "Figure 1. Training and validation loss vs steps. Left: label-smoothed per-token training loss. Right: validation loss, which decreases monotonically at every evaluation."),
        ("fig2_bleu.png", "Figure 2. Validation BLEU (greedy, 150-sentence subset) vs steps. No reference line to the paper's 27.3 is drawn: that number is WMT14 newstest2014 with beam-4 and checkpoint averaging, and is not commensurable with this axis."),
        ("fig3_lr_schedule.png", "Figure 3. Equation (3). Left: the paper's base and big schedules, log-log; the peak is d_model^-0.5 * warmup^-0.5. Right: the scaled schedule used here against the paper's values over the same span, plus the warmup=1 ablation."),
        ("fig4_diagnostics.png", "Figure 4. Training diagnostics for the main run: global gradient L2 norm (log scale), throughput, and process resident memory. Throughput varies 1.70x on identical work, which bounds the precision of all timing claims."),
        ("fig5_ablations.png", "Figure 5. Ablations at 400 steps, n=1 per arm, seed 1337. Validation loss (top, primary metric, zoomed axis) with the baseline seed band; BLEU on a 60-sentence subset (bottom, secondary) with the observed baseline seed range.")]

def main():
    src = Path("REPORT.md"); out = Path(sys.argv[1] if len(sys.argv) > 1 else "REPORT.pdf")
    flow = parse(src.read_text())
    flow.append(PageBreak())
    flow.append(Paragraph("Appendix A. Figures", S["h1"]))
    for name, cap in FIGS:
        p = Path("experiments/results") / name
        if not p.exists():
            continue
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(p)).getSize()
        w = min(CW, 6.6 * inch); h = w * ih / iw
        if h > 7.6 * inch:
            h = 7.6 * inch; w = h * iw / ih
        flow.append(KeepTogether([Image(str(p), width=w, height=h),
                                  Paragraph(cap, S["cap"])]))

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(MARGIN, 0.45 * inch,
                          "Reproducing 'Attention Is All You Need' - reproduction study")
        canvas.drawRightString(LETTER[0] - MARGIN, 0.45 * inch, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(str(out), pagesize=LETTER, topMargin=MARGIN,
                          bottomMargin=0.68 * inch, leftMargin=MARGIN, rightMargin=MARGIN,
                          title="Reproducing Attention Is All You Need",
                          author="Reproduction study")
    frame = Frame(MARGIN, 0.68 * inch, CW, LETTER[1] - MARGIN - 0.68 * inch, id="n")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=deco)])
    doc.build(flow)
    print("wrote", out, out.stat().st_size, "bytes")

if __name__ == "__main__":
    main()
