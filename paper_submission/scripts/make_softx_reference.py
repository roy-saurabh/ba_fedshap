#!/usr/bin/env python3
"""
Generate a SoftwareX-styled reference.docx for pandoc.
SoftwareX (Elsevier) requirements:
  - Font: Times New Roman 12pt (body), 12pt bold (headings)
  - Line spacing: double (body), single (abstract, captions)
  - Margins: 2.54 cm all sides
  - No automatic indentation for first paragraph after heading
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# Page margins: 2.54 cm
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

def set_font(run, name="Times New Roman", size=12, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic

def set_para_spacing(para, line_rule=WD_LINE_SPACING.DOUBLE, space_before=0, space_after=6):
    pf = para.paragraph_format
    pf.line_spacing_rule = line_rule
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.first_line_indent = None

def add_style(doc, name, base_name, font_name="Times New Roman", font_size=12,
              bold=False, italic=False, line_rule=WD_LINE_SPACING.DOUBLE,
              space_before=0, space_after=6, alignment=WD_ALIGN_PARAGRAPH.LEFT):
    styles = doc.styles
    if name in [s.name for s in styles]:
        style = styles[name]
    else:
        style = styles.add_style(name, 1)  # 1 = paragraph style
    style.font.name = font_name
    style.font.size = Pt(font_size)
    style.font.bold = bold
    style.font.italic = italic
    pf = style.paragraph_format
    pf.line_spacing_rule = line_rule
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.alignment = alignment
    return style

# Normal body text — double-spaced Times New Roman 12pt
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
pf = normal.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
pf.space_before = Pt(0)
pf.space_after = Pt(6)

# Heading 1 — numbered section heading
h1 = doc.styles["Heading 1"]
h1.font.name = "Times New Roman"
h1.font.size = Pt(12)
h1.font.bold = True
h1.font.all_caps = False
h1.paragraph_format.space_before = Pt(12)
h1.paragraph_format.space_after = Pt(6)
h1.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Heading 2
h2 = doc.styles["Heading 2"]
h2.font.name = "Times New Roman"
h2.font.size = Pt(12)
h2.font.bold = True
h2.font.italic = True
h2.paragraph_format.space_before = Pt(6)
h2.paragraph_format.space_after = Pt(3)
h2.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Abstract style (single-spaced, indented)
abstract_style = add_style(doc, "Abstract", "Normal",
                           line_rule=WD_LINE_SPACING.SINGLE,
                           space_before=6, space_after=12)
abstract_style.paragraph_format.left_indent = Cm(1.27)
abstract_style.paragraph_format.right_indent = Cm(1.27)

# Caption style
caption = doc.styles["Caption"]
caption.font.name = "Times New Roman"
caption.font.size = Pt(10)
caption.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Add a placeholder body paragraph so pandoc picks up Normal style
p = doc.add_paragraph("Body text paragraph.")
set_para_spacing(p)

out = "paper_submission/softx_reference.docx"
doc.save(out)
print(f"Saved: {out}")
