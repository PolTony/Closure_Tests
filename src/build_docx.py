#!/usr/bin/env python3
"""
Word deliverables (build_docx.py) — JURI Lab

Converts the two submission documents from Markdown to Word (.docx) in the project's house format: Times New
Roman 11 point, one-inch margins, black text, short bold section labels as headings with keep-with-next,
figures embedded at their place in the text, tables as native Word tables with repeated headers, cantSplit rows,
and landscape orientation for wide multi-column tables, and the references on their own page in 9 point.

Uses a zero-dependency OpenXML builder with its own minimal package and styles, so the documents can be
rebuilt in a clean workspace without a pre-existing Word file.

The default command validates the existing Word files and retains manual edits. The optional convert() API remains available to developers; submission Markdown files are not required.
"""

import os
import re
import struct
import zipfile
from xml.sax.saxutils import escape as xml_escape

import juri_paths

BASE = juri_paths.ROOT
SUB = os.path.join(BASE, "submission")
FONT = "Times New Roman"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def readable_math(text):
    """Use readable Unicode math rather than printing Markdown/LaTeX markup."""
    def plain(m):
        s = m.group(1).strip()
        for command, symbol in (("beta", "β"), ("Delta", "Δ"), ("delta", "δ"), ("kappa", "κ"),
                                ("lambda", "λ"), ("theta", "θ"), ("times", "×"), ("pm", "±"),
                                ("geq", "≥"), ("leq", "≤"), ("approx", "≈"), ("rightarrow", "→")):
            s = re.sub(r"\\" + command + r"\b", symbol, s)
        s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\(?:left|right|quad|qquad)\b", " ", s)
        s = re.sub(r"\\(?:ln|log|exp|min|max|sum)\b", lambda x: x.group()[1:], s)
        return " ".join(s.split())
    text = re.sub(r"\$\$([\s\S]*?)\$\$", plain, text)
    return re.sub(r"\$([^$\n]+)\$", plain, text)


def _png_dimensions(path):
    """Read (width_px, height_px) from a PNG header using the standard library."""
    with open(path, "rb") as f:
        hdr = f.read(24)
    if len(hdr) >= 24 and hdr[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", hdr[16:24])
        return max(1, w), max(1, h)
    return 1200, 700


def _compute_col_weights(header, rows):
    weights = []
    for j in range(len(header)):
        h_clean = re.sub(r"[`*]", "", header[j])
        body_vals = [re.sub(r"[`*]", "", r[j]) if j < len(r) else "" for r in rows]
        max_body_len = max((len(v) for v in body_vals), default=0)
        longest_token = max(
            (len(tok) for v in body_vals for tok in v.split()),
            default=len(h_clean.split()[0]) if h_clean.split() else 6,
        )
        base_len = max(max_body_len, int(len(h_clean) * 0.65), longest_token + 3)
        min_floor = 14 if j == 0 else max(9, longest_token + 2)
        cap = 44 if len(header) >= 8 else 52
        weights.append(min(max(base_len, min_floor), cap))
    return weights


def _convert_stdlib(md_path, docx_path, references_on_own_page=True, figure_width=6.3):
    """Pure standard-library OpenXML (.docx) builder with landscape wide tables, tblHeader, cantSplit, and keepNext."""
    # A self-contained package avoids inheriting author metadata, styles or relationships
    # from a prior deliverable and supports installed workspaces with no existing DOCX.
    template_entries = {
        "[Content_Types].xml": b'''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>''',
        "_rels/.rels": b'''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>''',
        "word/styles.xml": b'''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr></w:rPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="000000"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="000000"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="000000"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:color w:val="000000"/></w:rPr></w:style></w:styles>''',
    }

    doc_ns = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mo="http://schemas.microsoft.com/office/mac/office/2008/main" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:mv="urn:schemas-microsoft-com:mac:vml" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
    )

    body_parts = []
    media_files = []  # (rId, media_target, disk_path)

    def _sect_pr_xml(landscape=False):
        if landscape:
            return (
                '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>'
                '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
                'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
            )
        return (
            '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        )

    def _runs_xml(text, size_pt=11.0, base_bold=False):
        text = readable_math(text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        out = []
        for piece in INLINE.split(text):
            if not piece:
                continue
            bold = base_bold
            italic = False
            font = FONT
            sz = size_pt
            content = piece
            if piece.startswith("**") and piece.endswith("**"):
                content = piece[2:-2]
                bold = True
            elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
                content = piece[1:-1]
                italic = True
            elif piece.startswith("`") and piece.endswith("`"):
                content = piece[1:-1]
                font = "Consolas"
                sz = max(7.0, size_pt - 1.5)
            content = content.replace(r"\*", "*")
            rpr = [f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>']
            if bold:
                rpr.append("<w:b/>")
            if italic:
                rpr.append("<w:i/>")
            rpr.append(f'<w:sz w:val="{int(round(sz * 2))}"/>')
            rpr.append(f'<w:szCs w:val="{int(round(sz * 2))}"/>')
            out.append(f'<w:r><w:rPr>{"".join(rpr)}</w:rPr><w:t xml:space="preserve">{xml_escape(content)}</w:t></w:r>')
        return "".join(out)

    def _para_xml(text, style=None, size_pt=11.0, base_bold=False, keep_next=False, space_before=None, space_after=120, left_indent=None, first_line=None, page_break_before=False):
        ppr = []
        if style:
            ppr.append(f'<w:pStyle w:val="{style}"/>')
        if keep_next:
            ppr.append("<w:keepNext/>")
        if page_break_before:
            ppr.append("<w:pageBreakBefore/>")
        sp_attrs = []
        if space_before is not None:
            sp_attrs.append(f'w:before="{space_before}"')
        if space_after is not None:
            sp_attrs.append(f'w:after="{space_after}"')
        if sp_attrs:
            ppr.append(f'<w:spacing {" ".join(sp_attrs)}/>')
        if left_indent is not None or first_line is not None:
            ind_attrs = []
            if left_indent is not None:
                ind_attrs.append(f'w:left="{left_indent}"')
            if first_line is not None:
                if first_line < 0:
                    ind_attrs.append(f'w:hanging="{abs(first_line)}"')
                else:
                    ind_attrs.append(f'w:firstLine="{first_line}"')
            ppr.append(f'<w:ind {" ".join(ind_attrs)}/>')
        ppr_str = f'<w:pPr>{"".join(ppr)}</w:pPr>' if ppr else ""
        return f'<w:p>{ppr_str}{_runs_xml(text, size_pt=size_pt, base_bold=base_bold)}</w:p>'

    def _table_xml(header, rows, font_size=9.0, usable_width_in=6.5):
        weights = _compute_col_weights(header, rows)
        total = sum(weights)
        usable_dxa = int(round(usable_width_in * 1440))
        col_dxa = [max(360, int(round(usable_dxa * w / total))) for w in weights]
        grid_cols = "".join(f'<w:gridCol w:w="{w}"/>' for w in col_dxa)
        tbl_pr = (
            '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:type="dxa" w:w="' + str(sum(col_dxa)) + '"/>'
            '<w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="6" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
            '<w:insideH w:val="nil"/><w:insideV w:val="nil"/>'
            '<w:left w:val="nil"/><w:right w:val="nil"/>'
            '</w:tblBorders></w:tblPr>'
        )
        tr_list = []
        # Header row with tblHeader and cantSplit
        h_cells = []
        for j, h in enumerate(header):
            h_cells.append(
                f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{col_dxa[j]}"/>'
                '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:color="000000"/></w:tcBorders></w:tcPr>'
                f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="20" w:after="20"/></w:pPr>'
                f'{_runs_xml(h, size_pt=font_size, base_bold=True)}</w:p></w:tc>'
            )
        tr_list.append(f'<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>{"".join(h_cells)}</w:tr>')
        for row in rows:
            r_cells = []
            for j in range(len(header)):
                val = row[j] if j < len(row) else ""
                r_cells.append(
                    f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{col_dxa[j]}"/></w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                    f'{_runs_xml(val, size_pt=font_size, base_bold=False)}</w:p></w:tc>'
                )
            tr_list.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(r_cells)}</w:tr>')
        return f'<w:tbl>{tbl_pr}<w:tblGrid>{grid_cols}</w:tblGrid>{"".join(tr_list)}</w:tbl><w:p/>'

    lines = open(md_path, encoding="utf-8").read().split("\n")
    if lines and lines[0].strip() == "---":
        end = next((j for j in range(1, len(lines)) if lines[j].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1:]
    i = 0
    in_code = False
    code = []
    para_buf = []
    in_landscape = False

    def switch_orientation(want_landscape):
        nonlocal in_landscape
        if want_landscape != in_landscape:
            body_parts.append(f'<w:p><w:pPr>{_sect_pr_xml(landscape=in_landscape)}</w:pPr></w:p>')
            in_landscape = want_landscape

    def flush_para():
        nonlocal para_buf
        if para_buf:
            text = " ".join(x.strip() for x in para_buf)
            keep = text.startswith("**Table ") or text.startswith("**Figure ")
            body_parts.append(_para_xml(text, keep_next=keep))
            para_buf = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush_para()
            if in_code:
                for cln in code:
                    body_parts.append(_para_xml(f"`{cln if cln else ' '}`", size_pt=8.0, space_after=0, left_indent=360))
                body_parts.append("<w:p/>")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(ln)
            i += 1
            continue
        if ln.strip() == "---":
            flush_para()
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m:
            flush_para()
            switch_orientation(False)
            level = len(m.group(1))
            text = m.group(2).strip()
            if references_on_own_page and text.lower().startswith("references"):
                body_parts.append(
                    _para_xml(text, style="Heading2", size_pt=12.0, base_bold=True, keep_next=True, space_before=240, space_after=80, page_break_before=True)
                )
                i += 1
                while i < len(lines):
                    if lines[i].strip():
                        body_parts.append(
                            _para_xml(lines[i].strip(), size_pt=9.0, space_after=60, left_indent=432, first_line=-432)
                        )
                    i += 1
                continue
            h_style = "Title" if level == 1 else f"Heading{min(level, 3)}"
            h_sz = {1: 14.0, 2: 12.0, 3: 11.0}[min(level, 3)]
            body_parts.append(_para_xml(text, style=h_style, size_pt=h_sz, base_bold=True, keep_next=True, space_before=240, space_after=80))
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            flush_para()
            switch_orientation(False)
            img_path = os.path.join(os.path.dirname(md_path), m.group(2))
            if os.path.exists(img_path):
                idx = len(media_files) + 1
                rid = f"rId{idx + 8}"
                target = f"media/image{idx}.png"
                media_files.append((rid, target, img_path))
                w_px, h_px = _png_dimensions(img_path)
                cx = int(round(figure_width * 914400))
                cy = int(round(cx * (h_px / float(w_px))))
                drawing = (
                    f'<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:r><w:drawing>'
                    f'<wp:inline xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                    f'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                    f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{idx}" name="Picture {idx}"/>'
                    f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                    f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                    f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{xml_escape(os.path.basename(img_path))}"/><pic:cNvPicPr/></pic:nvPicPr>'
                    f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                    f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr>'
                    f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
                )
                body_parts.append(drawing)
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i].lstrip())
                i += 1
            cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in block]
            cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            if cells:
                header, body = cells[0], cells[1:]
                wide = len(header) >= 8
                caption = None
                if wide and body_parts and re.search(r'<w:t[^>]*>Table ', body_parts[-1]):
                    caption = body_parts.pop()
                switch_orientation(wide)
                if caption:
                    body_parts.append(caption)
                fsize = 8.0 if len(header) >= 10 else (8.5 if len(header) >= 8 else 9.0)
                uwidth = 9.0 if wide else 6.5
                body_parts.append(_table_xml(header, body, font_size=fsize, usable_width_in=uwidth))
                if wide:
                    switch_orientation(False)
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
        if m:
            flush_para()
            marker = "•" if m.group(2) in ("-", "*") else m.group(2)
            item_txt = f"{marker}  {m.group(3)}"
            i += 1
            while (
                i < len(lines)
                and lines[i].startswith("   ")
                and lines[i].strip()
                and not lines[i].lstrip().startswith("|")
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
            ):
                item_txt += " " + lines[i].strip()
                i += 1
            body_parts.append(_para_xml(item_txt, left_indent=432, first_line=-288))
            continue
        if ln.strip() == "":
            flush_para()
            i += 1
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", ln.strip()):
            flush_para()
            body_parts.append(_para_xml(ln.strip(), keep_next=True, space_before=200, space_after=40))
            i += 1
            continue
        if ln.startswith("*") and ln.endswith("*") and ln.count("*") == 2:
            flush_para()
            body_parts.append(_para_xml(ln, size_pt=10.0))
            i += 1
            continue
        para_buf.append(ln)
        i += 1

    flush_para()
    body_parts.append(_sect_pr_xml(landscape=in_landscape))
    doc_xml = (doc_ns + "".join(body_parts) + "</w:body></w:document>").encode("utf-8")

    # Build word/_rels/document.xml.rels
    rels_items = [
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
    ]
    for rid, target, _ in media_files:
        rels_items.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels_items)
        + "</Relationships>"
    ).encode("utf-8")

    template_entries["word/document.xml"] = doc_xml
    template_entries["word/_rels/document.xml.rels"] = rels_xml
    if "word/styles.xml" in template_entries:
        st_xml = template_entries["word/styles.xml"].decode("utf-8")
        st_xml = re.sub(r'\s+w:(asciiTheme|hAnsiTheme|eastAsiaTheme|cstheme)="[^"]*"', "", st_xml)
        template_entries["word/styles.xml"] = st_xml.encode("utf-8")
    if "[Content_Types].xml" in template_entries and media_files:
        ct_xml = template_entries["[Content_Types].xml"].decode("utf-8")
        if 'Extension="png"' not in ct_xml:
            ct_xml = ct_xml.replace(
                "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>'
            )
            template_entries["[Content_Types].xml"] = ct_xml.encode("utf-8")

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in template_entries.items():
            zout.writestr(name, data)
        for _, target, disk_path in media_files:
            with open(disk_path, "rb") as f:
                zout.writestr(f"word/{target}", f.read())

    print("wrote", os.path.relpath(docx_path, BASE), f"({os.path.getsize(docx_path) // 1024} KB)")


def convert(md_path, docx_path, references_on_own_page=True, figure_width=6.3):
    _convert_stdlib(md_path, docx_path, references_on_own_page=references_on_own_page, figure_width=figure_width)


def validate_submission():
    """Keep the submitted Word documents authoritative, including manual edits."""
    import xml.etree.ElementTree as ET
    for name in ("PROPOSAL_STAGE1.docx", "TECHNICAL_APPENDIX.docx"):
        path = os.path.join(SUB, name)
        with zipfile.ZipFile(path) as package:
            ET.fromstring(package.read("word/document.xml"))
        print(f"validated submission/{name}; document content retained")


if __name__ == "__main__":
    validate_submission()
