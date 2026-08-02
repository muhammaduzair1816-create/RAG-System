"""Builders for the PDF payloads the tests need.

``build_text_pdf`` emits a real PDF whose text lives in the content stream (the
normal case). ``build_scanned_pdf`` renders text into a bitmap and wraps that
bitmap in a PDF, which is structurally what a scanner produces: pypdf can find
no text in it at all.
"""

from __future__ import annotations

import io


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_text_pdf(pages: list[list[str]]) -> bytes:
    """A minimal PDF with selectable text, one content stream per page."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Font << /F1 3 0 R >> >>",
    }
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(len(pages)))
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()

    for index, lines in enumerate(pages):
        page_number, content_number = 5 + 2 * index, 6 + 2 * index
        objects[page_number] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources 4 0 R /Contents {content_number} 0 R >>"
        ).encode()
        body = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        body.extend(f"({_escape(line)}) Tj T*" for line in lines)
        body.append("ET")
        stream = "\n".join(body).encode("latin-1")
        objects[content_number] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    return _assemble(objects)


def _assemble(objects: dict[int, bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"

    xref_position = len(out)
    count = max(objects) + 1
    out += b"xref\n0 %d\n" % count
    out += b"0000000000 65535 f \n"
    for number in range(1, count):
        out += b"%010d 00000 n \n" % offsets.get(number, 0)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (count, xref_position)
    return bytes(out)


def build_scanned_pdf(pages: list[list[str]], size: tuple[int, int] = (1240, 1754)) -> bytes:
    """An image-only PDF: text is drawn into a bitmap, so no text is extractable.

    Requires Pillow. Mirrors what a flatbed scanner or a phone scan produces.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 40)
        heading_font = ImageFont.truetype("arialbd.ttf", 52)
    except OSError:  # pragma: no cover - font availability is platform-specific
        font = heading_font = ImageFont.load_default()

    images = []
    for lines in pages:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        y = 120
        for position, line in enumerate(lines):
            active = heading_font if position == 0 else font
            draw.text((100, y), line, fill="black", font=active)
            y += 90 if position == 0 else 62
        images.append(image)

    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:], resolution=150.0)
    return buffer.getvalue()
