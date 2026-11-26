"""Hojas de contacto de los PDFs que contienen principalmente imágenes."""
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

source = Path(__file__).resolve().parents[2] / "Clases"
dest = Path(__file__).resolve().parents[1] / "docs/revision_visual"
dest.mkdir(parents=True, exist_ok=True)
files = ["1.Máquinas_de_Soporte_Vectorial.pdf",
         "20261029_2_Geometría_del_Aprendizaje_Neuronal.pdf", "20261112_DelMLPalDL.pdf"]
for filename in files:
    pdf = pdfium.PdfDocument(source / filename)
    for start in range(0, len(pdf), 6):
        contact = Image.new("RGB", (1600, 1500), "#dce3ed")
        draw = ImageDraw.Draw(contact)
        for position, index in enumerate(range(start, min(start + 6, len(pdf)))):
            page = pdf[index]
            img = page.render(scale=1.3).to_pil()
            img.thumbnail((780, 460))
            x = position % 2 * 800 + 10
            y = position // 2 * 500 + 30
            contact.paste(img, (x, y))
            draw.text((x, y - 22), f"Página {index+1}", fill="black")
        path = dest / f"{Path(filename).stem}_{start+1:02}.jpg"
        contact.save(path)
        print(path.name)
