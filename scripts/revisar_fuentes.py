"""Extrae texto local de las fuentes del curso, conservando número de página."""
from pathlib import Path
from html.parser import HTMLParser
import argparse
import json
import re


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"h1", "h2", "h3", "p", "li", "div", "section", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def main():
    from pypdf import PdfReader
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    dest = Path(__file__).resolve().parents[1] / "docs" / "fuentes_texto"
    dest.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted(args.source.iterdir()):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            content = "\n\n".join(f"=== PÁGINA {i+1} ===\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
            inventory.append({"archivo": path.name, "paginas": len(reader.pages)})
        elif path.suffix.lower() == ".html":
            html = TextHTML()
            html.feed(path.read_text(encoding="utf-8-sig"))
            content = re.sub(r"\n\s*\n", "\n\n", "".join(html.parts))
            inventory.append({"archivo": path.name, "caracteres": len(content)})
        else:
            continue
        (dest / (path.stem + ".txt")).write_text(content, encoding="utf-8")
    (dest / "inventario.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
