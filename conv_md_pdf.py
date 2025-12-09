from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

def md_to_pdf(input_md: str, output_pdf: str) -> None:
    md_path = Path(input_md)
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    markdown_content = md_path.read_text(encoding="utf-8")

    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = md_path.stem
    pdf.add_section(Section(markdown_content, toc=True))
    pdf.save(output_pdf)

def main():
    input_md = "REPORT.md"
    output_pdf = "REPORT.pdf"
    md_to_pdf(input_md, output_pdf)
    print(f"Saved PDF to: {output_pdf}")

if __name__ == "__main__":
    main()
