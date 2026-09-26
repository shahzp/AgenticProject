import os
from docling.datamodel.base_models import InputFormat
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

input_pdf = "../../Documents/Report_2024-25.pdf"
output_markdown = "final_report_output.md"
chunk_size = 5  # Process only 5 pages at a time

# 1. Setup minimal pipeline options to save memory
pipeline_options = PdfPipelineOptions(
    do_formula_enrichment=False,
    do_code_enrichment=False,
    do_picture_classification=False,
    generate_parsed_pages=False  # Do not cache pages in RAM
)

def parse_docs():
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    print(f"Starting processing for {total_pages} pages...")

    # 2. Loop through the document in tiny chunks

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        temp_chunk_filename = f"temp_chunk_{start_page}.pdf"

        # Extract chunk to a temporary file
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        with open(temp_chunk_filename, "wb") as f:
            writer.write(f)

        # Initialize Docling fresh for this chunk to clear internal caches
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        try:
            result = converter.convert(temp_chunk_filename)
            md_content = result.document.export_to_markdown()

            # Append immediately to disk to free memory
            with open(output_markdown, "a", encoding="utf-8") as f_out:
                f_out.write(md_content + "\n\n")

            print(f"Successfully parsed pages {start_page + 1} to {end_page}")

        except Exception as e:
            print(f"Error on pages {start_page + 1}-{end_page}: {e}")

        finally:
            # Explicitly unload backend components and delete temporary file
            if 'result' in locals() and hasattr(result, 'input') and result.input._backend:
                result.input._backend.unload()

            if os.path.exists(temp_chunk_filename):
                os.remove(temp_chunk_filename)


print("Done! Check final_output.md")


if __name__=='__main__':
    parse_docs()

