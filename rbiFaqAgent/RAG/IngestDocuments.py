import os
import gc

from docling.datamodel.base_models import InputFormat
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# CONFIGURATION
# ============================================================

input_pdf = "Documents/Report_2024-25.pdf"
output_markdown = "rbiFaqAgent/Data/final_report_output.md"

# Process this many PDF pages in each Python run
START_PAGE = 201
END_PAGE = 220

# Inside each run, process 5 pages at a time
BATCH_SIZE = 5


# ============================================================
# DOCLING PIPELINE OPTIONS
# ============================================================

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_formula_enrichment=False,
    do_code_enrichment=False,
    do_picture_classification=False,
    generate_parsed_pages=False
)


# ============================================================
# PARSE DOCUMENT
# ============================================================

def parse_docs():

    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    print(f"Total PDF pages: {total_pages}")

    # --------------------------------------------------------
    # Validate requested page range
    # --------------------------------------------------------

    if START_PAGE < 1:
        raise ValueError("START_PAGE must be >= 1")

    if END_PAGE > total_pages:
        raise ValueError(
            f"END_PAGE {END_PAGE} exceeds "
            f"total PDF pages {total_pages}"
        )

    if START_PAGE > END_PAGE:
        raise ValueError(
            "START_PAGE cannot be greater than END_PAGE"
        )

    print(
        f"Processing pages "
        f"{START_PAGE} to {END_PAGE}"
    )

    # --------------------------------------------------------
    # Only delete output when starting from page 1
    #
    # This allows later runs to append to the same file.
    # --------------------------------------------------------

    if START_PAGE == 1 and os.path.exists(output_markdown):

        os.remove(output_markdown)

        print(
            f"Existing {output_markdown} deleted."
        )

    # --------------------------------------------------------
    # Convert 1-based page numbers to Python 0-based indexes
    # --------------------------------------------------------

    start_index = START_PAGE - 1
    end_index = END_PAGE

    # --------------------------------------------------------
    # Process BATCH_SIZE pages at a time
    # --------------------------------------------------------

    for batch_start in range(
        start_index,
        end_index,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            end_index
        )

        actual_start_page = batch_start + 1
        actual_end_page = batch_end

        temp_chunk_filename = (
            f"temp_chunk_{actual_start_page}_{actual_end_page}.pdf"
        )

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Processing pages "
            f"{actual_start_page} to "
            f"{actual_end_page}"
        )

        print(
            f"Temporary file: "
            f"{temp_chunk_filename}"
        )

        writer = None
        converter = None
        result = None

        try:

            # =================================================
            # 1. CREATE TEMPORARY PDF
            # =================================================

            writer = PdfWriter()

            for page_num in range(
                batch_start,
                batch_end
            ):

                writer.add_page(
                    reader.pages[page_num]
                )

            with open(
                temp_chunk_filename,
                "wb"
            ) as f:

                writer.write(f)

            print("Temporary PDF created.")

            # =================================================
            # 2. CREATE FRESH DOCLING CONVERTER
            # =================================================

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            # =================================================
            # 3. CONVERT PDF
            # =================================================

            print(
                f"Sending pages "
                f"{actual_start_page}-{actual_end_page} "
                f"to Docling..."
            )

            result = converter.convert(
                temp_chunk_filename
            )

            print("Docling conversion completed.")

            doc = result.document

            # =================================================
            # 4. EXPORT MARKDOWN
            # =================================================

            md_content = doc.export_to_markdown()

            print(
                f"Markdown generated: "
                f"{len(md_content)} characters"
            )

            # =================================================
            # 5. DIAGNOSTIC PAGE INFORMATION
            # =================================================

            page_numbers_found = set()

            for item, _level in doc.iterate_items():

                if (
                    hasattr(item, "prov")
                    and item.prov
                ):

                    for provenance in item.prov:

                        if hasattr(
                            provenance,
                            "page_no"
                        ):

                            page_numbers_found.add(
                                provenance.page_no
                            )

            print(
                "Docling pages detected in batch: "
                f"{sorted(page_numbers_found)}"
            )

            # =================================================
            # 6. APPEND MARKDOWN TO FINAL FILE
            # =================================================

            with open(
                output_markdown,
                "a",
                encoding="utf-8"
            ) as f_out:

                f_out.write(
                    "\n\n"
                    f"<!-- PDF BATCH "
                    f"{actual_start_page}-"
                    f"{actual_end_page} -->"
                    "\n\n"
                )

                f_out.write(md_content)

                f_out.write("\n\n")

            print(
                f"Successfully parsed pages "
                f"{actual_start_page} to "
                f"{actual_end_page}"
            )

        except Exception as e:

            print(
                f"\nERROR while processing pages "
                f"{actual_start_page}-"
                f"{actual_end_page}"
            )

            print(
                f"Error: {e}"
            )

            # Stop this run rather than continuing with
            # potentially incomplete data.

            break

        finally:

            # =================================================
            # 7. UNLOAD DOCLING BACKEND
            # =================================================

            try:

                if (
                    result is not None
                    and hasattr(result, "input")
                    and result.input is not None
                    and hasattr(
                        result.input,
                        "_backend"
                    )
                    and result.input._backend is not None
                ):

                    result.input._backend.unload()

                    print(
                        "Docling backend unloaded."
                    )

            except Exception as cleanup_error:

                print(
                    "Backend cleanup warning: "
                    f"{cleanup_error}"
                )

            # =================================================
            # 8. DELETE TEMPORARY PDF
            # =================================================

            if os.path.exists(
                temp_chunk_filename
            ):

                os.remove(
                    temp_chunk_filename
                )

                print(
                    "Temporary PDF deleted."
                )

            # =================================================
            # 9. RELEASE PYTHON REFERENCES
            # =================================================

            result = None
            converter = None
            writer = None

            gc.collect()

            print(
                "Garbage collection completed."
            )

    print(
        "\n========================================"
    )

    print(
        f"Run completed for pages "
        f"{START_PAGE}-{END_PAGE}"
    )

    print(
        f"Output file: {output_markdown}"
    )

    print(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parse_docs()