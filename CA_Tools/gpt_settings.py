import logging
from pathlib import Path

# Core Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    RapidOcrOptions,  # High-performance OCR backend
    TableStructureOptions,
    EasyOcrOptions,   # Fallback backend
)
from docling.document_converter import DocumentConverter, PdfFormatOption

def get_gpu_converter(
    aggressive_scale: float = 2.0,
    force_ocr: bool = True,
    use_rapid_ocr: bool = True
) -> DocumentConverter:
    """
    Creates a Docling DocumentConverter configured for aggressive OCR using GPU acceleration.

    This function addresses 'silent failure' modes in scanned or dirty-layer PDFs
    (e.g., 'Copy of sample_tax_doc.pdf') by forcing full-page rasterization and OCR.

    Args:
        aggressive_scale (float): Resolution multiplier.
                                  2.0 (~144 DPI) is standard aggressive.
                                  3.0 (~216 DPI) is maximum aggressive (watch VRAM).
        force_ocr (bool): If True, ignores existing text layers and runs OCR on the
                          entire page image. Mandatory for documents with garbled text.
        use_rapid_ocr (bool): If True, attempts to use RapidOCR (faster on GPU).
                              Falls back to EasyOCR if not installed.

    Returns:
        DocumentConverter: A configured instance ready for.convert().
    """

    # 1. Configure Hardware Acceleration
    # We explicitly lock the device to CUDA to prevent CPU fallback.
    # num_threads=8 ensures the CPU can feed the GPU fast enough during rasterization.
    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )

    # 2. Initialize PDF Pipeline Options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options

    # 3. Enable Core Extraction Features
    # do_ocr: Master switch for OCR.
    # do_table_structure: Essential for recovering the 'tables': failure.
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # 4. Configure Table Structure Recovery
    # do_cell_matching is critical. It forces the table model to map the
    # visual grid lines it detects back to the OCR'd text tokens.
    # Without this, you might get structure but no content, or content but no structure.
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )

    # 5. Set Aggressive Image Scaling
    # Increasing this from 1.0 to 2.0/3.0 provides the pixel density needed
    # to resolve small legal text (Section 6, footnotes) and separate
    # merged characters in low-quality scans.
    pipeline_options.images_scale = aggressive_scale

    # 6. Configure "Aggressive" OCR Engine
    if use_rapid_ocr:
        try:
            # RapidOCR (ONNX) is preferred for high-resolution images due to
            # better memory management and inference speed on CUDA.
            ocr_options = RapidOcrOptions(
                force_full_page_ocr=force_ocr
            )
            pipeline_options.ocr_options = ocr_options
            logging.info("Docling Config: Using RapidOCR with aggressive settings.")
        except ImportError:
            logging.warning("Docling Config: RapidOCR not found. Falling back to EasyOCR.")
            # Fallback to EasyOCR. Note: This will be slower.
            pipeline_options.ocr_options = EasyOcrOptions(
                force_full_page_ocr=force_ocr,
                use_gpu=True # Explicitly request GPU for EasyOCR
            )
    else:
        # Manual override to use EasyOCR
        pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=force_ocr
        )

    # 7. Disable Unnecessary Enrichments
    # To save GPU memory for the high-res OCR, we disable image description features.
    pipeline_options.do_picture_classification = False
    pipeline_options.do_picture_description = False

    # 8. Bind Options to PDF Format
    # This maps the configured pipeline specifically to PDF inputs.
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }

    # 9. Instantiate Converter
    converter = DocumentConverter(
        format_options=format_options
    )

    return converter