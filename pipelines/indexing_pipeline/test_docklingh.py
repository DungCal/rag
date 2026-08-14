# from docling.document_converter import DocumentConverter

# source = "/home/dungx/LGI/rag/t130sp_na_operator_manual.pdf"  # file path or URL
# converter = DocumentConverter()
# doc = converter.convert(source).document

# print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"

import logging
import os

# Enable standard info logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Optional: Set your Hugging Face Token to avoid rate limits

# (Now import and run Docling)
# from docling.document_converter import DocumentConverter

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_TRANSFORMERS,  # <-- change the model here
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source="/home/dungx/LGI/rag/t130sp_na_operator_manual.pdf").document