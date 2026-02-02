"""
OCR-based data extraction from PDFs and images (screenshots).

Extracts text via OCR, parses into structured tables, and returns
pandas DataFrames for use in the portfolio pipeline or API.
"""

from __future__ import annotations

import io
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of an extraction run: raw text, parsed DataFrame, and metadata."""

    success: bool
    raw_text: str = ""
    df: Optional[pd.DataFrame] = None
    page_count: int = 0
    error: Optional[str] = None
    source_type: str = "unknown"  # "pdf" | "image"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for API response (JSON-safe: NaN -> None)."""
        if self.df is not None and not self.df.empty:
            records = self.df.replace({np.nan: None}).to_dict(orient="records")
            columns = list(self.df.columns)
        else:
            records = []
            columns = []
        return {
            "success": self.success,
            "raw_text": self.raw_text[:5000] + ("..." if len(self.raw_text) > 5000 else ""),
            "dataframe": records,
            "columns": columns,
            "page_count": self.page_count,
            "error": self.error,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


class ExtractionService:
    """
    Extracts text from PDFs and images using OCR and optional native PDF parsing.
    Parses extracted text into a pandas DataFrame (table-like structure).
    """

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        ocr_lang: str = "eng",
        dpi: int = 200,
    ):
        """
        Args:
            tesseract_cmd: Path to tesseract executable (e.g. r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe").
                           If None, pytesseract uses system PATH.
            ocr_lang: Tesseract language code(s), e.g. "eng" or "eng+fra".
            dpi: DPI used when rendering PDF pages to images (higher = better quality, slower).
        """
        self._tesseract_cmd = tesseract_cmd
        self._ocr_lang = ocr_lang
        self._dpi = dpi
        self._tesseract_available: Optional[bool] = None

    def check_ocr_available(self) -> bool:
        """Return True if Tesseract OCR is available (for health checks)."""
        if self._tesseract_available is True:
            return True
        if self._tesseract_available is False:
            return False
        try:
            self._ensure_tesseract()
            return True
        except Exception:  # noqa: BLE001
            self._tesseract_available = False
            return False

    def _ensure_tesseract(self) -> None:
        """Set tesseract cmd and verify it's available."""
        if self._tesseract_available is False:
            raise RuntimeError(
                "Tesseract OCR is not available. Install Tesseract and add it to PATH, "
                "or set tesseract_cmd in ExtractionService."
            )
        try:
            import pytesseract
            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception as e:
            self._tesseract_available = False
            raise RuntimeError(
                f"Tesseract OCR could not be initialized: {e}. "
                "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki (Windows) or package manager."
            ) from e

    def extract_from_pdf(
        self,
        source: Union[Path, str, bytes],
    ) -> ExtractionResult:
        """
        Extract text from a PDF using native text extraction first, then OCR on rendered pages if needed.

        Args:
            source: File path or bytes of the PDF.

        Returns:
            ExtractionResult with raw_text, df (parsed table), and metadata.
        """
        try:
            if isinstance(source, (Path, str)):
                path = Path(source)
                if not path.exists():
                    return ExtractionResult(
                        success=False,
                        error=f"File not found: {path}",
                        source_type="pdf",
                    )
                pdf_bytes = path.read_bytes()
            else:
                pdf_bytes = source

            raw_text, page_count = self._pdf_to_text(pdf_bytes)
            if not raw_text or not raw_text.strip():
                raw_text, page_count = self._pdf_to_text_ocr(pdf_bytes)
            if not raw_text or not raw_text.strip():
                return ExtractionResult(
                    success=True,
                    raw_text="",
                    df=pd.DataFrame(),
                    page_count=page_count,
                    source_type="pdf",
                    metadata={"message": "No text could be extracted from the PDF."},
                )

            df = self._parse_text_to_dataframe(raw_text)
            return ExtractionResult(
                success=True,
                raw_text=raw_text,
                df=df,
                page_count=page_count,
                source_type="pdf",
                metadata={"parser": "table_from_lines"},
            )
        except Exception as e:
            logger.exception("PDF extraction failed")
            return ExtractionResult(
                success=False,
                error=str(e),
                source_type="pdf",
            )

    def extract_from_image(
        self,
        source: Union[Path, str, bytes],
    ) -> ExtractionResult:
        """
        Extract text from an image (e.g. screenshot) using OCR.

        Args:
            source: File path or bytes of the image.

        Returns:
            ExtractionResult with raw_text, df (parsed table), and metadata.
        """
        try:
            from PIL import Image

            if isinstance(source, (Path, str)):
                path = Path(source)
                if not path.exists():
                    return ExtractionResult(
                        success=False,
                        error=f"File not found: {path}",
                        source_type="image",
                    )
                img = Image.open(path).convert("RGB")
            else:
                img = Image.open(io.BytesIO(source)).convert("RGB")

            self._ensure_tesseract()
            import pytesseract
            raw_text = pytesseract.image_to_string(img, lang=self._ocr_lang)
            raw_text = (raw_text or "").strip()

            if not raw_text:
                return ExtractionResult(
                    success=True,
                    raw_text="",
                    df=pd.DataFrame(),
                    page_count=1,
                    source_type="image",
                    metadata={"message": "No text detected in the image."},
                )

            df = self._parse_text_to_dataframe(raw_text)
            return ExtractionResult(
                success=True,
                raw_text=raw_text,
                df=df,
                page_count=1,
                source_type="image",
                metadata={"parser": "table_from_lines"},
            )
        except Exception as e:
            logger.exception("Image extraction failed")
            return ExtractionResult(
                success=False,
                error=str(e),
                source_type="image",
            )

    def _pdf_to_text(self, pdf_bytes: bytes) -> tuple[str, int]:
        """Extract text natively from PDF (no OCR). Returns (text, page_count)."""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)
                parts: List[str] = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
                    # Also try tables
                    tables = page.extract_tables()
                    for table in tables or []:
                        if table:
                            parts.append(_table_to_text(table))
                return "\n".join(parts).strip(), page_count
        except ImportError:
            return "", 0
        except Exception as e:
            logger.debug("pdfplumber extraction failed: %s", e)
            return "", 0

    def _pdf_to_text_ocr(self, pdf_bytes: bytes) -> tuple[str, int]:
        """Render PDF pages to images and run OCR. Returns (text, page_count)."""
        self._ensure_tesseract()
        import pytesseract
        from PIL import Image

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF OCR. Install with: pip install pymupdf"
            ) from None

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        parts: List[str] = []
        zoom = self._dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=self._ocr_lang)
            if text and text.strip():
                parts.append(text.strip())
        doc.close()
        return "\n\n".join(parts), page_count

    def _parse_text_to_dataframe(self, text: str) -> pd.DataFrame:
        """
        Parse raw OCR/text into a table-like DataFrame.
        Uses line-based splitting and consistent column count to infer rows/columns.
        """
        if not text or not text.strip():
            return pd.DataFrame()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame()

        # Split each line into tokens (whitespace or tab)
        rows: List[List[str]] = []
        for line in lines:
            tokens = re.split(r"\s{2,}|\t", line)  # multiple spaces or tab
            if len(tokens) == 1:
                tokens = line.split()  # fallback: any whitespace
            rows.append(tokens)

        if not rows:
            return pd.DataFrame()

        # Normalize column count (use max and pad)
        max_cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < max_cols:
                r.append("")

        # First row as header if it looks like headers (mostly non-numeric)
        header_row = rows[0]
        use_first_as_header = _looks_like_header(header_row)
        if use_first_as_header and len(rows) > 1:
            columns = [_sanitize_column_name(c) for c in header_row]
            data = rows[1:]
        else:
            columns = [f"col_{i}" for i in range(max_cols)]
            data = rows

        df = pd.DataFrame(data, columns=columns)

        # Coerce numeric columns
        for col in df.columns:
            df[col] = _try_numeric(df[col])

        return df


def _table_to_text(table: List[List[Any]]) -> str:
    """Turn extracted table (list of rows) into line-based text for parsing."""
    return "\n".join("\t".join(str(cell) if cell is not None else "" for cell in row) for row in table)


def _looks_like_header(row: List[str]) -> bool:
    """Heuristic: row is header if majority of cells are not numeric."""
    numeric = 0
    for c in row:
        s = (c or "").strip()
        if s and _is_numeric_like(s):
            numeric += 1
    return numeric <= len(row) / 2


def _is_numeric_like(s: str) -> bool:
    """Check if string looks like a number (int, float, percentage)."""
    s = s.replace(",", "").replace("%", "").strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _try_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric where possible, leave rest as object."""
    return pd.to_numeric(series, errors="ignore")


def _sanitize_column_name(name: str) -> str:
    """Clean column name for DataFrame."""
    if not name or not isinstance(name, str):
        return "unnamed"
    s = re.sub(r"\W+", "_", name.strip()).strip("_") or "unnamed"
    return s[:100]
