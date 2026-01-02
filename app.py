import fitz
import re
from typing import List, Tuple

# OCR (EasyOCR) e imagem
import easyocr
from PIL import Image

# ============================
# Configurações
# ============================
# DPI para rasterização das páginas no OCR (200 conforme solicitado)
OCR_DPI = 200
# Idiomas para OCR (pt e en ajudam em documentos mistos; ajuste se necessário)
OCR_LANGS = ["pt", "en"]

# ============================
# Regex CPF (mantida)
# ============================
cpf_regex = r"""
\b
(?:
  \d{3}[\.\s-]?\d{3}[\.\s-]?\d{3}[\.\s-]?\d{2}  # com separadores
  |
  \d{11}                                        # colado
)
\b
"""

# ============================
# Regex CNPJ com prefixo opcional "CNPJ" (com/sem dois-pontos) e tolerância a espaços/hífens
# Parte numérica aceita separadores em 2-3-3-4-2 e barra entre 3º e 4º bloco; ou colado (14 dígitos)
# Obs: case-insensitive é controlado via re.IGNORECASE na compilação, não usando (?i) inline
# ============================
cnpj_regex = r"""
(?:CNPJ\s*:?\s*)?                # prefixo opcional, sem \b antes para não quebrar casos com símbolos
(?:
  \d{2}[\.\s-]?\d{3}[\.\s-]?\d{3}[\s/-]?\d{4}[\s-]?\d{2}  # com separadores
  |
  \d{14}                                                  # colado
)
"""

# ============================
# Regex combinada para CPF ou CNPJ
# ============================
combined_pattern = re.compile(
    rf"(?:{cpf_regex}|{cnpj_regex})",
    re.VERBOSE | re.IGNORECASE
)


def redact_rects(page: fitz.Page, rects: List[fitz.Rect]) -> None:
    """Adiciona tarjas pretas para todos os retângulos informados na página e aplica as redações."""
    for r in rects:
        page.add_redact_annot(r, fill=(0, 0, 0))
    if rects:
        page.apply_redactions()


def search_text_principal(page: fitz.Page) -> List[fitz.Rect]:
    """Busca principal usando regex nativa do PyMuPDF (quando disponível)."""
    flags = (
        (getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0))
        | (getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 0))
    )
    if hasattr(fitz, "TEXT_REGEX"):
        flags |= fitz.TEXT_REGEX

    rects: List[fitz.Rect] = []
    try:
        rects = page.search_for(combined_pattern.pattern, flags=flags) or []
    except Exception:
        # Alternativa via textpage.search quando disponível
        try:
            if hasattr(page, "get_textpage"):
                tp = page.get_textpage(
                    flags=(
                        getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0)
                        | getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 0)
                    )
                )
                if hasattr(tp, "search"):
                    rects = tp.search(combined_pattern.pattern, quads=False) or []
        except Exception:
            rects = []
    return rects


def fallback_literal_rects(page: fitz.Page) -> List[fitz.Rect]:
    """Fallback: extrai texto simples, roda regex em Python e busca literal no PDF os trechos encontrados,
    cobrindo casos onde rótulo e número estão fragmentados em spans distintos."""
    rects: List[fitz.Rect] = []
    try:
        page_text = page.get_text("text") or ""
    except Exception:
        page_text = ""

    if not page_text:
        return rects

    for match in re.finditer(combined_pattern, page_text):
        matched_text = match.group(0).strip()
        if not matched_text:
            continue
        try:
            literal_rects = page.search_for(matched_text) or []
            rects.extend(literal_rects)
        except Exception:
            continue
    return rects


def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Converte um Pixmap do PyMuPDF em PIL.Image, normalizando para RGB quando necessário."""
    if pix.n >= 4:
        # tem alpha; converte para RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def ocr_extract_boxes(img: Image.Image, reader: easyocr.Reader) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    """Executa OCR via EasyOCR e retorna a lista de bounding boxes, texto e confidência."""
    # reader.readtext aceita array numpy; PIL -> numpy
    import numpy as np
    np_img = np.array(img)
    # detail=1 retorna boxes; paragraph=False evita agrupamentos excessivos
    result = reader.readtext(np_img, detail=1, paragraph=False)
    # Cada item: (box, text, conf)
    return result


def boxes_to_rects_if_match(
    page: fitz.Page, boxes: List[Tuple[List[Tuple[int, int]], str, float]]
) -> List[fitz.Rect]:
    """Converte boxes do OCR em retângulos PDF quando o texto reconhecido contiver CPF/CNPJ (regex)."""
    rects: List[fitz.Rect] = []
    for box, text, conf in boxes:
        if not text:
            continue
        # Se qualquer parte do texto casa com a regex combinada, tarjar a box inteira
        if re.search(combined_pattern, text):
            # box é uma lista de 4 pontos [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] no espaço da imagem rasterizada
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            # Precisamos transformar coordenadas da imagem (em pixels) para coordenadas PDF.
            # A rasterização usará a Matriz de Zoom; manteremos a mesma matriz para mapear de volta.
            # Estratégia: construir um Rect em coordenadas da imagem e aplicar a inversa da matriz usada para render.
            rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects


def ocr_rects_for_page(doc: fitz.Document, page: fitz.Page, dpi: int, reader: easyocr.Reader) -> List[fitz.Rect]:
    """Rasteriza a página na DPI solicitada, executa OCR, detecta CPF/CNPJ e retorna retângulos em coordenadas PDF."""
    # Matriz de escala baseada em DPI (72 dpi base do PDF)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Renderizar a página para Pixmap com a matriz (respeita rotação/mediabox)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = pixmap_to_pil(pix)

    # OCR
    boxes = ocr_extract_boxes(img, reader)

    # Rects no espaço da imagem (em pixels)
    img_rects = boxes_to_rects_if_match(page, boxes)

    # Converter rects da imagem para o espaço PDF usando a inversa da matriz
    inv = fitz.Matrix(1 / zoom, 1 / zoom)
    pdf_rects: List[fitz.Rect] = [fitz.Rect(r) * inv for r in img_rects]
    return pdf_rects


def main() -> None:
    # Abrir o PDF de entrada
    doc = fitz.open("entrada.pdf")

    # Inicializar OCR uma única vez (carrega modelos)
    reader = easyocr.Reader(OCR_LANGS, gpu=False)  # ajuste gpu=True se tiver CUDA configurado

    for page in doc:
        all_rects: List[fitz.Rect] = []

        # 1) Busca principal (texto pesquisável)
        rects_primary = search_text_principal(page)
        all_rects.extend(rects_primary)

        # 2) Fallback literal (texto concatenado + busca literal para multi-span)
        rects_fallback = fallback_literal_rects(page)
        all_rects.extend(rects_fallback)

        # 3) OCR (páginas/áreas de imagem)
        ocr_rects = ocr_rects_for_page(doc, page, OCR_DPI, reader)
        all_rects.extend(ocr_rects)

        # Remover duplicidade aproximando coordenadas (round) para evitar múltiplas anotações sobrepostas
        unique = []
        seen = set()
        for r in all_rects:
            key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)

        # Aplicar tarjas
        redact_rects(page, unique)

    # Salvar o PDF de saída
    doc.save("saida_anonimizada.pdf")


if __name__ == "__main__":
    main()
import fitz
import re

# Regex CPF (mantida)
cpf_regex = r"""
\b
(?:
  \d{3}[\.\s-]?\d{3}[\.\s-]?\d{3}[\.\s-]?\d{2}  # com separadores
  |
  \d{11}                                        # colado
)
\b
"""

# Regex CNPJ com prefixo opcional "CNPJ" (com/sem dois-pontos) e tolerância a espaços/hífens
# Parte numérica aceita separadores em 2-3-3-4-2 e barra entre 3º e 4º bloco; ou colado (14 dígitos)
# Obs: case-insensitive é controlado via re.IGNORECASE na compilação, não usando (?i) inline
cnpj_regex = r"""
(?:CNPJ\s*:?\s*)?                # prefixo opcional, sem \b antes para não quebrar casos com símbolos
(?:
  \d{2}[\.\s-]?\d{3}[\.\s-]?\d{3}[\s/-]?\d{4}[\s-]?\d{2}  # com separadores
  |
  \d{14}                                                  # colado
)
"""

# Regex combinada para CPF ou CNPJ
combined_pattern = re.compile(
    rf"(?:{cpf_regex}|{cnpj_regex})",
    re.VERBOSE | re.IGNORECASE
)

# Abrir o PDF de entrada
doc = fitz.open("entrada.pdf")

for page in doc:
    # Busca primária: usar search_for com regex, preservando ligaduras e espaços
    flags = (
        (getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0))
        | (getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 0))
    )
    if hasattr(fitz, "TEXT_REGEX"):
        flags |= fitz.TEXT_REGEX

    text_instances = []
    try:
        text_instances = page.search_for(combined_pattern.pattern, flags=flags) or []
    except Exception:
        # Alternativa via textpage.search quando disponível
        try:
            if hasattr(page, "get_textpage"):
                tp = page.get_textpage(flags=(getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0) | getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 0)))
                if hasattr(tp, "search"):
                    text_instances = tp.search(combined_pattern.pattern, quads=False) or []
        except Exception:
            text_instances = []

    # Aplicar tarjas pretas para ocorrências encontradas pela busca primária
    for inst in text_instances:
        page.add_redact_annot(inst, fill=(0, 0, 0))

    # Fallback robusto: extrair texto plano e procurar com re.finditer,
    # depois buscar literalmente o trecho combinado (rótulo + número) para cobrir multi-span
    try:
        page_text = page.get_text("text") or ""
    except Exception:
        page_text = ""

    if page_text:
        for match in re.finditer(combined_pattern, page_text):
            matched_text = match.group(0).strip()
            if not matched_text:
                continue
            # Busca literal (sem regex) para obter os retângulos mesmo se label/número estiverem fragmentados
            try:
                literal_rects = page.search_for(matched_text) or []
                for r in literal_rects:
                    page.add_redact_annot(r, fill=(0, 0, 0))
            except Exception:
                pass

    # Aplicar redações da página
    page.apply_redactions()

# Salvar o PDF de saída
doc.save("saida_anonimizada.pdf")
