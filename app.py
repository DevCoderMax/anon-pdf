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
