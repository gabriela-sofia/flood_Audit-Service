"""Utilitarios minimos de proveniencia usados pelos artefatos publicos.

Este arquivo guarda somente a convencao deterministica de tempo e hash que os
scripts curados precisam para registrar resultados reproduziveis.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def agora_utc() -> str:
    """Instante UTC em formato estavel para manifestos."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_texto(texto: str) -> str:
    """Hash SHA-256 de texto UTF-8 com quebra de linha normalizada."""
    normalizado = texto.replace("\r\n", "\n")
    return "sha256:" + hashlib.sha256(normalizado.encode("utf-8")).hexdigest()


def canonicalizar(obj) -> str:
    """JSON canonico para que ordem de chaves nao mude o hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha256_json(obj) -> str:
    """Hash SHA-256 do JSON canonico."""
    return sha256_texto(canonicalizar(obj))
