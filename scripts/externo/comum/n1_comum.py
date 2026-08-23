"""
NIVEL 1 -- utilitarios comuns de aquisicao.

O Nivel 1 do levantamento de 2026-08-06 reune as fontes que resolvem a lacuna
L1 do projeto: NEGATIVO POR OBSERVACAO GARANTIDA.

A diferenca em relacao a tudo que o REV-P tinha ate agora e categorica. O
Protocolo C construia negativo por AUSENCIA de registro -- frageis por
construcao, e o proprio gate C4_BLOCKED_NO_FORMAL_NEGATIVES existe por causa
disso. As fontes do Nivel 1 trazem, no dado, a informacao de que uma area
FOI OLHADA e NAO estava inundada:

  - Sen1Floods11 : rotulo manual por pixel com classe nao-agua explicita
                   e mascara de validade (pixel sem dado e distinguivel de
                   pixel observado e seco).
  - UFO          : rotulo manual `inundated` / `non-inundated` em contexto
                   URBANO a 3 m -- o mais proximo do fenomeno de Recife.
  - Copernicus EMS: cada ativacao publica a AREA DE INTERESSE analisada; area
                   dentro da AOI e fora do poligono de inundacao e negativo
                   com garantia de analise.
  - GFD          : bandas `clear_views` / `clear_perc` dizem quantos dias o
                   pixel foi observado sem nuvem; pixel com clear_views>0 e
                   flooded=0 dentro da janela do evento e negativo observado.

Este modulo NAO baixa nada sozinho e NAO promove nada a negativo. Ele so
padroniza HTTP com retry, listagem de bucket publico, e registro de
proveniencia -- porque nenhuma base entra no pipeline sem fonte, data de
acesso e licenca escritas (regra do projeto).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
BASE_OUT = REPO / "local_runs"
TIMEOUT = 300
MAX_RETRIES = 5


def get_json(url: str, params: dict | None = None) -> dict | None:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    print(f"  ERRO get_json {url[:80]}: {type(last).__name__}")
    return None


def baixar(url: str, destino: Path, esperado: int | None = None) -> bool:
    """Download com retomada simples: pula se ja existe com o tamanho certo."""
    if destino.exists() and (esperado is None or destino.stat().st_size == esperado):
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                tmp = destino.with_suffix(destino.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(destino)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return False


def listar_bucket(bucket: str, prefix: str = "") -> list[dict]:
    """Lista objetos de um bucket GCS publico via API JSON (sem credencial)."""
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    itens: list[dict] = []
    token = None
    while True:
        params = {"maxResults": 1000, "prefix": prefix}
        if token:
            params["pageToken"] = token
        payload = get_json(url, params)
        if not payload:
            break
        for it in payload.get("items", []):
            itens.append(
                {
                    "name": it.get("name"),
                    "size": int(it.get("size", 0)),
                    "url": f"https://storage.googleapis.com/{bucket}/{it.get('name')}",
                }
            )
        token = payload.get("nextPageToken")
        if not token:
            break
    return itens


def gb(n_bytes: int) -> float:
    return round(n_bytes / 1024 ** 3, 2)


def escrever_proveniencia(
    outdir: Path,
    titulo: str,
    fonte: str,
    identificador: str,
    licenca: str,
    natureza: str,
    uso_pretendido: str,
    uso_proibido: str,
    extra: str = "",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "PROVENIENCIA.md").write_text(
        f"# Proveniencia - {titulo}\n"
        f"- Fonte: {fonte}\n"
        f"- Identificador: {identificador}\n"
        f"- Data de acesso: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- Licenca: {licenca}\n"
        f"- Natureza do dado: {natureza}\n"
        f"- Uso pretendido: {uso_pretendido}\n"
        f"- Uso PROIBIDO: {uso_proibido}\n"
        f"- Status: AQUISICAO_EXPLORATORIA - nenhum negativo promovido; "
        f"nenhuma variavel extraida; nenhum modelo treinado.\n"
        f"{extra}",
        encoding="utf-8",
    )


def salvar_json(caminho: Path, obj) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
