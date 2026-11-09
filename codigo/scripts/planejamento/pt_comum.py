"""Planejamento-tese (PT): utilidades da linhagem `local_runs/planejamento-tese/v1/`.

Reaproveita as métricas reescritas do fechamento do artigo (`fa_comum`) e só troca a linhagem de escrita.
Rede bloqueada; escrita restrita à linhagem; protocolo congelado não é sobrescrito com conteúdo diferente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for s in ("planejamento", "fechamento", "evolucao_local", "ia", "metodo", "mt12", "servico", "sar", "geo", "terreno",
          "manutencao", "chuva", "social", "neural", "suscetibilidade"):
    p = str(REPO / "scripts" / s)
    if p not in sys.path:
        sys.path.append(p)

import fa_comum  # noqa: E402
from fa_comum import agora, auc_pares, ap_por_prevalencia, boot_percentil_laco, leitura_cientifica, ler, rel, sha  # noqa: E402,F401

LINHAGEM_V1 = REPO / "local_runs" / "planejamento-tese" / "v1"   # fechamento de 19/10: só leitura (contexto_v2 congelado)
LINHAGEM = REPO / "local_runs" / "planejamento-tese" / "v2"      # execução de pendências
LINHAGEM_AOI = REPO / "local_runs" / "planejamento-tese" / "v3-aoi"   # evolução por AOI, cena e data exatas
EscritaRecusada = fa_comum.EscritaRecusada
NAO_E_LINHAGEM = ("analise retrospectiva da linhagem planejamento-tese sobre o acervo local; nao e modelo promovido, "
                  "rotulo de referencia, aquisicao real nem confirmacao")


def gravar(p, o, sobrescrever: bool = True) -> Path:
    p = Path(p)
    ok = (str(p.resolve()).startswith(str(LINHAGEM.resolve())) or p.resolve().is_relative_to(LINHAGEM_AOI.resolve())
          or (p.resolve().is_relative_to((LINHAGEM_V1 / "contexto_v2").resolve()) and LINHAGEM.name == "v2"))
    if not ok:
        raise EscritaRecusada(f"fora da linhagem: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        if isinstance(o, dict) and "tipo" not in o and "artefato" not in o:
            # Declaracao da linhagem (catraca de manifesto): o tipo e o caminho do artefato dentro da linhagem, que e o
            # que ele e; `nao_e` vale para todo artefato desta linhagem. Quem ja declara o proprio tipo nao e alterado.
            o = {"tipo": f"planejamento_tese:{p.resolve().relative_to(LINHAGEM.parent.resolve()).with_suffix('').as_posix()}",
                 "nao_e": NAO_E_LINHAGEM, **o}
        txt = json.dumps(o, ensure_ascii=False, indent=1, default=str)
        if p.exists() and not sobrescrever and p.read_text(encoding="utf-8") != txt:
            raise EscritaRecusada(f"conteudo congelado diferente: {p}")
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(txt, encoding="utf-8")
    elif p.suffix == ".csv":
        tmp = p.with_suffix(".csv.tmp")
        o.to_csv(tmp, index=False)
    elif p.suffix == ".parquet":
        tmp = p.with_suffix(".parquet.tmp")
        o.to_parquet(tmp, index=False)
    elif p.suffix in (".md", ".html", ".txt", ".tex"):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(o, encoding="utf-8")
    else:
        raise ValueError(p.suffix)
    tmp.replace(p)
    return p


def bloquear_rede():
    fa_comum.bloquear_rede()


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def bloquear_rede_externa():
    """Como `bloquear_rede`, mas admite loopback: no Windows o laço do asyncio (TestClient) abre um par de sockets
    em 127.0.0.1. O motor LLM local sai do ambiente para a síntese ficar no motor determinístico declarado."""
    import os
    import socket
    import eal_comum
    original = eal_comum._socket_connect_original
    os.environ.pop("LOCAL_LLM_URL", None)

    def _so_loopback(self, endereco):
        if isinstance(endereco, tuple) and endereco and str(endereco[0]) in LOOPBACK:
            return original(self, endereco)
        raise eal_comum.RedeBloqueada(f"rede externa desativada (tentativa de conexão a {endereco!r})")
    socket.socket.connect = _so_loopback
    socket.socket.connect_ex = _so_loopback
    os.environ.setdefault("NO_PROXY", "*")
    os.environ["GDAL_HTTP_TIMEOUT"] = "1"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".nenhuma"


def energia() -> dict:
    import executar_serializado as es
    return es.energia()
