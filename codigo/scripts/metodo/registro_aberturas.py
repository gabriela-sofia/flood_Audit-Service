"""Registro de abertura por evento, fonte, fase e protocolo — idempotente e seguro entre sessões.

O ledger `examinados_apos_congelamento.jsonl` guarda uma linha por avaliação com a lista `eventos`, e
é assim que os consumidores (reservas_examinadas, rodada3_ledger, obs01/obs03, mt04, mt11) o leem.
Ele não distingue a FASE da abertura (área do rótulo consultada, pontos amostrados, escores lidos),
não tem intenção/conclusão e aceita a mesma linha duas vezes. Em 17/10/2026 a seleção do MT11
consultou o rótulo de 46 eventos e nenhuma linha foi gravada até a avaliação, que nunca rodou.

Aqui:
  aberturas_por_fase.jsonl   uma linha por (evento, fonte, fase, protocolo, estado); estado
                             intencao -> concluido | erro; `quando_original_utc` quando a abertura
                             aconteceu antes do registro (conciliação)
  ledger compartilhado       ao CONCLUIR uma fase, uma linha agregada com o campo `eventos` (mesmo
                             formato dos consumidores) e `avaliacao` = "<avaliacao>:<fase>"; só se
                             a mesma (fonte, avaliacao, eventos) ainda não estiver lá
Escrita: trava por arquivo exclusivo (O_EXCL) ao lado do destino, releitura dentro da trava, append
e fsync. Nenhuma linha antiga é reescrita.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PASTA = REPO / "local_runs" / "reservas-examinadas"
LEDGER = PASTA / "examinados_apos_congelamento.jsonl"
FASES_ARQ = PASTA / "aberturas_por_fase.jsonl"
FASES = ("rotulo_consultado_na_selecao", "pontos_amostrados", "atributos_extraidos_fumaca", "escores_avaliados",
         # MT12 (18/10/2026): fases que não são leitura de rótulo e não podem ser confundidas com ela
         "metadado_lido", "geometria_neutra_consultada", "arquivo_adquirido", "rotulo_lido")
ESTADOS = ("intencao", "concluido", "erro")


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


class Trava:
    def __init__(self, alvo: Path, espera_s: float = 120.):
        self.arq, self.espera_s = alvo.with_suffix(alvo.suffix + ".trava"), espera_s

    def __enter__(self):
        fim = time.time() + self.espera_s
        while True:
            try:
                self.fd = os.open(self.arq, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()} {agora()}".encode())
                return self
            except FileExistsError:
                if time.time() > fim:
                    raise TimeoutError(f"trava ocupada: {self.arq}")
                time.sleep(.2)

    def __exit__(self, *exc):
        os.close(self.fd)
        os.remove(self.arq)


def _linhas(arq: Path) -> list[dict]:
    if not arq.exists():
        return []
    return [json.loads(t) for t in arq.read_text(encoding="utf-8").splitlines() if t.strip()]


def _anexar(arq: Path, obj: dict) -> None:
    with arq.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def registrar(eventos, fonte: str, fase: str, estado: str, avaliacao: str, protocolo_sha256: str,
              quando_original_utc: str | None = None, detalhe: dict | None = None,
              pasta: Path | None = None) -> dict:
    """Grava uma linha por evento novo para a chave; devolve quantas eram novas. Idempotente."""
    if fase not in FASES or estado not in ESTADOS:
        raise ValueError(f"fase/estado inválidos: {fase}/{estado}")
    pasta = pasta or PASTA
    fases_arq, ledger = pasta / FASES_ARQ.name, pasta / LEDGER.name
    pasta.mkdir(parents=True, exist_ok=True)
    eventos = sorted({str(e) for e in eventos})
    with Trava(fases_arq):
        ja = {(r["evento_id"], r["fonte"], r["fase"], r["protocolo_sha256"], r["estado"]) for r in _linhas(fases_arq)}
        novos = [e for e in eventos if (e, fonte, fase, protocolo_sha256, estado) not in ja]
        q = agora()
        for e in novos:
            _anexar(fases_arq, {"evento_id": e, "fonte": fonte, "fase": fase, "estado": estado, "avaliacao": avaliacao,
                                "protocolo_sha256": protocolo_sha256, "quando_utc": q,
                                "quando_original_utc": quando_original_utc, "detalhe": detalhe or {}})
    compat = False
    if estado == "concluido":
        aval = f"{avaliacao}:{fase}"
        with Trava(ledger):
            igual = any(r.get("fonte") == fonte and r.get("avaliacao") == aval and sorted(r.get("eventos", [])) == eventos
                        for r in _linhas(ledger))
            if not igual:
                _anexar(ledger, {"quando_utc": agora(), "fonte": fonte, "avaliacao": aval, "eventos": eventos,
                                 "fase": fase, "protocolo_sha256": protocolo_sha256,
                                 "quando_original_utc": quando_original_utc, "registro": "registro_aberturas.py"})
                compat = True
    return {"novos": len(novos), "ledger_compartilhado_gravado": compat}


def estado_por_evento(fonte: str, pasta: Path | None = None) -> dict:
    """Último estado por (evento, fase) para a fonte."""
    out = {}
    for r in _linhas((pasta or PASTA) / FASES_ARQ.name):
        if r["fonte"] == fonte:
            out[(r["evento_id"], r["fase"])] = r["estado"]
    return out
