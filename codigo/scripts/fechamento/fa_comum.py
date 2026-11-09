"""Fechamento do artigo (FA) — utilidades comuns, independentes do código que produziu os resultados.

Linhagem `local_runs/fechamento-artigo/v1/`. Nada aqui importa as funções de AUC, bootstrap ou contraste das
campanhas; elas são reescritas a partir das regras declaradas nos protocolos, para que a reprodução seja uma
verificação e não a reexecução do mesmo código. Rede bloqueada (reuso de `eal_comum.bloquear_rede`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for s in ("fechamento", "evolucao_local", "ia", "metodo", "mt12", "servico", "sar", "geo", "terreno", "manutencao", "chuva"):
    p = str(REPO / "scripts" / s)
    if p not in sys.path:
        sys.path.append(p)

LINHAGEM = REPO / "local_runs" / "fechamento-artigo" / "v1"


class EscritaRecusada(RuntimeError):
    pass


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


NAO_E_FA = ("verificacao do fechamento do artigo: reimplementacao independente das regras declaradas nos protocolos "
            "sobre resultados ja produzidos; nao e resultado cientifico novo, nao amplia populacao e nao substitui o "
            "protocolo original")


def _declarado(p, o):
    """Catraca de manifesto (19/10/2026): JSON sem `tipo` recebe o caminho dentro da linhagem como tipo e o `nao_e` da
    FA. Quem ja declara o proprio tipo nao e alterado; artefatos ja gravados nao sao reescritos."""
    if not isinstance(o, dict) or "tipo" in o or "artefato" in o:
        return o
    rel = Path(p).resolve().relative_to(LINHAGEM.resolve()).with_suffix("").as_posix()
    return {"tipo": f"fechamento_artigo:{rel}", "nao_e": o.get("nao_e", NAO_E_FA),
            **{k: v for k, v in o.items() if k != "nao_e"}}


def gravar(p, o, sobrescrever: bool = True) -> Path:
    """Só dentro da linhagem; `sobrescrever=False` recusa trocar conteúdo existente (protocolos congelados)."""
    p = Path(p)
    if not str(p.resolve()).startswith(str(LINHAGEM.resolve())):
        raise EscritaRecusada(f"fora da linhagem: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        txt = json.dumps(_declarado(p, o), ensure_ascii=False, indent=1, default=str)
        if p.exists() and not sobrescrever:
            # Protocolo congelado ANTES da declaracao automatica continua valendo: compara com as duas formas.
            if p.read_text(encoding="utf-8") not in (txt, json.dumps(o, ensure_ascii=False, indent=1, default=str)):
                raise EscritaRecusada(f"conteudo congelado diferente: {p}")
            return p
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(txt, encoding="utf-8")
    elif p.suffix == ".csv":
        tmp = p.with_suffix(".csv.tmp")
        o.to_csv(tmp, index=False)
    elif p.suffix == ".parquet":
        tmp = p.with_suffix(".parquet.tmp")
        o.to_parquet(tmp, index=False)
    else:
        raise ValueError(p.suffix)
    tmp.replace(p)
    return p


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ métricas reescritas

def auc_pares(y, s) -> float:
    """P(escore positivo > negativo) + 0,5 P(empate), por contagem com busca binária (sem ranks nem sklearn)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y], np.sort(s[~y])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    menor = np.searchsorted(neg, pos, side="left")
    igual = np.searchsorted(neg, pos, side="right") - menor
    return float((menor + 0.5 * igual).sum() / (len(pos) * len(neg)))


def ap_por_prevalencia(y, s) -> float:
    """Precisão média (soma da precisão nos positivos, ordem decrescente, empates em bloco) / prevalência."""
    y = np.asarray(y).astype(float)
    s = np.asarray(s, dtype=float)
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    # empates: limiares distintos; precisão avaliada no fim de cada bloco de empate
    fim = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[fim]
    n = fim + 1
    prec = tp / n
    rec = tp / y.sum()
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return ap / y.mean()


def boot_percentil_laco(v, reps: int, semente: int, nivel: float = .95) -> list[float]:
    """Percentil com laço de `integers` (mesma sequência pseudoaleatória declarada no MT01/MT05/MT04)."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(semente)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)])
    q = round((1 - nivel) / 2, 12)   # 0,025 exato (os protocolos escrevem o quantil literal)
    return [float(np.quantile(b, q)), float(np.quantile(b, 1 - q))]


def boot_percentil_choice(v, reps: int, semente: int, nivel: float) -> list[float]:
    """Percentil com `choice` em bloco (sequência declarada no MT11)."""
    v = np.asarray(v, float)
    b = np.random.default_rng(semente).choice(v, (reps, len(v))).mean(1)
    q = (1 - nivel) / 2
    return [float(np.quantile(b, q)), float(np.quantile(b, 1 - q))]


def veredito_superioridade(media, ic, margem=.01) -> str:
    if media >= margem and ic[0] > 0:
        return "sustentado"
    if ic[1] < margem:
        return "nao_sustentado"
    return "indecidivel"


def leitura_cientifica(media, ic, margem=.01) -> dict:
    """Separa o que o intervalo permite dizer (o pedido do artigo): ganho estatístico, relevante, mínimo > margem."""
    return {"ganho_estatistico": bool(ic[0] > 0), "media_acima_da_margem": bool(media >= margem),
            "ganho_minimo_acima_da_margem": bool(ic[0] > margem), "perda_estatistica": bool(ic[1] < 0),
            "inconclusivo": bool(ic[0] <= 0 <= ic[1])}


def bloquear_rede():
    import eal_comum
    eal_comum.bloquear_rede()


def inspecao() -> bool:
    return os.environ.get("FA_INSPECAO") == "1"
