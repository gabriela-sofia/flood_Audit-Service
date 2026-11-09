"""Linhagem transferencia-vulnerabilizados/v1: caminhos, gravacao imutavel e funcoes sem dependencia geoespacial.

As funcoes estatisticas ficam aqui, em numpy/pandas puros, para que os testes rodem
em qualquer ambiente do projeto e para que a verificacao independente possa usar
uma implementacao diferente (soma de ranks) contra elas.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "transferencia-vulnerabilizados" / "v1"
FONTES = RUNS / "transferencia-vulnerabilizados" / "fontes"
NAO_E = ("estudo de transferencia para territorios vulnerabilizados; nao e promocao de modelo, "
         "previsao de evento, profundidade, risco, dano, vitimas ou identificacao causal")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SETORES = {
    "PR": RUNS / "camada-social-fontes/ibge/PR_setores_CD2022.zip",
    "RS": RUNS / "social-fontes-complementares/2026-10-12/RS_setores_CD2022.zip",
    "AM": RUNS / "social-fontes-complementares/2026-10-12/AM_setores_CD2022.zip",
    "PE": FONTES / "setores_2022/PE_setores_CD2022.zip",
    "RJ": FONTES / "setores_2022/RJ_setores_CD2022.zip",
    "SP": FONTES / "setores_2022/SP_setores_CD2022.zip",
    "RO": FONTES / "setores_2022/RO_setores_CD2022.zip",
}
AGREGADO_BASICO = RUNS / "camada-social-fontes/ibge/Agregados_por_setores_basico_BR_20260520.zip"
AGREGADO_RENDA = RUNS / "camada-social-fontes/ibge/Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip"
SALARIO_MINIMO_2022 = 1212.0


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def ler(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def sem_nan(v):
    """Indefinido vira null no JSON (nunca 0): AUC sem duas classes nao e AUC zero."""
    if isinstance(v, dict):
        return {k: sem_nan(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sem_nan(x) for x in v]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def gravar(p, valor, raiz: Path = OUT):
    """Escreve so dentro da linhagem; JSON ganha tipo/nao_e/gerado_por; troca atomica."""
    p = Path(p).resolve()
    if not p.is_relative_to(raiz.resolve()):
        raise ValueError("escrita fora da linhagem transferencia-vulnerabilizados/v1")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        if not isinstance(valor, dict):
            valor = {"registros": valor}
        valor = sem_nan({"tipo": "transferencia_vulnerabilizados:" + p.relative_to(raiz.resolve()).with_suffix("").as_posix(),
                         "nao_e": NAO_E, "gerado_por": "scripts/vulnerabilizados", **valor})
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    elif p.suffix == ".csv":
        tmp = p.with_suffix(".csv.tmp")
        valor.to_csv(tmp, index=False)
    elif p.suffix == ".parquet":
        tmp = p.with_suffix(".parquet.tmp")
        valor.to_parquet(tmp, index=False)
    elif p.suffix in (".md", ".txt"):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(valor, encoding="utf-8")
    else:
        raise ValueError(p.suffix)
    tmp.replace(p)
    return p


def congelar(p, valor: dict) -> dict:
    """Primeira escrita fixa o conteudo; qualquer divergencia posterior falha (nova versao exigida)."""
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use outra versao")
        return existente
    gravar(p, {**valor, "quando_utc": agora()})
    return ler(p)


def bloquear_rede():
    import socket

    def recusada(*args, **kwargs):
        raise RuntimeError("rede bloqueada nesta etapa da linhagem")
    socket.socket.connect = recusada
    socket.socket.connect_ex = recusada


# ---------------------------------------------------------------- estatistica

def auc_mann_whitney(y, s) -> float | None:
    """AUC por U de Mann-Whitney com ranks medios (empates contam meio). None sem as duas classes."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_score(y, s) -> float | None:
    """Precisao media (degraus, sem interpolacao), como em sklearn.average_precision_score."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return None
    o = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def bootstrap_por_grupo(df: pd.DataFrame, grupo: str, estat, b: int, semente: int) -> np.ndarray:
    """Reamostra grupos inteiros (nunca pontos) com reposicao; devolve b estatisticas (NaN se indefinida)."""
    rng = np.random.default_rng(semente)
    ids = np.array(sorted(df[grupo].unique()))
    blocos = {g: idx for g, idx in df.groupby(grupo).indices.items()}
    out = np.full(b, np.nan)
    for i in range(b):
        escolha = rng.choice(ids, size=len(ids), replace=True)
        linhas = np.concatenate([blocos[g] for g in escolha])
        v = estat(df.iloc[linhas])
        out[i] = np.nan if v is None else v
    return out


class AucPonderavel:
    """AUC com pesos por ponto, empates contados como meio; ordem e blocos calculados uma vez.

    Com pesos inteiros (multiplicidade do grupo no bootstrap) e igual a AUC de Mann-Whitney da
    amostra replicada, sem materializa-la.
    """

    def __init__(self, y, s):
        self.y = np.asarray(y).astype(float)
        _, self.bloco = np.unique(np.asarray(s, dtype=float), return_inverse=True)
        self.nb = int(self.bloco.max()) + 1 if len(self.bloco) else 0

    def __call__(self, w) -> float:
        w1 = np.bincount(self.bloco, weights=w * self.y, minlength=self.nb)
        w0 = np.bincount(self.bloco, weights=w * (1 - self.y), minlength=self.nb)
        t1, t0 = w1.sum(), w0.sum()
        if t1 == 0 or t0 == 0:
            return np.nan
        abaixo = np.cumsum(w0) - w0
        return float((w1 * (abaixo + 0.5 * w0)).sum() / (t1 * t0))


def pesos_bootstrap(grupos: pd.Series, b: int, semente: int):
    """Gera b vetores de peso por ponto: multiplicidade do grupo numa reamostragem de grupos com reposicao."""
    cod, uniq = pd.factorize(grupos, sort=True)
    rng = np.random.default_rng(semente)
    g = len(uniq)
    for _ in range(b):
        cont = np.bincount(rng.integers(0, g, g), minlength=g).astype(float)
        yield cont[cod]


def bootstrap_auc(df: pd.DataFrame, grupo: str, colunas: list[str], b: int, semente: int) -> np.ndarray:
    """Matriz b x len(colunas) de AUC sob as mesmas reamostragens de grupos (pareado entre colunas)."""
    aucs = [AucPonderavel(df.y, df[c]) for c in colunas]
    return np.array([[a(w) for a in aucs] for w in pesos_bootstrap(df[grupo], b, semente)])


def bootstrap_media(df: pd.DataFrame, grupo: str, coluna: str, b: int, semente: int) -> np.ndarray:
    v = df[coluna].astype(float).to_numpy()
    return np.array([(w * v).sum() / w.sum() for w in pesos_bootstrap(df[grupo], b, semente)])


def intervalo(valores: np.ndarray, nivel: float) -> list[float] | None:
    v = valores[np.isfinite(valores)]
    if len(v) < 0.9 * len(valores) or len(v) == 0:
        return None
    a = (1 - nivel) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


def rotulo_referencia(agua_evento, agua_pre, permanente, valido_evento, valido_pre) -> np.ndarray:
    """Estados da referencia optica, na ordem de precedencia do protocolo.

    desconhecido: sem observacao valida na data do evento (nuvem, sombra, sem dado);
    permanente: agua na mascara JRC, ou agua tambem na cena pre-evento valida;
    pre_desconhecido: agua no evento, mas a cena pre-evento nao e valida (nao separa permanente);
    positivo: agua no evento, nao permanente; negativo: observado sem agua no evento.
    Ausencia de agua nunca e inferida onde a observacao nao e valida.
    """
    agua_evento = np.asarray(agua_evento, bool)
    agua_pre = np.asarray(agua_pre, bool)
    permanente = np.asarray(permanente, bool)
    valido_evento = np.asarray(valido_evento, bool)
    valido_pre = np.asarray(valido_pre, bool)
    out = np.full(agua_evento.shape, "negativo", dtype=object)
    out[agua_evento] = "positivo"
    out[agua_evento & ~valido_pre] = "pre_desconhecido"
    out[permanente | (valido_pre & agua_pre)] = "permanente"
    out[~valido_evento] = "desconhecido"
    return out
