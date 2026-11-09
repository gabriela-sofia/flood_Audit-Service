"""Verifica campanha encerrada sem importar treino, abrir reservas ou alterar suas fontes.

Recalcula AUC por contagem de pares com empates e P1/P2 a partir das métricas por
evento. Saída exclusivamente em um diretório novo da coordenação. Não executa modelos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAMPANHA = Path("local_runs/fechamento-aprendizado-transferencia/v1/campanha")


def hash_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def ler(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def auc_pares(y, scores):
    """P(escore positivo > negativo) + metade dos empates, sem sklearn/scipy."""
    y, s = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("escores ausentes/não finitos ou dimensões incompatíveis")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("alvo não binário")
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="stable")
    sy, ss = y[order], s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    n = np.diff(np.r_[starts, len(y)])
    positives = np.add.reduceat(sy, starts)
    negatives = n - positives
    previous = np.cumsum(negatives) - negatives
    return float(np.sum(positives * (previous + negatives / 2)) / (pos * neg))


def agregados(m: pd.DataFrame, sementes) -> dict[str, float]:
    """Não permite que pivot/mean escondam sementes faltantes ou linhas duplicadas."""
    g = m[(m.recorte == "todos") & m.destino_campanha.str.startswith("dobra_")
          & m.braco.isin(["conjunto", "parada"])].copy()
    g["semente"] = g.semente.astype(str)
    keys = ["destino", "evento_id", "braco", "semente"]
    if g.empty or g.duplicated(keys).any():
        raise ValueError("métrica primária ausente ou duplicada")
    expected = set(map(str, sementes))
    for _, sub in g.groupby(["destino", "evento_id", "braco"]):
        if set(sub.semente) != expected:
            raise ValueError("sementes incompletas na primária")
    if g.groupby(["destino", "evento_id"]).braco.nunique().ne(2).any():
        raise ValueError("braço primário ausente")
    if not np.isfinite(g.auc).all():
        raise ValueError("AUC primária não finita")
    e = g.groupby(["destino", "evento_id", "braco"]).auc.mean().unstack()
    d = (e.conjunto - e.parada).groupby(level="destino").mean()
    return {str(k): float(v) for k, v in d.sort_index().items()}


def resumir(d: dict, *, nivel=.975, reps=2000, minimo=8, margem=.01):
    v = np.asarray([d[k] for k in sorted(d)], dtype=float)
    if not len(v) or not np.isfinite(v).all():
        raise ValueError("contraste vazio ou não finito")
    r = {"n": len(v), "media": float(v.mean()), "por_unidade": d}
    if len(v) < minimo:
        return {**r, "veredito": "not_testable"}
    # Mesma distribuição e semente do protocolo, implementação de índices separada.
    rng = np.random.default_rng(7)
    means = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    q = (1 - nivel) / 2
    lo, hi = map(float, np.quantile(means, [q, 1 - q]))
    verdict = "sustentado" if r["media"] >= margem and lo > 0 else (
        "nao_sustentado" if hi < margem else "indecidivel")
    return {**r, "ic": [lo, hi], "nivel_ic": nivel, "veredito": verdict}


def comparar(obtido, salvo, tolerancia=1e-12):
    erros = []
    for k in ("n", "veredito"):
        if obtido.get(k) != salvo.get(k):
            erros.append(f"{k} diverge")
    for k in ("media", "ic"):
        a, b = obtido.get(k), salvo.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or not np.allclose(a, b, rtol=0, atol=tolerancia):
            erros.append(f"{k} diverge")
    if set(obtido["por_unidade"]) != set(salvo.get("por_unidade", {})):
        erros.append("identidade das unidades diverge")
    elif any(abs(v - salvo["por_unidade"][k]) > tolerancia for k, v in obtido["por_unidade"].items()):
        erros.append("efeito por unidade diverge")
    return erros


def plano(protocolo):
    out = []
    for destino, bracos in protocolo["plano"]:
        dests = [d for d in protocolo["destinos"] if d.startswith("dobra_")] if destino == "dobras" else [destino]
        for d in dests:
            for b in bracos:
                for s in protocolo["config"]["sementes"]:
                    out.append((d, b, str(s)))
    if len(out) != len(set(out)) or len(out) != protocolo["ajustes_previstos"]:
        raise ValueError("plano duplicado ou contagem incompatível")
    return out


def auditar(campanha: Path):
    fontes = {}
    def fonte(p):
        fontes[str(p.relative_to(campanha))] = hash_arquivo(p)
    for name in ("protocolo.json", "analise.json", "medidas_fisicas.parquet"):
        fonte(campanha / name)
    prot, analise = ler(campanha / "protocolo.json"), ler(campanha / "analise.json")
    m = pd.read_parquet(campanha / "medidas_fisicas.parquet")
    required = {"destino_campanha", "destino", "evento_id", "recorte", "braco", "semente", "auc"}
    if not required.issubset(m.columns):
        raise ValueError("esquema das métricas incompatível")
    erros, ajustes, n_auc, max_dif, temporal_hashes = [], 0, 0, 0., 0
    wall = 0.
    expected = plano(prot)
    actual = {(p.parent.parent.name, *p.parent.name.rsplit("_s", 1))
              for p in campanha.glob("*/*_s*/avaliacao.json")}
    if actual != set(expected):
        erros.append({"tipo": "completude", "faltantes": sorted(set(expected) - actual),
                      "inesperados": sorted(actual - set(expected))})
    groups = {(str(d), str(b), str(s)): sub for (d, b, s), sub in
              m.dropna(subset=["semente"]).groupby(["destino_campanha", "braco", "semente"])}
    for dest, b, seed in expected:
        folder = campanha / dest / f"{b}_s{seed}"
        try:
            ev, tr = ler(folder / "avaliacao.json"), ler(folder / "treino.json")
            if tr["braco"] != b or str(tr["semente"]) != seed or tr["passos"] != prot["config"]["passos"]:
                raise ValueError("identidade/passos do treino divergem")
            p = folder / "predicoes_fisicas.parquet"
            fonte(p); fonte(folder / "avaliacao.json"); fonte(folder / "treino.json")
            if fontes[str(p.relative_to(campanha))] != ev["predicoes_sha256"]:
                raise ValueError("hash das predições físicas diverge")
            pred = pd.read_parquet(p)
            if pred.ponto_id.duplicated().any():
                raise ValueError("ponto repetido na predição")
            measures = groups.get((dest, b, seed))
            if measures is None or measures.duplicated(["destino", "evento_id", "recorte"]).any():
                raise ValueError("métricas ausentes ou duplicadas")
            for row in measures.itertuples():
                sub = pred[pred.evento_id == row.evento_id]
                if row.recorte != "todos":
                    proc = row.recorte.split(":", 1)[1]
                    sub = sub[(sub.classe == 0) | (sub.processo_ponto == proc)]
                a = auc_pares(sub.classe.to_numpy(), sub.escore.to_numpy())
                if a is None or not np.isfinite(row.auc):
                    raise ValueError("AUC inválida nas métricas registradas")
                dif = abs(a - row.auc)
                max_dif = max(max_dif, dif); n_auc += 1
                if dif > 1e-12:
                    raise ValueError("AUC recalculada diverge")
            if ev.get("temporal_sha256"):
                tp = folder / "temporal_escore.npz"; fonte(tp)
                if fontes[str(tp.relative_to(campanha))] != ev["temporal_sha256"]:
                    raise ValueError("hash da predição temporal diverge")
                temporal_hashes += 1
            wall += float(tr["segundos"])
            ajustes += 1
        except (OSError, ValueError, KeyError, TypeError) as e:
            erros.append({"ajuste": f"{dest}/{b}_s{seed}", "erro": str(e)})
    primarias = {}
    try:
        d = agregados(m, prot["config"]["sementes"])
        pop = analise["primarias"]["populacao_supervisionada"]
        declarado = prot.get("primarias", {}).get("P1", {}).get("unidades")
        if isinstance(declarado, int) and len(d) != declarado:
            raise ValueError("número de componentes difere do protocolo P1")
        parts = [campanha / dest / "particao.json" for dest in prot["destinos"] if dest.startswith("dobra_")]
        if any(p.exists() for p in parts):
            pop_parts = set()
            for part_file in parts:
                fonte(part_file)
                pop_parts.update(ler(part_file)["populacao_supervisao_admissivel_destino"])
            if pop_parts != set(pop):
                raise ValueError("população P2 difere das partições congeladas")
        if len(pop) != len(set(pop)) or not set(pop).issubset(d):
            raise ValueError("população P2 duplicada ou fora da P1")
        for name, values in (("P1_43", d), ("P2_supervisionada", {k: d[k] for k in pop})):
            cfg = prot["config"]
            calc = resumir(values, nivel=cfg["nivel_primarias"], reps=cfg["bootstrap"],
                           minimo=cfg["minimo_unidades"], margem=cfg["margem_auc"])
            difs = comparar(calc, analise["primarias"][name])
            primarias[name] = {**calc, "divergencias": difs}
            erros.extend({"primaria": name, "erro": e} for e in difs)
    except (ValueError, KeyError, TypeError) as e:
        erros.append({"tipo": "primarias", "erro": str(e)})
    for rel, hashed in fontes.items():
        if hash_arquivo(campanha / rel) != hashed:
            erros.append({"tipo": "fonte_alterada_durante_auditoria", "arquivo": rel})
    return {"tipo": "verificacao_fechamento_coordenacao_v1", "quando_utc": datetime.now(timezone.utc).isoformat(),
            "verificador_sha256": hash_arquivo(Path(__file__)),
            "campanha": str(campanha), "integridade_numerica": "conferida" if not erros else "divergente",
            "ajustes_previstos": len(expected), "ajustes_verificados": ajustes,
            "auc_fisicas_recalculadas": n_auc, "max_dif_auc": max_dif,
            "primarias": primarias, "predicoes_temporais_hashes_conferidos": temporal_hashes,
            "alvos_temporais_reconstruidos": False,
            "segundos_relogio_somados_dos_ajustes": wall, "erros": erros, "fontes": fontes,
            "limites": ["não confirma validade física dos rótulos nem causalidade",
                       "não inclui novas unidades de confirmação",
                       "bootstrap reproduz a regra histórica e seus limites de treinos sobrepostos",
                       "hash temporal não equivale a reconstrução do alvo ou de todas as AUC temporais"],
            "promovido": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campanha", type=Path, default=ROOT / CAMPANHA)
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args()
    if args.saida.resolve().is_relative_to(args.campanha.resolve()):
        ap.error("saída deve ficar fora da campanha examinada")
    if args.saida.exists():
        ap.error("saída já existe; use versão nova")
    result = auditar(args.campanha.resolve())
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("integridade_numerica", "ajustes_verificados", "auc_fisicas_recalculadas", "max_dif_auc", "erros")}, ensure_ascii=False))
    return 0 if not result["erros"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
