"""Confere o quadro territorial de desenvolvimento sem abrir novas reservas.

O manifesto descreve a coleta; a auditoria confere o que chegou ao arquivo.
Uma visada seca nao comprova ausencia durante todo o pico. Pontos repetidos
entre tiles com buffer nao sao novas observacoes independentes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

COLUNAS = ("ponto_id", "evento_id", "unidade", "lon", "lat", "flooded",
           "clear_views", "jrc_perm_water", "estado")
ESTADOS = ("positivo", "negativo_observado", "sem_visada", "agua_permanente", "fora_da_cena")


def classes_observacionais(d: pd.DataFrame) -> pd.Series:
    """Reconstrucao independente da admissao: desconhecido nunca vira negativo."""
    a, c, p = (d[k].to_numpy(float) for k in ("flooded", "clear_views", "jrc_perm_water"))
    valido = (np.isfinite(a) & np.isfinite(c) & np.isfinite(p)
              & np.isin(a, (0, 1)) & np.isin(p, (0, 1)) & (c >= 0) & (c == np.floor(c)))
    s = np.full(len(d), "fora_da_cena", dtype=object)
    s[valido & (p == 1)] = "agua_permanente"
    s[valido & (p == 0) & (a == 1)] = "positivo"
    s[valido & (p == 0) & (a == 0) & (c > 0)] = "negativo_observado"
    s[valido & (p == 0) & (a == 0) & (c == 0)] = "sem_visada"
    return pd.Series(s, index=d.index, name="estado_reconstruido")


def conferir(d: pd.DataFrame, manifesto: dict, eventos_conhecidos: set[str]) -> dict:
    esperado = classes_observacionais(d)
    problemas, alertas = [], []
    divergencias = int((d.estado != esperado).sum())
    if divergencias:
        problemas.append(f"{divergencias} classes divergem das bandas observacionais")
    ids = int(d.ponto_id.duplicated().sum())
    if ids or d.ponto_id.isna().any():
        problemas.append("ponto_id repetido ou ausente")
    coordenadas = d[["lon", "lat"]].to_numpy(float)
    if not (np.isfinite(coordenadas).all() and (d.lon.abs() <= 180).all() and (d.lat.abs() <= 90).all()):
        problemas.append("coordenadas invalidas")
    desconhecidos = sorted(set(d.evento_id.dropna()) - eventos_conhecidos)
    if desconhecidos or d.evento_id.isna().any():
        problemas.append("evento fora do desenvolvimento conhecido ou sem identidade")
    contagens = {str(k): int(v) for k, v in d.estado.value_counts().items()}
    if contagens != manifesto.get("por_estado"):
        problemas.append("contagens por estado divergem do manifesto")
    if manifesto.get("registro_sha256") != sha256_json(manifesto.get("registro", [])):
        problemas.append("hash do registro de coleta divergente")
    for registro in manifesto.get("registro", []):
        sub = d[(d.evento_id == registro["evento"]) & (d.unidade == registro["unidade"])]
        if len(sub) != registro["pontos"] or sub.estado.value_counts().to_dict() != registro["por_estado"]:
            problemas.append(f"registro diverge: {registro['evento']} {registro['unidade']}")
    registrados = {(x["evento"], x["unidade"]) for x in manifesto.get("registro", [])}
    if registrados != set(zip(d.evento_id, d.unidade)):
        problemas.append("pares evento/unidade nao coincidem com o registro")
    # Mesma grade EE entre buffers: sete casas bastam para identificar a celula repetida,
    # mas nao para provar independencia espacial dos pontos nao repetidos.
    chaves = d[["evento_id", "lon", "lat"]].copy()
    chaves[["lon", "lat"]] = chaves[["lon", "lat"]].round(7)
    extras = int(chaves.duplicated().sum())
    repetidos = chaves.duplicated(keep=False)
    conflitos = 0
    if repetidos.any():
        grupos = chaves.loc[repetidos].assign(estado=d.loc[repetidos, "estado"])
        conflitos = int((grupos.groupby(["evento_id", "lon", "lat"]).estado.nunique() > 1).sum())
        alertas.append("coordenadas repetidas no evento: declarar ponderacao e auditar sobreposicao dos buffers")
    if conflitos:
        problemas.append("mesma coordenada/evento com classes conflitantes")
    classes = d[d.estado.isin(("positivo", "negativo_observado"))]
    if len(classes) != (manifesto.get("admissao") or {}).get("ensinam"):
        problemas.append("admissao total diverge do manifesto")
    if int((classes.estado == "positivo").sum()) != (manifesto.get("admissao") or {}).get("positivos"):
        problemas.append("positivos admitidos divergem do manifesto")
    eventos = classes.groupby("evento_id").estado.agg(["size", "nunique"])
    return {"estado": "divergente" if problemas else "coerente_com_alertas" if alertas else "coerente",
            "problemas": problemas, "alertas": alertas, "linhas": int(len(d)),
            "eventos": int(d.evento_id.nunique()), "pares_evento_unidade": len(registrados),
            "por_estado": contagens, "classes_divergentes": divergencias,
            "ids_repetidos_excedentes": ids, "coordenadas_repetidas_excedentes": extras,
            "grupos_com_classes_conflitantes": conflitos,
            "eventos_fora_do_desenvolvimento": desconhecidos,
            "eventos_com_duas_classes": int((eventos["nunique"] == 2).sum()),
            "falhas_da_coleta": manifesto.get("falhas", []),
            "nao_e": "nao valida o sensor, a cronologia do pico ou independencia espacial"}


def executar(pasta: Path, ds06: Path) -> dict:
    pontos, man = pasta / "pontos.parquet", pasta / "manifesto.json"
    codigo = REPO / "scripts/geo/tc01_territorio_consultavel.py"
    alvos = {"pontos": pontos, "manifesto": man, "ds06": ds06, "codigo_coleta_atual": codigo,
             "codigo_auditoria": Path(__file__)}
    hashes = {k: sha256_arquivo(p) for k, p in alvos.items()}
    if any(v is None for v in hashes.values()):
        raise FileNotFoundError("auditoria exige os arquivos de desenvolvimento existentes")
    manifesto = json.loads(man.read_text(encoding="utf-8"))
    conhecidos = pd.read_parquet(ds06, columns=["fonte", "evento_id"])
    eventos = set(conhecidos.loc[conhecidos.fonte == "gfd", "evento_id"])
    d = pd.read_parquet(pontos, columns=list(COLUNAS))
    r = {"tipo": "auditoria_tc01", "regime": "auditoria_do_desenvolvimento", "origens": hashes,
         "auditoria": conferir(d, manifesto, eventos),
         "codigo_da_coleta_congelado_no_manifesto": False,
         "limite_de_linhagem": "hash do codigo atual nao prova qual codigo executou a coleta antiga",
         "rotulos_novos_lidos": False, "dados_alterados": False, "promovido": False}
    for k, p in alvos.items():
        if hashes[k] != sha256_arquivo(p):
            raise RuntimeError(f"arquivo mudou durante auditoria: {p}")
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    return r


def selecionar_celulas(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma celula por evento, escolha pela identidade do tile, nao pela classe."""
    x = d.assign(_lon=d.lon.round(7), _lat=d.lat.round(7))
    chaves = ["evento_id", "_lon", "_lat"]
    repetidos = x.duplicated(chaves, keep=False)
    if (x.loc[repetidos].groupby(chaves).estado.nunique() > 1).any():
        raise ValueError("classes conflitantes: curadoria exige revisao da observacao")
    x = x.sort_values(chaves + ["unidade", "ponto_id"], kind="stable")
    removidos = x.duplicated(chaves)
    principais = x.loc[~removidos, chaves + ["ponto_id"]].rename(columns={"ponto_id": "ponto_id_mantido"})
    aliases = x.loc[removidos, chaves + ["ponto_id", "unidade"]].merge(principais, on=chaves, validate="many_to_one")
    return (x.loc[~removidos].drop(columns=["_lon", "_lat"]).reset_index(drop=True),
            aliases.rename(columns={"_lon": "lon", "_lat": "lat"}).reset_index(drop=True))


def curar(pasta: Path, auditoria: dict, saida: Path) -> Path:
    """Derivacao separada: consumidores antigos nao sao redirecionados."""
    if auditoria["auditoria"]["problemas"]:
        raise ValueError("dados divergentes; nao curar automaticamente")
    origem = pasta / "pontos.parquet"
    if sha256_arquivo(origem) != auditoria["origens"]["pontos"]:
        raise RuntimeError("pontos mudaram depois da auditoria")
    receita = {"tipo": "tc01_curado", "versao": "v1", "origem_sha256": auditoria["origens"]["pontos"],
               "auditoria_sha256": auditoria["hash_resumo"], "codigo_sha256": sha256_arquivo(Path(__file__)),
               "regra": "evento e coordenadas arredondadas a 7 casas; manter menor unidade/ponto_id, sem usar classe",
               "regime": "derivacao_de_desenvolvimento", "promovido": False,
               "consumidores_redirecionados": False,
               "nao_e": "nao restaura independencia entre eventos nem valida observacao ou territorio novo"}
    destino = saida / sha256_json(receita).split(":")[1]
    if destino.exists():
        man = json.loads((destino / "manifesto.json").read_text(encoding="utf-8"))
        for nome, chave in (("pontos.parquet", "pontos_sha256"), ("aliases.parquet", "aliases_sha256")):
            if sha256_arquivo(destino / nome) != man.get(chave):
                raise RuntimeError("curadoria existente divergente; nao sobrescrever")
        return destino / "manifesto.json"
    d = pd.read_parquet(origem)
    curado, aliases = selecionar_celulas(d)
    if sha256_arquivo(origem) != auditoria["origens"]["pontos"]:
        raise RuntimeError("pontos mudaram durante a curadoria")
    destino.mkdir(parents=True)
    curado.to_parquet(destino / "pontos.parquet", index=False)
    aliases.to_parquet(destino / "aliases.parquet", index=False)
    r = {**receita, "linhas_origem": int(len(d)), "linhas_curadas": int(len(curado)),
         "repeticoes_removidas": int(len(aliases)), "por_estado": curado.estado.value_counts().to_dict(),
         "pontos_sha256": sha256_arquivo(destino / "pontos.parquet"),
         "aliases_sha256": sha256_arquivo(destino / "aliases.parquet"),
         "proxima_avaliacao": "novo experimento de desenvolvimento com linhagem, mesmo quadro e controle de repeticao"}
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    man = destino / "manifesto.json"
    man.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--curar", type=Path, help="versao derivada sem repeticoes; nao altera consumidores")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "scripts/suscetibilidade"))
    from ds06_dataset_de_treino import versao_atual
    r = executar(REPO / "local_runs/tc01-territorio-consultavel/desenvolvimento",
                 versao_atual() / "pontos.parquet")
    destino = args.saida / r["hash_resumo"].split(":")[1] / "resultado.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if {k: v for k, v in antigo.items() if k != "quando_utc"} != {k: v for k, v in r.items() if k != "quando_utc"}:
            raise RuntimeError("snapshot da auditoria divergente")
    else:
        destino.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GRAVADO={destino}")
    print(json.dumps(r["auditoria"], ensure_ascii=False))
    if args.curar:
        print(f"CURADORIA={curar(REPO / 'local_runs/tc01-territorio-consultavel/desenvolvimento', r, args.curar)}")
    return int(bool(r["auditoria"]["problemas"]))


if __name__ == "__main__":
    raise SystemExit(main())
