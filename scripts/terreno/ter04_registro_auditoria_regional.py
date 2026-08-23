"""
TER-04 -- Registro consolidado da auditoria de terreno, uma linha por regiao.

O QUE ESTE SCRIPT RESOLVE:

a auditoria bit a bit existe, mas espalhada. Recife tem a prova de reproducao
contra o raster de referencia (r = 1,0, max_abs_diff = 0,0, `recife_validation`).
Curitiba e Petropolis passaram pelo mesmo motor generico do `ter01` e a
comparacao contra a cadeia anterior ficou no `comparacao.json` do `ter03`. As
119 AOIs externas tem a sua no `resumo_todas.json` do `ter02`. Tres formatos,
tres lugares, e nenhuma resposta unica para "esta regiao esta auditada?".

Este script le os manifestos que o `ter01` ja grava -- e nao recalcula nada --
e monta um registro so, com as grandezas que decidem comparabilidade:

    resolucao efetiva      pixel do raster derivado
    limiar de canal        em AREA CONTRIBUINTE, que e grandeza fisica
    percentil equivalente  so como diagnostico da densidade de drenagem
    versao do WhiteboxTools, CRS, sha256 das saidas

POR QUE O LIMIAR EM AREA E A COLUNA QUE IMPORTA:

os manifestos antigos das tres regioes brasileiras registravam todos
"drenagem_percentil = 98,0". Convertido para area contribuinte, o criterio era
outro em cada uma -- 0,1123 km2 em Recife, 0,0847 em Curitiba e 0,7507 em
Petropolis, 6,7 vezes mais esparso. Como HAND e a altura acima do canal mais
proximo, tres limiares sao tres definicoes de canal, logo tres variaveis
diferentes com o mesmo nome de coluna. Se este registro mostrar area igual em
todas as linhas, a comparabilidade esta provada por construcao. Se nao mostrar,
o registro aponta exatamente onde.

O QUE ELE NAO FAZ: nao deriva, nao reamostra, nao decide se a diferenca e
aceitavel. Le manifesto, tabula e sinaliza divergencia contra a convencao.

Uso:
    python scripts/terreno/ter04_registro_auditoria_regional.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DERIV = RUNS / "ter-01-cadeia-harmonizada"
CMP_BR = RUNS / "ter-03-brasil-harmonizado" / "comparacao.json"
CMP_EXT = RUNS / "ter-02-comparacao" / "resumo_todas.json"
CMP_EXT_SERRA = RUNS / "ter-02-comparacao" / "resumo.json"
OUT = RUNS / "ter-04-registro-auditoria"

# convencao fixada no ter01. Divergir daqui invalida comparacao entre regioes.
AREA_CANAL_KM2 = 0.1123
RESOLUCAO_M = 30.0

# regioes com ponto rotulado, que e onde a comparabilidade tem consequencia
REGIOES_BR = ("recife_harmonizado", "curitiba_harmonizado", "petropolis_harmonizado")


def ler_manifesto(pasta: Path) -> dict | None:
    p = pasta / "run_manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def linha(nome: str, man: dict) -> dict:
    px = man.get("pixel_size") or [None, None]
    saidas = man.get("sha256") or man.get("saidas") or {}
    return {
        "regiao": nome,
        "crs": man.get("crs"),
        "pixel_m": px[0],
        "shape": "x".join(str(v) for v in (man.get("shape") or [])),
        "area_canal_km2": man.get("stream_area_km2"),
        "canal_celulas": (round(man["drenagem_limiar_celulas"], 1)
                          if man.get("drenagem_limiar_celulas") is not None else None),
        "percentil_equivalente": (round(man["stream_percentile_equivalente"], 1)
                                  if man.get("stream_percentile_equivalente") is not None
                                  else None),
        "wbt": man.get("versao_wbt") or man.get("whitebox_version"),
        "segundos": man.get("elapsed_s"),
        "n_saidas_com_hash": len(saidas) if isinstance(saidas, dict) else None,
    }


def divergencias(d: pd.DataFrame) -> list[str]:
    fora = []
    for _, r in d.iterrows():
        if r["area_canal_km2"] is not None and abs(
                float(r["area_canal_km2"]) - AREA_CANAL_KM2) > 1e-6:
            fora.append(f"{r['regiao']}: area de canal {r['area_canal_km2']} km2 "
                        f"!= convencao {AREA_CANAL_KM2}")
        if r["pixel_m"] is not None and abs(float(r["pixel_m"]) - RESOLUCAO_M) > 1e-6:
            fora.append(f"{r['regiao']}: pixel {r['pixel_m']} m != {RESOLUCAO_M} m")
    return fora


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===TER04_REGISTRO_AUDITORIA===")
    if not DERIV.exists():
        print(f"ABORTADO: {DERIV} ausente. Rode ter01.")
        return 1

    linhas, sem_manifesto = [], []
    for pasta in sorted(p for p in DERIV.iterdir() if p.is_dir()):
        man = ler_manifesto(pasta)
        if man is None:
            sem_manifesto.append(pasta.name)
            continue
        linhas.append(linha(pasta.name, man))
    d = pd.DataFrame(linhas)
    if d.empty:
        print("ABORTADO: nenhuma derivacao com manifesto.")
        return 1

    # Tres granularidades de derivacao, e a distincao importa: a AOI do CEMS
    # tem dezenas de km, o chip do Nivel 1 tem 3 a 5 km, e numa janela pequena
    # o canal relevante pode ficar de fora. Misturar as tres numa contagem so
    # esconderia exatamente o grupo em que o limite de HAND e mais apertado.
    def escopo(n: str) -> str:
        if n in REGIOES_BR:
            return "brasil"
        if n.startswith("uk_"):
            return "piloto_uk"
        if n.startswith("n1chip_"):
            return "chip_nivel1"
        return "aoi_externa"

    d["escopo"] = [escopo(n) for n in d["regiao"]]

    print(f"DERIVACOES={len(d)}  "
          + ", ".join(f"{k}={v}" for k, v in d.escopo.value_counts().items()))
    if sem_manifesto:
        print(f"SEM_MANIFESTO ({len(sem_manifesto)}): {sem_manifesto[:5]}")

    print("\n--- REGIOES COM PONTO ROTULADO ---")
    br = d[d.escopo.isin(["brasil", "piloto_uk"])]
    if len(br):
        print(br[["regiao", "crs", "pixel_m", "shape", "area_canal_km2",
                  "canal_celulas", "percentil_equivalente", "wbt"]]
              .to_string(index=False))

    print("\n--- CONVENCAO ---")
    fora = divergencias(d)
    if fora:
        print(f"DIVERGENCIAS ({len(fora)}):")
        for f in fora[:20]:
            print(f"   {f}")
    else:
        print(f"todas as {len(d)} derivacoes usam area de canal "
              f"{AREA_CANAL_KM2} km2 a {RESOLUCAO_M} m -- "
              "HAND comparavel entre regioes por construcao")

    # o percentil equivalente NAO deve ser constante: se ele varia, e porque a
    # densidade de drenagem varia entre regioes, que e informacao real. Um
    # percentil identico em todas seria o sintoma do erro antigo.
    p = d["percentil_equivalente"].dropna()
    if len(p) > 1:
        print(f"\npercentil equivalente: min={p.min():.1f} mediana={p.median():.1f} "
              f"max={p.max():.1f}  (variar e esperado: e a densidade de drenagem)")

    # ---- comparacoes ja medidas, trazidas para o mesmo lugar ----
    comparacoes: dict = {}
    if CMP_BR.exists():
        comparacoes["brasil_ter03"] = json.loads(CMP_BR.read_text(encoding="utf-8"))
        print("\n--- BRASIL: cadeia antiga contra harmonizada (ter03) ---")
        for reg, v in comparacoes["brasil_ter03"].items():
            for f in v.get("features", []):
                print(f"   {reg:10s} {f['variavel']:16s} pearson={f['pearson']:6.3f} "
                      f"razao={f['razao']}  nodata={f['perdidos_nodata']}")
    else:
        print(f"\nAVISO: {CMP_BR} ausente -- Brasil sem comparacao registrada")

    for chave, caminho in (("externo_todas", CMP_EXT), ("externo_serra", CMP_EXT_SERRA)):
        if caminho.exists():
            comparacoes[chave] = json.loads(caminho.read_text(encoding="utf-8"))
    ext = comparacoes.get("externo_todas") or comparacoes.get("externo_serra")
    if ext:
        print(f"\n--- EXTERNO: produto global contra cadeia propria "
              f"({ext.get('escopo', 'ingremes')}, {ext.get('pontos', 0):,} pontos) ---")
        for f, v in ext.get("por_feature", {}).items():
            print(f"   {f:16s} pearson_mediano={v['pearson_mediano']:6.3f} "
                  f"med {v['mediana_global']:8.2f} -> {v['mediana_wbt']:8.2f}")

    # ---- gravacao ----
    d.to_csv(OUT / "registro_derivacoes.csv", index=False)
    (OUT / "registro_auditoria.json").write_text(json.dumps({
        "convencao": {"area_canal_km2": AREA_CANAL_KM2, "resolucao_m": RESOLUCAO_M},
        "n_derivacoes": int(len(d)),
        "por_escopo": {k: int(v) for k, v in d["escopo"].value_counts().items()},
        "sem_manifesto": sem_manifesto,
        "divergencias_da_convencao": fora,
        "percentil_equivalente": {
            "min": float(p.min()), "mediana": float(p.median()),
            "max": float(p.max()), "nota": (
                "variacao esperada: o percentil e estatistica da janela, a area "
                "e a grandeza fisica. Percentil identico em todas as regioes "
                "seria o sintoma do erro corrigido em 12/08/2026")} if len(p) > 1 else None,
        "comparacoes": comparacoes,
        "limite_declarado": (
            "reproducao bit a bit contra raster de referencia so existe para "
            "Recife (recife_validation, r=1,0). Para as demais regioes o que "
            "existe e comparacao contra a cadeia ANTERIOR, que e outra coisa: "
            "mostra o efeito da troca, nao prova a implementacao"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
