"""MT11, emenda 2: deriva o terreno das janelas pela cadeia canônica, com transporte FABDEM conferido.

Não edita `cadeia_em_lote` (hash no pré-registro). Como a rodada 3, troca em tempo de execução só
`baixar_fabdem.baixar`, e só quando a origem data.bris.ac.uk não responde. O substituto:
  1. lê pelo espelho GEE (`rodada3_fabdem_gee.baixar`) para um diretório TEMPORÁRIO;
  2. valida forma, transform, CRS, nodata e cobertura (não pode ser todo nodata);
  3. publica (os.replace) em `local_runs/mdt-fabdem/` só depois de validar.
Habilitação exige as três conferências gravadas: amostral (40), regional (60) e caminho de gravação (5),
todas idênticas. Cada janela termina classificada: completo | transporte_falhou | sem_cobertura_comprovada
| derivacao_falhou | parcial. Janela parcial é refeita (só ela) até 3 passadas. Estado vivo em
`local_runs/mt11-transdutivo/terreno_estado.json` (pid, comando, início, progresso, fim, rc).
Guarda de disco: para com menos de 15 GB livres e registra pendência.

Uso (pela trava):
    python scripts/manutencao/executar_serializado.py -- python scripts/metodo/mt11_terreno_executar.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for s in ("metodo", "ia", "terreno", "servico", "geo", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
from proveniencia import agora_utc  # noqa: E402

MT11 = REPO / "local_runs" / "mt11-transdutivo"
ESTADO = MT11 / "terreno_estado.json"
TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
ORIGEM = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N40E020-N50E030_FABDEM_V1-2.zip"
DISCO_MIN_GB = 15
PASSADAS = 3


def gravar_estado(**kw):
    e = json.loads(ESTADO.read_text(encoding="utf-8")) if ESTADO.exists() else {}
    e.update(kw, atualizado_utc=agora_utc())
    tmp = ESTADO.with_suffix(".part")
    tmp.write_text(json.dumps(e, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, ESTADO)


def origem_responde() -> bool:
    try:
        req = urllib.request.Request(ORIGEM, headers={"Range": "bytes=0-1023"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status in (200, 206)
    except Exception:  # noqa: BLE001 - qualquer recusa é "não responde"
        return False


def transporte_conferido() -> tuple[bool, dict]:
    arqs = {"amostral": MT11 / "fabdem_transporte.json", "regional": MT11 / "fabdem_transporte_regional.json",
            "gravacao": MT11 / "fabdem_caminho_gravacao.json"}
    if not all(a.exists() for a in arqs.values()):
        return False, {"motivo": "conferência ausente"}
    a, r, g = (json.loads(p.read_text(encoding="utf-8")) for p in arqs.values())
    ok = (a["comparadas"] >= 40 and a["identicas"] == a["comparadas"] and r["comparadas"] >= 40 and r["identicas"] == r["comparadas"]
          and g["total"] >= 5 and g["identicas"] == g["total"])
    return ok, {"amostral": [a["identicas"], a["comparadas"]], "regional": [r["identicas"], r["comparadas"]],
                "gravacao": [g["identicas"], g["total"]]}


def baixar_validado(plano: dict, saida: Path) -> dict:
    """Espelho -> temporário -> validação -> publicação."""
    import numpy as np
    import rasterio
    import rodada3_fabdem_gee as fg
    with tempfile.TemporaryDirectory(prefix="mt11_fab_", dir=str(saida)) as tmp:
        fg.baixar(plano, Path(tmp))   # SystemExit("... sem cobertura na bbox") quando o espelho é todo nodata
        nome = f"{plano['nome']}_fabdem_4326.tif"
        with rasterio.open(Path(tmp) / nome) as s:
            a = s.read(1, out_shape=(max(1, s.height // 8), max(1, s.width // 8)))
            if s.crs.to_epsg() != 4326 or abs(s.transform.a - 1 / 3600) > 1e-12 or s.nodata != fg.NODATA:
                raise RuntimeError(f"grade inesperada no espelho: {s.crs} {s.transform.a} {s.nodata}")
            if not (a != s.nodata).any():
                raise SystemExit(f"{plano['nome']}: FABDEM GEE sem cobertura na bbox")
            forma = [s.height, s.width]
        for f in (nome, f"{plano['nome']}_fabdem_manifesto.json"):
            os.replace(Path(tmp) / f, saida / f)
    return {"arquivo": str(saida / nome), "shape": forma}


def classificar(erro: str | None, pasta: Path) -> str:
    import mt11_emenda as em
    if erro is None:
        return "completo" if em.terreno_ok(pasta)["estado"] == "completo" else "parcial"
    if "sem cobertura na bbox" in erro and "GEE" in erro:
        return "sem_cobertura_comprovada"
    if any(k in erro for k in ("EEException", "HttpError", "computePixels", "Connection", "vsicurl", "oceano", "Timeout", "urlopen")):
        return "transporte_falhou"
    return "derivacao_falhou"


def main() -> int:
    import baixar_fabdem as fab
    import cadeia_em_lote as cl
    import mt11_emenda as em
    gravar_estado(pid=os.getpid(), comando=" ".join(sys.argv), inicio_utc=agora_utc(), fim_utc=None, rc=None, fase="inicio")
    transporte = "origem"
    if not origem_responde():
        ok, conf = transporte_conferido()
        if not ok:
            gravar_estado(fim_utc=agora_utc(), rc=2, fase="recusado", motivo="origem fora e transporte não conferido", conferencia=conf)
            return 2
        fab.baixar = lambda plano, saida=fab.SAIDA: baixar_validado(plano, saida)
        transporte = "espelho_gee_conferido"
    janelas = json.loads((MT11 / "janelas.json").read_text(encoding="utf-8"))
    estado_janelas = {}
    for passada in range(1, PASSADAS + 1):
        livre_gb = shutil.disk_usage(REPO).free / 1024 ** 3
        if livre_gb < DISCO_MIN_GB:
            gravar_estado(fase="parado_por_disco", livre_gb=round(livre_gb, 1))
            break
        plano = cl.planejar(janelas, cl.MARGEM_DE_BACIA_GRAUS)
        for j in plano["janelas"]:
            if j["ja_existe"] and em.terreno_ok(TER / j["nome"])["estado"] != "completo":
                j["ja_existe"] = False          # manifesto sem derivados íntegros: refazer só esta janela
        pendentes = [j for j in plano["janelas"] if not j["ja_existe"]]
        gravar_estado(fase=f"passada_{passada}", transporte=transporte, pendentes=len(pendentes), livre_gb=round(livre_gb, 1))
        if not pendentes:
            break
        plano["janelas"] = pendentes
        r = cl.executar(plano, refazer=True)
        erros = {f["nome"]: f["erro"] for f in r["falhas_detalhe"]}
        for j in pendentes:
            estado_janelas[j["nome"]] = {"passada": passada, "estado": classificar(erros.get(j["nome"]), TER / j["nome"]),
                                         "erro": (erros.get(j["nome"]) or "")[:240] or None}
        gravar_estado(janelas=estado_janelas, derivados=r["derivados"], falhas=r["falhas"])
        if all(v["estado"] in ("completo", "sem_cobertura_comprovada") for v in estado_janelas.values()):
            break
    final = {j["nome"]: em.terreno_ok(TER / j["nome"])["estado"] for j in janelas}
    rc = 0 if all(v == "completo" for v in final.values()) else 1
    gravar_estado(fim_utc=agora_utc(), rc=rc, fase="fim", estado_final=final,
                  completas=sum(v == "completo" for v in final.values()), total=len(final))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
