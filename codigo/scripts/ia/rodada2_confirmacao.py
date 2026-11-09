"""Rodada 2, parte 6: pacote de confirmacao protegida dos pesos Q (MT05) e das candidatas do fechamento.

Linhagem `referencia_fisica_forcamento_v1` (17/10/2026). Nao abre reserva e nao le rotulo de unidade nova.

  congelar     hashes dos pesos candidatos: Q_base/Q_conexao (MT05, `pesos_e_padronizacao.json`) e
               conjunto/parada/fisica_so do fechamento (`final/*/final.pt`); escore reproduzivel sem o objeto
               de treino (Q) e por carga do estado com hash conferido (rede).
  elegibilidade lista CEMS publica (so metadados) x tudo que o projeto ja examinou (codigos em nomes de
               artefatos de local_runs, eventos dos 510.348 pontos, reservas de outros protocolos).
  importar     mascara de um evento novo (GeoTIFF 1 = inundado observado, 0 = seco observado, outro = desconhecido)
               + terreno ter-01 da AOI (derivado por `scripts/terreno/derivar_aoi_nova.py`) -> pontos com as regras
               MT01 (teto 2.000 por evento-classe, semente 20261010; desconhecido nunca vira negativo).
  avaliar      AUC dentro do evento para A0 (-HAND), Q_base, Q_conexao, conjunto, parada; media por componente;
               contrastes Q_conexao - Q_base e conjunto - Q_conexao; margem 0,01; protocolo deve ser congelado ANTES.

Uso:
    python scripts/ia/rodada2_confirmacao.py congelar|elegibilidade
    python scripts/ia/rodada2_confirmacao.py importar --mascara M.tif --terreno DIR --evento ID --componente C
    python scripts/ia/rodada2_confirmacao.py avaliar --pontos P.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "sar", "servico", "metodo", "neural"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import conjunta_base as cb  # noqa: E402
import rodada2_referencia_optica as ro  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

CONF = ro.RF / "confirmacao"
PESOS_Q = REPO / "local_runs/mecanismos-transferiveis/v5/sem_amostra_v1/candidatas/pesos_e_padronizacao.json"
FINAL = REPO / "local_runs/fechamento-aprendizado-transferencia/v1/final"
TETO_TESTE = 2000
SEMENTE = 20261010


def gravar(p, o):
    ro.gravar(p, o)


def congelar() -> dict:
    r = {"tipo": "pesos_candidatos_congelados", "quando_utc": agora_utc(),
         "Q": {"arquivo": str(PESOS_Q.relative_to(REPO)), "sha256": sha256_arquivo(PESOS_Q)},
         "rede": {b: {"arquivo": str((FINAL / b / "final.pt").relative_to(REPO)), "sha256": sha256_arquivo(FINAL / b / "final.pt"),
                      "hash_estado": json.loads((FINAL / b / "treino.json").read_text(encoding="utf-8"))["hash_estado"]}
                  for b in ("conjunto", "parada", "fisica_so")},
         "entradas": list(cb.TERRENO) + list(cb.COBERTURA),
         "proibido_na_consulta": ["SAR", "mascara do evento", "rotulo", "elevacao relativa a amostra", "sociais"]}
    gravar(CONF / "pesos_congelados.json", r)
    return r


def escore_q(nome: str, x: pd.DataFrame) -> np.ndarray:
    p = json.loads(PESOS_Q.read_text(encoding="utf-8"))[nome]
    pad = p["padronizador"]
    cols = pad["colunas"]
    z = (x[cols].fillna(pd.Series(pad["mediana"], index=cols)) - pd.Series(pad["media"], index=cols)) / pd.Series(pad["desvio"], index=cols)
    z = z.fillna(0.)
    s = sum(z[k] * v for k, v in p["coeficientes"].items())
    for it in p["interacoes"]:  # [{"colunas": [a, b], "peso": w}]
        a, b = it["colunas"]
        s = s + z[a] * z[b] * it["peso"]
    return np.asarray(s, float)


def elegibilidade() -> dict:
    lista = json.loads((CONF / "lista_cems_20261017.json").read_text(encoding="utf-8"))["itens"]
    men = set()
    for root, dirs, files in os.walk(REPO / "local_runs"):
        if "triagem_cems" in root or "catalog" in root.lower() or "confirmacao" in root:
            continue
        for n in dirs + files:
            men.update(re.findall(r"EMSR\d{3}", n))
    pts = pd.read_parquet(cb.BASE / "pontos.parquet", columns=["evento_id"])
    men.update(re.findall(r"EMSR\d{3}", " ".join(pts.evento_id.unique())))
    reserva = REPO / "local_runs/mecanismos-transferiveis/confirmacao/reserva.json"
    reservadas = set(re.findall(r"EMSR\d{3}", reserva.read_text(encoding="utf-8")))
    tri = pd.read_csv(REPO / "local_runs/mecanismos-transferiveis/v3/triagem_cems/triagem.csv").set_index("code")
    geoid = set(tri[tri.motivo_exclusao == "reserva_sar_geoid"].index)
    linhas = []
    for a in lista:
        if a.get("category") != "Flood":
            continue
        c = a["code"]
        estado = ("examinado" if c in men and c not in reservadas | geoid else
                  "reservado_mt_confirmacao" if c in reservadas else "reservado_sar_geoid" if c in geoid else "nao_examinado")
        linhas.append({"code": c, "subcategoria": a.get("subCategory"), "nome": a.get("name"), "evento": a.get("eventTime"),
                       "paises": a.get("countries"), "estado": estado})
    t = pd.DataFrame(linhas)
    t.to_csv(CONF / "elegibilidade_cems.csv", index=False)
    r = {"tipo": "elegibilidade_por_metadados", "quando_utc": agora_utc(), "inundacoes_no_catalogo": len(t),
         "por_estado": t.estado.value_counts().to_dict(),
         "nao_examinadas_e_nao_reservadas": t[t.estado == "nao_examinado"].code.tolist(),
         "reservadas": t[t.estado.str.startswith("reservado")][["code", "subcategoria", "nome", "estado"]].to_dict("records"),
         "lista_sha256": sha256_arquivo(CONF / "lista_cems_20261017.json")}
    gravar(CONF / "elegibilidade.json", r)
    return {k: r[k] for k in ("inundacoes_no_catalogo", "por_estado", "nao_examinadas_e_nao_reservadas")}


def importar(mascara: Path, terreno: Path, evento: str, componente: str, ano: int | None = None) -> Path:
    """Mascara de evento novo -> pontos MT01 com os 8 atributos REAIS (sem elevacao da amostra).

    Emenda rodada 3 (17/10/2026): a versao anterior gravava f_construido/f_areia = NaN para todos os pontos, o que
    imputava a mediana do padronizador nos dois atributos e nao e a consulta original dos pesos Q. Agora cobertura
    (GHSL na epoca <= ano, SoilGrids), distancias de drenagem/canal e contexto vem de
    `mt03_atributos_novos.completar`, a mesma funcao do MT03/MT04; o ano e obrigatorio.
    """
    import mt03_atributos_novos as an
    import rasterio
    import rasterio.transform
    from pyproj import Transformer
    from rasterio.warp import Resampling, reproject
    if ano is None:
        raise SystemExit("--ano obrigatorio: cobertura depende da epoca do evento")
    ctx_p50 = None
    man = json.loads((Path(terreno) / "run_manifest.json").read_text(encoding="utf-8"))
    with rasterio.open(Path(terreno) / "slope_deg_wbt.tif") as s:
        sl = s.read(1).astype(float); sl[sl == cb.SEM_DADO] = np.nan
        ctx_p50 = float(np.nanmedian(sl))  # declive mediano do raster INTEIRO da unidade, sem rotulo (regra MT01)
        tr, crs, forma = s.transform, s.crs, s.shape
    f, _ = cb.terreno_nativo(Path(terreno), ctx_p50)
    with rasterio.open(mascara) as m:
        rot = np.full(forma, 255, np.uint8)
        reproject(m.read(1), rot, src_transform=m.transform, src_crs=m.crs, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.nearest, dst_nodata=255)
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    for classe in (1, 0):
        ii = np.flatnonzero((rot.ravel() == classe) & np.isfinite(f["f_hand"].ravel()))
        ii = np.sort(rng.choice(ii, min(TETO_TESTE, len(ii)), replace=False)) if len(ii) else ii
        d = pd.DataFrame({k: f[k].ravel()[ii] for k in cb.TERRENO}).assign(classe=classe, pixel=ii)
        linhas.append(d)
    p = pd.concat(linhas, ignore_index=True).assign(evento_id=evento, unidade=componente, terreno_manifesto=man.get("region_label"))
    lin, col = np.divmod(p.pixel.to_numpy(), forma[1])
    xs, ys = rasterio.transform.xy(tr, lin, col)
    p["lon"], p["lat"] = Transformer.from_crs(crs, 4326, always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    p["ponto_id"] = evento.replace(":", "_") + "_" + p.pixel.astype(str)
    p = an.completar(p[["ponto_id", "lon", "lat", "classe", "evento_id", "unidade"]], Path(terreno), ano,
                     evento.replace(":", "_"), CONF / "lotes_cobertura")
    if p[["construido_v2", "areia_v2"]].isna().all().any():
        raise SystemExit("cobertura ausente em todos os pontos: consulta dos pesos Q nao seria a original")
    out = CONF / "importados" / f"{evento.replace(':', '_')}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out, index=False)
    return out


def avaliar(pontos: Path) -> dict:
    import torch
    sys.path.insert(0, str(REPO / "scripts" / "ia"))
    import fechamento_campanha as fc
    from sklearn.metrics import roc_auc_score
    import mt05_transferencia_sem_amostra as m5
    p = pd.read_parquet(pontos)
    x = m5.caracteristicas(p)  # emenda rodada 3: escore_q espera f_*; antes recebia as colunas brutas
    esc = {"A0_regra_hand": -p.hand_m.to_numpy(), "Q_base": escore_q("Q_base", x), "Q_conexao": escore_q("Q_conexao", x)}
    for b in ("conjunto", "parada"):
        e = torch.load(FINAL / b / "final.pt", weights_only=False, map_location="cpu")
        m = fc.construir(b, 7)
        m.load_state_dict(e["modelo"])
        with torch.no_grad():
            T = torch.tensor(x[list(cb.TERRENO)].to_numpy(np.float32)); C = torch.tensor(x[list(cb.COBERTURA)].to_numpy(np.float32))
            esc[b] = m.fisica(T, C).numpy()
    linhas = []
    for (ev, un), g in p.assign(**{k: v for k, v in esc.items()}).groupby(["evento_id", "unidade"]):
        if g.classe.nunique() == 2 and len(g) >= 20:
            linhas.append({"evento_id": ev, "unidade": un, **{k: float(roc_auc_score(g.classe, g[k])) for k in esc}})
    return {"eventos": linhas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["congelar", "elegibilidade", "importar", "avaliar"])
    ap.add_argument("--mascara"); ap.add_argument("--terreno"); ap.add_argument("--evento"); ap.add_argument("--componente")
    ap.add_argument("--pontos"); ap.add_argument("--ano", type=int)
    a = ap.parse_args()
    if a.acao == "importar":
        print(importar(Path(a.mascara), Path(a.terreno), a.evento, a.componente, a.ano))
    elif a.acao == "avaliar":
        print(json.dumps(avaliar(Path(a.pontos)), indent=1))
    else:
        print(json.dumps({"congelar": congelar, "elegibilidade": elegibilidade}[a.acao](), ensure_ascii=False, indent=1, default=str)[:3000])
