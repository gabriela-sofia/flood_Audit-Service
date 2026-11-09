"""O substrato onde as frentes deixam de estar separadas por codigo.

-------------------------------------------------------------------------
O DIAGNOSTICO QUE ORIGINOU ISTO
-------------------------------------------------------------------------

O pre-treino mascarado le `dados_sen1floods11` e SO ele: 252 chips, 2 canais.
Enquanto isso, `ter-01-cadeia-harmonizada` guarda 786 diretorios derivados pelo
MESMO contrato -- 30 m, limiar de canal 0,1123 km2, `ln(SCA/tan(slope))` --, e
446 deles sao exatamente os chips do Sen1Floods11, com join de 100%.

    786 diretorios, 7 camadas de terreno em TODOS os 786

As frentes nao estavam separadas por dado. Estavam separadas por codigo.

-------------------------------------------------------------------------
A ESCALA E 30 m, E ESSA E A DECISAO QUE CUSTA
-------------------------------------------------------------------------

O S1 e o terreno do mesmo chip NAO estao na mesma grade:

    S1        EPSG:4326, 512x512, ~10 m, cobre 5,12 km
    terreno   UTM,       312x306,  30 m, cobre 9,36 km

Empilhar exige escolher uma. Subir o terreno para 10 m criaria detalhe que a
cadeia nao mediu -- e o limiar de canal de 0,1123 km2 vira um numero de CELULAS
que so vale a 30 m. `derivar_aoi_nova` ja avisa disso, e `EMSR789_AOI06` esta
em disco como o erro cometido.

Entao o substrato e 30 m, e o S1 desce por MEDIA. Tres consequencias, todas
declaradas:

1. PERDE-SE RESOLUCAO DO RADAR. Agua fina de rua a 10 m vira media de celula de
   30 m. Para reconstrucao mascarada isso e aceitavel; para segmentacao de agua
   e uma perda real.

2. E ARENA NOVA PARA A CATRACA. Os pesos de `unet_v1` e os do pote crescente
   foram medidos a 10 m em 512x512. Nada aqui e comparavel com eles, e a
   catraca tem de recusar arena montada sobre a outra.

3. A AREA DE SOLO POR RECORTE FICA UNIFORME. Um recorte de 128x128 cobre 3,84
   km em chip, em AOI e em regiao servida -- e Reed et al. 2023 (Scale-MAE) e
   explicito em que a area de Terra coberta, e nao o pixel, e o que define a
   escala. Hoje o pote nao tem essa propriedade.

-------------------------------------------------------------------------
OS CANAIS, E O QUE CADA LUGAR TEM
-------------------------------------------------------------------------

    VV, VH                    radar         446 chips + AOI com s1_grd proprio
    dem_filled                terreno       os 786
    hand_dinf                 terreno       os 786
    slope_deg_wbt             terreno       os 786
    twi_dinf                  terreno       os 786
    dinf_sca                  terreno       os 786
    agua_permanente_jrc       observacao    1 (so o EMSR720, por enquanto)

O `EMSR720` e o unico lugar com os OITO canais: `adquirir_s1_para_aoi` trouxe o
S1 de 08/05/2024 -- a data de que o analista do EMS desenhou -- ja na grade do
terreno, e `agua_permanente_jrc` trouxe a mascara. Ele e o primeiro chip
brasileiro completo do projeto.

`dinf_accum_cells` e `streams_dinf` ficam de FORA por decisao, e nao por falta:
o primeiro e `dinf_sca` multiplicado por constante da celula, e o segundo e
`dinf_accum_cells` limiarizado. Tres canais para uma informacao gastariam
capacidade para a rede aprender que sao o mesmo.

`dem_filled` e `hand_dinf` FICAM os dois, mesmo correlacionados. A colinearidade
medida em 0,87-0,92 nas centrais e um problema de IDENTIFICACAO de coeficiente,
que aparece na regressao e na sonda linear -- nao na reconstrucao. Mas quem
usar o embedding numa sonda linear herda a mesma nao-identificacao, e isso vai
dito aqui em vez de descoberto de novo.

-------------------------------------------------------------------------
MODALIDADE AUSENTE E O MOTIVO DE `valido` SER POR CANAL
-------------------------------------------------------------------------

Recife tem terreno e nao tem S1. Um chip tem os dois. Com a validade colapsada
por pixel -- que era o que `preparar` fazia com `.all(axis=0)` -- a modalidade
ausente invalidaria o lugar INTEIRO, e as 340 AOIs e regioes sairiam do pote
sem que nada denunciasse.

`perda_mascarada` passou a aceitar `(B, C, H, W)`, de forma aditiva e provada
bit a bit igual no caminho antigo. Aqui cada canal carrega a propria validade.

-------------------------------------------------------------------------
O POTE E 592, E NAO 786
-------------------------------------------------------------------------

    786  diretorios derivados
    -89  validacao   ┐ o terreno DESTES chips existe, e deixa-lo entrar
    -90  teste       │ vaza o conjunto onde a medida final acontece
    -15  bolivia     ┘
    ---
    592  = 252 chips de treino + 336 AOIs e chips sem rotulo + 4 regioes

O `EMSR720` esta entre as AOIs: e o primeiro lugar brasileiro do pote, e o
unico com mascara de agua permanente por enquanto.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao treina, nao promove peso e nao decide arena. Ele responde "que canais este
lugar tem, e em que grade" -- e recusa entregar o que vazaria.

Uso:
    python scripts/sar/substrato_multimodal.py            # inventario
    python scripts/sar/substrato_multimodal.py --amostra n1chip_Ghana_103272_LabelHand
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from dados_sen1floods11 import caminho, nomes_do_split  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

RAIZ = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
SAIDA = REPO / "local_runs" / "substrato-multimodal"
RESOLUCAO_M = 30.0

RADAR = ["VV", "VH"]
TERRENO = {
    "dem_filled": "dem_filled.tif",
    "hand_dinf": "hand_dinf.tif",
    "slope_deg_wbt": "slope_deg_wbt.tif",
    "twi_dinf": "twi_dinf.tif",
    "dinf_sca": "dinf_sca.tif",
}
OBSERVACAO = {"agua_permanente_jrc": "agua_permanente_jrc.tif"}
# Radar adquirido para a AOI e ja alinhado a grade do terreno. Tem
# precedencia sobre o S1 do Sen1Floods11, que precisa ser reamostrado.
S1_PROPRIO = "s1_grd.tif"
CANAIS = RADAR + list(TERRENO) + list(OBSERVACAO)

# Fora do pote por VAZAMENTO, e nao por qualidade: e nestes que a medida final
# do ajuste fino acontece.
SPLITS_PROIBIDOS = ("validacao", "teste", "bolivia")


def papel(nome: str, splits: dict) -> str:
    if nome.startswith("n1chip_") and nome.endswith("_LabelHand"):
        base = nome[len("n1chip_"):-len("_LabelHand")]
        for s, conjunto in splits.items():
            if base in conjunto:
                return f"chip_{s}"
        return "chip_sem_rotulo"
    if "harmonizado" in nome:
        return "regiao_servida"
    return "aoi_cems"


def _splits() -> dict:
    return {s: set(nomes_do_split(s))
            for s in ("treino", "validacao", "teste", "bolivia")}


def nome_do_chip(diretorio: str) -> str | None:
    if diretorio.startswith("n1chip_") and diretorio.endswith("_LabelHand"):
        return diretorio[len("n1chip_"):-len("_LabelHand")]
    return None


def origem_do_radar(diretorio: str) -> str | None:
    """De onde vem VV e VH neste lugar -- e se houve reamostragem.

    Duas corridas com o mesmo numero de canais podem ter passado por
    tratamentos diferentes, e sem isso declarado elas ficam indistinguiveis no
    arquivo.
    """
    if (RAIZ / diretorio / S1_PROPRIO).exists():
        return "s1_grd.tif proprio, ja na grade do terreno, sem reamostragem"
    chip = nome_do_chip(diretorio)
    if chip and caminho(chip, "s1").exists():
        return "Sen1Floods11, reamostrado de ~10 m para 30 m por media"
    return None


def inventario() -> dict:
    splits = _splits()
    dirs = sorted(p for p in RAIZ.iterdir() if p.is_dir())
    por_papel, por_canal, lugares = {}, {c: 0 for c in CANAIS}, []
    for d in dirs:
        p = papel(d.name, splits)
        por_papel[p] = por_papel.get(p, 0) + 1
        chip = nome_do_chip(d.name)
        tem = {c: (d / TERRENO[c]).exists() for c in TERRENO}
        tem.update({c: (d / OBSERVACAO[c]).exists() for c in OBSERVACAO})
        tem_radar = ((d / S1_PROPRIO).exists()
                     or (bool(chip) and caminho(chip, "s1").exists()))
        for c in RADAR:
            por_canal[c] += int(tem_radar)
        for c, v in tem.items():
            por_canal[c] += int(v)
        lugares.append({"diretorio": d.name, "papel": p,
                        "no_pote": not p.endswith(SPLITS_PROIBIDOS),
                        "radar": tem_radar,
                        "canais": sum(tem.values()) + 2 * int(tem_radar)})
    no_pote = [x for x in lugares if x["no_pote"]]
    return {
        "tipo": "inventario_do_substrato_multimodal",
        "quando_utc": agora_utc(),
        "resolucao_m": RESOLUCAO_M,
        "por_que_30m": ("e a escala da cadeia canonica; o limiar de canal de "
                        "0,1123 km2 vira numero de CELULAS e so vale a 30 m, e "
                        "subir o terreno para os 10 m do S1 criaria detalhe que "
                        "nao foi medido"),
        "canais": CANAIS,
        "canais_fora_por_decisao": {
            "dinf_accum_cells": "e dinf_sca vezes constante da celula",
            "streams_dinf": "e dinf_accum_cells limiarizado",
        },
        "diretorios": len(dirs),
        "por_papel": por_papel,
        "lugares_no_pote": len(no_pote),
        "presenca_por_canal": por_canal,
        "lugares_com_radar_no_pote": sum(1 for x in no_pote if x["radar"]),
        "lugares_so_terreno_no_pote": sum(1 for x in no_pote
                                          if not x["radar"]),
        "fora_do_pote_por_vazamento": {
            s: por_papel.get(f"chip_{s}", 0) for s in SPLITS_PROIBIDOS},
        "e_arena_nova": ("os pesos ja medidos vieram de 10 m em 512x512; nada "
                         "deste substrato e comparavel com eles, e a catraca "
                         "tem de recusar arena montada sobre a outra"),
        "nao_e": ("nao treina, nao promove peso e nao decide arena: responde "
                  "que canais cada lugar tem e em que grade"),
    }


def carregar(diretorio: str, estat: dict | None = None) -> tuple:
    """Empilha os canais do lugar na grade de 30 m, com validade POR CANAL.

    Devolve `(x, valido)` com forma `(C, H, W)` cada. Canal ausente entra como
    zero e validade FALSA -- nunca como zero silencioso, que a rede aprenderia
    a reconstruir.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    pasta = RAIZ / diretorio
    with rasterio.open(pasta / TERRENO["dem_filled"]) as ref:
        perfil = {"crs": ref.crs, "transform": ref.transform,
                  "height": ref.height, "width": ref.width}
    h, w = perfil["height"], perfil["width"]

    x = np.zeros((len(CANAIS), h, w), dtype=np.float32)
    valido = np.zeros((len(CANAIS), h, w), dtype=bool)

    # DOIS CAMINHOS PARA O RADAR, e a ordem importa.
    #
    # `s1_grd.tif` na propria pasta e o caminho preferido: ele foi adquirido
    # por `adquirir_s1_para_aoi` JA na grade do terreno, e entra sem
    # reamostragem nenhuma. E o unico caminho que existe para AOI e regiao
    # servida, que nao sao chips do Sen1Floods11.
    #
    # O S1 do Sen1Floods11 e o caminho de compatibilidade: ele vem em 4326 a
    # ~10 m e DESCE para 30 m por MEDIA. A media preserva o nivel da celula; o
    # caminho inverso -- subir o terreno para 10 m -- inventaria detalhe que a
    # cadeia nao mediu.
    proprio = pasta / S1_PROPRIO
    chip = nome_do_chip(diretorio)
    if proprio.exists():
        with rasterio.open(proprio) as src:
            for i in range(2):
                v = src.read(i + 1).astype(np.float32)
                bom = np.isfinite(v)
                if src.nodata is not None:
                    bom &= v != src.nodata
                x[i][bom] = v[bom]
                valido[i] = bom
    elif chip and caminho(chip, "s1").exists():
        with rasterio.open(caminho(chip, "s1")) as src:
            for i in range(2):
                destino = np.full((h, w), np.nan, dtype=np.float32)
                reproject(source=rasterio.band(src, i + 1), destination=destino,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=perfil["transform"],
                          dst_crs=perfil["crs"], dst_nodata=np.nan,
                          resampling=Resampling.average)
                bom = np.isfinite(destino)
                x[i][bom] = destino[bom]
                valido[i] = bom

    for j, (nome, arquivo) in enumerate(
            list(TERRENO.items()) + list(OBSERVACAO.items()), start=2):
        alvo = pasta / arquivo
        if not alvo.exists():
            continue
        with rasterio.open(alvo) as src:
            v = src.read(1).astype(np.float32)
            nd = src.nodata
        bom = np.isfinite(v)
        if nd is not None:
            bom &= v != nd
        # HAND negativo e artefato do preenchimento, nao terreno abaixo do rio.
        if nome == "hand_dinf":
            bom &= v >= 0
        x[j][bom] = v[bom]
        valido[j] = bom

    if estat:
        media = np.array(estat["media"], dtype=np.float32).reshape(-1, 1, 1)
        desvio = np.array(estat["desvio"], dtype=np.float32).reshape(-1, 1, 1)
        x = np.where(valido, (x - media) / desvio, 0.0).astype(np.float32)
    return x, valido


def estatisticas(lugares: list[str], limite: int | None = None) -> dict:
    """Media e desvio POR CANAL, so onde ha leitura, so no pote.

    Canal ausente nao entra na conta -- incluir zero de ausencia deslocaria a
    media na direcao de quantos lugares nao tem aquela modalidade, que e
    propriedade do inventario e nao do sinal.
    """
    # AMOSTRA POR PASSO REGULAR, e nao pelos primeiros N.
    #
    # `pote()` devolve em ordem alfabetica, e `EMSR*` vem antes de `n1chip_*`:
    # os 120 primeiros nao tinham UM pixel de radar, e a media de VV e VH saiu
    # zero com contagem zero. E o mesmo defeito que `dividir_o_pote` ja
    # documenta para os chips retidos -- o passo regular atravessa os grupos.
    if limite and limite < len(lugares):
        passo = len(lugares) / limite
        escolhidos = [lugares[int(i * passo)] for i in range(limite)]
    else:
        escolhidos = lugares

    somas = np.zeros(len(CANAIS), dtype=np.float64)
    somas_q = np.zeros(len(CANAIS), dtype=np.float64)
    contagem = np.zeros(len(CANAIS), dtype=np.int64)
    for nome in escolhidos:
        x, valido = carregar(nome)
        for c in range(len(CANAIS)):
            v = x[c][valido[c]].astype(np.float64)
            if v.size:
                somas[c] += v.sum()
                somas_q[c] += (v ** 2).sum()
                contagem[c] += v.size
    seguro = np.maximum(contagem, 1)
    media = somas / seguro
    var = np.maximum(somas_q / seguro - media ** 2, 1e-12)
    faltando = [c for c, n in zip(CANAIS, contagem) if n == 0]
    return {"canais": CANAIS,
            "media": [round(float(v), 6) for v in media],
            "desvio": [round(float(v), 6) for v in np.sqrt(var)],
            "celulas_por_canal": [int(v) for v in contagem],
            "lugares": len(escolhidos),
            "amostragem": ("passo regular sobre o pote ordenado" if
                           limite and limite < len(lugares) else "pote inteiro"),
            # Canal sem UMA celula na amostra nao tem normalizacao possivel, e
            # deixar isso implicito devolveria media 0 e desvio 0 -- que dividiria
            # o canal por zero mais adiante sem denunciar nada.
            "canais_sem_leitura_na_amostra": faltando}


def pote() -> list[str]:
    """Os lugares que podem entrar no pre-treino, sem os splits de medida."""
    splits = _splits()
    return sorted(d.name for d in RAIZ.iterdir()
                  if d.is_dir()
                  and not papel(d.name, splits).endswith(SPLITS_PROIBIDOS))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--amostra", help="carrega um lugar e descreve os canais")
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    p.add_argument("--gravar", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--nao-gravar", dest="gravar", action="store_false",
                   help="nao escreve artefato; so imprime o resumo")
    a = p.parse_args()

    if a.amostra:
        x, valido = carregar(a.amostra)
        print(f"{a.amostra}: forma {x.shape}")
        print(f"  radar: {origem_do_radar(a.amostra) or 'AUSENTE'}")
        for i, c in enumerate(CANAIS):
            frac = float(valido[i].mean())
            marca = "AUSENTE" if frac == 0 else f"{frac:6.1%} valido"
            print(f"  {c:<22}{marca}")
        return 0

    inv = inventario()
    inv["hash_resumo"] = sha256_json(
        {k: v for k, v in inv.items() if k != "quando_utc"})
    print(json.dumps({k: v for k, v in inv.items()}, indent=2,
                     ensure_ascii=False))
    if a.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        (SAIDA / "inventario.json").write_text(
            json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nGRAVADO={SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
