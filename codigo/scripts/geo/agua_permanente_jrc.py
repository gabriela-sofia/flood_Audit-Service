"""Agua permanente para qualquer AOI, com o limiar calibrado no proprio projeto.

-------------------------------------------------------------------------
O BURACO QUE ISTO FECHA
-------------------------------------------------------------------------

O projeto ja sabe separar agua permanente de inundacao -- mas so onde o
Sen1Floods11 entrega a mascara pronta:

    dados_sen1floods11.ler_jrc   `JRCWaterHand`, binario, um raster por chip
    baseline_otsu                `valido & (jrc == 0)` para as metricas
    ds01.MAPA_CLASSE             AGUA_PERMANENTE -> classe 9
    avaliar_rede                 `micro_sem_agua_permanente`

Fora do Sen1Floods11 a mascara nao existe. O EMSR720 e o caso: o produto do EMS
traz so `Riverine flood` e `Landslide` em `obj_desc`, entao o leito do rio caiu
no NEGATIVO e o coeficiente do `twi_dinf` INVERTEU -- de +0,4408 para -0,1727.
Um leito de rio tem TWI altissimo e nao foi delineado como inundacao.

O conhecimento existia; faltava a camada. Este modulo traz a camada.

-------------------------------------------------------------------------
O LIMIAR NAO FOI ESCOLHIDO: FOI CALIBRADO CONTRA O QUE O PROJETO JA ACEITA
-------------------------------------------------------------------------

O `JRCWaterHand` do Sen1Floods11 e binario e ja vem limiarizado -- o dataset
nao publica de que corte do JRC ele saiu. Escolher um numero "razoavel" aqui
seria inventar a definicao de agua permanente do projeto.

Entao o corte foi MEDIDO: para um chip de cada pais com agua permanente real,
a banda `occurrence` do JRC Global Surface Water foi lida na janela do chip e
comparada com o `JRCWaterHand`, por IoU, de 5 em 5.

    limiar    IoU medio       limiar    IoU medio
        30       0,6774           55       0,8545
        40       0,7741           60       0,8165
        45       0,8296           70       0,6653
        50       0,8616  <-       80       0,4271

Pico unico e bem comportado em `occurrence >= 50`, com IoU 0,8616 sobre 9
paises (Ghana, India, Mekong, Nigeria, Paquistao, Paraguai, Espanha, Sri Lanka
e EUA). E esse o corte que reproduz a mascara que o resto do projeto usa, e por
isso e o que viaja daqui para regiao nenhuma tem chip.

`calibrar()` refaz essa medida a qualquer momento -- o numero acima nao esta
gravado como constante de fe.

-------------------------------------------------------------------------
A REGRA DE USO VEM DO `baseline_otsu`, E NAO E NOVA
-------------------------------------------------------------------------

    c_sem_perm = contagens(pred, verdade, valido & (jrc == 0))

Agua permanente sai das DUAS classes -- exatamente como o deslizamento sai no
pacote areal. Ela nao e positivo, porque a agua ja estava la antes do evento; e
nao e negativo util, porque e agua. Tratar leito de rio como "nao inundado"
ensina o inverso da fisica.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao rotula evento, nao promove ponto e nao decide classe: entrega uma mascara
binaria alinhada a um raster de referencia, e quem consome declara o que faz
com ela.

Uso:
    python scripts/geo/agua_permanente_jrc.py --calibrar
    python scripts/geo/agua_permanente_jrc.py \\
        --referencia local_runs/ter-01-cadeia-harmonizada/EMSR720/dem_filled.tif \\
        --saida local_runs/ter-01-cadeia-harmonizada/EMSR720/agua_permanente_jrc.tif
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

BASE = ("https://storage.googleapis.com/global-surface-water"
        "/downloads2021/occurrence")
LIMIAR_CALIBRADO = 50
SAIDA_CALIBRACAO = REPO / "local_runs" / "agua-permanente-jrc"
FONTE = ("JRC Global Surface Water v1.4 (Pekel et al. 2016), banda "
         "`occurrence`, epoca 1984-2021")


def tile_de(lon: float, lat: float) -> str:
    """O tile de 10x10 graus do JRC, nomeado pelo canto NOROESTE."""
    oeste = math.floor(lon / 10) * 10
    norte = math.ceil(lat / 10) * 10
    ns = f"{abs(norte)}{'S' if norte < 0 else 'N'}"
    ew = f"{abs(oeste)}{'W' if oeste < 0 else 'E'}"
    return f"occurrence_{ew}_{ns}v1_4_2021.tif"


def tiles_da_janela(oeste: float, sul: float, leste: float,
                    norte: float) -> list[str]:
    """Todos os tiles de 10x10 graus que a janela em lat/lon toca."""
    colunas = range(math.floor(oeste / 10), math.floor(leste / 10) + 1)
    linhas = range(math.floor(sul / 10), math.ceil(norte / 10))
    return sorted({tile_de(c * 10 + 5, l * 10 + 5)
                   for c in colunas for l in linhas})


def url_de(lon: float, lat: float) -> str:
    return f"/vsicurl/{BASE}/{tile_de(lon, lat)}"


def calibrar(limiares=range(5, 100, 5)) -> dict:
    """O corte de `occurrence` que reproduz o `JRCWaterHand` do Sen1Floods11.

    Um chip por pais, entre os que tem agua permanente de verdade -- chip sem
    agua permanente nao distingue limiar nenhum e so infla a media.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    from dados_sen1floods11 import caminho, ler_jrc, nomes_do_split

    escolhidos = {}
    for nome in nomes_do_split("treino"):
        pais = nome.split("_")[0]
        if pais in escolhidos:
            continue
        if int((ler_jrc(nome) == 1).sum()) > 2000:
            escolhidos[pais] = nome

    por_limiar, usados, falhas = {}, [], []
    for pais, nome in escolhidos.items():
        j = ler_jrc(nome)
        with rasterio.open(caminho(nome, "jrc")) as src:
            b = src.bounds
        try:
            with rasterio.open(url_de((b.left + b.right) / 2,
                                      (b.top + b.bottom) / 2)) as src:
                janela = from_bounds(b.left, b.bottom, b.right, b.top,
                                     transform=src.transform)
                occ = src.read(1, window=janela, out_shape=j.shape,
                               resampling=Resampling.nearest)
        except Exception as e:                                   # noqa: BLE001
            falhas.append({"pais": pais, "erro": type(e).__name__})
            continue
        usados.append({"pais": pais, "chip": nome,
                       "pixels_permanentes": int((j == 1).sum())})
        verdade = j == 1
        for t in limiares:
            pred = (occ >= t) & (occ <= 100)
            uniao = int((pred | verdade).sum())
            por_limiar.setdefault(int(t), []).append(
                float((pred & verdade).sum() / uniao) if uniao else float("nan"))

    curva = {t: round(float(np.nanmean(v)), 4) for t, v in por_limiar.items()}
    melhor = max(curva, key=lambda t: curva[t]) if curva else None
    return {
        "tipo": "calibracao_de_agua_permanente_jrc",
        "quando_utc": agora_utc(),
        "pergunta": ("que corte da banda `occurrence` do JRC reproduz o "
                     "`JRCWaterHand` binario que o Sen1Floods11 entrega?"),
        "por_que_calibrar": ("o dataset nao publica de que corte a mascara "
                             "dele saiu; escolher um numero razoavel seria "
                             "inventar a definicao de agua permanente do "
                             "projeto"),
        "fonte": FONTE,
        "chips_usados": usados,
        "falhas": falhas,
        "curva_iou_por_limiar": curva,
        "limiar_escolhido": melhor,
        "iou_no_limiar": curva.get(melhor) if melhor else None,
        "nao_e": ("nao e validacao do JRC nem do Sen1Floods11: e a traducao de "
                  "um para o outro, para que a mascara viaje para regiao sem "
                  "chip"),
    }


def mascara(referencia: Path, saida: Path,
            limiar: int = LIMIAR_CALIBRADO) -> dict:
    """Mascara binaria de agua permanente alinhada ao raster de referencia."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds

    with rasterio.open(referencia) as ref:
        perfil = ref.profile.copy()
        alvo_crs, alvo_transform = ref.crs, ref.transform
        altura, largura = ref.height, ref.width
        oeste, sul, leste, norte = transform_bounds(
            ref.crs, "EPSG:4326", *ref.bounds)

    # TODOS os tiles de 10 graus que a janela toca, e nao so o do centro.
    # Com o do centro, 10 das 147 AOIs nao-chip (medido em 18/09/2026) ficavam
    # com a parte do outro lado da linha de 10 graus zerada -- leito de rio
    # tratado como terra, sem aviso. Cada tile escreve so onde cobre.
    occ = np.full((altura, largura), 255, dtype="uint8")
    tiles = tiles_da_janela(oeste, sul, leste, norte)
    for nome_tile in tiles:
        with rasterio.open(f"/vsicurl/{BASE}/{nome_tile}") as src:
            reproject(source=rasterio.band(src, 1), destination=occ,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=alvo_transform, dst_crs=alvo_crs,
                      src_nodata=255, dst_nodata=255, init_dest_nodata=False,
                      resampling=Resampling.nearest)
    tile = tiles[0] if len(tiles) == 1 else tiles

    # `occ <= 100` exclui o codigo de ausencia de dado do produto; sem isso
    # celula sem observacao viraria agua permanente por ser um numero grande.
    perm = ((occ >= limiar) & (occ <= 100)).astype("uint8")
    perfil.update(dtype="uint8", count=1, nodata=255, compress="deflate")
    saida.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(saida, "w", **perfil) as fora:
        fora.write(perm, 1)

    return {
        "tipo": "mascara_de_agua_permanente_jrc",
        "quando_utc": agora_utc(),
        "referencia": str(referencia),
        "arquivo": str(saida),
        "fonte": FONTE,
        "tile": tile,
        "limiar_occurrence": limiar,
        "limiar_veio_de": ("calibracao contra o `JRCWaterHand` do "
                           "Sen1Floods11; ver "
                           "local_runs/agua-permanente-jrc/calibracao.json"),
        "crs": str(alvo_crs), "shape": [altura, largura],
        "celulas_permanentes": int(perm.sum()),
        "fracao_permanente": round(float(perm.mean()), 6),
        "regra_de_uso": ("agua permanente sai das DUAS classes, como em "
                         "`baseline_otsu`: `valido & (jrc == 0)`. Nao e "
                         "positivo porque a agua ja estava la, e nao e "
                         "negativo util porque e agua"),
        "nao_e": ("nao rotula evento e nao decide classe: entrega a mascara, e "
                  "quem consome declara o que faz com ela"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calibrar", action="store_true")
    p.add_argument("--referencia", type=Path)
    p.add_argument("--saida", type=Path)
    p.add_argument("--limiar", type=int, default=LIMIAR_CALIBRADO)
    a = p.parse_args()

    if a.calibrar:
        r = calibrar()
        r["hash_resumo"] = sha256_json(
            {k: v for k, v in r.items() if k != "quando_utc"})
        SAIDA_CALIBRACAO.mkdir(parents=True, exist_ok=True)
        (SAIDA_CALIBRACAO / "calibracao.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    if not (a.referencia and a.saida):
        raise SystemExit("informe --calibrar, ou --referencia e --saida")
    r = mascara(a.referencia, a.saida, a.limiar)
    r["hash_resumo"] = sha256_json(
        {k: v for k, v in r.items() if k != "quando_utc"})
    (a.saida.parent / "agua_permanente_manifesto.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
