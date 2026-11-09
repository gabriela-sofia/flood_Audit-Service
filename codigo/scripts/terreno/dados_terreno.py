"""B1 -- contrato de dado do terreno, para o encoder auto-supervisionado.

-------------------------------------------------------------------------
O QUE MUDA EM RELACAO A ROTA CONGELADA
-------------------------------------------------------------------------

`scripts/dino/` testou embedding de imagem optica como feature, tres vezes, e
deu nulo nas tres. O diagnostico registrado no LEIA-ME e que o gargalo estava
na granularidade do ponto SEDEC, e nao no embedding.

A leitura que sobra e dupla, e a segunda metade e a que abre esta frente:
nenhuma rede salva um rotulo pontual, E embedding de imagem optica nao e a
representacao que o fenomeno usa. Agua nao se acumula onde a cena parece
molhada; ela se acumula onde a FORMA DO TERRENO a leva.

Entao a esteira e a mesma -- manifesto, preflight, extracao, QA, proveniencia
-- e o objeto e outro: a pilha fisico-hidrologica que o `ter-01` ja derivou.

-------------------------------------------------------------------------
AS CINCO CAMADAS, e o que cada uma responde
-------------------------------------------------------------------------

    dem_filled          altitude com depressoes preenchidas
    slope_deg_wbt       declividade local, graus
    hand_dinf           altura acima da drenagem mais proxima
    dinf_sca            area de contribuicao especifica
    dinf_accum_cells    acumulacao de fluxo, em celulas

HAND e SCA sao os dois descritores que a literatura do projeto ja sustenta
(Renno 2008, Beven e Kirkby 1979). As outras tres nao sao redundantes: uma
encosta ingreme com HAND alto e um lugar; uma varzea plana com HAND alto e
outro, e so `slope` separa os dois.

-------------------------------------------------------------------------
O QUE A FRENTE A ENSINOU, APLICADO AQUI
-------------------------------------------------------------------------

1. NODATA NAO E ZERO. No SAR era NaN; aqui e `-32768`, e zero e uma altitude
   plausivel ao nivel do mar. Toda leitura devolve o dado E a mascara, pela
   mesma razao que `ler_rotulo` devolve o par: uma funcao que devolvesse so o
   array tornaria `array > 0` a leitura obvia.

2. A ESCALA E DECLARADA. Terreno vem a 30 m em UTM local por chip; o SAR da
   frente A vem a ~10 m em EPSG:4326. Os dois NAO estao na mesma grade, e este
   modulo nao finge que estao -- a frente B trabalha na grade do terreno, e
   qualquer cruzamento com a frente A tera de reprojetar explicitamente.

3. OS MESMOS SPLITS. Treino, validacao, teste e o holdout da Bolivia sao os do
   Sen1Floods11, importados de `dados_sen1floods11`. Nao por conveniencia: e o
   que permite, depois, perguntar se o embedding de terreno ajuda exatamente
   nos chips em que a rede de SAR falhou.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from dados_sen1floods11 import nomes_do_split  # noqa: E402
from proveniencia import sha256_arquivo  # noqa: E402

RAIZ = REPO / "local_runs" / "ter-01-cadeia-harmonizada"

SEM_DADO = -32768.0

# ordem fixa: ela vira a ordem dos canais do encoder, e trocar a ordem entre
# treino e uso produziria um embedding que parece funcionar e nao significa nada.
CAMADAS = ("dem_filled", "slope_deg_wbt", "hand_dinf", "dinf_sca",
           "dinf_accum_cells")

# `dinf_sca` e `dinf_accum_cells` sao fortemente assimetricas -- uma celula de
# fundo de vale acumula ordens de grandeza mais que uma de topo. Sem log, a
# normalizacao vira "quase tudo em zero e alguns extremos", e o encoder aprende
# a reconstruir o zero.
CAMADAS_EM_LOG = ("dinf_sca", "dinf_accum_cells")


class ContratoDeTerrenoViolado(Exception):
    """Falha dura. Chip fora do contrato nao entra silenciosamente."""


@dataclass(frozen=True)
class ChipDeTerreno:
    """Pilha de terreno de um chip, com a mascara sempre junto."""

    nome: str
    pilha: np.ndarray            # (5, H, W) float32
    valido: np.ndarray           # (H, W) bool -- valido em TODAS as camadas
    crs: str
    resolucao_m: float

    @property
    def fracao_valida(self) -> float:
        return float(self.valido.mean())


def diretorio(nome: str) -> Path:
    """`Bolivia_103757` -> o diretorio do `ter-01` correspondente.

    Resolve tres formas: chip rotulado (`n1chip_..._LabelHand`), chip sem
    rotulo (`n1chip_...`) e AOI derivada com o nome proprio (`EMSR734_AOI02`,
    `petropolis`). A terceira existe porque a mesma cadeia derivou terreno para
    ativacoes do CEMS e para as regioes servidas, e esse dado ficava invisivel
    para o treino so por causa do prefixo do diretorio.
    """
    com_rotulo = RAIZ / f"n1chip_{nome}_LabelHand"
    if com_rotulo.exists():
        return com_rotulo
    sem_rotulo = RAIZ / f"n1chip_{nome}"
    if sem_rotulo.exists():
        return sem_rotulo
    return RAIZ / nome


def tem_terreno(nome: str) -> bool:
    d = diretorio(nome)
    return d.exists() and all((d / f"{c}.tif").exists() for c in CAMADAS)


def _ler_camada(caminho: Path) -> tuple[np.ndarray, np.ndarray, str, float]:
    if not caminho.exists():
        raise ContratoDeTerrenoViolado(f"ausente: {caminho}")
    with rasterio.open(caminho) as fonte:
        bruto = fonte.read(1).astype(np.float32)
        crs = str(fonte.crs)
        resolucao = abs(float(fonte.transform.a))
    # duas formas de ausencia, e as duas contam: o sentinela do WhiteboxTools
    # e o NaN que sobrevive a alguma operacao intermediaria.
    valido = np.isfinite(bruto) & (bruto != SEM_DADO)
    return bruto, valido, crs, resolucao


def ler_terreno(nome: str) -> ChipDeTerreno:
    """A pilha das cinco camadas, com a mascara de validade conjunta.

    A mascara e a INTERSECCAO: um pixel so vale se vale nas cinco. Aceitar
    pixel valido em quatro camadas obrigaria a preencher a quinta, e o
    preenchimento entraria no encoder como se fosse terreno.
    """
    d = diretorio(nome)
    camadas, mascaras = [], []
    crs = resolucao = None
    forma = None
    for camada in CAMADAS:
        bruto, valido, c, r = _ler_camada(d / f"{camada}.tif")
        if forma is None:
            forma, crs, resolucao = bruto.shape, c, r
        elif bruto.shape != forma:
            raise ContratoDeTerrenoViolado(
                f"{nome}: {camada} tem {bruto.shape}, esperado {forma}")
        elif c != crs:
            raise ContratoDeTerrenoViolado(
                f"{nome}: {camada} em {c}, esperado {crs}")
        camadas.append(bruto)
        mascaras.append(valido)

    pilha = np.stack(camadas).astype(np.float32)
    valido = np.logical_and.reduce(mascaras)

    for i, camada in enumerate(CAMADAS):
        if camada in CAMADAS_EM_LOG:
            # log1p so onde vale; o resto fica como esta e sera mascarado.
            seguro = np.where(valido & (pilha[i] > 0), pilha[i], 0.0)
            pilha[i] = np.log1p(seguro)

    return ChipDeTerreno(nome=nome, pilha=pilha, valido=valido,
                         crs=crs or "", resolucao_m=float(resolucao or 0.0))


SEM_ROTULO = "sem_rotulo"

# A cadeia canonica, como o `ter-01` a declara em cada `run_manifest.json`.
# Terreno derivado por outra cadeia produz uma coluna com o NOME certo e o
# SIGNIFICADO diferente -- o erro numero um do `ds03`, que ja aconteceu tres
# vezes neste projeto e esta em disco agora mesmo: `EMSR789_AOI06` usou
# percentil 98 em vez do limiar de area, e extraiu a rede de drenagem com 826
# celulas em vez de ~125. O HAND daquele AOI descreve outra coisa.
RESOLUCAO_CANONICA_M = 30.0
LIMIAR_CANAL_CANONICO_KM2 = 0.1123
FORMULA_TWI_CANONICA = "ln(SCA / tan(slope))"
MANIFESTO_DA_CADEIA = "run_manifest.json"


def cadeia_declarada(nome: str, pasta: Path | None = None) -> dict:
    """O que o proprio artefato diz sobre como foi derivado, e se isso serve.

    Le, nao infere. Um diretorio sem `run_manifest.json` nao e reprovado por
    suspeita: e reprovado porque nao ha como saber por qual cadeia ele veio, e
    misturar cadeias no treino do encoder produz uma representacao que parece
    funcionar e nao significa nada.

    `pasta` existe para quem deriva fora do `ter-01` -- um diretorio de teste,
    ou uma derivacao que ainda nao foi aceita. Sem ela a checagem procurava
    sempre na raiz oficial e recusava por ausencia o que estava ali do lado.
    """
    caminho = (pasta or diretorio(nome)) / MANIFESTO_DA_CADEIA
    if not caminho.exists():
        return {"nome": nome, "no_contrato": False,
                "por_que": f"sem {MANIFESTO_DA_CADEIA}: cadeia desconhecida"}
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except Exception as erro:
        return {"nome": nome, "no_contrato": False,
                "por_que": f"{MANIFESTO_DA_CADEIA} ilegivel: {erro}"}

    pixel = dados.get("pixel_size") or []
    limiar = dados.get("stream_area_km2")
    desvios = []
    if not (len(pixel) == 2
            and all(abs(float(p) - RESOLUCAO_CANONICA_M) < 1e-6 for p in pixel)):
        desvios.append(f"resolucao {pixel}, esperado {RESOLUCAO_CANONICA_M} m")
    if limiar is None:
        desvios.append(
            "limiar de canal por percentil e nao por area: a rede de drenagem "
            f"extraida difere da de {LIMIAR_CANAL_CANONICO_KM2} km2, e HAND "
            "passa a medir altura acima de outra drenagem")
    elif abs(float(limiar) - LIMIAR_CANAL_CANONICO_KM2) > 1e-9:
        desvios.append(f"limiar de canal {limiar} km2, esperado "
                       f"{LIMIAR_CANAL_CANONICO_KM2}")
    if dados.get("twi_formula") != FORMULA_TWI_CANONICA:
        desvios.append(f"formula do TWI {dados.get('twi_formula')!r}")

    return {
        "nome": nome,
        "no_contrato": not desvios,
        "por_que": "; ".join(desvios) or "cadeia canonica declarada",
        "crs": dados.get("crs"),
        "forma": dados.get("shape"),
        "wbt": dados.get("wbt_version"),
    }


def aois_derivadas(so_no_contrato: bool = True) -> list[str]:
    """Terreno derivado que NAO e chip do Sen1Floods11 -- e nunca foi treinado.

    Sao as ativacoes do CEMS e as regioes servidas (`petropolis`, `recife`,
    `curitiba`, `uk`), derivadas pela mesma cadeia e guardadas no mesmo
    diretorio. `chips_sem_rotulo` nunca as viu porque exige o prefixo
    `n1chip_`, e o prefixo e convencao de nome, nao propriedade do dado.

    Isso importa alem do volume: o Sen1Floods11 nao tem um chip do Brasil, e
    estas AOIs tem. O encoder passa a ver forma de terreno brasileira sem que
    ninguem rotule nada.
    """
    if not RAIZ.exists():
        return []
    saida = []
    for d in sorted(RAIZ.iterdir()):
        if not d.is_dir() or d.name.startswith("n1chip_"):
            continue
        if not all((d / f"{c}.tif").exists() for c in CAMADAS):
            continue
        if so_no_contrato and not cadeia_declarada(d.name)["no_contrato"]:
            continue
        saida.append(d.name)
    return saida


LADO_DE_RECORTE = 128
TETO_DE_RECORTES = 64


def forma_do_chip(nome: str) -> tuple[int, int] | None:
    """Altura e largura, lidas do `run_manifest.json` -- sem abrir raster.

    Construir o indice de amostragem abrindo 785 rasters custaria minutos a
    cada treino. O manifesto ja declara `shape`, e ele e o mesmo artefato que
    declara a cadeia.
    """
    caminho = diretorio(nome) / MANIFESTO_DA_CADEIA
    if not caminho.exists():
        return None
    try:
        forma = json.loads(caminho.read_text(encoding="utf-8")).get("shape")
    except Exception:
        return None
    if not forma or len(forma) != 2:
        return None
    return int(forma[0]), int(forma[1])


def recortes_por_chip(nomes: list[str], lado: int = LADO_DE_RECORTE,
                      teto: int | None = TETO_DE_RECORTES) -> dict:
    """Quantas amostras cada chip contribui por epoca, proporcional a AREA.

    -------------------------------------------------------------------------
    O VIES QUE ISTO CORRIGE
    -------------------------------------------------------------------------

    O treino sorteava UM recorte de 128x128 por diretorio por epoca. Como o
    sorteio e de posicao, ao longo das epocas ele cobre area -- mas cobre a
    mesma quantidade de area em `EMSR778_AOI01`, que tem 4136x6883, e num chip
    do Sen1Floods11, que tem 512x512.

    O resultado e amostragem 108 vezes menos densa por pixel no primeiro.
    Petropolis sai 10,8x sub-amostrada; Recife, 3,5x. As AOIs somam 84% da area
    do conjunto e recebiam 21% das amostras.

    Isso explica por que acrescentar 123 AOIs nao mudou a representacao: elas
    entraram no conjunto carregando a maior parte da area nova, e continuaram
    valendo uma amostra por epoca cada uma.

    -------------------------------------------------------------------------
    POR QUE HA TETO
    -------------------------------------------------------------------------

    Proporcional puro entrega 84% do peso as AOIs e 34% a CINCO rasters. Trocar
    um vies por outro nao e conserto: o encoder aprenderia a forma de cinco
    lugares. Com teto de 64 recortes -- quatro vezes um chip 512x512 -- as AOIs
    ficam com 60% do peso e os cinco maiores com 6,1%.

    Este e o mesmo raciocinio do teto da normalizacao, e nao coincidencia: as
    duas coisas sofrem do mesmo desequilibrio de tamanho.
    """
    saida = {}
    for nome in nomes:
        forma = forma_do_chip(nome)
        if forma is None:
            saida[nome] = 1
            continue
        n = (forma[0] // lado) * (forma[1] // lado)
        saida[nome] = max(1, min(n, teto) if teto else max(n, 1))
    return saida


def chips_sem_rotulo() -> list[str]:
    """Chips do `ter-01` que tem terreno e NAO tem rotulo manual.

    Sao 215, de 14 eventos que nao aparecem em nenhum split do Sen1Floods11 --
    Beira, Houston, Katmandu, Nova Gales do Sul e outros. Eles nao podem entrar
    na avaliacao, porque nao ha alvo contra o que medir.

    O que eles podem, e e o ponto inteiro desta frente: entrar no TREINO
    auto-supervisionado. O encoder aprende forma de terreno sem rotulo nenhum,
    entao dado sem rotulo e dado util para ele -- e e a unica familia de
    aprendizado com essa propriedade.

    O nome termina em `_LabelHand` para os rotulados e em outra coisa para
    estes; a distincao vem do diretorio, nao de um julgamento.
    """
    if not RAIZ.exists():
        return []
    saida = []
    for d in sorted(RAIZ.iterdir()):
        if not d.is_dir() or not d.name.startswith("n1chip_"):
            continue
        if d.name.endswith("_LabelHand"):
            continue
        if all((d / f"{c}.tif").exists() for c in CAMADAS):
            saida.append(d.name[len("n1chip_"):])
    return saida


def diretorio_de(nome: str) -> Path:
    """Diretorio de um chip, com ou sem rotulo. Alias historico de `diretorio`.

    Um unico ponto de resolucao: eram duas copias da mesma regra, e quando
    `diretorio` passou a resolver AOI derivada esta ficou para tras -- o treino
    leria um conjunto e a auditoria outro. Corrigir o padrao, e nao a
    ocorrencia: aqui isso significa nao ter a segunda copia.
    """
    return diretorio(nome)


def nomes_com_terreno(split: str) -> list[str]:
    """Os chips do split que tem terreno derivado, na ordem do split."""
    return [n for n in nomes_do_split(split) if tem_terreno(n)]


def cobertura_por_split() -> dict:
    """Quantos chips de cada split tem terreno. Verificado, nao assumido."""
    saida = {}
    for split in ("treino", "validacao", "teste", "bolivia"):
        nomes = nomes_do_split(split)
        com = [n for n in nomes if tem_terreno(n)]
        saida[split] = {"total": len(nomes), "com_terreno": len(com),
                        "faltando": sorted(set(nomes) - set(com))}
    return saida


def auditar_chip(nome: str) -> dict:
    """Retrato verificavel, com hash de cada camada."""
    chip = ler_terreno(nome)
    d = diretorio(nome)
    por_camada = {}
    for i, camada in enumerate(CAMADAS):
        vals = chip.pilha[i][chip.valido]
        por_camada[camada] = {
            "min": round(float(vals.min()), 4) if vals.size else None,
            "max": round(float(vals.max()), 4) if vals.size else None,
            "media": round(float(vals.mean()), 4) if vals.size else None,
            "em_log": camada in CAMADAS_EM_LOG,
            "hash": sha256_arquivo(d / f"{camada}.tif"),
        }
    return {
        "nome": nome,
        "forma": list(chip.pilha.shape),
        "crs": chip.crs,
        "resolucao_m": chip.resolucao_m,
        "fracao_valida": round(chip.fracao_valida, 6),
        "camadas": por_camada,
        "nao_e": ("pilha de terreno nao e observacao de agua; ela descreve a "
                  "forma do lugar, nao o que aconteceu nele"),
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default=None)
    args = parser.parse_args()

    if args.chip:
        print(json.dumps(auditar_chip(args.chip), indent=2, ensure_ascii=False))
        return 0

    cobertura = cobertura_por_split()
    resumo = {s: {"total": v["total"], "com_terreno": v["com_terreno"],
                  "faltando": len(v["faltando"])}
              for s, v in cobertura.items()}
    exemplo = nomes_com_terreno("teste")[:1]
    print(json.dumps({
        "cobertura": resumo,
        "camadas": list(CAMADAS),
        "camadas_em_log": list(CAMADAS_EM_LOG),
        "exemplo": auditar_chip(exemplo[0]) if exemplo else None,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
