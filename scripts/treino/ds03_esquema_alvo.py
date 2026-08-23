"""
DS-03 -- O esquema-alvo da tabela unica, congelado antes de qualquer reducao.

POR QUE ESTE ARQUIVO EXISTE E NAO CONTEM NENHUMA LINHA DE DADO:

o projeto tem hoje seis pipelines com esquemas proprios -- Recife (v12),
Curitiba (SIAC 156), Copernicus EMS, EA/UK, Sen1Floods11 e UFO. Cada um foi
construido para responder a uma pergunta local e nomeou suas colunas em
funcao dela. Reduzir os seis a uma tabela so e facil de fazer errado de um
jeito que nao aparece: basta uma coluna com o mesmo nome significando duas
coisas, e o erro atravessa o modelo inteiro sem disparar excecao.

Ja aconteceu tres vezes neste projeto, e as tres estao documentadas:

  1. `hand_m` derivado por duas cadeias diferentes (ter01)
  2. `hand_m` com limiar de canal diferente em cada regiao, todos rotulados
     "percentil 98" (ext_hand_incomparavel_entre_regioes_v1.md)
  3. `rain_max_24h_chirps` contendo CHIRPS em 181 pontos e ERA5-Land em 97,
     dentro da MESMA regiao (aud_chuva01)

A resposta estrutural nao e revisar melhor: e congelar o contrato primeiro e
popular depois. Um passo por vez, definir antes de preencher. Este modulo e o
contrato. `ds04` reduz cada fonte a ele; `ds05` admite, deduplica e consolida.

REGRA DE OURO DO CONTRATO:
    nenhuma coluna de valor pode existir sem a coluna de procedencia que diz
    de onde aquele valor veio. Onde a fonte nao tem a variavel, o campo e
    NULO DECLARADO -- nunca estimado, nunca imputado, nunca zero.

DELTA v2 (2026-08-19) -- elevacao relativa, permanente:

  `elevation_m` absoluta e o QUARTO caso desta mesma classe de erro (as tres
  do topo deste docstring), achado em `app01-transferencia-curitiba-sem-
  rotulo-local`: Curitiba (~900 m) teve 0% dos pontos dentro do intervalo
  5-95% de um treino externo perto do nivel do mar (diferenca padronizada de
  media = 2,76 desvios). HAND ja e relativo por construcao (altura acima da
  drenagem); elevacao sozinha nao era, e por isso quebra entre regioes de
  altitude de base diferente.

  Correcao: `elevation_baseline_m` (P1 de elevation_m DENTRO da mesma fonte)
  e `elevation_rel_m` (elevation_m - elevation_baseline_m). Aplicada dentro
  de `moldar()` no ds04, para todo redutor, presente ou futuro -- nao e uma
  correcao pontual de Curitiba, e uma regra permanente do contrato: toda
  regiao nova (Petropolis e as que vierem depois) ganha o mesmo tratamento
  sem que ninguem precise lembrar de pedir. `elevation_m` continua no
  contrato, inalterada -- e a leitura absoluta, util dentro de uma fonte;
  `elevation_rel_m` e a leitura que pode ser comparada ou agrupada ENTRE
  fontes. Ver `regra_elevacao` em `contrato()`.

Uso:
    python scripts/suscetibilidade/ds03_esquema_alvo.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "ds-03-esquema"

VERSAO = "v2"


@dataclass(frozen=True)
class Coluna:
    nome: str
    tipo: str
    papel: str
    obrigatoria: bool
    descricao: str
    dominio: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------
# Dominios fechados. Valor fora do dominio e erro de reducao, nao dado novo.
# --------------------------------------------------------------------------

# Hierarquia herdada do ds01, sem alteracao. O que muda e so o nome da coluna:
# `tipo_negativo` -> `nivel_negativo`, porque a Secao II do planejamento fala
# em NIVEL e a hierarquia e ordenada.
NIVEL_NEGATIVO = (
    "observado",               # area declarada analisada e sem inundacao detectada
    "exclusao_qualificada",    # fora de outline E de zona modelada E pareado (N1-N4)
    "ausencia",                # so nao ha registro. NAO constitui negativo
    "nao_aplicavel",           # a linha nao e negativa (classe 1, 2 ou 9)
)

UNIDADE_AGRUPAMENTO = (
    "evento",              # UK: EV_*; o grupo e a cheia registrada
    "aoi",                 # CEMS: EMSR*_AOI*; o grupo e a area mapeada
    "chip",                # Sen1Floods11 / UFO: o grupo e o recorte rotulado
    "unidade_observacao",  # Curitiba: observation_unit_key
    "ponto",               # Recife v12: cada ponto e seu proprio grupo (LOO)
)

CLASSE_RELEVO = ("INGREME", "PLANO_OU_ONDULADO", "NAO_CLASSIFICADO")

# De onde veio a classificacao de relevo. Sem isto, "INGREME" medido numa
# janela de raster e "INGREME" inferido dos pontos amostrados viram a mesma
# palavra, e o criterio (relevo >= 400 m E frac_gt15 >= 0,25) foi calibrado
# na janela, nao na amostra.
CLASSE_RELEVO_BASE = (
    "janela_raster_cems03",  # medido por cems03 --verificar, criterio original
    "pontos_amostrados",     # mesmo criterio aplicado aos pontos: estimador diferente
    "declarada",             # perfil da regiao, com fonte no texto
    "ausente",
)

MECANISMO = (
    "FLUVIAL_ENXURRADA",
    "PLUVIAL_URBANO",
    "MISTO_NAO_SEPARAVEL",
    "NAO_CLASSIFICADO",
)

# Cadeia que produziu elevacao/declividade/HAND/TWI daquela linha.
# `global` e `wbt30` NAO sao intercambiaveis: sao instrumentos diferentes.
#
# Desde 13/08/2026 o projeto tem RESOLUCAO UNICA: toda linha canonica e wbt30.
# A coluna nao foi removida por isso -- ao contrario. Uma procedencia que so
# admite um valor e a forma de a violacao aparecer como violacao no dia em que
# alguem introduzir uma segunda cadeia; uma coluna ausente nao acusa nada.
#
# `nativa_10m` sobrevive apenas nos arquivos de VARIANTE, que existem para
# medir o que a harmonizacao custou e nunca entram na tabela unica.
CADEIA_TERRENO = ("wbt30", "nativa_10m", "global", "ausente")
CADEIA_CANONICA = "wbt30"

# Fonte da chuva. Coluna obrigatoria justamente porque a violacao ja existe no
# dado atual: duas fontes ocupando a mesma coluna dentro da mesma regiao.
FONTE_CHUVA = (
    "chirps_v2_gauge_satellite_blend",
    "chirps_v3_rnl",
    "open_meteo_era5_land",
    "ausente",
)

CLASSE = {
    1: "INUNDACAO",
    2: "MOVIMENTO_DE_MASSA",   # classe propria: nem positivo, nem negativo
    0: "NAO_INUNDADO",
    9: "AGUA_PERMANENTE",      # fora de treino, mantida para auditoria
    -1: "SEM_DADO",            # o chip existe e o rotulo nao; herdado do ds01
}

# So estas entram em ajuste. As outras duas nao sao lixo -- sao classes com
# significado proprio que um modelo binario nao sabe representar: agua
# permanente nao e inundacao, e SEM_DADO nao e negativo.
CLASSES_TREINO = (0, 1, 2)

# As seis variaveis fisicas que o planejamento nomeia. Ordem fixa.
VARIAVEIS_FISICAS = (
    "elevation_m", "slope_deg", "hand_m", "twi_dinf",
    "rain_max_24h", "rain_decay_index",
)

# As quatro que vem da cadeia de terreno e precisam da mesma derivacao para
# serem comparaveis entre regioes.
VARIAVEIS_TERRENO = ("elevation_m", "slope_deg", "hand_m", "twi_dinf")

# As mesmas quatro, na versao que pode ser AGRUPADA OU COMPARADA ENTRE
# fontes/regioes. So elevacao muda (absoluta -> relativa ao nivel de
# referencia da propria fonte); slope/HAND/TWI ja eram comparaveis. Usar esta
# tupla, nao VARIAVEIS_TERRENO, em qualquer ajuste que misture mais de uma
# fonte -- ver regra_elevacao em contrato().
VARIAVEIS_TERRENO_TRANSFERIVEL = ("elevation_rel_m", "slope_deg", "hand_m", "twi_dinf")


ESQUEMA: tuple[Coluna, ...] = (
    # ---- identidade ----
    Coluna("ponto_id", "str", "identidade", True,
           "identificador do ponto, unico em toda a tabela consolidada"),
    Coluna("fonte", "str", "identidade", True,
           "pipeline de origem: recife | curitiba | cems | uk | sen1floods11 | ufo"),
    Coluna("aoi", "str", "identidade", True,
           "area de interesse declarada a que o ponto pertence"),

    # ---- procedencia ----
    Coluna("procedencia", "str", "procedencia", True,
           "como o rotulo foi produzido (herda fonte_label do ds01)"),
    Coluna("resolucao_nativa_m", "float", "procedencia", True,
           "resolucao da observacao que originou o ponto, em metros"),
    Coluna("cadeia_terreno", "enum", "procedencia", True,
           "cadeia que derivou as quatro variaveis de terreno desta linha",
           CADEIA_TERRENO),
    Coluna("fonte_chuva", "enum", "procedencia", True,
           "produto de precipitacao que originou rain_max_24h e rain_decay_index",
           FONTE_CHUVA),

    # ---- rotulo ----
    Coluna("classe", "int", "rotulo", True,
           "1 inundacao | 2 movimento de massa | 0 nao inundado | 9 agua permanente"),
    Coluna("classe_nome", "str", "rotulo", True, "nome da classe por extenso"),
    Coluna("nivel_negativo", "enum", "rotulo", True,
           "hierarquia do negativo; 'ausencia' nao constitui registro de ausencia",
           NIVEL_NEGATIVO),

    # ---- agrupamento e estrato ----
    Coluna("grupo_cv", "str", "agrupamento", True,
           "unidade de validacao; nenhum split pode partir um grupo"),
    Coluna("unidade_agrupamento", "enum", "agrupamento", True,
           "o que o grupo_cv representa", UNIDADE_AGRUPAMENTO),
    Coluna("classe_relevo", "enum", "estrato", True,
           "serra ou planicie, criterio relevo_local>=400m E frac_gt15>=0,25",
           CLASSE_RELEVO),
    Coluna("classe_relevo_base", "enum", "estrato", True,
           "como classe_relevo foi obtida", CLASSE_RELEVO_BASE),
    Coluna("mecanismo", "enum", "estrato", True,
           "mecanismo declarado com evidencia, nunca inferido de desempenho",
           MECANISMO),

    # ---- espaco e tempo ----
    Coluna("lat", "float", "geo", True, "latitude WGS84"),
    Coluna("lon", "float", "geo", True, "longitude WGS84"),
    Coluna("data_evento", "str", "geo", False,
           "data do evento; nulo declarado onde a fonte nao registra"),

    # ---- as seis variaveis fisicas ----
    Coluna("elevation_m", "float", "fisica", False, "elevacao, metros"),
    Coluna("slope_deg", "float", "fisica", False, "declividade, graus"),
    Coluna("hand_m", "float", "fisica", False,
           "altura acima da drenagem mais proxima, metros"),
    Coluna("twi_dinf", "float", "fisica", False,
           "indice topografico de umidade, SCA D-infinito"),
    Coluna("rain_max_24h", "float", "fisica", False,
           "precipitacao maxima em 24 h na janela do evento, mm"),
    Coluna("rain_decay_index", "float", "fisica", False,
           "indice de precipitacao antecedente com decaimento (API)"),

    # ---- elevacao relativa (v2, permanente -- ver DELTA v2 no topo) ----
    Coluna("elevation_baseline_m", "float", "procedencia", False,
           "P1 de elevation_m dentro da mesma fonte; nivel de referencia "
           "local subtraido para obter elevation_rel_m"),
    Coluna("elevation_rel_m", "float", "fisica_derivada", False,
           "elevation_m - elevation_baseline_m; comparavel entre fontes de "
           "altitude de base diferente, ao contrario de elevation_m absoluta"),
)

COLUNAS = tuple(c.nome for c in ESQUEMA)
OBRIGATORIAS = tuple(c.nome for c in ESQUEMA if c.obrigatoria)
DOMINIOS = {c.nome: c.dominio for c in ESQUEMA if c.dominio}


# --------------------------------------------------------------------------
# Criterios de admissao. Declarados aqui, aplicados no ds05.
# Sao os tres que a Secao II do planejamento ja promete, sem acrescimo.
# --------------------------------------------------------------------------
CRITERIOS_ADMISSAO = {
    "A_dentro_da_aoi": (
        "o ponto tem AOI declarada e coordenada finita dentro dela"),
    "B_mesma_cadeia": (
        "as quatro variaveis de terreno vem de uma cadeia declarada e nao "
        "'global' -- global e outro instrumento, nao outra resolucao"),
    "C_grupo_identificavel": (
        "existe grupo_cv nao nulo, para que o split nao parta um grupo"),
}

# Gate adicional do pool fluvial a 30 m. Nao e criterio de admissao geral:
# Recife e pluvial e entra na tabela unica em cadeia nativa de 10 m por
# decisao declarada (ext_resolucao_e_mecanismo_decisao_v1.md secao 3).
GATE_POOL_FLUVIAL_30M = {
    "cadeia_terreno": "wbt30",
    "mecanismo": "FLUVIAL_ENXURRADA",
    "terreno_completo": list(VARIAVEIS_TERRENO),
}


def contrato() -> dict:
    return {
        "versao": VERSAO,
        "n_colunas": len(ESQUEMA),
        "colunas": [
            {"nome": c.nome, "tipo": c.tipo, "papel": c.papel,
             "obrigatoria": c.obrigatoria, "descricao": c.descricao,
             "dominio": list(c.dominio) or None}
            for c in ESQUEMA
        ],
        "classes": {str(k): v for k, v in CLASSE.items()},
        "variaveis_fisicas": list(VARIAVEIS_FISICAS),
        "variaveis_terreno": list(VARIAVEIS_TERRENO),
        "variaveis_terreno_transferivel": list(VARIAVEIS_TERRENO_TRANSFERIVEL),
        "criterios_admissao": CRITERIOS_ADMISSAO,
        "gate_pool_fluvial_30m": GATE_POOL_FLUVIAL_30M,
        "regra_de_ouro": (
            "nenhuma coluna de valor sem a coluna de procedencia "
            "correspondente; ausencia e nulo declarado, nunca estimativa"),
        "cadeia_canonica": CADEIA_CANONICA,
        "regra_resolucao": (
            "resolucao unica em todo o projeto desde 13/08/2026: 30 m, cadeia "
            "WhiteboxTools, limiar de canal em 0,1123 km2. Substitui a regra "
            "anterior de resolucao por mecanismo. Linha canonica fora de "
            "wbt30 e erro de reducao"),
        "regra_twi": (
            "TWI a 10 m e TWI a 30 m nao sao a mesma variavel (pearson 0,293 "
            "em Recife, 0,205 em Curitiba). A regra era: ou todo o TWI vem de "
            "30 m, ou TWI sai do conjunto. Com resolucao unica ela passa a ser "
            "satisfeita por construcao, e nao por vigilancia"),
        "regra_chuva": (
            "duas fontes de precipitacao nao podem ocupar a mesma coluna. "
            "Pool so e licito dentro de uma fonte_chuva"),
        "regra_elevacao": (
            "elevation_m absoluta nao e comparavel entre fontes de altitude "
            "de base diferente (achado app01, 19/08/2026: Curitiba ~900 m, "
            "0% dos pontos dentro do intervalo 5-95% de treino externo perto "
            "do nivel do mar, diferenca padronizada = 2,76). elevation_rel_m "
            "(elevation_m menos o P1 da propria fonte) e obrigatoria em "
            "qualquer ajuste que agrupe ou compare mais de uma fonte -- use "
            "VARIAVEIS_TERRENO_TRANSFERIVEL, nao VARIAVEIS_TERRENO, nesses "
            "casos. elevation_m absoluta continua valida dentro de uma unica "
            "fonte. Regra permanente, aplicada a toda fonte no ds04 por "
            "construcao, nao por lembranca"),
    }


def cabecalho_csv() -> str:
    return ",".join(COLUNAS) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c = contrato()
    (OUT / f"esquema_alvo_{VERSAO}.json").write_text(
        json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / f"esquema_alvo_{VERSAO}.csv").write_text(
        cabecalho_csv(), encoding="utf-8")

    print(f"===DS03_ESQUEMA_ALVO_{VERSAO}===")
    print(f"{len(ESQUEMA)} colunas, {len(OBRIGATORIAS)} obrigatorias, "
          f"{len(DOMINIOS)} com dominio fechado, 0 linhas de dado")
    papel = {}
    for col in ESQUEMA:
        papel.setdefault(col.papel, []).append(col.nome)
    for p, cols in papel.items():
        print(f"  {p:12s} {len(cols):2d}  {', '.join(cols)}")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
