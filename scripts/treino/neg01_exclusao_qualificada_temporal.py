"""
NEG-01 -- Tira o negativo de Curitiba da lacuna, sem promove-lo a observacao.

O PROBLEMA:

os 442 negativos de Curitiba estavam marcados `ausencia`, e ausencia de
registro nao e negativo -- e lacuna de dado. "Nao ha registro de enchente aqui"
fala do arquivo, nao do lugar. Com a regra aplicada, Curitiba e Recife ficaram
com ZERO negativos utilizaveis, e o MVP passou a se apoiar num modelo de
positivo contra nao-rotulado.

Mas `ausencia` estava classificando esses pontos de forma pessimista demais, e
isso tambem e erro. O que existe no dado nao e silencio: **e um chamado ao 156,
feito daquele endereco, naquele dia, sobre outro assunto** -- coleta, transito,
arvore, iluminacao. Isso nao e a mesma coisa que nao ter registro nenhum.

O QUE UM CHAMADO SOBRE OUTRO ASSUNTO PROVA, e o que nao prova:

  prova    o canal estava funcionando naquele dia
  prova    aquele endereco tem morador que usa o canal
  prova    a categoria de alagamento existia e estava sendo usada na cidade
  NAO prova que alguem vistoriou o ponto procurando inundacao

Por isso o destino correto NAO e `observado`. E `exclusao_qualificada` -- o
mesmo nivel do piloto ingles, que tambem e inferencia a partir de criterio, e
nao observacao direta. O `ds01` ja ordena os tres niveis; este script move
Curitiba um degrau, com criterio, e nao dois.

OS CRITERIOS, espelhando o N1-N4 do piloto ingles e declarados antes de rodar:

  N1  canal ativo          volume total de chamados na cidade naquele dia >= p25
                           do ano. Dia de sistema fora do ar nao serve.
  N2  categoria em uso     houve pelo menos um chamado HIDROLOGICO na cidade
                           naquele dia. Se ninguem reportou alagamento em lugar
                           nenhum, o silencio pode ser da categoria e nao do
                           lugar.
  N3  endereco reporta     o proprio ponto e um chamado daquele endereco naquele
                           dia, sobre assunto nao-hidrologico.
  N4  afastamento          distancia minima de qualquer positivo da MESMA data.

Ponto que falha em qualquer um continua `ausencia`. O script conta quantos
passam e quantos caem em cada criterio -- reprovar e resultado, nao erro.

POR QUE O N2 E O CRITERIO QUE MAIS IMPORTA:

sem ele, um dia em que o 156 recebeu mil chamados e nenhum de alagamento
contaria como evidencia contra inundacao. Mas esse dia pode ser justamente um
dia sem chuva -- e ai a informacao e trivial -- ou um dia em que a categoria
falhou. Exigir que a cidade tenha registrado alagamento em ALGUM lugar naquele
dia garante que o canal estava recebendo e classificando esse tipo de
ocorrencia quando o ponto ficou em silencio.

NAO faz: nao promove nada a `observado`, nao altera o ds04 (grava um registro
que o ds04 le), e nao mexe em Recife -- para Recife nao existe no repositorio o
fluxo completo de chamados da SEDEC, entao os criterios N1 e N2 nao teriam como
ser avaliados. Fica declarado como pendencia, nao resolvido em silencio.

Uso:
    python scripts/suscetibilidade/neg01_exclusao_qualificada_temporal.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
RED = RUNS / "ds-04-reducao"
OUT = RUNS / "neg-01-exclusao-qualificada"

# Uma regiao por entrada. O que muda entre elas nao e o criterio -- e o canal
# que responde cada pergunta, e o vocabulario com que ele nomeia inundacao.
REGIOES: dict[str, dict] = {
    "curitiba": {
        "bruto": RUNS / "susc_20n_siac156_negative_expansion" / "raw",
        "sep": ";", "encoding": "utf-8-sig",
        "col_data": "DataCriacao", "col_categoria": "Subdivisao",
        "col_extra": "Assunto",
        "categorias_inundacao": ("alagamentos",
                                 "enchentes, inundacoes ou alagamentos"),
        "n3_nivel": "endereco",
        "n3_nota": ("o ponto E um chamado do proprio endereco naquele dia, "
                    "sobre assunto nao-hidrologico"),
        "canal": "SIAC 156 de Curitiba",
    },
    "recife": {
        "bruto": RUNS / "acq-recife-01-sedec" / "raw",
        "sep": ";", "encoding": "utf-8",
        # `Data_da_Acao` e a unica coluna de data presente e coerente nos
        # doze anos; `Data` guarda texto de ocorrencia em metade deles
        "col_data": "Data_da_Acao", "col_categoria": "ocorrencia_detectada",
        "colunas_por_conteudo": True,
        "col_extra": None,
        # os seis valores de `Ocorrencia` que nomeiam alagamento. NAO se usa
        # `Solicitacao`: la o texto livre traz nome de rua, e procurar "agua"
        # captura "rua aguas claras", "rio paraguacu" e "araguaiana" -- 1.736
        # falsos so na primeira. Mesmo erro que "bueiro" causou em Curitiba.
        "categorias_inundacao": ("alagamentos", "monitoramento alagado",
                                 "vistoria setor alagado", "imoveis alagados",
                                 "vistoria alagado",
                                 "monitoramento setor alagado"),
        "n3_nivel": "bairro",
        "n3_nota": ("o ponto veio do 156 da EMLURB casado por BAIRRO, nao por "
                    "endereco (dataset_role_source TASK2_v9). E um degrau mais "
                    "fraco que Curitiba e fica declarado como tal"),
        "canal": "Atendimentos da Defesa Civil do Recife (SEDEC)",
        # so estes vieram de chamado real do cidadao; os demais sao base
        # canonica e nao satisfazem o N3
        "n3_role_sources": ("TASK2_v9_emlurb_156_bairro_overlap_matched",),
        "harmonizado": RUNS / "ter-03-brasil-harmonizado" / "recife_harmonizado.csv",
    },
}

# distancia minima a um positivo da mesma data. 500 m e a mesma ordem do
# afastamento usado no piloto ingles; abaixo disso o ponto pode estar na mesma
# quadra alagada.
AFASTAMENTO_M = 500.0

# percentil do volume diario abaixo do qual o dia nao conta como canal ativo
P_VOLUME = 0.25

# PARES (assunto, subdivisao) que marcam chamado de INUNDACAO. Sao os dois que
# o susc_20k validou contra o catalogo de eventos -- 1.005 e 233 registros --
# e nao busca por substring.
#
# A primeira versao deste script usava substring solta e errou por tres razoes
# ao mesmo tempo, todas visiveis so quando o criterio passou em 100% dos dias:
#   "agua"/"chuva"  casavam ZERO -- o arquivo estava sendo lido com o encoding
#                   errado, e o acento chegava corrompido (ver carregar_bruto).
#   "bueiro"        casava "fauna sinantropica | risco para leptospirose/\n#                   roedores em bueiro" -- 3.179 chamados sobre roedores.
#   "drenagem"      casava a categoria inteira, 16.718 por ano, quase toda de
#                   manutencao de rotina (limpeza de caixa de captacao, erosao
#                   em galeria). Presente todo dia, por isso o N2 nunca
#                   reprovava: media a existencia da categoria, nao a chegada
#                   de relato de alagamento.
SUBDIVISOES_INUNDACAO = ("alagamentos", "enchentes, inundacoes ou alagamentos")


# Assinaturas de conteudo para achar a coluna certa quando o cabecalho mente.
# Necessario no Recife: os anuais da SEDEC vem em DOIS layouts alternados --
# em 2015/2019/2020/2022 a ocorrencia esta em `Ocorrencia`, e nos demais anos
# a MESMA informacao aparece em `Solicitacao`, com `Data` guardando o texto da
# ocorrencia. Confiar no nome da coluna mistura as duas coisas: foi assim que
# um levantamento inicial "achou" nome de rua entre os alagamentos.
ASSINATURA_OCORRENCIA = re.compile(
    r"monitoramento|vistoria|lona|alagad|alagament|desliza|imoveis", re.I)
ASSINATURA_DATA_ISO = re.compile(r"^20\d\d-\d\d-\d\d")


def detectar_coluna(d: pd.DataFrame, assinatura: re.Pattern,
                    minimo: float = 0.3) -> str | None:
    """A coluna cujo CONTEUDO casa com a assinatura, e nao cujo nome casa."""
    melhor, escore = None, minimo
    for c in d.columns:
        f = d[c].astype(str).str.contains(assinatura, na=False).mean()
        if f > escore:
            melhor, escore = c, f
    return melhor


def sem_acento(s: str) -> str:
    """NFKD e minuscula. Nada de remendo de mojibake -- ver `carregar_bruto`."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower().strip()


def carregar_bruto(cfg: dict) -> pd.DataFrame:
    """So as colunas necessarias -- as bases somam centenas de MB.\n
    ENCODING: declarado por regiao. O 156 de Curitiba e UTF-8; a primeira
    versao o leu como latin-1, o que NAO levanta excecao -- latin-1 decodifica
    qualquer byte -- e entregava 'Enchentes, inundaÃ§Ãµes ou alagamentos'. Uma
    subdivisao inteira, 233 chamados, deixava de casar em silencio.
    """
    por_conteudo = cfg.get("colunas_por_conteudo", False)
    cols = [cfg["col_data"], cfg["col_categoria"]]
    if cfg.get("col_extra"):
        cols.append(cfg["col_extra"])
    partes = []
    for p in sorted(cfg["bruto"].glob("*.csv")):
        try:
            uso = (None if por_conteudo
                   else (lambda c: c.strip("﻿") in cols))
            d = pd.read_csv(p, sep=cfg["sep"], encoding=cfg["encoding"],
                            low_memory=False, usecols=uso)
        except Exception as exc:  # noqa: BLE001
            print(f"   {p.name}: PULADO ({type(exc).__name__})")
            continue
        d.columns = [c.strip("﻿") for c in d.columns]
        if por_conteudo:
            c_oc = detectar_coluna(d, ASSINATURA_OCORRENCIA)
            # a data tambem e detectada por conteudo: `Data_da_Acao` existe em
            # todos os anos mas nem sempre guarda data. Assumir o nome deixou
            # 60% das linhas sem data e trouxe um "1900-07-01" para o intervalo.
            # `Data_da_Acao` e 100% ISO nos doze anos -- preferida sempre que
            # existir. A deteccao por conteudo fica de reserva, porque com duas
            # colunas empatadas em 1,0 ela escolhia a primeira, que em alguns
            # anos e `Data` guardando texto de ocorrencia.
            c_dt = ("Data_da_Acao" if "Data_da_Acao" in d.columns
                    else detectar_coluna(d, ASSINATURA_DATA_ISO, minimo=0.5))
            if c_oc is None or c_dt is None:
                print(f"   {p.name}: PULADO (nao achei ocorrencia/data)")
                continue
            d = d[[c_dt, c_oc]].rename(
                columns={c_dt: cfg["col_data"], c_oc: cfg["col_categoria"]})
            print(f"   {p.name}: {len(d):,} registros  (ocorrencia em '{c_oc}')")
        else:
            if cfg["col_data"] not in d.columns:
                print(f"   {p.name}: PULADO (sem {cfg['col_data']})")
                continue
            print(f"   {p.name}: {len(d):,} registros")
        partes.append(d)
    if not partes:
        return pd.DataFrame()
    d = pd.concat(partes, ignore_index=True)
    d["data"] = pd.to_datetime(d[cfg["col_data"]], errors="coerce",
                               dayfirst=True).dt.date
    return d[d["data"].notna()]


def perfil_diario(bruto: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Volume total e de INUNDACAO por dia. E o que sustenta N1 e N2."""
    cat = bruto[cfg["col_categoria"]].map(sem_acento)
    bruto = bruto.assign(hidro=cat.isin(cfg["categorias_inundacao"]))
    n = int(bruto["hidro"].sum())
    print(f"   registros de inundacao: {n:,} "
          f"({100*n/max(len(bruto),1):.3f}% do total)")
    if n == 0:
        raise SystemExit(
            "ABORTADO: nenhum registro de inundacao casou com "
            f"{cfg['categorias_inundacao']}. O vocabulario mudou ou a leitura "
            "quebrou -- corrigir antes de usar o N2, e nao seguir com um "
            "criterio que nao mede nada.")
    g = bruto.groupby("data").agg(total=(cfg["col_categoria"], "size"),
                                  hidro=("hidro", "sum"))
    return g.reset_index()


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def avaliar_regiao(regiao: str, cfg: dict) -> dict | None:
    p = RED / f"{regiao}.csv"
    if not p.exists():
        print(f"   ABORTADO: {p} ausente. Rode ds04.")
        return None
    d = pd.read_csv(p, low_memory=False)
    # TODOS os negativos, e nao so os que ainda estao em `ausencia`. O ds04 ja
    # aplica a promocao, entao filtrar por estado atual tornaria o script
    # dependente de quantas vezes ele rodou -- na segunda execucao os
    # aprovados sumiam do denominador e o resultado mudava sozinho.
    cand = d[d.classe == 0].copy()
    pos = d[d.classe == 1].copy()
    print(f"   candidatos em ausencia: {len(cand):,}  "
          f"positivos para afastamento: {len(pos):,}")
    if cand.empty:
        print("   nada a reclassificar")
        return None

    if not cfg["bruto"].exists():
        print(f"   ABORTADO: base bruta ausente em {cfg['bruto']}")
        return None
    print(f"   --- lendo {cfg['canal']} ---")
    bruto = carregar_bruto(cfg)
    if bruto.empty:
        print("   ABORTADO: base bruta vazia")
        return None
    perfil = perfil_diario(bruto, cfg)
    print(f"   {len(bruto):,} registros em {len(perfil):,} dias, "
          f"{perfil.data.min()} a {perfil.data.max()}")

    limiar = float(perfil["total"].quantile(P_VOLUME))
    dias_hidro = set(perfil.loc[perfil["hidro"] > 0, "data"])
    print(f"   N1: volume diario >= p{int(P_VOLUME*100)} = {limiar:,.0f}")
    print(f"   N2: {len(dias_hidro):,} dias com registro de inundacao "
          f"({100*len(dias_hidro)/len(perfil):.1f}% dos dias)")

    cand["data"] = pd.to_datetime(cand["data_evento"], errors="coerce",
                                  format="mixed").dt.date
    pos["data"] = pd.to_datetime(pos["data_evento"], errors="coerce",
                                 format="mixed").dt.date
    vol = dict(zip(perfil["data"], perfil["total"]))

    cand["n1_canal_ativo"] = cand["data"].map(vol).fillna(0) >= limiar
    cand["n2_categoria_em_uso"] = cand["data"].isin(dias_hidro)

    # N3 -- o ponto veio de um relato de cidadao? O nivel difere por regiao e
    # a coluna `n3_nivel` diz qual: endereco em Curitiba, bairro em Recife.
    if cfg["n3_nivel"] == "endereco":
        cand["n3_ponto_reporta"] = (
            cand["procedencia"].astype(str)
            .str.contains("siac156", case=False, na=False) & cand["data"].notna())
    else:
        # em Recife a procedencia reduzida e uniforme; o vinculo com chamado
        # real esta no `dataset_role_source` do arquivo harmonizado
        papeis = set(cfg.get("n3_role_sources") or ())
        h = pd.read_csv(cfg["harmonizado"], low_memory=False)
        mapa = dict(zip((f"{regiao}__" + h["ponto_id"].astype(str)),
                        h["papel_fonte_dataset"].astype(str)))
        origem = cand["ponto_id"].astype(str).map(mapa)
        cand["papel_fonte_dataset"] = origem
        cand["n3_ponto_reporta"] = origem.isin(papeis) & cand["data"].notna()

    dist = []
    for _, r in cand.iterrows():
        mesma = pos[pos["data"] == r["data"]]
        dist.append(np.inf if mesma.empty else float(np.min(haversine_m(
            r["lat"], r["lon"], mesma["lat"].to_numpy(), mesma["lon"].to_numpy()))))
    cand["dist_positivo_mesma_data_m"] = dist
    cand["n4_afastamento"] = cand["dist_positivo_mesma_data_m"] >= AFASTAMENTO_M

    criterios = ["n1_canal_ativo", "n2_categoria_em_uso",
                 "n3_ponto_reporta", "n4_afastamento"]
    cand["aprovado"] = cand[criterios].all(axis=1)
    cand["nivel_negativo_novo"] = np.where(
        cand["aprovado"], "exclusao_qualificada", "ausencia")
    cand["motivo_reprovacao"] = [
        ";".join(c for c in criterios if not r[c]) or ""
        for _, r in cand.iterrows()]

    print("\n   --- CRITERIOS ---")
    for c in criterios:
        n = int(cand[c].sum())
        print(f"   {c:22s} passa {n:>4,} / {len(cand):,} "
              f"({100*n/len(cand):.1f}%)")
    n_ok = int(cand["aprovado"].sum())
    print(f"   APROVADOS nos quatro: {n_ok:,} / {len(cand):,}")
    if n_ok < len(cand):
        for m, k in cand.loc[~cand.aprovado, "motivo_reprovacao"].value_counts().items():
            print(f"      {m}: {k:,}")
    print(f"   => {n_ok:,} em exclusao_qualificada, {len(cand)-n_ok:,} seguem ausencia"
          + ("" if n_ok >= 30 else "  (abaixo do minimo de 30 para a condicao de rotulo)"))

    cols = ["ponto_id", "data_evento", "lat", "lon", "procedencia", *criterios,
            "dist_positivo_mesma_data_m", "aprovado", "nivel_negativo_novo",
            "motivo_reprovacao"]
    cols = [c for c in cols if c in cand.columns]
    cand[cols].to_csv(OUT / f"{regiao}_reclassificacao_negativo.csv", index=False)
    return {
        "regiao": regiao, "canal": cfg["canal"], "candidatos": int(len(cand)),
        "aprovados_exclusao_qualificada": n_ok,
        "seguem_ausencia": int(len(cand) - n_ok),
        "criterios": {c: int(cand[c].sum()) for c in criterios},
        "limiar_volume_diario": limiar, "percentil_volume": P_VOLUME,
        "afastamento_m": AFASTAMENTO_M,
        "dias_com_inundacao": len(dias_hidro), "dias_no_perfil": int(len(perfil)),
        "categorias_inundacao": list(cfg["categorias_inundacao"]),
        "n3_nivel": cfg["n3_nivel"], "n3_nota": cfg["n3_nota"],
    }


def main() -> int:
    args = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===NEG01_EXCLUSAO_QUALIFICADA=== esquema={VERSAO}")
    alvos = ([args[args.index("--regiao") + 1]] if "--regiao" in args
             else list(REGIOES))
    saida = {}
    for regiao in alvos:
        if regiao not in REGIOES:
            print(f"regiao desconhecida: {regiao}. Conhecidas: {list(REGIOES)}")
            return 2
        print(f"\n{'='*66}\n[{regiao}] canal: {REGIOES[regiao]['canal']}")
        r = avaliar_regiao(regiao, REGIOES[regiao])
        if r:
            saida[regiao] = r

    print(f"\n{'='*66}\n--- RESUMO ---")
    for reg, r in saida.items():
        print(f"   {reg:10s} {r['aprovados_exclusao_qualificada']:>4,} de "
              f"{r['candidatos']:>4,} sobem para exclusao qualificada "
              f"(N3 por {r['n3_nivel']})")
    (OUT / "resumo.json").write_text(json.dumps({
        "esquema": VERSAO, "regioes": saida,
        "nao_e": (
            "exclusao qualificada NAO e observacao. Ninguem vistoriou o ponto "
            "procurando inundacao; o que se sabe e que o canal estava ativo, "
            "que o ponto veio de relato de cidadao e que o relato era de outra "
            "coisa. E o mesmo nivel do piloto ingles, um degrau abaixo do CEMS"),
        "diferenca_entre_regioes": (
            "o N3 de Curitiba casa por ENDERECO e o de Recife por BAIRRO. Sao "
            "forcas diferentes de evidencia sob o mesmo nome, e a coluna "
            "n3_nivel registra qual foi usada em cada uma"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
