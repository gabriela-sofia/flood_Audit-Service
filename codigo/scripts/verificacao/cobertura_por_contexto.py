"""Cobertura por CONTEXTO: onde o projeto pode aprender, comparar, transferir
ou recusar -- e por que.

-------------------------------------------------------------------------
A MUDANCA DE PERGUNTA
-------------------------------------------------------------------------

Ate aqui "cobertura" queria dizer "quantas regioes o produto atende". Isso
mede oferta, e oferta nao e conhecimento: Nova York tinha modelo servido e
nao e lugar que o produto oferece; as cinco macrorregioes tem a evidencia
mais forte do projeto e nao tem grade nenhuma.

A pergunta que este arquivo responde e outra:

    em quais combinacoes de territorio, mecanismo e relevo o projeto tem
    evidencia suficiente para APRENDER, COMPARAR, TRANSFERIR ou RECUSAR?

A unidade deixa de ser a regiao servida e passa a ser a CELULA DE CONTEXTO.
Uma celula nao e um lugar no mapa: e um lugar COM um mecanismo COM um
relevo -- porque a mesma cidade responde a perguntas diferentes conforme o
fenomeno, e tratar "Curitiba" como uma coisa so foi o que permitiu servi-la
com um modelo que a ordena no acaso.

-------------------------------------------------------------------------
AS CAPACIDADES SAO UM CONJUNTO, E NAO UM ROTULO
-------------------------------------------------------------------------

Uma celula pode ser ao mesmo tempo boa para aprender e para comparar. Forcar
um rotulo unico obrigaria a escolher entre dois fatos verdadeiros, e o que se
perde numa escolha dessas e sempre a informacao mais especifica.

    aprender    ha as duas classes, o negativo e OBSERVADO, e o orcamento de
                EPV do projeto comporta ao menos uma variavel
    comparar    duas ou mais fontes distintas medem a mesma celula com as
                duas classes -- da para contrastar instrumento e processo
    transferir  ha terreno derivado e ha pontos, mas nao ha negativo
                observado nem orcamento: a celula pode RECEBER escore de
                outra, declarando transferencia
    recusar     falta terreno, ou falta uma das classes, ou o unico negativo
                e `ausencia` -- e ausencia de registro nunca foi ausencia de
                agua

Nenhuma delas e escrita por celula. Todas saem dos mesmos criterios que o
`svc01` ja usa para decidir o que serve: `EPV_MINIMO` de 10 grupos por
variavel, e a hierarquia de negativo do `ds03`. Um segundo numero aqui faria
o projeto ter duas definicoes de "evidencia suficiente".

-------------------------------------------------------------------------
O QUE E OFERTADO E NAO E USADO -- e por que isso fica no artefato
-------------------------------------------------------------------------

Vegetacao, bioma e solo sao contexto real e o projeto TEM dado deles. So que
nenhum liga na tabela de hoje:

  * MapBiomas Colecao 9 existe para 300 patches de tres cidades, chaveado por
    `patch_id` -- coluna que a tabela unica nao tem (ela tem `ponto_id`), e
    marcado `review_only`;
  * WorldCover foi lido para `ufo` e `sen1floods11` (duas fontes
    ESTRANGEIRAS), e virou estatistica agregada por chip, nunca coluna;
  * ha um raster MapBiomas de bbox de Petropolis, de uma fase encerrada.

Declarar isso no proprio artefato e o ponto. Um eixo que some da tabela vira
um eixo que ninguem lembra de adquirir; um eixo que aparece como
`ofertado_sem_dado`, com o motivo, e uma lacuna com endereco.

Uso:
    python scripts/verificacao/cobertura_por_contexto.py
    python scripts/verificacao/cobertura_por_contexto.py --minimo-para-listar 50
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
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

UNICA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
MATRIZ = (REPO / "local_runs" / "temporalidade-integracao"
          / "scientific_support_matrix.json")
SAIDA = REPO / "local_runs" / "cobertura-por-contexto"

# O MESMO numero do `svc01`. Importado e nao copiado: duas constantes iguais
# em arquivos diferentes ficam iguais ate o dia em que uma muda.
try:
    from svc01_construir_modelos_servidos import EPV_MINIMO
except Exception:  # pragma: no cover - so quando o svc01 nao importa
    EPV_MINIMO = 10

TERRENO = ["hand_m", "twi_dinf", "slope_deg", "elevation_rel_m"]

APRENDER = "aprender"
COMPARAR = "comparar"
TRANSFERIR = "transferir"
RECUSAR = "recusar"


# --------------------------------------------------- eixos declarados
#
# A lista e o contrato deste artefato: todo eixo de contexto que o projeto
# reconhece aparece aqui, com ou sem dado. `chave` e None quando o eixo nao
# tem coluna -- e e isso que o torna visivel como lacuna em vez de ausente.

EIXOS = [
    {"eixo": "territorio", "chave": "territorio",
     "o_que_e": "macrorregiao do IBGE por ponto-em-poligono; fora do Brasil, "
                "o contexto externo declarado pela fonte"},
    {"eixo": "mecanismo", "chave": "mecanismo",
     "o_que_e": "o processo fisico declarado com evidencia, nunca inferido "
                "de desempenho"},
    {"eixo": "relevo", "chave": "classe_relevo",
     "o_que_e": "serra ou planicie, por relevo_local e fracao de declive"},
    {"eixo": "como_o_relevo_foi_obtido", "chave": "classe_relevo_base",
     "o_que_e": "amostrado, janela de raster, declarado ou ausente -- a "
                "procedencia da propria classificacao"},
    {"eixo": "escala_do_dado", "chave": "resolucao_nativa_m",
     "o_que_e": "resolucao nativa do rotulo em metros; 250 m do MODIS nao "
                "responde a mesma pergunta que 3 m"},
    {"eixo": "tipo_de_negativo", "chave": "nivel_negativo",
     "o_que_e": "observado, exclusao qualificada ou ausencia -- e ausencia "
                "nunca constitui registro de ausencia"},
    {"eixo": "precisao_da_data", "chave": "precisao_data",
     "o_que_e": "se a data e do evento, do lote, da ativacao ou nao "
                "declarada"},
    {"eixo": "forcamento_de_chuva", "chave": "fonte_chuva",
     "o_que_e": "qual produto de chuva cobre a celula, ou ausente"},
    {"eixo": "procedencia", "chave": "procedencia",
     "o_que_e": "o processo que produziu o rotulo, por valor"},
    {"eixo": "terreno", "chave": None, "derivado_de": TERRENO,
     "o_que_e": "HAND, TWI, declividade e elevacao relativa -- entram como "
                "DESCRICAO quantitativa da celula, e nao como chave"},
]

EIXOS_OFERTADOS_SEM_DADO = [
    {"eixo": "vegetacao_e_cobertura_do_solo",
     "existe_no_projeto": True,
     "onde": "local_runs/suscetibilidade/19f_mapbiomas_territorial "
             "(MapBiomas Colecao 9, 2022)",
     "por_que_nao_liga": "300 patches de tres cidades, chaveados por "
                         "`patch_id`; a tabela unica e por `ponto_id` e nao "
                         "tem essa chave. O artefato e `review_only`",
     "o_que_resolveria": "extrair MapBiomas nos pontos da tabela atual (GEE), "
                         "como coluna com procedencia, e nao como resumo"},
    {"eixo": "cobertura_construida_worldcover",
     "existe_no_projeto": True,
     "onde": "local_runs/drv-01-cobertura/cobertura.json",
     "por_que_nao_liga": "lida so para `ufo` e `sen1floods11` -- duas fontes "
                         "estrangeiras -- e agregada por chip, nunca por "
                         "ponto; nenhuma fonte brasileira foi coberta",
     "o_que_resolveria": "ler a mesma cobertura nos pontos brasileiros e "
                         "gravar por ponto"},
    {"eixo": "bioma",
     "existe_no_projeto": False,
     "onde": "malhas IBGE em disco sao municipio e UF "
             "(local_runs/br-02-negativo-nacional)",
     "por_que_nao_liga": "a malha de biomas nao foi adquirida",
     "o_que_resolveria": "adquirir a malha de biomas do IBGE pelo mesmo "
                         "caminho do `aq_malha_territorial_ibge`"},
    {"eixo": "solo",
     "existe_no_projeto": False,
     "onde": None,
     "por_que_nao_liga": "nenhuma fonte de solo foi adquirida",
     "o_que_resolveria": "declarar a fonte antes de adquirir; sem isso o "
                         "eixo entraria sem procedencia"},
]


BORDA_DA_MALHA_GRAUS = 0.1
MALHA_UF = REPO / "local_runs" / "br-02-negativo-nacional" / "malha_uf.geojson"


def _perto_da_malha(lon: np.ndarray, lat: np.ndarray,
                    malha: Path = MALHA_UF) -> np.ndarray:
    """True onde o ponto esta a ate `BORDA_DA_MALHA_GRAUS` da malha do IBGE.

    Sem a malha em disco nao se sabe o que e borda, e a resposta e False:
    o ponto vira externo, e nao Brasil por omissao.
    """
    import json as _json

    import shapely
    from shapely.geometry import shape

    if not malha.exists() or len(lon) == 0:
        return np.zeros(len(lon), dtype=bool)
    g = _json.loads(malha.read_text(encoding="utf-8"))
    brasil = shapely.union_all([shape(f["geometry"]) for f in g["features"]])
    ok = np.isfinite(lon) & np.isfinite(lat)
    saida = np.zeros(len(lon), dtype=bool)
    pts = shapely.points(lon[ok], lat[ok])
    saida[ok] = shapely.dwithin(brasil, pts, BORDA_DA_MALHA_GRAUS)
    return saida


def _territorio(d: pd.DataFrame) -> pd.Series:
    """Macrorregiao por ponto-em-poligono; fora do Brasil, contexto externo.

    NAO e caixa envolvente. A caixa do Centro-Oeste e a do Nordeste se cruzam
    em Tocantins, e um ponto ali receberia territorio conforme a ordem de um
    dicionario -- o mesmo defeito que ja custou uma maturidade errada.
    """
    try:
        import br04_o_brasil_como_professor as br
        com = br.com_regiao(d[["lat", "lon"]].copy())
        sigla = com.get("regiao_br")
    except Exception as e:  # pragma: no cover - malha ausente
        print(f"  AVISO: malha do IBGE indisponivel ({e}); "
              "territorio brasileiro fica 'brasil_sem_malha'")
        sigla = pd.Series([None] * len(d), index=d.index)

    nome = {"N": "norte", "NE": "nordeste", "CO": "centro_oeste",
            "SE": "sudeste", "S": "sul"}
    saida = pd.Series(
        [nome.get(s) if s else None for s in sigla], index=d.index,
        dtype="object")
    # Quem nao caiu em macrorregiao brasileira e contexto externo, e o nome
    # dele e a FONTE -- que e o que de fato identifica o processo de
    # observacao la fora. Chamar tudo de "exterior" apagaria a diferenca
    # entre Nova York e o Sen1Floods11.
    # FORA DA MALHA NAO E ESTRANGEIRO, e confundir os dois inverte o contexto.
    #
    # A primeira versao mandava todo ponto sem macrorregiao para
    # `externo:<fonte>`, e 9.722 linhas do GFD -- que e brasileiro por
    # construcao (`gfd01_eventos_do_brasil`) -- sairam rotuladas como
    # externas. Ponto de costa ou de divisa cai fora do poligono do IBGE sem
    # deixar de ser Brasil; a malha tem borda, o pais nao some nela.
    #
    # O segundo teste e a caixa do Brasil, e ela so entra DEPOIS do poligono:
    # caixa sozinha erraria em Tocantins, mas como desempate de quem ja nao
    # casou com poligono nenhum ela separa "borda" de "outro continente".
    #
    # A CAIXA ERA LARGA DEMAIS, e o erro foi medido em 01/10/2026: a caixa
    # do Brasil (-34..6, -74..-34) contem o Paraguai inteiro, metade da
    # Bolivia e o nordeste argentino. A celula `brasil_fora_da_malha` tinha
    # 31.899 linhas, 2.769 delas em volta de Assuncao e 9.512 a oeste de -60
    # -- contadas como Brasil no ganho e na lista de promocao. Borda e agora
    # DISTANCIA ATE A MALHA: ate `BORDA_DA_MALHA_GRAUS` (~11 km, a margem da
    # simplificacao da malha e da linha de costa) e Brasil; alem disso e o
    # pais vizinho, que continua no projeto como contexto externo.
    fora = saida.isna()
    lat = pd.to_numeric(d.loc[fora, "lat"], errors="coerce")
    lon = pd.to_numeric(d.loc[fora, "lon"], errors="coerce")
    perto = _perto_da_malha(lon.to_numpy(), lat.to_numpy())
    na_borda = pd.Series(perto, index=lat.index).fillna(False).astype(bool)
    idx_borda = fora[fora].index[na_borda.to_numpy()]
    idx_externo = fora[fora].index[~na_borda.to_numpy()]
    saida.loc[idx_borda] = "brasil_fora_da_malha"
    saida.loc[idx_externo] = (
        "externo:" + d.loc[idx_externo, "fonte"].astype(str))
    return saida


def _faixa(s: pd.Series) -> dict | None:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return None
    return {"n": int(len(v)),
            "p5": round(float(v.quantile(0.05)), 3),
            "mediana": round(float(v.median()), 3),
            "p95": round(float(v.quantile(0.95)), 3)}


def capacidades(cel: pd.DataFrame) -> tuple[list, dict]:
    """As capacidades desta celula, derivadas -- e a razao de cada uma.

    Devolve (lista de capacidades, dicionario de criterios medidos). O
    segundo existe para que discordar seja possivel com fundamento: quem
    achar o veredito errado ve qual numero o produziu.
    """
    tem_terreno = bool(
        cel[[c for c in TERRENO if c in cel.columns]].notna().any(axis=1).any())
    classes = set(cel.classe.dropna().unique().tolist())
    duas_classes = {0, 1} <= classes

    negativos = cel[cel.classe == 0]
    observado = int((negativos.nivel_negativo == "observado").sum())
    so_ausencia = (not negativos.empty
                   and set(negativos.nivel_negativo.dropna().unique())
                   <= {"ausencia"})

    por_classe = cel.groupby("classe").grupo_cv.nunique()
    orcamento = (int(por_classe.min()) // EPV_MINIMO
                 if len(por_classe) >= 2 else 0)

    fontes_com_as_duas = sorted({
        f for f, g in cel.groupby("fonte")
        if {0, 1} <= set(g.classe.dropna().unique().tolist())})

    criterios = {
        "tem_terreno": tem_terreno,
        "tem_as_duas_classes": duas_classes,
        "negativos_observados": observado,
        "negativo_so_por_ausencia": bool(so_ausencia),
        "grupos_da_classe_rara": (int(por_classe.min())
                                  if len(por_classe) >= 2 else 0),
        "orcamento_de_variaveis": orcamento,
        "epv_minimo_do_projeto": EPV_MINIMO,
        "fontes_com_as_duas_classes": fontes_com_as_duas,
    }

    caps = []
    if tem_terreno and duas_classes and observado > 0 and orcamento >= 1:
        caps.append(APRENDER)
    if tem_terreno and len(fontes_com_as_duas) >= 2:
        caps.append(COMPARAR)
    if tem_terreno and APRENDER not in caps:
        caps.append(TRANSFERIR)
    if not caps:
        caps.append(RECUSAR)
    return caps, criterios


def construir(minimo_para_listar: int = 0) -> dict:
    if not UNICA.exists():
        raise SystemExit(f"ABORTADO: {UNICA} ausente")
    d = pd.read_parquet(UNICA)
    d = d[(d.admitido == True) & d.classe.isin([0, 1])].copy()  # noqa: E712
    print(f"lendo {len(d):,} linhas admitidas com rotulo...")
    d["territorio"] = _territorio(d)

    celulas = []
    chave = ["territorio", "mecanismo", "classe_relevo"]
    for (terr, mec, rel), cel in d.groupby(chave, dropna=False):
        caps, crit = capacidades(cel)
        celulas.append({
            "celula": {"territorio": terr, "mecanismo": mec,
                       "relevo": rel},
            "n": int(len(cel)),
            "grupos": int(cel.grupo_cv.nunique()),
            "positivos": int((cel.classe == 1).sum()),
            "negativos": int((cel.classe == 0).sum()),
            "negativo_por_nivel": {
                str(k): int(v) for k, v in
                cel[cel.classe == 0].nivel_negativo.value_counts().items()},
            "fontes": sorted(cel.fonte.dropna().astype(str).unique().tolist()),
            "procedencias": sorted(
                cel.procedencia.dropna().astype(str).unique().tolist()),
            "escalas_nativas_m": sorted(
                float(x) for x in
                cel.resolucao_nativa_m.dropna().unique().tolist()),
            "precisao_da_data": sorted(
                cel.precisao_data.dropna().astype(str).unique().tolist()),
            "forcamento_de_chuva": sorted(
                cel.fonte_chuva.dropna().astype(str).unique().tolist()),
            "como_o_relevo_foi_obtido": sorted(
                cel.classe_relevo_base.dropna().astype(str).unique().tolist()),
            "janela": {
                "inicio": (str(pd.to_datetime(
                    cel.data_evento, errors="coerce", format="mixed",
                    utc=True).min())[:10]),
                "fim": (str(pd.to_datetime(
                    cel.data_evento, errors="coerce", format="mixed",
                    utc=True).max())[:10]),
            },
            "terreno": {c: _faixa(cel[c]) for c in TERRENO if c in cel.columns},
            "capacidades": caps,
            "criterios_medidos": crit,
        })

    celulas.sort(key=lambda c: -c["n"])
    por_cap: dict = {}
    for c in celulas:
        for cap in c["capacidades"]:
            por_cap.setdefault(cap, []).append(
                "|".join(str(v) for v in c["celula"].values()))

    total_linhas = sum(c["n"] for c in celulas)
    return {
        "tipo": "cobertura_por_contexto",
        "quando_utc": agora_utc(),
        "pergunta": ("em quais combinacoes de territorio, mecanismo e relevo "
                     "o projeto tem evidencia para aprender, comparar, "
                     "transferir ou recusar"),
        "unidade": "celula de contexto = territorio x mecanismo x relevo",
        "eixos_declarados": EIXOS,
        "eixos_ofertados_sem_dado": EIXOS_OFERTADOS_SEM_DADO,
        "criterio_das_capacidades": {
            APRENDER: ("terreno + as duas classes + negativo OBSERVADO + "
                       f"orcamento de EPV >= 1 (grupos da rara // "
                       f"{EPV_MINIMO})"),
            COMPARAR: "terreno + duas ou mais fontes com as duas classes",
            TRANSFERIR: "terreno, mas sem sustentar `aprender`",
            RECUSAR: "sem terreno, ou sem uma das classes",
            "epv_vem_de": "svc01.EPV_MINIMO, importado e nao copiado",
        },
        "total_de_celulas": len(celulas),
        "linhas_cobertas": total_linhas,
        "celulas_por_capacidade": {k: sorted(v) for k, v in
                                   sorted(por_cap.items())},
        "contagem_por_capacidade": {k: len(v) for k, v in
                                    sorted(por_cap.items())},
        "celulas": [c for c in celulas if c["n"] >= minimo_para_listar],
        "nao_e": ("nao e cobertura de PRODUTO e nao promove nada: uma celula "
                  "com `aprender` diz que ha evidencia para ajustar ali, e "
                  "nao que exista modelo ajustado"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimo-para-listar", type=int, default=0,
                    help="omite do artefato celulas menores que isto")
    ns = ap.parse_args()

    r = construir(ns.minimo_para_listar)
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items()
                                    if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "cobertura.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    (SAIDA / "commands.txt").write_text(
        "python scripts/verificacao/cobertura_por_contexto.py\n",
        encoding="utf-8")

    print("\n===COBERTURA POR CONTEXTO===")
    print(f"{r['total_de_celulas']} celulas cobrindo "
          f"{r['linhas_cobertas']:,} linhas")
    print("capacidades: " + " · ".join(
        f"{k}={v}" for k, v in r["contagem_por_capacidade"].items()))
    print()
    largura = max((len("|".join(str(x) for x in c["celula"].values()))
                   for c in r["celulas"]), default=20)
    print(f"{'celula':{largura}s} {'n':>9s} {'grp':>6s} {'obs':>8s}  capacidades")
    for c in r["celulas"][:25]:
        k = "|".join(str(x) for x in c["celula"].values())
        print(f"{k:{largura}s} {c['n']:>9,} {c['grupos']:>6} "
              f"{c['criterios_medidos']['negativos_observados']:>8,}  "
              + ",".join(c["capacidades"]))
    print()
    print("eixos ofertados e ainda sem dado ligavel:")
    for e in r["eixos_ofertados_sem_dado"]:
        print(f"  - {e['eixo']}: {e['por_que_nao_liga'][:88]}")
    print(f"\nGRAVADO={SAIDA}/cobertura.json")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
