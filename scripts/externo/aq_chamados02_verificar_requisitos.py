"""Os seis requisitos de `aud_desenho03`, rodados contra o dado que chegou.

POR QUE ESTA VERIFICACAO EXISTE:

`aud_desenho03` avalia uma fonte candidata pelo que se sabe dela ANTES de
baixar, e por isso quase tudo la fica `None` -- desconhecido. Este modulo fecha
o ciclo: com o arquivo em disco, cada requisito deixa de ser desconhecido e
passa a ser medido.

E ele mede com os MESMOS criterios que o projeto ja aplica as seis fontes
servidas, em `aud_desenho01`: pontos por data para saber se a data e da
ocorrencia ou carimbo de lote, classes na mesma celula de chuva para o eixo do
terreno, classes no mesmo dia para o eixo da chuva. Fonte nova nao ganha regra
mais frouxa por ser nova.

-------------------------------------------------------------------------
COMO OS POSITIVOS E O FUNDO SAO SEPARADOS
-------------------------------------------------------------------------

Pelo campo de categoria da propria base, nunca por geometria ou por chuva. E o
padrao de Curitiba: chamado hidrologico e positivo, chamado de outro assunto na
mesma base e fundo. `PADROES_HIDRICOS` guarda os termos por fonte, e eles sao
declarados aqui em vez de inferidos, para que a decisao fique visivel.

O QUE ESTE MODULO NAO FAZ: nao promove nada a rotulo, nao entra no `ds01`, nao
gera variavel e nao decide adquirir mais. Ele mede e declara.

Uso:
    python scripts/externo/aq_chamados02_verificar_requisitos.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "treino"))

from chuva.procedencia_das_datas import granularidade_da_data  # noqa: E402
from aud_desenho01_eixos_de_contraste import (  # noqa: E402
    PCT_MINIMO_DO_EIXO, pct_em_estrato_misto,
)

DADOS = REPO / "local_runs" / "aq-chamados-01"
OUT = REPO / "local_runs" / "aq-chamados-02"
RESOLUCAO_CHUVA = 0.1

# Termos que marcam o chamado como OCORRENCIA na categoria da propria base.
# Declarados, e nao inferidos: quem le precisa ver o que entrou como positivo.
#
# -------------------------------------------------------------------------
# POR QUE SAO PAULO ENCOLHEU, e por que encolher esta certo
# -------------------------------------------------------------------------
#
# A lista anterior de Sao Paulo era `enchente, alagamento, inunda, boca de
# lobo, bueiro, agua parada, drenagem, piscin`. Lida contra o que ela de fato
# casa na base, num trimestre:
#
#   'enchente'      843 linhas, e 762 delas sao
#                   `financas | processo administrativo | complementar
#                    documento para pedido de ISENCAO DE IPTU por enchente`
#                   -- papelada fiscal, nao ocorrencia. Outras 81 sao
#                   `solicitar limpeza da via publica APOS enchentes`.
#   'boca de lobo'  5.459, todas `solicitar LIMPEZA de bueiros`
#   'bueiro'        5.461, idem
#   'drenagem'      6.769, `limpeza de bueiros` e `REFORMAR galerias`
#   'agua parada'   0 linhas
#   'piscin'        0 linhas
#   'alagamento'    11 linhas, `urgencia | DEFESA CIVIL | alagamento`
#   'inunda'         9 linhas, `urgencia | DEFESA CIVIL | inundacao`
#
# Vinte linhas de ocorrencia contra doze mil de manutencao e papelada. O modelo
# `pluvial_urbano_sao_paulo` estava ajustado sobre pedido de limpeza de bueiro:
# isso mede onde a rede existe e onde entope, e nao onde alaga.
#
# Pedido de manutencao nao vai para o FUNDO, e a diferenca importa. Fundo e
# `chamado de OUTRO assunto` -- oportunidade de observacao sem relacao com o
# fenomeno. Um pedido de limpeza de bueiro tem relacao com o fenomeno e nao e
# a ocorrencia dele: ele sai das duas classes.
PADROES_HIDRICOS = {
    "nyc311": ("street flooding", "sewer backup", "catch basin", "flooding"),
    "sp156": ("defesa civil | alagamento", "defesa civil | inunda"),
}

# Chamado relacionado ao fenomeno que NAO e ocorrencia dele. Sai do positivo e
# sai do fundo -- e nao vira classe nenhuma.
PADROES_EXCLUIDOS = {
    "sp156": ("boca de lobo", "bueiro", "drenagem", "galeria",
              "isencao de iptu", "isen", "limpeza da via publica apos",
              "enchente", "inunda", "alagamento", "agua parada", "piscin",
              "corrego"),
}

# Como cada fonte nomeia o que precisamos. O portal de SP declara `format: CSV`
# e serve XLSX -- o arquivo comeca com `PK`, assinatura de ZIP.
#
# CADA PAPEL TEM UMA LISTA DE NOMES, e nao um nome so. O mesmo dataset chega
# por dois caminhos com cabecalhos diferentes: a API do Socrata devolve
# `complaint_type` / `latitude`, e a exportacao do portal devolve
# `Problem (formerly Complaint Type)` / `Latitude`. Com um nome unico, o
# arquivo de exportacao nao casa em nenhuma coluna, `marcar_hidrico` devolve
# tudo False e 142.579 linhas saem da contagem SEM ERRO NENHUM -- foi
# exatamente o que aconteceu, e o efeito medido foi o modelo servido de Nova
# York ser ajustado sobre 2,5% do rotulo disponivel.
ESQUEMAS = {
    "nyc311": {
        "lat": ["latitude", "Latitude"],
        "lon": ["longitude", "Longitude"],
        "data": ["created_date", "Created Date"],
        "categoria": ["complaint_type", "Problem (formerly Complaint Type)",
                      "descriptor", "Problem Detail (formerly Descriptor)"],
        "leitor": "csv"},
    "nyc311_fundo": {
        "lat": ["latitude", "Latitude"],
        "lon": ["longitude", "Longitude"],
        "data": ["created_date", "Created Date"],
        "categoria": ["complaint_type", "Problem (formerly Complaint Type)",
                      "descriptor", "Problem Detail (formerly Descriptor)"],
        "leitor": "csv"},
    "sp156": {
        "lat": ["Latitude"], "lon": ["Longitude"],
        "data": ["Data de Abertura"],
        "categoria": ["Tema", "Assunto", "Serviço", "Servico"],
        "leitor": "xlsx"},
    "fortaleza_copdc": {"lat": [], "lon": [], "data": [], "categoria": [],
                        "leitor": "csv"},
}


def coluna(d: pd.DataFrame, esq: dict, papel: str) -> str | None:
    """O primeiro nome declarado para o papel que existe no quadro."""
    for nome in esq.get(papel) or []:
        if nome in d.columns:
            return nome
    return None


def colunas(d: pd.DataFrame, esq: dict, papel: str) -> list:
    """Todos os nomes declarados para o papel presentes no quadro."""
    return [n for n in (esq.get(papel) or []) if n in d.columns]


def formato_real(caminho: Path) -> str:
    """O que o arquivo E, e nao o que a extensao diz.

    O portal de Sao Paulo declara `format: CSV` e serve XLSX. Ler como texto
    levanta `UnicodeDecodeError`, e a excecao era engolida -- 25 dos 26
    trimestres sumiam sem deixar rastro.
    """
    with caminho.open("rb") as f:
        return "zip/xlsx" if f.read(2) == b"PK" else "texto"


# Dialetos de texto tentados, em ordem. O SP156 tem TRES formatos ao longo dos
# anos -- XLSX com extensao `.csv`, CSV UTF-8 e CSV latin-1 separado por ponto
# e virgula --, e o log de leitura foi quem mostrou isso: 21 dos 26 trimestres
# falhavam com `UnicodeDecodeError` e a excecao era engolida.
DIALETOS = (
    {"sep": ",", "encoding": "utf-8"},
    {"sep": ";", "encoding": "utf-8"},
    {"sep": ";", "encoding": "latin-1"},
    {"sep": ",", "encoding": "latin-1"},
)


# Quantas linhas de CORPO o detector le antes de aceitar um dialeto. O
# cabecalho nao basta: em `Data de Abertura;Tema;Latitude;Longitude` todos os
# caracteres sao ASCII, entao `utf-8` aprova o cabecalho e estoura na primeira
# linha acentuada do corpo -- e a excecao chegaria depois da escolha ja feita.
LINHAS_DE_PROVA = 50


def cabecalho_do_texto(caminho: Path) -> tuple[dict, list] | tuple[None, None]:
    """O dialeto que produz mais de uma coluna E le o corpo sem estourar.

    Uma coluna so significa separador errado: o pandas leu a linha inteira
    como um campo. Testar pelo numero de colunas evita ter de adivinhar pela
    extensao ou pelo pais.
    """
    for dialeto in DIALETOS:
        try:
            prova = pd.read_csv(caminho, nrows=LINHAS_DE_PROVA, **dialeto)
        except (UnicodeDecodeError, pd.errors.ParserError, OSError):
            continue
        if len(prova.columns) > 1:
            return dialeto, list(prova.columns)
    return None, None


def ler(caminho: Path, leitor: str, *, colunas_uteis=None,
        registro: list | None = None) -> pd.DataFrame | None:
    """Le o arquivo inteiro, pelo formato REAL, e registra o que aconteceu.

    Sem teto de linhas. O teto anterior de 200.000 era silencioso: sobre um
    arquivo ordenado por data decrescente ele virava "so os dias mais
    recentes", e nada no manifesto dizia isso. Quem precisar de teto declara
    o teto no manifesto.
    """
    nota = {"arquivo": caminho.name, "leitor_declarado": leitor}
    try:
        nota["formato_real"] = formato_real(caminho)
        if nota["formato_real"] == "zip/xlsx":
            d = pd.read_excel(caminho, usecols=colunas_uteis)
            nota["lido_como"] = "xlsx"
        else:
            dialeto, cabecalho = cabecalho_do_texto(caminho)
            if dialeto is None:
                raise ValueError(
                    "nenhum dialeto de texto produziu mais de uma coluna; "
                    f"tentados: {DIALETOS}")
            d = pd.read_csv(caminho, low_memory=False, usecols=colunas_uteis,
                            **dialeto)
            nota["lido_como"] = f"csv sep={dialeto['sep']!r} {dialeto['encoding']}"
            nota["colunas_no_cabecalho"] = len(cabecalho or [])
        nota["linhas"] = int(len(d))
        nota["ok"] = True
    except Exception as e:                                       # noqa: BLE001
        # A falha vai para o registro em vez de virar `None` mudo: um arquivo
        # que nao entrou tem de aparecer em algum lugar.
        nota.update({"ok": False, "erro": f"{type(e).__name__}: {str(e)[:200]}"})
        if registro is not None:
            registro.append(nota)
        return None
    if registro is not None:
        registro.append(nota)
    return d


def _uteis(esq: dict) -> set:
    return {n for papel in ("lat", "lon", "data", "categoria")
            for n in (esq.get(papel) or [])}


def carregar(fonte: str, registro: list | None = None,
             reduzir=None) -> pd.DataFrame | None:
    """Todos os arquivos da fonte, com projecao de colunas.

    A projecao existe por memoria: os 26 trimestres do SP156 somam quinze
    milhoes de linhas, e so cinco colunas interessam. `usecols` com nomes que
    podem nao existir quebraria o pandas, entao a selecao e feita pelo
    cabecalho de cada arquivo.

    `reduzir(quadro, esquema)` roda POR ARQUIVO, antes da concatenacao. Sem
    ele, promover Sao Paulo exigiria segurar quinze milhoes de linhas de texto
    em memoria de uma vez para depois jogar fora 54% delas.
    """
    esq = ESQUEMAS.get(fonte)
    if not esq:
        return None
    uteis = _uteis(esq)
    partes = []
    for c in sorted(glob.glob(str(DADOS / fonte / "*.csv"))):
        caminho = Path(c)
        if "_amostra" in caminho.name and len(partes):
            continue
        d = ler(caminho, esq["leitor"],
                colunas_uteis=lambda n: n in uteis, registro=registro)
        if d is None or not len(d):
            continue
        if reduzir is not None:
            d = reduzir(d, esq)
            if registro:
                registro[-1]["linhas_apos_reducao"] = int(len(d))
        if len(d):
            partes.append(d)
    return pd.concat(partes, ignore_index=True) if partes else None


def no_futuro(ts: pd.Series) -> pd.Series:
    """Marca ocorrencia datada depois de agora.

    O SP156 tem milhares delas, ate 2026-12-06. Ocorrencia datada depois de
    hoje nao e ocorrencia -- e ela nao e inofensiva: a janela de aquisicao da
    chuva vai ate a maior data da celula, e uma data futura estoura o arquivo
    do provedor com HTTP 400, deixando a celula inteira sem serie.

    A regra mora AQUI, e nao em cada consumidor, porque ela ja divergiu uma
    vez: o autopareamento descartava data futura e a promocao nao.
    """
    if ts.empty:
        return ts.astype(bool)
    agora = pd.Timestamp.now(tz=ts.dt.tz) if ts.dt.tz else pd.Timestamp.now()
    return ts > agora


def _alternativa(padroes) -> str:
    """Alternativa de regex com os termos ESCAPADOS.

    `"|".join(padroes)` sem escapar transforma o separador de coluna dos
    proprios termos em alternancia: `defesa civil | alagamento` passaria a
    casar `defesa civil ` OU ` alagamento`, e o rotulo de Sao Paulo saltava de
    vinte linhas por trimestre para cento e setenta mil.
    """
    return "|".join(re.escape(str(t)) for t in padroes)


def _texto_das_categorias(d: pd.DataFrame, esq: dict) -> pd.Series:
    """As colunas de categoria concatenadas, em minusculas e sem acento.

    Sem a remocao de acento os termos declarados teriam de existir em duas
    grafias -- `urgencia` e `urgência` --, e a lista viraria o lugar onde o
    rotulo se perde por causa de um til. E o separador ` | ` e o que permite
    um termo declarar a COLUNA de origem: `defesa civil | alagamento` casa a
    urgencia da Defesa Civil, e nao a palavra alagamento em qualquer contexto.
    """
    import unicodedata

    cols = [c for c in esq.get("categoria") or [] if c in d.columns]
    if not cols:
        return pd.Series("", index=d.index, dtype="object")
    texto = d[cols[0]].fillna("").astype(str).str.lower()
    for c in cols[1:]:
        texto = texto + " | " + d[c].fillna("").astype(str).str.lower()
    return texto.map(lambda t: "".join(
        c for c in unicodedata.normalize("NFKD", t)
        if not unicodedata.combining(c)))


def marcar_excluido(d: pd.DataFrame, fonte: str, esq: dict) -> pd.Series:
    """Chamado relacionado ao fenomeno que nao e a ocorrencia dele.

    Ele nao pode ser positivo -- nao houve alagamento declarado -- e nao pode
    ser fundo -- fundo e `outro assunto`, e limpeza de bueiro nao e outro
    assunto. Sai das duas classes.
    """
    padroes = PADROES_EXCLUIDOS.get(fonte.replace("_fundo", ""), ())
    if not padroes:
        return pd.Series(False, index=d.index)
    texto = _texto_das_categorias(d, esq)
    excluido = texto.str.contains(_alternativa(padroes), regex=True, na=False)
    # a ocorrencia declarada vence a exclusao: `defesa civil | alagamento`
    # contem `alagamento`, que tambem esta na lista de exclusao
    return excluido & ~marcar_hidrico(d, fonte, esq)


def marcar_hidrico(d: pd.DataFrame, fonte: str, esq: dict) -> pd.Series:
    padroes = PADROES_HIDRICOS.get(fonte.replace("_fundo", ""), ())
    if not padroes:
        return pd.Series(False, index=d.index)
    texto = _texto_das_categorias(d, esq)
    return texto.str.contains(_alternativa(padroes), regex=True, na=False)


def verificar(fonte: str, d: pd.DataFrame) -> dict:
    esq = ESQUEMAS[fonte]
    r: dict = {"linhas_lidas": int(len(d)), "colunas": int(d.shape[1])}

    c_lat, c_lon = coluna(d, esq, "lat"), coluna(d, esq, "lon")
    c_data = coluna(d, esq, "data")
    tem_geo, tem_data = bool(c_lat and c_lon), bool(c_data)
    r["geolocalizacao_por_ocorrencia"] = tem_geo
    r["colunas_resolvidas"] = {"lat": c_lat, "lon": c_lon, "data": c_data,
                               "categoria": colunas(d, esq, "categoria")}
    if not (tem_geo and tem_data):
        r["veredito"] = "nao_serve"
        r["motivo"] = ("sem coordenada por ocorrencia" if not tem_geo
                       else "sem data por ocorrencia")
        return r

    z = d.copy()
    z["_lat"] = pd.to_numeric(z[c_lat], errors="coerce")
    z["_lon"] = pd.to_numeric(z[c_lon], errors="coerce")
    z["_ts"] = pd.to_datetime(z[c_data], errors="coerce", utc=True,
                              format="mixed")
    z = z[z._lat.notna() & z._lon.notna() & z._ts.notna()]
    r["com_geo_e_data"] = int(len(z))
    r["pct_com_geo_e_data"] = round(100 * len(z) / max(len(d), 1), 1)
    if not len(z):
        r["veredito"] = "nao_serve"
        r["motivo"] = "nenhuma linha tem coordenada e data validas"
        return r

    z["classe"] = marcar_hidrico(z, fonte, esq).astype(int)
    z["dia"] = z._ts.dt.strftime("%Y-%m-%d")
    z["cel"] = ((z._lat / RESOLUCAO_CHUVA).round().astype(str) + "_"
                + (z._lon / RESOLUCAO_CHUVA).round().astype(str))
    z["cel_dia"] = z.cel + "@" + z.dia

    datas = int(z.dia.nunique())
    # o intervalo e obrigatorio aqui: sem ele a razao de pontos por data
    # acusaria lote numa cidade que so tem muito chamado por dia
    janela = int((z._ts.max() - z._ts.min()).days) + 1
    r.update(granularidade_da_data(len(z), datas, janela))
    r["datas_distintas"] = datas
    r["celulas"] = int(z.cel.nunique())
    r["positivos_hidricos"] = int(z.classe.sum())
    r["fundo"] = int((z.classe == 0).sum())
    r["intervalo"] = [z.dia.min(), z.dia.max()]

    if not r["positivos_hidricos"] or not r["fundo"]:
        r["veredito"] = "indeterminado"
        r["motivo"] = ("a categoria nao separou as duas classes nesta amostra; "
                       "revisar PADROES_HIDRICOS para esta fonte")
        return r

    r["pct_pontos_em_celula_mista"] = pct_em_estrato_misto(z, "cel")
    r["pct_pontos_em_data_mista"] = pct_em_estrato_misto(z, "dia")
    r["pct_pontos_com_chuva_neutralizada"] = pct_em_estrato_misto(z, "cel_dia")

    serve_terreno = r["pct_pontos_em_celula_mista"] >= PCT_MINIMO_DO_EIXO
    serve_chuva = (r["granularidade"] == "por_ocorrencia"
                   and r["pct_pontos_em_data_mista"] >= PCT_MINIMO_DO_EIXO)
    r["eixo_terreno"] = "serve" if serve_terreno else "nao_serve"
    r["eixo_chuva"] = "serve" if serve_chuva else "nao_serve"
    r["veredito"] = ("serve_nos_dois_eixos" if serve_terreno and serve_chuva
                     else "serve_so_terreno" if serve_terreno else "nao_serve")
    return r


def main() -> int:
    if not DADOS.exists():
        print(f"nada adquirido em {DADOS}")
        return 2
    resultados, leitura = {}, {}
    for fonte in sorted(ESQUEMAS):
        if not (DADOS / fonte).exists():
            continue
        registro: list = []
        d = carregar(fonte, registro=registro)
        leitura[fonte] = {
            "arquivos": registro,
            "arquivos_lidos": sum(1 for x in registro if x.get("ok")),
            "arquivos_com_falha": [x for x in registro if not x.get("ok")],
            "linhas_lidas": sum(x.get("linhas", 0) for x in registro),
        }
        if d is None:
            resultados[fonte] = {"veredito": "nao_lido",
                                 "motivo": "nenhum arquivo legivel"}
            continue
        resultados[fonte] = verificar(fonte, d)

    relatorio = {
        "artefato": "aq_chamados02_verificar_requisitos",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterios": {"origem": "aud_desenho01/aud_desenho03",
                      "pct_minimo_do_eixo": PCT_MINIMO_DO_EIXO,
                      "teto_de_linhas_por_arquivo": None,
                      "por_que_sem_teto": (
                          "o teto de 200.000 era silencioso: sobre arquivo "
                          "ordenado por data decrescente ele virava 'so os "
                          "dias mais recentes', e nada dizia isso")},
        "leitura_por_fonte": leitura,
        "padroes_hidricos": {k: list(v) for k, v in PADROES_HIDRICOS.items()},
        "por_fonte": resultados,
        "nao_e": ("nao promove nada a rotulo, nao entra no ds01, nao gera "
                  "variavel e nao decide adquirir mais"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "requisitos_verificados.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    print(f"{'fonte':16s} {'linhas':>9s} {'pos':>8s} {'fundo':>8s} "
          f"{'pt/data':>8s} {'terreno':>8s} {'chuva':>8s}  veredito")
    for f, v in resultados.items():
        print(f"{f:16s} {v.get('linhas_lidas', 0):9d} "
              f"{v.get('positivos_hidricos', 0):8d} {v.get('fundo', 0):8d} "
              f"{str(v.get('pontos_por_data', '-')):>8s} "
              f"{str(v.get('pct_pontos_em_celula_mista', '-')):>8s} "
              f"{str(v.get('pct_pontos_em_data_mista', '-')):>8s}  "
              f"{v.get('veredito')}"
              + (f" ({v['motivo'][:60]})" if v.get("motivo") else ""))
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
