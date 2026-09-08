"""Casos datados das centrais de atendimento, e a auditoria do que foi adquirido.

POR QUE ESTE MODULO EXISTE, e por que ele nao substitui `aq_chamados01-03`:

a cadeia `aq_chamados` promove uma central a FONTE do `ds01` -- positivos, fundo,
terreno, esquema. Este modulo pede outra coisa e so ela: os CASOS datados, com
lugar e dia, sem fundo nenhum. O desenho autopareado nao usa classe negativa;
ele compara o caso consigo mesmo em outras datas. Pedir fundo seria carregar
para dentro do desenho exatamente o viés que ele existe para cancelar.

E ao ler os arquivos brutos ele mede uma coisa que a cadeia atual nao mede: se a
janela de aquisicao dos positivos e a do fundo COINCIDEM. Quando nao coincidem,
a classe passa a ser previsivel pela data, e qualquer variavel que varie no
tempo -- a chuva em primeiro lugar -- herda esse poder sem ter fisica nenhuma.

-------------------------------------------------------------------------
O DEFEITO QUE ESTA AUDITORIA ENCONTROU EM NOVA YORK
-------------------------------------------------------------------------

Tres causas somadas, todas verificaveis no proprio manifesto de aquisicao:

  1. o arquivo COMPLETO dos positivos (142.579 linhas, 2020-2026) tem cabecalho
     de EXPORTACAO -- `Problem (formerly Complaint Type)`, `Latitude` --, e o
     esquema de `aq_chamados02` so conhece o cabecalho de API -- `complaint_type`,
     `latitude`. Nenhuma coluna casa, `marcar_hidrico` devolve tudo False, e o
     arquivo inteiro sai da contagem sem erro nenhum. Sobra a AMOSTRA de 5.000
     linhas;
  2. o fundo foi pedido com `$limit=400000 ... $order=created_date DESC`, e lido
     com `nrows=200000`: ficam as 200 mil linhas MAIS RECENTES da janela;
  3. a janela pedida do fundo termina em 2026-07-30 e a dos positivos vai ate
     2026-08-29.

Resultado medido: 73,8% dos positivos caem em dia SEM nenhum negativo, e um
preditor que so conhece o NUMERO DO DIA separa as classes com AUC 0,887.

Isso nao invalida a fisica -- invalida a leitura causal de qualquer ganho de
AUC obtido comparando as duas classes nessa base. E a razao pela qual o
autopareamento e o desenho certo aqui, e nao um refinamento opcional.

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao promove fonte, nao escreve no `ds01`, nao rotula ponto novo e nao altera a
tabela unica. Le bruto, agrega por celula-dia e declara procedencia.

Uso:
    python scripts/chuva/casos_das_centrais.py --fonte nyc
    python scripts/chuva/casos_das_centrais.py --auditar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import SPATIAL_RESOLUTION_DEGREES  # noqa: E402

BRUTOS = REPO / "local_runs" / "aq-chamados-01"
TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
OUT = REPO / "local_runs" / "chuva-casos-centrais"

# Cabecalho de API e cabecalho de exportacao para a MESMA base. Guardar os dois
# e o que impede um arquivo inteiro de sair da contagem porque a coluna mudou
# de nome no caminho do portal para o disco.
ALIASES = {
    "data": ["created_date", "Created Date", "Data de Abertura"],
    "lat": ["latitude", "Latitude"],
    "lon": ["longitude", "Longitude"],
    "categoria": ["complaint_type", "Problem (formerly Complaint Type)",
                  "Tema", "Assunto"],
    "detalhe": ["descriptor", "Problem Detail (formerly Descriptor)",
                "Serviço", "Servico"],
}

FONTES = {
    "nyc": {
        "regiao": "nyc",
        # A leitura passa por `aq_chamados02.carregar`, que ja resolve os dois
        # cabecalhos, o formato real e o dialeto. Duplicar aqui um leitor
        # proprio foi o que fez este modulo enxergar so um trimestre de Sao
        # Paulo enquanto a promocao ja lia os vinte e seis.
        "fonte_do_leitor": "nyc311",
        "arquivo": BRUTOS / "nyc311" / "9qq5-d465_completo.csv",
        "leitor": "csv",
        "dataset": "NYC Open Data 9qq5-d465 (Sewer / Street Flooding)",
        "termos": ("street flooding", "sewer backup", "catch basin", "flooding",
                   "highway flooding"),
        # Subtipo importa: `sewer backup` EXIGE que exista rede para haver a
        # quem reclamar; `street flooding` nao exige. Separar os dois e o teste
        # de atalho de deteccao que a drenagem levantou.
        "subtipos": {
            "sewer_backup": ("sewer backup",),
            "catch_basin": ("catch basin",),
            "street_flooding": ("street flooding", "highway flooding"),
        },
    },
    "sao_paulo": {
        "regiao": "sao_paulo",
        "fonte_do_leitor": "sp156",
        "arquivo": BRUTOS / "sp156" / "9aac32a3_Dados do SP156 - 2º TRI 2021.csv",
        "leitor": "xlsx",
        "dataset": "SP156/SIGRC, os 26 trimestres de 2020 a 2026",
        # `termos` vem de `aq_chamados02.PADROES_HIDRICOS` -- ver `_termos()`.
        # A lista literal que existia aqui era a ANTIGA, larga, e ela divergiu
        # da promocao no instante em que a promocao foi corrigida: este modulo
        # seguia contando pedido de limpeza de bueiro como caso.
        "subtipos": {
            "alagamento": ("alagamento",),
            "inundacao": ("inunda",),
        },
    },
}


def _aq02():
    sys.path.insert(0, str(REPO / "scripts" / "externo"))
    import aq_chamados02_verificar_requisitos as modulo

    return modulo


def _texto(d, fonte: str):
    """O texto de categoria pelo MESMO construtor da promocao.

    Ele junta as colunas com ` | ` e tira acento -- e os termos declarados
    dependem disso: `defesa civil | alagamento` so casa se o separador de
    coluna estiver la, e `urgencia` so casa se o til tiver saido.
    """
    aq02 = _aq02()
    return aq02._texto_das_categorias(d, aq02.ESQUEMAS[FONTES[fonte]["fonte_do_leitor"]])


def _casa(texto, termos):
    """Casamento com os termos ESCAPADOS: o `|` deles nao e alternancia."""
    return texto.str.contains(_aq02()._alternativa(termos), regex=True, na=False)


def _termos(fonte: str) -> tuple:
    """Os termos de ocorrencia, lidos do MESMO lugar que a promocao usa.

    Duplicar a lista aqui foi o que fez este modulo contar 183.662 casos em
    Sao Paulo enquanto a promocao contava 5.965: a duplicata nao acompanhou a
    separacao entre ocorrencia e manutencao.
    """
    sys.path.insert(0, str(REPO / "scripts" / "externo"))
    from aq_chamados02_verificar_requisitos import PADROES_HIDRICOS

    return tuple(PADROES_HIDRICOS[FONTES[fonte]["fonte_do_leitor"]])


def _formato_real(caminho: Path) -> str:
    """O que o arquivo E, e nao o que a extensao diz."""
    with caminho.open("rb") as f:
        assinatura = f.read(4)
    if assinatura[:2] == b"PK":
        return "zip/xlsx"
    return "texto"


def _coluna(d: pd.DataFrame, papel: str) -> str | None:
    for nome in ALIASES[papel]:
        if nome in d.columns:
            return nome
    return None


def _reduzir_para_casos(fonte: str):
    """Guarda so o que o autopareamento usa, POR ARQUIVO.

    Sao Paulo sao quinze milhoes de linhas em 26 arquivos, e o desenho precisa
    de tres coisas: onde, quando e de que tipo. Concatenar o bruto inteiro
    para depois jogar 98% fora nao cabe em memoria.
    """
    cfg = FONTES[fonte]

    def reduzir(d, _esq):
        col_data, col_lat = _coluna(d, "data"), _coluna(d, "lat")
        col_lon = _coluna(d, "lon")
        if not (col_data and col_lat and col_lon):
            return d.iloc[0:0]
        texto = _texto(d, fonte)
        hidrico = _casa(texto, _termos(fonte))
        # manutencao e papelada saem do conjunto de casos pelo mesmo criterio
        # da promocao: limpeza de bueiro nao e ocorrencia de alagamento
        hidrico = hidrico & ~_aq02().marcar_excluido(
            d, FONTES[fonte]["fonte_do_leitor"],
            _aq02().ESQUEMAS[FONTES[fonte]["fonte_do_leitor"]])
        z = pd.DataFrame({
            "lat": pd.to_numeric(d[col_lat], errors="coerce"),
            "lon": pd.to_numeric(d[col_lon], errors="coerce"),
            "ts": pd.to_datetime(d[col_data], errors="coerce", format="mixed"),
            "texto": texto.str.strip(),
        })[hidrico]
        return z[z.lat.notna() & z.lon.notna() & z.ts.notna()]

    return reduzir


def carregar_bruto(fonte: str) -> tuple[pd.DataFrame, dict]:
    """Os casos hidricos de TODOS os arquivos da fonte.

    A leitura e delegada a `aq_chamados02.carregar`, que resolve cabecalho,
    formato real e dialeto -- os tres defeitos que faziam arquivo inteiro
    sumir em silencio. Manter um leitor proprio aqui era duplicata, e a
    duplicata divergiu: este modulo lia um trimestre de Sao Paulo enquanto a
    promocao ja lia os vinte e seis.
    """
    sys.path.insert(0, str(REPO / "scripts" / "externo"))
    import aq_chamados02_verificar_requisitos as aq02

    cfg = FONTES[fonte]
    chave = cfg["fonte_do_leitor"]
    registro: list = []
    d = aq02.carregar(chave, registro=registro,
                      reduzir=_reduzir_para_casos(fonte))
    if d is None or d.empty:
        raise FileNotFoundError(
            f"nenhum caso hidrico legivel para {fonte} em "
            f"{aq02.DADOS / chave}")
    proc = {
        "fonte_do_leitor": chave,
        "pasta": str((aq02.DADOS / chave).relative_to(REPO)),
        "arquivos_lidos": sum(1 for x in registro if x.get("ok")),
        "arquivos_com_falha": [x for x in registro if not x.get("ok")],
        "linhas_lidas": sum(x.get("linhas", 0) for x in registro),
        "casos_hidricos_apos_reducao": int(len(d)),
        "por_arquivo": [{k: x.get(k) for k in
                         ("arquivo", "lido_como", "linhas",
                          "linhas_apos_reducao")} for x in registro],
        "sem_limite_de_linhas": True,
        "dataset": cfg["dataset"],
    }
    return d, proc


def normalizar(d: pd.DataFrame, fonte: str) -> pd.DataFrame:
    """Casos hidricos com dia, lugar e subtipo -- so isso, e nada de classe 0.

    Aceita tanto o quadro ja reduzido por `_reduzir_para_casos` (colunas
    `lat/lon/ts/texto`) quanto um quadro bruto com cabecalho de qualquer um dos
    dois dialetos -- e o que permite testar a resolucao de cabecalho sem
    carregar arquivo nenhum.
    """
    cfg = FONTES[fonte]
    if {"lat", "lon", "ts", "texto"}.issubset(d.columns):
        z = d[["lat", "lon", "ts", "texto"]].copy()
    else:
        col_data, col_lat = _coluna(d, "data"), _coluna(d, "lat")
        col_lon, col_cat = _coluna(d, "lon"), _coluna(d, "categoria")
        col_det = _coluna(d, "detalhe")
        faltando = [p for p, c in (("data", col_data), ("lat", col_lat),
                                   ("lon", col_lon)) if c is None]
        if faltando:
            raise ValueError(
                f"{fonte}: colunas ausentes para {faltando}; cabecalho lido: "
                f"{list(d.columns)[:8]}")
        texto = _texto(d, fonte)
        hidrico = _casa(texto, _termos(fonte))
        z = pd.DataFrame({
            "lat": pd.to_numeric(d[col_lat], errors="coerce"),
            "lon": pd.to_numeric(d[col_lon], errors="coerce"),
            "ts": pd.to_datetime(d[col_data], errors="coerce", format="mixed"),
            "texto": texto.str.strip(),
        })[hidrico]
    z = z[z.lat.notna() & z.lon.notna() & z.ts.notna()].copy()
    # DATA NO FUTURO NAO E OCORRENCIA. A regra vive em `aq_chamados02` para
    # nao divergir entre este caminho e o da promocao -- ela ja divergiu uma
    # vez, e o efeito foi vinte das 58 celulas de Sao Paulo ficarem sem serie
    # de chuva, entre elas as centrais.
    sys.path.insert(0, str(REPO / "scripts" / "externo"))
    from aq_chamados02_verificar_requisitos import no_futuro

    futuras = no_futuro(z.ts)
    n_futuras = int(futuras.sum())
    if n_futuras:
        z = z[~futuras].copy()
    z.attrs["casos_com_data_no_futuro"] = n_futuras
    z["dia"] = z.ts.dt.strftime("%Y-%m-%d")
    z["subtipo"] = "outro"
    for nome, termos in cfg["subtipos"].items():
        z.loc[z.texto.str.contains("|".join(termos), regex=True, na=False),
              "subtipo"] = nome
    z["fonte"] = fonte
    r = SPATIAL_RESOLUTION_DEGREES
    z["cl_lat"] = (z.lat / r).round() * r
    z["cl_lon"] = (z.lon / r).round() * r
    return z[["fonte", "cl_lat", "cl_lon", "dia", "subtipo", "lat", "lon"]]


def por_celula_dia(casos: pd.DataFrame) -> pd.DataFrame:
    """Casos por (celula do produto, dia) -- a unidade que a chuva resolve.

    Mais fino que isso seria pseudo-replicacao com relacao ao forcing: dentro
    de uma celula-dia o produto devolve UM numero, e dois chamados a 200 m um
    do outro tem a mesma exposicao por construcao.
    """
    largo = (casos.groupby(["fonte", "cl_lat", "cl_lon", "dia", "subtipo"])
             .size().unstack("subtipo", fill_value=0))
    largo.columns = [f"casos_{c}" for c in largo.columns]
    total = (casos.groupby(["fonte", "cl_lat", "cl_lon", "dia"]).size()
             .rename("casos"))
    return pd.concat([total, largo], axis=1).reset_index()


def celulas_e_janela(cd: pd.DataFrame) -> pd.DataFrame:
    """Celulas e janela util, no formato que `serie_por_celula.construir` espera."""
    return (cd.groupby(["fonte", "cl_lat", "cl_lon"], as_index=False)
            .agg(pontos=("casos", "sum"), positivos=("casos", "sum"),
                 primeiro_dia=("dia", "min"), ultimo_dia=("dia", "max"))
            .sort_values(["fonte", "cl_lat", "cl_lon"]).reset_index(drop=True))


# ------------------------------------------------- auditoria da aquisicao

def auditar_janelas(fonte_regiao: str = "nyc") -> dict:
    """As duas classes da tabela unica cobrem a MESMA janela de dias?

    Um preditor que so conhece o numero do dia nao tem fisica nenhuma. Se ele
    separa as classes, entao qualquer variavel que varie no tempo separa
    tambem -- e o ganho medido nao pode ser lido como efeito da variavel.
    """
    if not TABELA.exists():
        return {"auditado": False, "motivo": "tabela unica ausente"}
    from sklearn.metrics import roc_auc_score

    d = pd.read_csv(TABELA, low_memory=False)
    s = d[(d.fonte == fonte_regiao) & d.classe.isin([0, 1])].copy()
    if s.empty:
        return {"auditado": False, "motivo": f"fonte {fonte_regiao} ausente"}
    s["dia"] = pd.to_datetime(s.data_evento, errors="coerce",
                              format="mixed", utc=True).dt.normalize()
    s = s[s.dia.notna()]
    dias_pos = set(s.loc[s.classe == 1, "dia"])
    dias_neg = set(s.loc[s.classe == 0, "dia"])
    n_pos = int(s.classe.eq(1).sum())
    n_neg = int(s.classe.eq(0).sum())
    pos_fora = int((s.classe.eq(1) & ~s.dia.isin(dias_neg)).sum())
    neg_fora = int((s.classe.eq(0) & ~s.dia.isin(dias_pos)).sum())
    ordinal = (s.dia - s.dia.min()).dt.days.astype(float)
    # O sinal e escolhido pela direcao observada; o que interessa e a distancia
    # de 0,5, e nao o lado.
    auc_data = float(roc_auc_score(s.classe, ordinal))
    auc_data = max(auc_data, 1.0 - auc_data)
    return {
        "auditado": True,
        "fonte": fonte_regiao,
        "n_positivos": n_pos, "n_negativos": n_neg,
        "dias_com_positivo": len(dias_pos), "dias_com_negativo": len(dias_neg),
        "dias_com_as_duas_classes": len(dias_pos & dias_neg),
        "positivos_em_dia_sem_negativo": pos_fora,
        "pct_positivos_em_dia_sem_negativo": round(100.0 * pos_fora / max(n_pos, 1), 1),
        "negativos_em_dia_sem_positivo": neg_fora,
        "pct_negativos_em_dia_sem_positivo": round(100.0 * neg_fora / max(n_neg, 1), 1),
        "auc_de_um_preditor_que_so_conhece_o_dia": round(auc_data, 4),
        "leitura": (
            "um preditor sem fisica nenhuma separa as classes: qualquer "
            "variavel que varie no tempo herda esse poder, e ganho de AUC "
            "nesta base nao sustenta leitura causal"
            if auc_data >= 0.7 else
            "as janelas se sobrepoem o bastante para que a data sozinha nao "
            "separe as classes"),
        "consequencia": (
            "o desenho autopareado nao usa a classe negativa e por isso nao "
            "herda este defeito; leituras comparando classes nesta base ficam "
            "condicionadas a correcao da janela de aquisicao"),
        "cobertura_do_rotulo": cobertura_do_rotulo(fonte_regiao, n_pos),
    }


def cobertura_do_rotulo(fonte: str, positivos_na_tabela: int) -> dict:
    """Quantos casos existem no bruto, e quantos chegaram ao modelo servido.

    A pergunta nao e academica: se o bruto tem quarenta vezes mais caso do que
    a tabela, o modelo servido foi ajustado numa fracao do rotulo disponivel --
    e a fracao nao foi escolhida, foi o que sobrou de uma falha de leitura.
    """
    cfg = FONTES.get(fonte)
    if cfg is None:
        return {"medida": False, "motivo": f"fonte {fonte} nao registrada"}
    caminho = Path(cfg["arquivo"])
    agregado = OUT / f"casos_celula_dia_{fonte}.csv"
    if not agregado.exists():
        return {"medida": False,
                "motivo": (f"rode `casos_das_centrais.py --fonte {fonte}` para "
                           "contar os casos do bruto"),
                "arquivo_bruto": str(caminho.relative_to(REPO))
                if caminho.exists() else None}
    cd = pd.read_csv(agregado)
    disponiveis = int(cd.casos.sum())
    return {
        "medida": True,
        "casos_no_bruto_com_lugar_e_dia": disponiveis,
        "positivos_na_tabela_unica": positivos_na_tabela,
        "fracao_usada": round(positivos_na_tabela / max(disponiveis, 1), 4),
        "janela_do_bruto": [cd.dia.min(), cd.dia.max()],
        "dias_no_bruto": int(cd.dia.nunique()),
        "leitura": (
            "a tabela unica usa uma fracao do rotulo disponivel; a fracao nao "
            "foi escolhida por criterio cientifico, e o que sobrou da leitura "
            "do bruto"
            if positivos_na_tabela < 0.5 * disponiveis else
            "a tabela unica cobre a maior parte do rotulo disponivel"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="nyc", choices=sorted(FONTES))
    ap.add_argument("--auditar", action="store_true",
                    help="so a auditoria de janelas, sem ler o bruto")
    ns = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    auditoria = {f: auditar_janelas(f) for f in sorted(FONTES)}
    if ns.auditar:
        (OUT / "auditoria_janelas.json").write_text(
            json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")
        for f, a in auditoria.items():
            if a.get("auditado"):
                print(f"{f:10s} pos_sem_negativo_no_dia="
                      f"{a['pct_positivos_em_dia_sem_negativo']:5.1f}%  "
                      f"auc_so_com_a_data={a['auc_de_um_preditor_que_so_conhece_o_dia']}")
        print(f"GRAVADO={OUT / 'auditoria_janelas.json'}")
        return 0

    bruto, proc = carregar_bruto(ns.fonte)
    print(f"{ns.fonte}: {proc['linhas_lidas']:,} linhas em "
          f"{proc['arquivos_lidos']} arquivo(s) de {proc['pasta']}"
          + (f" · {len(proc['arquivos_com_falha'])} ilegivel(is)"
             if proc["arquivos_com_falha"] else ""))
    casos = normalizar(bruto, ns.fonte)
    cd = por_celula_dia(casos)
    print(f"casos hidricos com lugar e dia: {len(casos):,}")
    print(f"celulas-dia com caso: {len(cd):,} em "
          f"{cd.groupby(['cl_lat', 'cl_lon']).ngroups} celulas, "
          f"{cd.dia.nunique():,} dias distintos")
    print(casos.subtipo.value_counts().to_string())

    destino = OUT / f"casos_celula_dia_{ns.fonte}.csv"
    cd.sort_values(["cl_lat", "cl_lon", "dia"]).to_csv(destino, index=False)
    manifesto = {
        "artefato": "chuva_casos_das_centrais",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": ns.fonte,
        "procedencia": proc,
        "termos_hidricos": list(_termos(ns.fonte)),
        "subtipos": {k: list(v) for k, v in FONTES[ns.fonte]["subtipos"].items()},
        "casos": int(len(casos)),
        "casos_descartados_por_data_no_futuro": int(
            casos.attrs.get("casos_com_data_no_futuro", 0)),
        "por_que_descartar_data_futura": (
            "ocorrencia datada depois de hoje nao e ocorrencia, e a data "
            "futura ainda estoura a janela de aquisicao da chuva: o provedor "
            "devolve 400 e a celula inteira fica sem serie"),
        "celulas_dia": int(len(cd)),
        "celulas": int(cd.groupby(["cl_lat", "cl_lon"]).ngroups),
        "dias_distintos": int(cd.dia.nunique()),
        "primeiro_dia": cd.dia.min(), "ultimo_dia": cd.dia.max(),
        "por_subtipo": casos.subtipo.value_counts().to_dict(),
        "auditoria_de_janelas": auditoria,
        "nao_e": ("nao e rotulo de referencia, nao promove fonte e nao escreve "
                  "no ds01: e o conjunto de CASOS datados do autopareamento"),
        "sem_classe_negativa": (
            "o desenho autopareado compara o caso consigo mesmo em outras "
            "datas; carregar fundo traria para dentro o viés que ele cancela"),
        "saida": str(destino.relative_to(REPO)),
        "sha256_saida": hashlib.sha256(destino.read_bytes()).hexdigest(),
    }
    (OUT / f"manifesto_{ns.fonte}.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MANIFESTO={OUT / f'manifesto_{ns.fonte}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
