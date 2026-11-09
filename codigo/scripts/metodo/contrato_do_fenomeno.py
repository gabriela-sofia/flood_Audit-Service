"""Contrato do fenomeno: o que o REV-P chama de inundacao, e o que cada observacao ensina.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

O projeto deriva terreno para centenas de milhares de pontos e treina sobre
eles. O terreno descreve CONDICOES; o que transforma condicao em conhecimento
e a ligacao com uma observacao que diga se houve agua ali, durante qual
evento, vista por qual instrumento. Sem um contrato explicito dessa ligacao,
tres erros ja aconteceram sem disparar excecao:

  1. ausencia de registro lida como negativo (o corte prospectivo de
     Curitiba usava chamado de outro assunto como "nao inundou");
  2. movimento de massa dentro do pool de inundacao (1.592 linhas do CEMS
     elegiveis ao mesmo ajuste que cheia de rio);
  3. o mesmo tipo de instrumento rotulado com mecanismos diferentes (central
     156 de Sao Paulo e PLUVIAL_URBANO com justificativa; a de Curitiba e
     FLUVIAL_ENXURRADA sem nenhuma).

Este modulo declara, uma vez:

  MECANISMOS    o que cada classe de fenomeno e, o que a gera, quem consegue
                observa-la e que sinal fisico se espera de cada variavel
  PAPEIS        o que cada linha ensina (vem do ds03: positivo, negativo
                observado, negativo por exclusao, desconhecido, agua
                permanente, outro fenomeno)
  INSTRUMENTOS  por fonte: o que observa, o que NAO ve, que negativo produz,
                como data, se distingue o mecanismo que lhe foi atribuido

e audita a tabela unica contra isso. A auditoria nao conserta nada: diz onde
a tabela contradiz o contrato, quanto, e o que cada mecanismo pode aprender.

-------------------------------------------------------------------------
O QUE ISTO NAO E
-------------------------------------------------------------------------

Nao e resultado de treino nem prova de causa. As hipoteses de sinal sao o
que a fisica PREVE; se os dados as sustentam e outra pergunta, respondida
pelo ciclo de contexto e pela generalizacao protegida.

Uso:
    python scripts/metodo/contrato_do_fenomeno.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
sys.path.insert(0, str(REPO / "scripts" / "servico"))

import ds03_esquema_alvo as ds03  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
SAIDA = REPO / "local_runs" / "contrato-do-fenomeno"
DOC = REPO / "docs" / "CONTRATO_DO_FENOMENO.md"

# Grupos independentes por classe para um mecanismo ser declarado aprendivel.
# E o EPV_MINIMO do svc01: o contrato nao pode ser mais frouxo que o ajuste.
MINIMO_GRUPOS = 10

COLUNAS_LIDAS = ["fonte", "mecanismo", "classe", "nivel_negativo", "classe_relevo",
                 "precisao_data", "grupo_cv", "admitido", "elegivel_pool_fluvial"]

# --------------------------------------------------------------------------
# Mecanismos. `hipoteses` e o sinal que a fisica preve para a associacao
# MARGINAL da variavel padronizada, dentro de um contexto homogeneo
# (contexto x fonte); None quer dizer que nao ha previsao defensavel, e nao
# que o sinal seja nulo.
#
# `condicional` e o que a fisica preve com as outras variaveis no modelo. So
# o HAND tem previsao ali: TWI, declive e elevacao tem correlacao de 0,7 a
# 0,8 com ele, e o sinal condicional dessas pode trocar por supressao sem que
# o fenomeno mude. Em 02/10/2026 as tres "inversoes" medidas eram isso, ou
# mistura de fontes, ou referencia de elevacao errada (`sinais_marginais`).
# --------------------------------------------------------------------------
MECANISMOS = {
    "FLUVIAL_ENXURRADA": {
        "definicao": ("agua que sai do canal e ocupa a planicie ou o fundo de "
                      "vale, inclusive a cheia rapida de enxurrada"),
        "processo_gerador": "vazao do rio excede a calha",
        "observavel_por": ["sensor areal optico ou SAR", "outline oficial de cheia"],
        "hipoteses": {
            "hand_m": (-1, "quanto mais acima da drenagem, mais agua precisa subir"),
            "twi_dinf": (+1, "area que acumula fluxo e umidade topografica"),
            "slope_deg": (-1, "encosta ingreme escoa em vez de reter"),
            "elevation_local_m": (-1, "posicao baixa em relacao ao piso do proprio lugar"),
            # O piso da FONTE e nacional no GFD: entre bacias a variavel mede em
            # que bacia o ponto esta, e nao onde ele fica dentro dela.
            "elevation_rel_m": (None, "piso da fonte mistura bacias de nivel de base diferente"),
            # Cobertura e solo nao condicionam ate onde o rio chega: a cheia
            # ocupa a planicie esteja ela construida ou nao. Sem previsao,
            # mede-se so a direcao -- declarar sinal depois de ver o dado
            # seria ajustar a hipotese ao resultado.
            "construido_frac": (None, "ocupacao humana nao condiciona o alcance do rio"),
            "argila_pct_0_30": (None, "textura governa infiltracao, nao o alcance da cheia de rio"),
        },
        "condicional": {
            "hand_m": (-1, "variavel primaria: a distancia vertical a drenagem nao e "
                           "explicada pelas outras"),
        },
        "forcamento": "chuva e vazao do evento; nao sao condicionantes do lugar",
    },
    "PLUVIAL_URBANO": {
        "definicao": ("agua da chuva que excede a drenagem urbana e acumula na "
                      "rua, sem que o canal transborde"),
        "processo_gerador": "intensidade da chuva excede a capacidade da rede",
        "observavel_por": ["registro de ocorrencia (central 156/311, defesa civil)"],
        "hipoteses": {
            "twi_dinf": (+1, "acumulacao local de escoamento superficial"),
            "slope_deg": (-1, "ponto plano retem a lamina d'agua"),
            # A drenagem que importa aqui e a rede de galerias, que o DEM nao
            # ve: o canal derivado do terreno nao e a referencia do processo.
            "hand_m": (None, "rede de galerias nao e a drenagem derivada do DEM"),
            "construido_frac": (+1, "superficie impermeavel transforma chuva em escoamento"),
            "argila_pct_0_30": (+1, "solo argiloso infiltra menos e escoa mais"),
        },
        "condicional": {},
        "forcamento": "intensidade da chuva em minutos a horas",
    },
    "MISTO_NAO_SEPARAVEL": {
        "definicao": "o instrumento ve agua e nao separa de onde ela veio",
        "processo_gerador": "nao identificavel pela fonte",
        "observavel_por": ["imagem urbana rotulada sem atribuicao de processo"],
        "hipoteses": {},
        "condicional": {},
        "forcamento": "nao separavel",
    },
    "NAO_CLASSIFICADO": {
        "definicao": "mecanismo nao declarado pela reducao",
        "processo_gerador": "desconhecido",
        "observavel_por": [],
        "hipoteses": {},
        "condicional": {},
        "forcamento": "desconhecido",
    },
}

# --------------------------------------------------------------------------
# Instrumentos. Um por fonte da tabela unica. `mecanismo_justificado` e falso
# quando nada no instrumento separa o mecanismo atribuido de outro: o rotulo
# pode ate estar certo, mas a fonte nao o sustenta sozinha.
# --------------------------------------------------------------------------
INSTRUMENTOS = {
    "gfd": {
        "instrumento": "Global Flood Database: MODIS optico, extensao maxima na janela",
        "mecanismo": "FLUVIAL_ENXURRADA", "mecanismo_justificado": True,
        "por_que": "cheia areal de rio; MODIS nao ve alagamento raso urbano",
        "negativos": ("observado",), "datas": ("janela_do_evento",),
        "separa_agua_permanente": True,
        # Medido em 03/10/2026: no GFD de serra o "inundado" tinha declive
        # mediano 5,7 graus contra 3,3 do seco (nos Andes, 14 contra 6), o
        # unico caso entre fontes e relevos em que o positivo e mais ingreme.
        # CEMS a 30 m e Sen1Floods11 a 10 m, no mesmo relevo, dao 4,3 contra
        # 16,1 e 2,8 contra 19,4. O pixel de 250 m mistura fundo de vale e
        # encosta; o ponto de 30 m herda a encosta.
        "relevos_resolvidos": ds03.RELEVOS_RESOLVIDOS["gfd"],
        "nao_ve": ("nuvem persistente, agua sob dossel, lamina urbana rasa e vale de "
                   "serra: o pixel de 250 m mistura fundo de vale e encosta"),
    },
    "cems": {
        "instrumento": "Copernicus EMS: mapeamento de ativacao por SAR ou optico",
        "mecanismo": "FLUVIAL_ENXURRADA", "mecanismo_justificado": True,
        "por_que": "delineacao de cheia areal em ativacao de emergencia",
        "negativos": ("observado",), "datas": ("ativacao_cems", "cena_do_produto_ems"),
        "separa_agua_permanente": True,
        "nao_ve": ("o instante da ocorrencia: a data e a da ativacao ou a da cena que o "
                   "analista desenhou, nunca o pico da cheia"),
    },
    "sen1floods11": {
        "instrumento": "Sen1Floods11: chips Sentinel-1/2 rotulados a mao",
        "mecanismo": "FLUVIAL_ENXURRADA", "mecanismo_justificado": True,
        "por_que": "eventos de cheia selecionados pelo produtor do dataset",
        "negativos": ("observado",), "datas": ("chip_stac",),
        "separa_agua_permanente": True,
        "nao_ve": "pixel sem rotulo (classe -1) fica desconhecido",
    },
    "uk": {
        "instrumento": "Environment Agency: outlines de cheia registrados",
        "mecanismo": "FLUVIAL_ENXURRADA", "mecanismo_justificado": True,
        "por_que": "outline oficial de cheia de rio",
        "negativos": ("exclusao_qualificada",), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "o lado de fora do outline nao foi vistoriado: negativo inferido",
    },
    "ufo": {
        "instrumento": "UFO: observacoes urbanas de agua em imagem de alta resolucao",
        "mecanismo": "MISTO_NAO_SEPARAVEL", "mecanismo_justificado": True,
        "por_que": "ve agua na cidade e nao diz se veio do rio ou da chuva",
        "negativos": ("observado",), "datas": ("nome_do_chip",),
        "separa_agua_permanente": False,
        "nao_ve": "o processo gerador",
    },
    "curitiba": {
        "instrumento": "SIAC 156 de Curitiba: ocorrencias hidrologicas registradas",
        # Era FLUVIAL_ENXURRADA ate 03/10/2026. Nada no registro separa rio de
        # galeria, e o dado confirmou: HAND marginal nulo (-0,03, IC cruza
        # zero) e inundado e seco com o mesmo terreno (HAND 5,7 contra 5,9 m).
        "mecanismo": "MISTO_NAO_SEPARAVEL", "mecanismo_justificado": True,
        "por_que": ("o positivo e 'ocorrencia hidrologica' generica: nada no registro "
                    "separa rio de galeria, ao contrario de SP/NYC/Chicago"),
        "negativos": ("ausencia", "exclusao_qualificada"), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "onde nao houve chamado, ninguem foi ver",
    },
    "recife": {
        "instrumento": "SEDEC Recife e base v12: pontos de alagamento",
        "mecanismo": "PLUVIAL_URBANO", "mecanismo_justificado": True,
        "por_que": "alagamento urbano registrado pela defesa civil",
        "negativos": ("ausencia", "exclusao_qualificada"), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "onde nao houve registro, ninguem foi ver",
    },
    "sao_paulo": {
        "instrumento": "SP156: chamados por categoria hidrica",
        "mecanismo": "PLUVIAL_URBANO", "mecanismo_justificado": True,
        "por_que": "categoria do chamado nomeia alagamento e boca de lobo",
        "negativos": ("ausencia",), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "chamado de outro assunto nao e vistoria sem agua",
    },
    "nyc": {
        "instrumento": "NYC311: street flooding e sewer backup",
        "mecanismo": "PLUVIAL_URBANO", "mecanismo_justificado": True,
        "por_que": "categoria do chamado nomeia o processo",
        "negativos": ("ausencia",), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "chamado de outro assunto nao e vistoria sem agua",
    },
    "chicago": {
        "instrumento": "Chicago 311: reclamacoes de agua",
        "mecanismo": "PLUVIAL_URBANO", "mecanismo_justificado": True,
        "por_que": "categoria do chamado nomeia o processo",
        "negativos": ("ausencia",), "datas": ("nao_declarada",),
        "separa_agua_permanente": False,
        "nao_ve": "chamado de outro assunto nao e vistoria sem agua",
    },
}


def relevo_resolvido(fonte: str, relevo: str) -> bool:
    """O instrumento consegue observar agua neste relevo, na escala dele? (ds03)"""
    return ds03.relevo_resolvido(fonte, relevo)


def _violacao(regra: str, mascara: pd.Series, d: pd.DataFrame, grave: bool,
              explicacao: str) -> dict | None:
    n = int(mascara.sum())
    if not n:
        return None
    return {"regra": regra, "grave": grave, "linhas": n,
            "por_fonte": {str(k): int(v) for k, v in
                          d.loc[mascara, "fonte"].value_counts().items()},
            "explicacao": explicacao}


def auditar(d: pd.DataFrame) -> dict:
    """Confronta a tabela com o contrato. Nao filtra nem corrige linha."""
    d = d.copy()
    d["papel"] = ds03.papeis_evidenciais(d)
    declarada = d["fonte"].isin(INSTRUMENTOS)

    def do_instrumento(campo: str) -> pd.Series:
        return d["fonte"].map(lambda f: INSTRUMENTOS.get(f, {}).get(campo))

    def fora(campo: str, coluna: str) -> pd.Series:
        aceitos = do_instrumento(campo)
        return declarada & pd.Series(
            [v not in (a or ()) for v, a in zip(d[coluna], aceitos)], index=d.index)

    negativos = d["classe"] == 0
    violacoes = [
        _violacao("fonte_sem_instrumento", ~declarada, d, True,
                  "fonte sem instrumento declarado: nao se sabe o que ela observa"),
        _violacao("mecanismo_diverge_do_instrumento",
                  declarada & (d["mecanismo"] != do_instrumento("mecanismo")), d, True,
                  "a linha declara um mecanismo que o instrumento nao observa"),
        _violacao("classe_e_nivel_incoerentes", d["papel"] == "incoerente", d, True,
                  "classe e nivel de negativo se contradizem"),
        _violacao("negativo_que_o_instrumento_nao_produz",
                  negativos & fora("negativos", "nivel_negativo"), d, True,
                  "nivel de negativo diferente do que o instrumento consegue dar"),
        _violacao("data_com_procedencia_nao_declarada",
                  fora("datas", "precisao_data"), d, False,
                  "precisao da data diferente da que o instrumento declara"),
        _violacao("pool_fluvial_com_evidencia_que_nao_ensina",
                  d["elegivel_pool_fluvial"].astype(bool)
                  & ~d["papel"].isin(ds03.PAPEIS_QUE_ENSINAM), d, True,
                  "linha elegivel ao ajuste de inundacao sem ensinar inundacao"),
        _violacao("relevo_que_o_instrumento_nao_resolve",
                  d["elegivel_pool_fluvial"].astype(bool) & pd.Series(
                      [not relevo_resolvido(f, r) for f, r in zip(d["fonte"], d["classe_relevo"])]
                      if "classe_relevo" in d else [False] * len(d), index=d.index), d, False,
                  "escala do instrumento grossa demais para o relevo: o ponto herda a encosta"),
        _violacao("mecanismo_sem_justificativa_do_instrumento",
                  declarada & (do_instrumento("mecanismo_justificado") == False),  # noqa: E712
                  d, False,
                  "o instrumento nao separa o mecanismo atribuido de outro"),
    ]
    violacoes = [v for v in violacoes if v]

    mecanismos = {}
    for mec, s in d.groupby("mecanismo"):
        adm = s[s["admitido"].astype(bool)]
        por_papel = {p: int((s["papel"] == p).sum()) for p in ds03.PAPEL_EVIDENCIAL}
        grupos = {p: int(adm.loc[adm["papel"] == p, "grupo_cv"].nunique())
                  for p in ("positivo_episodico", "negativo_observado",
                            "negativo_por_exclusao")}
        basta = {p: n >= MINIMO_GRUPOS for p, n in grupos.items()}
        if basta["positivo_episodico"] and basta["negativo_observado"]:
            estado, falta = "aprendivel", None
        elif basta["positivo_episodico"] and basta["negativo_por_exclusao"]:
            estado, falta = "aprendivel_com_ressalva", (
                "negativo observado durante o evento; so ha negativo inferido por exclusao")
        elif basta["positivo_episodico"]:
            estado, falta = "not_testable", (
                f"negativo observado em ao menos {MINIMO_GRUPOS} grupos: area "
                "vistoriada durante o evento, sem agua")
        else:
            estado, falta = "not_testable", (
                f"positivo observado em ao menos {MINIMO_GRUPOS} grupos")
        mecanismos[mec] = {
            "linhas_por_papel": por_papel,
            "grupos_admitidos_por_papel": grupos,
            "fontes": sorted(s["fonte"].unique().tolist()),
            "estado": estado, "o_que_falta": falta,
            "hipoteses": {v: {"sinal_esperado": h[0], "por_que": h[1]}
                          for v, h in MECANISMOS.get(mec, {}).get("hipoteses", {}).items()},
            "hipoteses_condicionais": {
                v: {"sinal_esperado": h[0], "por_que": h[1]}
                for v, h in MECANISMOS.get(mec, {}).get("condicional", {}).items()},
        }

    por_fonte = {
        f: {p: int(n) for p, n in s["papel"].value_counts().items()}
        for f, s in d.groupby("fonte")}
    corpo = {
        "tipo": "auditoria_contrato_do_fenomeno",
        "papeis": {k: {"significado": v[0], "ensina": v[1]}
                   for k, v in ds03.PAPEL_EVIDENCIAL.items()},
        "mecanismos": mecanismos,
        "instrumentos": INSTRUMENTOS,
        # Versao das DEFINICOES e HIPOTESES, separada do estado da tabela:
        # muda so quando muda o que o projeto afirma esperar, e quem mede
        # (relacoes_sustentadas) carrega esta versao no registro.
        "versao_das_hipoteses": versao_das_hipoteses(),
        "papel_por_fonte": por_fonte,
        "violacoes": violacoes,
        "graves": sorted(v["regra"] for v in violacoes if v["grave"]),
        "linhas": int(len(d)),
        "nao_e": ("nao e treino, validacao nem prova de causa: diz o que cada "
                  "observacao pode ensinar e onde a tabela contradiz isso"),
    }
    corpo["hash_resumo"] = sha256_json(corpo)
    return corpo


def _n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def markdown(r: dict, origem: str) -> str:
    L = ["# Contrato do fenômeno", "",
         f"**Gerado de** `scripts/metodo/contrato_do_fenomeno.py` em {r['quando_utc']}, "
         f"sobre `{origem}` ({_n(r['linhas'])} linhas). **Não editar à mão.**",
         "", f"**Não é.** {r['nao_e']}.", "",
         "## O que cada observação ensina", "",
         "| papel | significado | ensina inundação |", "|---|---|---|"]
    L += [f"| `{k}` | {v['significado']} | {v['ensina']} |" for k, v in r["papeis"].items()]
    L += ["", "## Mecanismos", ""]
    for mec, m in r["mecanismos"].items():
        dec = MECANISMOS.get(mec, {})
        L += [f"### {mec} — `{m['estado']}`", "",
              f"{dec.get('definicao', '')}. Gerado por: {dec.get('processo_gerador', '')}. "
              f"Forçamento: {dec.get('forcamento', '')}.", "",
              f"Fontes: {', '.join(m['fontes'])}.", ""]
        g = m["grupos_admitidos_por_papel"]
        L += [f"Grupos admitidos — positivos {g['positivo_episodico']}, "
              f"negativos observados {g['negativo_observado']}, "
              f"negativos por exclusão {g['negativo_por_exclusao']}."]
        if m["o_que_falta"]:
            L += ["", f"**Falta:** {m['o_que_falta']}."]
        if m["hipoteses"]:
            L += ["", "| variável | sinal marginal esperado | por quê |", "|---|---:|---|"]
            for v, h in m["hipoteses"].items():
                s = {1: "+", -1: "−", None: "sem previsão"}[h["sinal_esperado"]]
                L.append(f"| `{v}` | {s} | {h['por_que']} |")
            for v, h in m["hipoteses_condicionais"].items():
                L += ["", f"Com as outras variáveis no modelo, só `{v}` tem sinal previsto "
                      f"({'−' if h['sinal_esperado'] < 0 else '+'}): {h['por_que']}. "
                      "TWI, declive e elevação são colineares com ele, e o sinal condicional "
                      "deles pode trocar por supressão."]
        L.append("")
    L += ["## Instrumentos", "",
          "| fonte | instrumento | mecanismo | justificado | negativo | não vê |",
          "|---|---|---|---|---|---|"]
    for f, i in INSTRUMENTOS.items():
        L.append(f"| `{f}` | {i['instrumento']} | {i['mecanismo']} | "
                 f"{'sim' if i['mecanismo_justificado'] else '**não**'} | "
                 f"{', '.join(i['negativos'])} | {i['nao_ve']} |")
    L += ["", "## Onde a tabela contradiz o contrato", ""]
    if not r["violacoes"]:
        L.append("Nenhuma violação.")
    for v in r["violacoes"]:
        fontes = ", ".join(f"{k} {_n(n)}" for k, n in v["por_fonte"].items())
        L.append(f"- **{v['regra']}**{' (grave)' if v['grave'] else ''}: "
                 f"{_n(v['linhas'])} linhas — {v['explicacao']}. Por fonte: {fontes}.")
    L += ["", f"`{r['hash_resumo']}`", ""]
    return "\n".join(L)


def versao_das_hipoteses() -> str:
    """Hash das definicoes, hipoteses marginais e condicionais de cada mecanismo."""
    return sha256_json({m: {k: v.get(k) for k in ("definicao", "processo_gerador", "hipoteses",
                                                  "condicional", "forcamento")}
                        for m, v in MECANISMOS.items()})


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "metodo"))
    import rastreio_de_impacto as _ri
    _entrada, _saida = _ri.ganchos("contrato", "contrato_do_fenomeno")
    _entrada()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tabela", type=Path, default=TABELA)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--doc", type=Path, default=DOC)
    args = ap.parse_args()
    if not args.tabela.exists():
        ap.error(f"tabela unica ausente: {args.tabela}")
    d = pd.read_parquet(args.tabela, columns=COLUNAS_LIDAS)
    r = auditar(d)
    r["quando_utc"] = agora_utc()
    r["tabela"] = args.tabela.relative_to(REPO).as_posix() if args.tabela.is_relative_to(REPO) else str(args.tabela)
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "contrato.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc.write_text(markdown(r, r["tabela"]), encoding="utf-8")
    print(json.dumps({"graves": r["graves"],
                      "estados": {m: v["estado"] for m, v in r["mecanismos"].items()}},
                     ensure_ascii=False))
    _saida()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
