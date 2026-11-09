"""Admissao cientifica: para que cada observacao pode ser usada, e por que nao.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

Ate 03/10/2026 cada consumidor da tabela unica filtrava por conta propria, e
os filtros divergiam:

  `celula_para_treino`     classe 0/1, admitido, negativo do svc01 -- nao
                           olhava se o instrumento resolve o relevo (GFD de
                           serra entrava no treino de serra)
  `generalizacao_protegida` positivo ou negativo OBSERVADO, por conta propria
  `curva_de_aprendizado`   o que o ds06 entregasse
  `relacoes_sustentadas`   separa direto de registro, mas na leitura do card

O ds03 ja centralizou o PAPEL (o que a linha ensina). Faltava centralizar o
USO: uma linha pode ensinar inundacao e ainda assim nao servir para teste
temporal (sem data), para afirmacao (negativo inferido) ou para teste de
serra (instrumento grosso demais para o relevo).

-------------------------------------------------------------------------
CAPACIDADE DA FONTE NAO E VALIDADE DA OBSERVACAO
-------------------------------------------------------------------------

`CAPACIDADE` diz o que o INSTRUMENTO consegue observar: classes, negativo,
escala, relevo resolvido, data, metodo de delineacao e dependencias. Vale
para todas as linhas da fonte.

`validade(d)` diz, LINHA a LINHA, se aquela observacao cumpre o que o uso
exige: papel, classe, mecanismo, escala x relevo, terreno completo, grupo,
evento, data resolvida. Uma fonte capaz produz linha invalida (CEMS sem cena
no produto); uma fonte limitada produz linha valida para alguns usos (UK:
treino com ressalva, nunca afirmacao).

-------------------------------------------------------------------------
USOS
-------------------------------------------------------------------------

  exploracao          descrever, contar, mapear. Qualquer linha legivel.
                      Nunca sustenta conclusao.
  treino              ajustar coeficiente. Papel que ensina, classe 0/1,
                      admitida no ds05, terreno completo, grupo, mecanismo
                      justificado pelo instrumento, relevo resolvido.
  avaliacao_espacial  medir em grupo/territorio reservado. Treino + negativo
                      OBSERVADO (oportunidade de observacao comparavel).
  avaliacao_temporal  medir em evento/janela reservado. Espacial + evento
                      identificado + data com procedencia declarada. O DEM e
                      atual: a hipotese de estabilidade vai declarada no
                      resultado, nunca suposta comprovada.
  avaliacao_externa   medir em outro instrumento. Espacial + o instrumento
                      declara negativo observado. Quem e "outro" decide o
                      executor do corte; a linha so diz se pode ser alvo.
  afirmacao           sustentar relacao/claim. Espacial + negativo observado
                      pelo instrumento + sem dependencia circular declarada.

Ausencia de registro, falta de leitura e nodata nunca viram negativo: o papel
`desconhecido` bloqueia todos os usos menos exploracao. Agua permanente,
outro fenomeno e simulacao ficam com classe propria e fora do binario.

Uso:
    python scripts/metodo/admissao_cientifica.py        # grava o retrato
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "metodo"))

import ds03_esquema_alvo as ds03  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

VERSAO = "v1"
TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
SAIDA = REPO / "local_runs" / "admissao-cientifica"
DOC = REPO / "docs" / "ADMISSAO_CIENTIFICA.md"

USOS = ("exploracao", "treino", "avaliacao_espacial", "avaliacao_temporal",
        "avaliacao_externa", "afirmacao")
TERRENO = ("hand_m", "twi_dinf", "slope_deg", "elevation_m")
DATAS_RESOLVIDAS = tuple(p for p in ds03.PRECISAO_DATA if p != "nao_declarada")

# O uso temporal depende da PERGUNTA. A data de cena do CEMS e a data em que
# a imagem foi tomada -- observacao, nao inicio nem pico do evento; a janela
# do GFD e o inicio de uma extensao maxima acumulada. Servem para ordenar
# eventos no tempo; nao servem para alinhar forcamento ao instante da cheia.
PERGUNTAS_TEMPORAIS = {
    "ordem_entre_eventos": {
        "datas": DATAS_RESOLVIDAS,
        "o_que_e": "reservar eventos posteriores; a data so precisa ordenar eventos"},
    "forcamento_no_instante": {
        "datas": ("registro_local",),
        "o_que_e": ("ligar chuva/nivel ao instante da ocorrencia; exige data da propria "
                    "ocorrencia, e cena, ativacao ou inicio de janela nao sao")},
    "previsao": {
        "datas": DATAS_RESOLVIDAS,
        "o_que_e": ("estimar antes do evento: alem da data, todo preditor tem de existir antes "
                    "dele -- o DEM nao existia para evento anterior a sua publicacao")},
}
# FABDEM v1 (publicado em 2022) sobre o Copernicus GLO-30 (aberto em 2021,
# aquisicoes TanDEM-X 2010-2015). Para a pergunta `previsao`, evento anterior
# a isto usaria um preditor do futuro.
DEM_PUBLICADO_EM = "2022-01-01"

# Hipotese que todo uso temporal carrega. Nao e checagem: e o que o
# resultado precisa dizer, porque a data do evento nao torna o DEM historico.
HIPOTESE_DEM_ESTAVEL = ("terreno derivado de DEM atual (FABDEM/MERIT) aplicado a "
                        "evento antigo: supoe-se que corte, aterro e canalizacao "
                        "nao mudaram o relevo no ponto")

# --------------------------------------------------------------------------
# Capacidade de cada fonte. O que o contrato do fenomeno declara por
# instrumento, mais duas perguntas que ele nao fazia: como a mancha foi
# delineada e de que ela depende. Uma referencia delineada com HAND nao pode
# testar se HAND explica inundacao.
# --------------------------------------------------------------------------
CAPACIDADE = {
    "gfd": {"delineacao": "observada: limiar MODIS 250 m, extensao maxima na janela",
            "escala_m": 250, "dependencias": ["JRC para agua permanente"],
            "circular_com": []},
    "cems": {"delineacao": "observada: analista sobre SAR/optico, por AOI",
             "escala_m": 30, "dependencias": ["metodo por AOI nao declarado no pacote"],
             "circular_com": []},
    "sen1floods11": {"delineacao": "observada: rotulo manual sobre S1/S2",
                     "escala_m": 10, "dependencias": [], "circular_com": []},
    "uk": {"delineacao": "registro oficial de outline; fora dele nao vistoriado",
           "escala_m": None, "dependencias": [], "circular_com": []},
    "ufo": {"delineacao": "observada em imagem urbana; processo nao separavel",
            "escala_m": None, "dependencias": [], "circular_com": []},
    "curitiba": {"delineacao": "registro de chamado; nao delineia", "escala_m": None,
                 "dependencias": ["propensao a registrar"], "circular_com": []},
    "recife": {"delineacao": "registro de ponto de alagamento; nao delineia",
               "escala_m": None, "dependencias": ["propensao a registrar"], "circular_com": []},
    "sao_paulo": {"delineacao": "chamado 156; nao delineia", "escala_m": None,
                  "dependencias": ["propensao a registrar"], "circular_com": []},
    "nyc": {"delineacao": "chamado 311; nao delineia", "escala_m": None,
            "dependencias": ["propensao a registrar"], "circular_com": []},
    "chicago": {"delineacao": "chamado 311; nao delineia", "escala_m": None,
                "dependencias": ["propensao a registrar"], "circular_com": []},
}

# Fontes candidatas ja auditadas e RECUSADAS como referencia. Ficam aqui para
# que nenhuma aquisicao futura as admita sem reproduzir o motivo.
RECUSADAS = {
    "opera_dswx_hls_serra": (
        "03/10/2026: em serra brasileira, controle seco de ago/2024 deu 278 km2 de "
        "agua nova contra 103 km2 no evento de mai/2024; mascara de sombra nao muda"),
    "ufrgs_rs2024_observada": (
        "IoU 0,23-0,30 com a agua do S2 de 06/05/2024; trecho so-UFRGS tem HAND p90 "
        "15 m: extensao sobre o MDE, nao delineacao observada"),
    "gfm_copernicus": (
        "nao auditada: algoritmos do GFM usam mascara de HAND -- circular para testar "
        "HAND; exclusoes do radar precisam virar classe desconhecida"),
    "geoglows_vazao": ("simulacao forcada por ERA5; no Taquari atrasou o pico de 2024 "
                       "em 5 dias e nao viu set/2023: nao e observacao"),
    # Auditadas em 03/10/2026 (rodada 2), todas publicas e sem credencial:
    "mapbiomas_rs2024_area_atingida": (
        "nota tecnica MapBiomas (jun/2024): deteccao de mudanca S1/S2/Planet, depois "
        "exclui pixel em encosta por limiar de HAND e separa deslizamento por declive de 7% "
        "-- circular para testar HAND e declive; inclui solo saturado e mancha modelada de "
        "POA (Possanti 2024); sem metrica de acuracia. So exploracao"),
    "zenodo_17177153": ("titulo promete mapeamento de areas inundadas RS 2024, mas o pacote "
                        "publicado contem ganho/perda de vegetacao por hexagono de 2 km: nao ha "
                        "delineacao de cheia"),
    "zenodo_11151528": ("limiar de NDWI de um sensor (S2) que inclui area umida e agua permanente, "
                        "publicado so como imagem (PNG/PDF), sem vetor: nao e independente do que "
                        "o projeto computa e nao e utilizavel como camada"),
}


class AdmissaoRecusada(RuntimeError):
    """Um consumidor pediu um uso que as linhas nao sustentam."""


def _papel(d: pd.DataFrame) -> pd.Series:
    if "papel" in d:
        return d["papel"].astype(str)
    return ds03.papeis_evidenciais(d)


def _instrumento(fonte: str) -> dict:
    import contrato_do_fenomeno as cf
    return cf.INSTRUMENTOS.get(fonte, {})


def _coluna(d: pd.DataFrame, *nomes: str, ausente="__ausente__") -> pd.Series:
    for n in nomes:
        if n in d:
            return d[n]
    return pd.Series(ausente, index=d.index, dtype=object)


def validade(d: pd.DataFrame, preditores: tuple | list | None = None,
             pergunta_temporal: str = "ordem_entre_eventos") -> pd.DataFrame:
    """Uma coluna booleana por uso e uma coluna de motivo do primeiro bloqueio.

    Aceita a tabela unica (ds05) ou o dataset do ds06: as colunas que mudam de
    nome entre os dois sao lidas pelos dois nomes. Coluna de PROCEDENCIA
    ausente (relevo, data) nao e "tudo bem": vira valor nao verificavel e
    bloqueia o uso que depende dela. `preditores` e o vetor que o consumidor
    vai usar; sem ele, as colunas de terreno presentes.
    """
    papel = _papel(d)
    neg = _coluna(d, "qualidade_negativo", "nivel_negativo").astype(str)
    grupo = _coluna(d, "grupo_validacao", "grupo_cv", ausente=None)
    admitido = d["admitido"].astype(bool) if "admitido" in d else pd.Series(True, index=d.index)
    fontes = d["fonte"].astype(str)
    instr = {f: _instrumento(f) for f in fontes.unique()}
    mec_ok = pd.Series([bool(instr[f].get("mecanismo_justificado", False)) and
                        instr[f].get("mecanismo") == m
                        for f, m in zip(fontes, _coluna(d, "mecanismo").astype(str))], index=d.index)
    relevo_ok = pd.Series([ds03.relevo_resolvido(f, r) for f, r in
                           zip(fontes, _coluna(d, "classe_relevo").astype(str))], index=d.index)
    usados = [c for c in (preditores or TERRENO) if c in d]
    if preditores and len(usados) < len(preditores):
        raise KeyError(f"preditores ausentes da tabela: {sorted(set(preditores) - set(usados))}")
    terreno_ok = pd.Series(np.isfinite(d[usados].to_numpy(dtype=float)).all(axis=1)
                           if usados else False, index=d.index)
    if pergunta_temporal not in PERGUNTAS_TEMPORAIS:
        raise ValueError(f"pergunta temporal desconhecida: {pergunta_temporal}")
    data_ok = _coluna(d, "precisao_data").astype(str).isin(PERGUNTAS_TEMPORAIS[pergunta_temporal]["datas"])
    if "data_evento" in d:
        data_ok &= d["data_evento"].notna()
    elif "janela_inicio" in d:
        data_ok &= d["janela_inicio"].notna()
    # Na tabela do ds05 nao ha `evento_id`: a identidade la e a data + o
    # grupo, ja cobrados nas outras regras. No dataset do ds06 ela e cobrada.
    evento_ok = (d["evento_id"].notna() if "evento_id" in d
                 else pd.Series(True, index=d.index))
    preditor_antes = pd.Series(True, index=d.index)
    if pergunta_temporal == "previsao":
        quando = pd.to_datetime(_coluna(d, "data_evento", "janela_inicio", ausente=None).astype("string"),
                                errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
        preditor_antes = quando >= pd.Timestamp(DEM_PUBLICADO_EM)
    neg_instr_obs = fontes.map(lambda f: "observado" in instr[f].get("negativos", ()))
    circular = fontes.map(lambda f: bool(CAPACIDADE.get(f, {}).get("circular_com")))

    ensina = papel.isin(ds03.PAPEIS_QUE_ENSINAM)
    observado = (papel == "positivo_episodico") | (papel == "negativo_observado")

    regras = [
        # (uso, condicao, motivo quando falha) -- em ordem: o primeiro motivo vale
        ("treino", ensina, "papel_nao_ensina:" + papel),
        ("treino", d["classe"].isin([0, 1]), "classe_fora_do_binario"),
        ("treino", admitido, "rejeitada_no_ds05"),
        ("treino", mec_ok, "mecanismo_nao_justificado_pelo_instrumento"),
        ("treino", relevo_ok, "escala_do_instrumento_nao_resolve_o_relevo"),
        ("treino", terreno_ok, "terreno_incompleto"),
        ("treino", grupo.notna(), "sem_grupo_de_validacao"),
        ("avaliacao_espacial", observado, "negativo_nao_observado:" + neg),
        ("avaliacao_temporal", data_ok, f"data_nao_serve_a_pergunta:{pergunta_temporal}"),
        ("avaliacao_temporal", evento_ok, "evento_nao_identificado"),
        ("avaliacao_temporal", preditor_antes, "preditor_publicado_depois_do_evento"),
        ("avaliacao_externa", neg_instr_obs, "instrumento_nao_produz_negativo_observado"),
        ("afirmacao", neg_instr_obs, "instrumento_nao_produz_negativo_observado"),
        ("afirmacao", ~circular, "referencia_circular_com_preditor"),
    ]
    # Cadeia de dependencia entre usos: cada um exige o anterior.
    exige = {"treino": None, "avaliacao_espacial": "treino",
             "avaliacao_temporal": "avaliacao_espacial",
             "avaliacao_externa": "avaliacao_espacial", "afirmacao": "avaliacao_espacial"}
    out = pd.DataFrame(index=d.index)
    out["uso_exploracao"] = True
    for uso in USOS[1:]:
        base = out[f"uso_{exige[uso]}"] if exige[uso] else pd.Series(True, index=d.index)
        motivo = (out[f"motivo_{exige[uso]}"] if exige[uso]
                  else pd.Series("", index=d.index, dtype=object)).copy()
        ok = base.copy()
        for u, cond, porque in regras:
            if u != uso:
                continue
            cond = cond.astype(bool)
            falha = ok & ~cond
            motivo[falha] = (porque[falha] if isinstance(porque, pd.Series) else porque)
            ok &= cond
        out[f"uso_{uso}"] = ok
        out[f"motivo_{uso}"] = motivo.where(~ok, "")
    return out


def exigir(d: pd.DataFrame, uso: str, quem: str, preditores=None,
           pergunta_temporal: str = "ordem_entre_eventos") -> pd.DataFrame:
    """Ponto de entrada: falha se ALGUMA linha nao sustenta o uso pedido.

    O consumidor filtra antes com `filtrar` e chama `exigir` sobre o que vai
    ajustar ou medir. Quem passa linha nao autorizada recebe excecao com a
    contagem por motivo, e nao um resultado.
    """
    if uso not in USOS:
        raise ValueError(f"uso desconhecido: {uso}")
    v = validade(d, preditores, pergunta_temporal)
    ruim = ~v[f"uso_{uso}"]
    if ruim.any():
        motivos = v.loc[ruim, f"motivo_{uso}"].value_counts().head(8).to_dict()
        raise AdmissaoRecusada(
            f"{quem}: {int(ruim.sum())} de {len(d)} linhas nao sustentam '{uso}': {motivos}")
    return d


def filtrar(d: pd.DataFrame, uso: str, preditores=None,
            pergunta_temporal: str = "ordem_entre_eventos") -> tuple[pd.DataFrame, dict]:
    """As linhas que sustentam o uso, e a contagem do que saiu por motivo."""
    v = validade(d, preditores, pergunta_temporal)
    ok = v[f"uso_{uso}"]
    saiu = v.loc[~ok, f"motivo_{uso}"].value_counts().to_dict()
    return d.loc[ok], {str(k): int(n) for k, n in saiu.items()}


def retrato(d: pd.DataFrame) -> dict:
    v = validade(d)
    por_fonte = {}
    for f, idx in d.groupby("fonte").groups.items():
        s = v.loc[idx]
        por_fonte[str(f)] = {
            "linhas": int(len(s)),
            **{u: int(s[f"uso_{u}"].sum()) for u in USOS},
            "bloqueios": {u: {str(k): int(n) for k, n in
                              s.loc[~s[f"uso_{u}"], f"motivo_{u}"].value_counts().head(5).items()}
                          for u in USOS[1:]},
        }
    return {
        "tipo": "admissao_cientifica", "versao": VERSAO, "quando_utc": agora_utc(),
        "entrada": str(TABELA.relative_to(REPO)), "linhas": int(len(d)),
        "por_uso": {u: int(v[f"uso_{u}"].sum()) for u in USOS},
        "temporal_por_pergunta": {q: {"linhas": int(validade(d, None, q)["uso_avaliacao_temporal"].sum()),
                                      "o_que_e": c["o_que_e"]}
                                  for q, c in PERGUNTAS_TEMPORAIS.items()},
        "por_fonte": por_fonte,
        "capacidade": CAPACIDADE, "recusadas": RECUSADAS,
        "hipotese_temporal": HIPOTESE_DEM_ESTAVEL,
        "nao_e": ("nao e filtro de conveniencia: e a regra unica de uso. Consumidor "
                  "que ajusta ou mede chama `exigir` e falha com linha nao autorizada"),
    }


def markdown(r: dict) -> str:
    L = ["# Admissão científica", "",
         f"**Gerado de** `scripts/metodo/admissao_cientifica.py` ({r['versao']}) em "
         f"{r['quando_utc']}, sobre `{r['entrada']}` ({r['linhas']:,} linhas). "
         "**Não editar à mão.**".replace(",", "."), "",
         f"**Não é.** {r['nao_e']}", "",
         "## Linhas por uso autorizado", "",
         "| uso | linhas |", "|---|---:|"]
    L += [f"| `{u}` | {n:,} |".replace(",", ".") for u, n in r["por_uso"].items()]
    L += ["", "## Uso temporal depende da pergunta", "", "| pergunta | linhas | o que exige |", "|---|---:|---|"]
    L += [f"| `{q}` | {x['linhas']:,} | {x['o_que_e']} |".replace(",", ".", 1) for q, x in r["temporal_por_pergunta"].items()]
    L += ["", "## Por fonte", "", "| fonte | linhas | " + " | ".join(USOS[1:]) + " |",
          "|---|---:|" + "---:|" * len(USOS[1:])]
    for f, s in sorted(r["por_fonte"].items(), key=lambda x: -x[1]["linhas"]):
        L.append(f"| `{f}` | {s['linhas']:,} | ".replace(",", ".")
                 + " | ".join(f"{s[u]:,}".replace(",", ".") for u in USOS[1:]) + " |")
    L += ["", "## Primeiro bloqueio por fonte e uso", ""]
    for f, s in sorted(r["por_fonte"].items()):
        linhas = [f"- `{u}`: " + ", ".join(f"{k} ({n:,})".replace(",", ".") for k, n in b.items())
                  for u, b in s["bloqueios"].items() if b]
        if linhas:
            L += [f"**{f}**", "", *linhas, ""]
    L += ["## Fontes recusadas como referência", ""]
    L += [f"- `{k}`: {v}" for k, v in r["recusadas"].items()]
    L += ["", "## Hipótese que todo uso temporal carrega", "", r["hipotese_temporal"], "",
          f"`{r['hash_resumo']}`", ""]
    return "\n".join(L)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "metodo"))
    import rastreio_de_impacto as _ri
    _entrada, _saida = _ri.ganchos("admissao", "admissao_cientifica")
    _entrada()
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cols = ["ponto_id", "fonte", "mecanismo", "classe", "nivel_negativo", "classe_relevo",
            "precisao_data", "data_evento", "grupo_cv", "admitido", *TERRENO]
    d = pd.read_parquet(TABELA, columns=cols)
    r = retrato(d)
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "retrato.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(markdown(r), encoding="utf-8")
    for u, n in r["por_uso"].items():
        print(f"  {u:20s} {n:>10,}")
    print(f"GRAVADO={SAIDA / 'retrato.json'}")
    _saida()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
