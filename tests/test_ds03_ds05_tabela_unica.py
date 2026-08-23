"""Testes da tabela unica: contrato (ds03), reducao (ds04) e admissao (ds05).

O que estes testes protegem, e por que cada um existe:

o risco desta etapa nao e o script quebrar -- e ele produzir uma tabela
plausivel e errada. Uma coluna com duas fontes dentro, um negativo promovido a
observado, um ponto contado duas vezes: nada disso levanta excecao. Por isso os
testes checam INVARIANTES do dado produzido, e nao so se o codigo roda.

Os artefatos vivem em `local_runs/`, que e git-ignored. Quando faltarem, o
teste marca skip com o comando que os gera, em vez de falhar -- falhar por
ausencia de dado local confundiria quem clonasse o repositorio.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs" if (ROOT / "local_runs").is_dir() else ROOT / "modelo" / "execucoes"
RED = RUNS / "ds-04-reducao"
UNI = RUNS / "ds-05-tabela-unica"
ESQ = RUNS / "ds-03-esquema"

sys.path.insert(0, str(ROOT / "scripts" / "treino"))
from ds03_esquema_alvo import (  # noqa: E402
    CLASSE, CLASSES_TREINO, COLUNAS, DOMINIOS, OBRIGATORIAS,
    VARIAVEIS_TERRENO, VARIAVEIS_TERRENO_TRANSFERIVEL, VERSAO, contrato,
)
from ds04_reduzir_por_fonte import REDUTORES, VARIANTES  # noqa: E402


def _arquivos_reduzidos() -> list[Path]:
    """So os arquivos que o ds04 ATUAL declara produzir.

    Nao usa glob("*.csv"): local_runs pode conter orfaos de convencoes de
    nomeacao anteriores (ex.: `recife__variante_wbt30.csv`, anterior a
    reversao de 12/08/2026, sem correspondente em REDUTORES/VARIANTES hoje) --
    esses arquivos nao sao regenerados nem apagados por este script, e testar
    contra eles testaria uma versao do pipeline que ja nao existe.
    """
    return [RED / f"{nome}.csv" for nome in {**REDUTORES, **VARIANTES}
            if (RED / f"{nome}.csv").exists()]


def _exige(p: Path, comando: str) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {comando}")


@pytest.fixture(scope="module")
def unica() -> pd.DataFrame:
    p = UNI / f"tabela_unica_{VERSAO}.csv"
    _exige(p, "python scripts/treino/ds05_admissao_consolidacao.py")
    return pd.read_csv(p, low_memory=False)


# --------------------------------------------------------------------------
# contrato
# --------------------------------------------------------------------------

def test_contrato_nao_tem_coluna_de_valor_sem_procedencia():
    """A regra de ouro do ds03, verificada e nao so escrita no docstring."""
    c = contrato()
    nomes = {x["nome"] for x in c["colunas"]}
    assert {"elevation_m", "slope_deg", "hand_m", "twi_dinf"} <= nomes
    # terreno tem cadeia_terreno; chuva tem fonte_chuva
    assert "cadeia_terreno" in nomes
    assert "fonte_chuva" in nomes
    assert set(c["variaveis_terreno"]) <= set(c["variaveis_fisicas"])


def test_contrato_gerado_bate_com_o_modulo():
    p = ESQ / f"esquema_alvo_{VERSAO}.json"
    _exige(p, "python scripts/treino/ds03_esquema_alvo.py")
    gravado = json.loads(p.read_text(encoding="utf-8"))
    assert gravado == contrato(), (
        "o contrato gravado divergiu do modulo: rode o ds03 de novo")


def test_esquema_alvo_nao_contem_dado():
    p = ESQ / f"esquema_alvo_{VERSAO}.csv"
    _exige(p, "python scripts/treino/ds03_esquema_alvo.py")
    linhas = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 1, "o arquivo de contrato ganhou linha de dado"
    assert linhas[0].split(",") == list(COLUNAS)


# --------------------------------------------------------------------------
# reducao
# --------------------------------------------------------------------------

def test_toda_fonte_reduzida_respeita_o_contrato():
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    achou = False
    for p in sorted(_arquivos_reduzidos()):
        d = pd.read_csv(p, low_memory=False)
        achou = True
        assert list(d.columns) == list(COLUNAS), f"{p.name} fora do contrato"
        for col, dom in DOMINIOS.items():
            fora = set(d.loc[d[col].notna(), col].unique()) - set(dom)
            assert not fora, f"{p.name}: {col} tem valor fora do dominio: {fora}"
        fora_cls = set(d["classe"].dropna().astype(int).unique()) - set(CLASSE)
        assert not fora_cls, f"{p.name}: classe fora do dominio: {fora_cls}"
    assert achou, "nenhuma fonte reduzida encontrada"


def test_curitiba_nunca_entra_como_negativo_observado():
    """O negativo de Curitiba nunca pode chegar a `observado`.

    O campo de origem diz que houve chamado no 156 e o assunto nao era
    hidrologico. Isso nunca e "a area foi analisada e nao havia inundacao" --
    ninguem vistoriou o ponto. `observado` esta fora por definicao.

    O que MUDA e o degrau abaixo: desde o `neg01`, parte dos pontos sobe de
    `ausencia` para `exclusao_qualificada` quando os quatro criterios N1-N4
    provam que o silencio informa. A invariante testada e o teto, nao a
    contagem -- ela muda quando a base do 156 e reprocessada.
    """
    p = RED / "curitiba.csv"
    _exige(p, "python scripts/treino/ds04_reduzir_por_fonte.py --fonte curitiba")
    d = pd.read_csv(p, low_memory=False)
    neg = d[d["classe"] == 0]
    assert len(neg) > 0
    niveis = set(neg["nivel_negativo"].unique())
    assert "observado" not in niveis, (
        f"Curitiba chegou a negativo observado: {niveis}")
    assert niveis <= {"ausencia", "exclusao_qualificada"}, (
        f"nivel inesperado no negativo de Curitiba: {niveis}")


def test_reclassificacao_de_negativo_so_sobe_de_ausencia():
    """O `neg01` e caminho de sentido unico: nunca rebaixa, nunca vira observado."""
    p = RUNS / "neg-01-exclusao-qualificada" / "curitiba_reclassificacao_negativo.csv"
    _exige(p, "python scripts/treino/neg01_exclusao_qualificada_temporal.py")
    r = pd.read_csv(p, low_memory=False)
    assert set(r["nivel_negativo_novo"].unique()) <= {"ausencia",
                                                      "exclusao_qualificada"}
    # aprovado <=> subiu; reprovado <=> ficou onde estava
    assert (r.loc[r["aprovado"].astype(bool), "nivel_negativo_novo"]
            == "exclusao_qualificada").all()
    assert (r.loc[~r["aprovado"].astype(bool), "nivel_negativo_novo"]
            == "ausencia").all()
    # e todo aprovado passou nos QUATRO criterios, nao em tres
    crit = [c for c in r.columns if c.startswith(("n1_", "n2_", "n3_", "n4_"))]
    assert len(crit) == 4
    assert r.loc[r["aprovado"].astype(bool), crit].all(axis=1).all()


def test_toda_fonte_canonica_esta_em_wbt30():
    """Resolucao unica no projeto inteiro (decisao de 13/08/2026).

    Substitui a regra anterior, em que a resolucao seguia o mecanismo e Recife
    ficava a 10 m. Uma base replicavel nao pode ter duas resolucoes convivendo:
    `hand_m` passaria a significar coisas diferentes conforme a linha.
    """
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    for p in sorted(_arquivos_reduzidos()):
        if "__variante_" in p.stem:
            continue
        d = pd.read_csv(p, low_memory=False)
        cadeias = set(d["cadeia_terreno"].unique()) - {"global", "ausente"}
        assert cadeias <= {"wbt30"}, (
            f"{p.name} tem cadeia derivada fora de wbt30: {cadeias}")
    assert set(pd.read_csv(RED / "recife.csv")["mecanismo"].unique()) == {"PLUVIAL_URBANO"}


def test_a_variante_de_10m_de_recife_continua_disponivel():
    """A harmonizacao custa declividade e TWI em Recife; o custo tem de ficar
    mensuravel, entao a cadeia de 10 m e preservada como variante."""
    p = RED / "recife__variante_nativa_10m.csv"
    _exige(p, "python scripts/treino/ds04_reduzir_por_fonte.py")
    v = pd.read_csv(p, low_memory=False)
    assert set(v["cadeia_terreno"].unique()) == {"nativa_10m"}
    c = pd.read_csv(RED / "recife.csv", low_memory=False)
    assert len(v) == len(c)
    # se as duas cadeias dessem o mesmo numero, harmonizar nao teria custo --
    # e o ter03 ja mediu que tem (slope pearson 0,518)
    assert not v["slope_deg"].equals(c["slope_deg"])


def test_nivel1_tem_aoi_por_chip_e_relevo_medido():
    """Depois do ter06 as duas fontes de Nivel 1 deixam de ser opacas.

    Antes tinham `aoi` igual ao grupo (pais, no Sen1Floods11) e relevo
    NAO_CLASSIFICADO por falta de declividade. Com a cadeia derivada por chip a
    AOI passa a ser o recorte rotulado e o criterio de relevo volta a ser
    aplicavel -- sobre pontos amostrados, que e outro estimador e por isso fica
    marcado em `classe_relevo_base`.
    """
    for fonte in ("sen1floods11", "ufo"):
        p = RED / f"{fonte}.csv"
        _exige(p, "python scripts/terreno/ter06_harmonizar_chips_nivel1.py")
        d = pd.read_csv(p, low_memory=False)
        com_terreno = d[d["cadeia_terreno"] == "wbt30"]
        if com_terreno.empty:
            pytest.skip(f"{fonte} ainda sem cadeia derivada neste ambiente")
        # A AOI nunca pode ser mais grossa que o grupo de validacao. Se for
        # igual nao ha erro: no UFO o chip E o evento, entao as duas coincidem.
        # No Sen1Floods11 o grupo e o pais e a AOI tem de ser estritamente
        # mais fina -- 446 chips contra 11 paises.
        assert d["aoi"].nunique() >= d["grupo_cv"].nunique(), (
            f"{fonte}: AOI mais grossa que o grupo de validacao")
        if fonte == "sen1floods11":
            assert d["aoi"].nunique() > d["grupo_cv"].nunique()
        classificado = com_terreno[com_terreno["classe_relevo"] != "NAO_CLASSIFICADO"]
        assert len(classificado) > 0
        assert set(classificado["classe_relevo_base"].unique()) == {"pontos_amostrados"}


def test_hand_nulo_onde_a_rede_de_drenagem_e_degenerada():
    """Chip cuja janela nao sustenta rede de drenagem sai com HAND nulo.

    HAND e altura acima do canal mais proximo. Numa janela de ~9 km o canal
    real pode estar fora, e um numero plausivel e errado e pior que a ausencia.
    """
    p = RUNS / "ter-02-comparacao" / "resumo_nivel1.json"
    _exige(p, "python scripts/terreno/ter06_harmonizar_chips_nivel1.py")
    m = json.loads(p.read_text(encoding="utf-8"))
    degenerados = {d["chip"] for d in m["chips_rede_degenerada"]}
    if not degenerados:
        pytest.skip("nenhum chip com rede degenerada nesta execucao")
    h = pd.read_csv(RUNS / "ter-02-comparacao" / "dataset_harmonizado_nivel1.csv",
                    low_memory=False)
    afetados = h[h["chip"].astype(str).isin(degenerados)]
    assert afetados["hand_m_wbt"].isna().all()


def test_nenhuma_variavel_fisica_foi_imputada_com_zero():
    """Ausencia e nulo declarado. Zero em HAND ou chuva seria um valor real."""
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    for p in sorted(_arquivos_reduzidos()):
        d = pd.read_csv(p, low_memory=False)
        for v in ("hand_m", "rain_max_24h", "rain_decay_index"):
            # zero e valor legitimo isolado; o que denuncia imputacao e a
            # coluna INTEIRA em zero, ou zero exatamente onde a fonte nao tem
            # a variavel.
            if d[v].notna().any():
                assert not (d[v].fillna(0) == 0).all(), f"{p.name}: {v} toda zero"


# --------------------------------------------------------------------------
# admissao e consolidacao
# --------------------------------------------------------------------------

def test_ponto_id_e_unico_na_tabela_consolidada(unica):
    assert unica["ponto_id"].is_unique, (
        "ponto_id repetido: dois pontos com o mesmo identificador seriam "
        "contados duas vezes em qualquer agregacao")


def test_nenhum_ponto_existe_em_duas_fontes(unica):
    """Passo 1.4 do checklist. Coincidencia geografica, nao so de id."""
    g = unica.copy()
    g["celula"] = (
        (g["lat"] / 1e-4).round().astype("Int64").astype(str) + "_"
        + (g["lon"] / 1e-4).round().astype("Int64").astype(str))
    cruzadas = g.groupby("celula")["fonte"].nunique()
    assert int((cruzadas > 1).sum()) == 0, (
        f"{int((cruzadas > 1).sum())} celulas de ~11 m aparecem em mais de "
        "uma fonte")


def test_toda_rejeicao_tem_motivo_nomeado(unica):
    """Passo 1.3: 'o restante fica como proveniencia, com o motivo nomeado'."""
    rej = unica[~unica["admitido"]]
    assert len(rej) > 0, "nenhuma rejeicao: o filtro de admissao nao rodou?"
    assert rej["motivo_rejeicao"].notna().all()
    assert (rej["motivo_rejeicao"].astype(str).str.strip() != "").all()
    adm = unica[unica["admitido"]]
    assert (adm["motivo_rejeicao"].fillna("") == "").all(), (
        "linha admitida com motivo de rejeicao preenchido")


def test_a_conta_fecha_por_fonte(unica):
    """entrada = admitido + rejeitado, em cada fonte. E o relatorio de E2."""
    p = UNI / "relatorio_contagem_por_fonte.csv"
    _exige(p, "python scripts/treino/ds05_admissao_consolidacao.py")
    t = pd.read_csv(p)
    real = unica.groupby("fonte").agg(
        entrada=("ponto_id", "size"), admitido=("admitido", "sum"))
    for _, r in t.iterrows():
        assert int(r["entrada"]) == int(real.loc[r["fonte"], "entrada"])
        assert int(r["admitido"]) == int(real.loc[r["fonte"], "admitido"])
        assert int(r["admitido"]) <= int(r["entrada"])


def test_admitido_satisfaz_os_tres_criterios(unica):
    adm = unica[unica["admitido"]]
    assert adm["aoi"].notna().all()
    assert adm["grupo_cv"].notna().all()
    assert adm["lat"].notna().all() and adm["lon"].notna().all()
    assert not (adm["cadeia_terreno"].isin(["global", "ausente"])).any(), (
        "cadeia global admitida: e outro instrumento, nao outra resolucao")
    assert adm[list(VARIAVEIS_TERRENO)].notna().all(axis=1).all()
    assert adm["classe"].isin(CLASSES_TREINO).all()


def test_pool_fluvial_e_uma_cadeia_so_e_um_mecanismo_so(unica):
    pool = unica[unica["elegivel_pool_fluvial"]]
    assert len(pool) > 0
    assert set(pool["cadeia_terreno"].unique()) == {"wbt30"}
    assert set(pool["mecanismo"].unique()) == {"FLUVIAL_ENXURRADA"}
    assert "recife" not in set(pool["fonte"].unique()), (
        "Recife e pluvial urbano; entrar no pool fluvial desfaria a separacao "
        "por mecanismo")


def test_ufo_nunca_entra_num_modelo_por_mecanismo(unica):
    """A propria base declara drivers pluvial, fluvial e mare sem separacao."""
    ufo = unica[unica["fonte"] == "ufo"]
    if ufo.empty:
        pytest.skip("fonte ufo nao reduzida neste ambiente")
    assert set(ufo["mecanismo"].unique()) == {"MISTO_NAO_SEPARAVEL"}
    assert int(ufo["elegivel_pool_fluvial"].sum()) == 0


def test_todo_valor_de_chuva_tem_fonte_declarada(unica):
    """A coluna pode misturar produtos; o que nao pode e misturar em silencio.

    A direcao que importa e valor -> fonte. O contrario nao vale como regra:
    Recife tem 9 pontos sem valor de chuva e com `rain_data_source` gravado,
    porque a fonte estava escolhida e a serie e que faltou naquele dia.
    """
    com_chuva = unica[unica["rain_max_24h"].notna()
                      | unica["rain_decay_index"].notna()]
    if com_chuva.empty:
        pytest.skip("nenhuma linha com chuva")
    assert (com_chuva["fonte_chuva"] != "ausente").all(), (
        "valor de chuva sem fonte declarada")


def test_recife_tem_fonte_de_chuva_unica(unica):
    """Guarda de regressao sobre a correcao do achado do aud_chuva01.

    Ate 2026-08-15 Recife carregava CHIRPS v2 (181 pontos) e Open-Meteo/
    ERA5-Land (97 pontos) na mesma coluna, com a fonte associada ao rotulo
    (AUC do indicador de fonte 0,826 > AUC da propria chuva 0,738). Corrigido
    em 2026-08-16 por `chuva02_padronizar_fonte_unica_recife.py`: os 278
    pontos passam a vir todos de Open-Meteo/ERA5-Land (decisao registrada em
    `ext_chuva_fonte_unica_recife_v1.md`). Se este teste falhar, a mistura
    voltou -- reveja o que reescreveu `recife_harmonizado.csv` antes de seguir.
    """
    rec = unica[(unica["fonte"] == "recife") & unica["rain_max_24h"].notna()]
    if rec.empty:
        pytest.skip("recife nao reduzido neste ambiente")
    fontes = set(rec["fonte_chuva"].unique())
    assert len(fontes) == 1, (
        f"Recife voltou a ter mais de uma fonte de chuva na mesma coluna: {fontes}. "
        "Rode aud_chuva01 para medir o confundimento antes de admitir isso na base")
    assert fontes == {"open_meteo_era5_land"}, (
        f"fonte unica esperada e open_meteo_era5_land, achei {fontes}")


def test_toda_fonte_tem_produto_de_chuva_unico(unica):
    """A garantia de Recife vale para a base inteira desde o `chuva04`.

    O teste acima nasceu quando so Recife tinha o problema. Depois do
    `chuva04_adquirir_era5_global.py` (2026-08-16) as seis fontes passaram a
    usar Open-Meteo/ERA5-Land com a mesma janela de 14 dias e o mesmo fator de
    decaimento -- entao o invariante deixou de ser sobre Recife e passou a ser
    sobre o projeto. Manter os dois: o de Recife guarda a correcao historica,
    este guarda a propriedade atual da base.
    """
    com_chuva = unica[unica["rain_max_24h"].notna()]
    if com_chuva.empty:
        pytest.skip("nenhuma linha com chuva")
    por_fonte = com_chuva.groupby("fonte")["fonte_chuva"].nunique()
    misturadas = por_fonte[por_fonte > 1]
    assert misturadas.empty, (
        f"fontes com mais de um produto de precipitacao: {dict(misturadas)}. "
        "Rode aud_chuva01 para medir o confundimento antes de admitir na base")


def test_chuva_de_toda_a_base_vem_do_mesmo_produto(unica):
    """Comparar chuva entre fontes exige que seja a mesma grandeza medida igual.

    Sem isto, o coeficiente de chuva de um modelo multirregiao mistura produto
    com regiao, que e a versao entre fontes do achado do `aud_chuva01`.
    """
    com_chuva = unica[unica["rain_max_24h"].notna()]
    if com_chuva.empty:
        pytest.skip("nenhuma linha com chuva")
    produtos = set(com_chuva["fonte_chuva"].unique())
    assert produtos == {"open_meteo_era5_land"}, (
        f"a base deixou de ter produto unico de precipitacao: {produtos}")


def test_manifesto_registra_hash_do_consolidado():
    p = UNI / f"manifesto_{VERSAO}.json"
    _exige(p, "python scripts/treino/ds05_admissao_consolidacao.py")
    m = json.loads(p.read_text(encoding="utf-8"))
    alvo = f"tabela_unica_{VERSAO}.csv"
    assert alvo in m["arquivos"]
    assert len(m["arquivos"][alvo]["sha256"]) == 64
    assert m["admitidos"] <= m["entrada"]
    assert m["pool_fluvial"] <= m["admitidos"]


def test_consolidacao_e_deterministica():
    """Rodar de novo com a mesma entrada tem de dar o mesmo hash."""
    p = UNI / f"manifesto_{VERSAO}.json"
    _exige(p, "python scripts/treino/ds05_admissao_consolidacao.py")
    alvo = f"tabela_unica_{VERSAO}.csv"
    antes = json.loads(p.read_text(encoding="utf-8"))["arquivos"][alvo]["sha256"]
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/treino/ds05_admissao_consolidacao.py")],
        cwd=ROOT, check=True, capture_output=True)
    depois = json.loads(p.read_text(encoding="utf-8"))["arquivos"][alvo]["sha256"]
    assert antes == depois


def test_nada_de_local_runs_esta_staged():
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert "local_runs/" not in staged


# --------------------------------------------------------------------------
# elevacao relativa (v2, 2026-08-19) -- ver DELTA v2 no ds03 e
# docs/metodologia_cientifica/elevacao_relativa_regra_permanente_v1.md
#
# Estes testes protegem uma regra permanente, nao um teste pontual de
# Curitiba: qualquer fonte futura (Petropolis, ou outra) tem de ganhar
# elevation_rel_m automaticamente, e o achado que motivou a regra (Curitiba
# fora do dominio de elevacao absoluta de fontes ao nivel do mar) tem de
# continuar corrigido enquanto o pipeline evoluir.
# --------------------------------------------------------------------------

def test_elevation_rel_m_e_reconstruivel_a_partir_do_baseline():
    """elevation_rel_m tem de ser SEMPRE elevation_m - elevation_baseline_m.

    Nao pode virar um numero solto: a regra de ouro do contrato exige que
    toda coluna de valor seja rastreavel ate a procedencia.
    """
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    achou = False
    for p in sorted(_arquivos_reduzidos()):
        d = pd.read_csv(p, low_memory=False)
        com_elev = d[d["elevation_m"].notna()]
        if com_elev.empty:
            continue
        achou = True
        reconstruido = com_elev["elevation_m"] - com_elev["elevation_baseline_m"]
        assert (reconstruido - com_elev["elevation_rel_m"]).abs().max() < 1e-6, (
            f"{p.name}: elevation_rel_m nao reconstroi a partir do baseline")
    assert achou, "nenhuma fonte com elevation_m para checar"


def test_elevation_baseline_e_constante_dentro_da_fonte():
    """O baseline e o P1 de elevation_m DENTRO da mesma fonte -- um so valor
    por fonte, nunca um numero que varia linha a linha dentro dela."""
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    for p in sorted(_arquivos_reduzidos()):
        d = pd.read_csv(p, low_memory=False)
        base = d["elevation_baseline_m"].dropna().unique()
        assert len(base) <= 1, (
            f"{p.name}: elevation_baseline_m varia dentro da fonte: {base}")


def test_elevation_rel_m_nulo_apenas_onde_elevation_m_e_nulo():
    """Nulo declarado, nunca imputado: se elevation_m falta, elevation_rel_m
    tem de faltar tambem -- nunca zero, nunca o baseline sozinho."""
    _exige(RED, "python scripts/treino/ds04_reduzir_por_fonte.py")
    for p in sorted(_arquivos_reduzidos()):
        d = pd.read_csv(p, low_memory=False)
        assert (d["elevation_m"].isna() == d["elevation_rel_m"].isna()).all(), (
            f"{p.name}: nulidade de elevation_rel_m nao acompanha elevation_m")


def test_curitiba_ganha_dominio_comparavel_com_elevacao_relativa(unica):
    """Guarda de regressao do achado central (app01, 19/08/2026): elevation_m
    absoluta tinha 0% dos pontos de Curitiba dentro do intervalo 5-95% de
    treino externo (CEMS+UK+Sen1Floods11), diferenca padronizada ~2,4 desvios
    -- Curitiba fica a ~900 m, as fontes externas ficam perto do nivel do
    mar. elevation_rel_m resolve isso por construcao. Se este teste falhar,
    a correcao de elevacao relativa parou de funcionar ou uma fonte nova
    entrou com baseline mal calculado.
    """
    pool = unica[unica["elegivel_pool_fluvial"]]
    ext = pool[pool["fonte"].isin(["cems", "uk", "sen1floods11"])]
    cur = pool[pool["fonte"] == "curitiba"]
    if ext.empty or cur.empty:
        pytest.skip("pool fluvial sem as fontes necessarias neste ambiente")

    p5, p95 = ext["elevation_rel_m"].quantile([0.05, 0.95])
    cobertura_rel = ((cur["elevation_rel_m"] >= p5)
                     & (cur["elevation_rel_m"] <= p95)).mean()
    assert cobertura_rel > 0.80, (
        f"cobertura de Curitiba em elevation_rel_m caiu para {cobertura_rel:.1%} "
        "-- a correcao de elevacao relativa parece ter parado de funcionar")

    # a variavel absoluta continua com o problema original -- documentado
    # aqui, nao escondido, para que a diferenca entre as duas fique visivel
    # no proprio teste
    p5a, p95a = ext["elevation_m"].quantile([0.05, 0.95])
    cobertura_abs = ((cur["elevation_m"] >= p5a)
                     & (cur["elevation_m"] <= p95a)).mean()
    assert cobertura_abs < 0.10, (
        "elevation_m absoluta deixou de ter o problema de dominio esperado -- "
        "confirme se isso e uma mudanca real de dado antes de comemorar")


def test_variaveis_terreno_transferivel_troca_so_a_elevacao():
    """A tupla usada em ajuste multirregiao difere da tupla bruta em UM
    unico elemento -- se mais de um mudar, algo alem da elevacao foi alterado
    sem que a regra permanente tenha sido revisada."""
    assert len(VARIAVEIS_TERRENO) == len(VARIAVEIS_TERRENO_TRANSFERIVEL)
    diferentes = [a for a, b in zip(VARIAVEIS_TERRENO, VARIAVEIS_TERRENO_TRANSFERIVEL)
                 if a != b]
    assert diferentes == ["elevation_m"], (
        f"esperava so elevation_m trocada por elevation_rel_m, achei: {diferentes}")
    assert VARIAVEIS_TERRENO_TRANSFERIVEL[
        VARIAVEIS_TERRENO.index("elevation_m")] == "elevation_rel_m"
