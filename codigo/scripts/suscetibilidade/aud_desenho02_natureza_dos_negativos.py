"""O que a amostra negativa e, e que leitura do escore ela autoriza.

POR QUE ESTA AUDITORIA EXISTE:

`classe == 0` junta na mesma coluna tres coisas que a hidrologia e a estatistica
de distribuicao tratam de modo diferente, e o `ds03` ja as nomeia na hierarquia
de `nivel_negativo`:

  observado              alguem olhou aquela area e nao viu agua. E AUSENCIA
                         OBSERVADA -- o unico negativo no sentido pleno.
  exclusao_qualificada   fora de outline e de zona modelada, pareado pelos
                         criterios N1-N4. E negativo por exclusao: nao houve
                         observacao direta, houve argumento.
  ausencia               so nao ha registro. O proprio `ds03` avisa: "nao
                         constitui registro de ausencia".

O terceiro caso tem nome na literatura e nao e "negativo ruim": e FUNDO
(background) -- uma amostra do ambiente disponivel, nao uma observacao de
que ali nao inundou. Modelos ajustados com presenca-e-fundo estimam adequacao
RELATIVA, e a prevalencia nao e identificavel a partir dos dados: o nivel
absoluto do escore depende de quantos pontos de fundo alguem resolveu sortear,
que e decisao de amostragem e nao propriedade do terreno.

-------------------------------------------------------------------------
POR QUE ISSO IMPORTA AQUI, E NAO E DISCUSSAO ACADEMICA
-------------------------------------------------------------------------

A medida separa as fontes em duas populacoes limpas:

  cems, sen1floods11, ufo, uk    0% de fundo. Negativos observados ou
                                 qualificados. Escore admite leitura absoluta.
  curitiba                       74,2% de fundo
  recife                         98,4% de fundo

As duas regioes BRASILEIRAS servidas ao usuario sao as duas que ajustam com
fundo. E o `svc02` serve `escore_detalhe.tipo = "probabilidade_projetada"`.

O `svc01` ja declara que "um escore de regressao logistica herda a prevalencia
do conjunto de treino". Isso esta certo e e mais fraco do que o caso: com
ausencia observada a prevalencia e estimavel e o intercepto se corrige; com
fundo ela nao e identificavel de forma alguma, e nenhuma correcao de intercepto
resolve, porque o dado nao carrega a informacao.

O frontend ja diz a coisa certa -- "nao e probabilidade, o escore ordena areas
dentro da regiao servida". A pratica do produto esta correta. O que falta e o
fundamento viajar junto do numero, em vez de morar numa string de glossario que
a proxima tela pode nao repetir.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao remove ponto, nao reclassifica nivel, nao reajusta modelo e nao muda o que
o servico responde. Ele mede a natureza da amostra e declara a leitura que ela
autoriza, para que quem sirva o numero possa anexar essa leitura.

Uso:
    python scripts/suscetibilidade/aud_desenho02_natureza_dos_negativos.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import METHOD_VERSION  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.parquet"
OUT = REPO / "local_runs" / "aud-desenho-02"

# Os dois niveis que constituem observacao de que nao houve agua. `ausencia`
# fica de fora por decisao do proprio `ds03`, e nao por criterio deste modulo.
NIVEIS_DE_AUSENCIA_REAL = ("observado", "exclusao_qualificada")

# Qualquer fracao de fundo ja desloca o nivel absoluto do escore, porque o
# intercepto passa a absorver a razao de amostragem. Acima do segundo corte a
# amostra e majoritariamente fundo e so a ordem sobrevive. Os cortes sao
# declarados, nao estimados: servem para nomear regimes, nao para testar nada.
PCT_FUNDO_QUE_CONDICIONA = 0.0
PCT_FUNDO_QUE_ORDINALIZA = 50.0

LEITURA = {
    "absoluta": ("o escore pode ser lido no proprio valor: os negativos sao "
                 "ausencia observada ou qualificada"),
    "absoluta_condicionada": ("o valor depende da razao de amostragem entre "
                              "presencas e fundo; comparavel dentro da regiao, "
                              "nao interpretavel como frequencia de evento"),
    "ordinal": ("so a ordem entre celulas e sustentada: com amostra "
                "majoritariamente de fundo, a prevalencia nao e identificavel "
                "e o nivel absoluto nao e propriedade do terreno"),
}


def natureza(pct_fundo: float) -> str:
    if pct_fundo > PCT_FUNDO_QUE_ORDINALIZA:
        return "ordinal"
    if pct_fundo > PCT_FUNDO_QUE_CONDICIONA:
        return "absoluta_condicionada"
    return "absoluta"


def negativos_da_fonte(s: pd.DataFrame) -> dict:
    """A composicao da amostra negativa e a leitura que ela autoriza."""
    neg = s[s.classe == 0]
    if not len(neg):
        return {"negativos": 0, "leitura_admitida": "sem_negativos",
                "fundamento": "a fonte nao contribui com negativos"}
    contagem = {k: int(v) for k, v in neg.nivel_negativo.value_counts().items()}
    fundo = contagem.get("ausencia", 0)
    pct = round(100 * fundo / len(neg), 1)
    leitura = natureza(pct)
    return {
        "positivos": int((s.classe == 1).sum()),
        "negativos": int(len(neg)),
        "por_nivel": contagem,
        "ausencia_observada": sum(contagem.get(k, 0)
                                  for k in NIVEIS_DE_AUSENCIA_REAL),
        "fundo": fundo,
        "pct_fundo": pct,
        "amostra": "presenca_e_fundo" if pct > 0 else "presenca_e_ausencia",
        "leitura_admitida": leitura,
        "fundamento": LEITURA[leitura],
    }


def auditar(d: pd.DataFrame) -> dict:
    por_fonte = {f: negativos_da_fonte(s) for f, s in d.groupby("fonte")}
    ordinais = sorted(f for f, v in por_fonte.items()
                      if v["leitura_admitida"] == "ordinal")
    return {
        "por_fonte": por_fonte,
        "agregado": {
            "fontes_com_leitura_ordinal": ordinais,
            "fontes_com_leitura_absoluta": sorted(
                f for f, v in por_fonte.items()
                if v["leitura_admitida"] == "absoluta"),
            "alerta": (
                f"as fontes {ordinais} ajustam com amostra majoritariamente de "
                f"fundo; o escore servido para elas nao sustenta leitura "
                f"absoluta") if ordinais else None,
        },
    }


def confronto_com_o_servico(r: dict) -> dict:
    """Onde a declaracao servida e a natureza da amostra nao batem."""
    ordinais = r["agregado"]["fontes_com_leitura_ordinal"]
    return {
        "campo_servido": "svc02.escore_detalhe.tipo = 'probabilidade_projetada'",
        "frontend_ja_declara": (
            "cartao.js: 'O escore ordena areas dentro da regiao servida (...) "
            "Nao e probabilidade, nao tem periodo de retorno e nao e calibrado "
            "contra frequencia de evento'"),
        "svc01_ja_declara": (
            "'um escore de regressao logistica herda a prevalencia do conjunto "
            "de treino; declarar e o minimo para quem le o numero'"),
        "onde_a_declaracao_fica_curta": (
            "herdar prevalencia pressupoe que ela seja estimavel, o que vale "
            "para ausencia observada; com fundo ela nao e identificavel e "
            "nenhuma correcao de intercepto a recupera"),
        "fontes_afetadas": ordinais,
        "correcao_proposta": (
            "a composicao da amostra negativa e a leitura admitida viajarem no "
            "artefato servido, ao lado do escore, em vez de viverem numa "
            "string de glossario do frontend"),
        "nao_e_regressao_do_produto": (
            "o frontend ja apresenta o escore como ordenamento; o que falta e "
            "o fundamento acompanhar o numero"),
    }


def main() -> int:
    if not TABELA.exists():
        print(f"tabela unica ausente: {TABELA}")
        return 2
    d = pd.read_parquet(TABELA)
    d = d[d.classe.isin([0, 1])].copy()
    r = auditar(d)

    relatorio = {
        "artefato": "aud_desenho02_natureza_dos_negativos",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "pergunta": "a amostra negativa e ausencia observada ou fundo?",
        "cortes_declarados": {
            "pct_fundo_que_condiciona": PCT_FUNDO_QUE_CONDICIONA,
            "pct_fundo_que_ordinaliza": PCT_FUNDO_QUE_ORDINALIZA,
        },
        **r,
        "confronto_com_o_servico": confronto_com_o_servico(r),
        "nao_e": ("nao remove ponto, nao reclassifica nivel, nao reajusta "
                  "modelo e nao muda o que o servico responde"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "natureza_dos_negativos.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/suscetibilidade/aud_desenho02_natureza_dos_negativos.py\n",
        encoding="utf-8")

    print(f"{'fonte':13s} {'neg':>6s} {'fundo':>6s} {'%fundo':>7s}  leitura admitida")
    for f, v in sorted(r["por_fonte"].items()):
        print(f"{f:13s} {v['negativos']:6d} {v['fundo']:6d} "
              f"{v['pct_fundo']:6.1f}%  {v['leitura_admitida']}")
    if r["agregado"]["alerta"]:
        print(f"\n{r['agregado']['alerta']}")
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
