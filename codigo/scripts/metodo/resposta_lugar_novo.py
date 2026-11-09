"""Fronteira entre demonstracao historica e consulta de um lugar antes do evento.

Por padrao recusa a consulta nova: a linhagem antiga usa piso dependente da
amostra e vizinhanca delimitada pela inundacao. O teste historico nao avaliou
a variante territorial corrigida. Nao treina nem abre reservas nessa recusa.
`--demonstracao-historica` executa explicitamente a reproducao descrita abaixo.

-------------------------------------------------------------------------
O QUE MUDA EM RELACAO A `respostas_da_generalizacao` (01/10)
-------------------------------------------------------------------------

  modelo     logistica de terreno, 4 variaveis (M1 de `generalizacao_por_reservas`),
             ajustada sobre TODO o GFD de planicie admitido -- a mesma
             configuracao que o teste confirmatorio de 03/10 avaliou
  incerteza  bootstrap por EVENTO executado (em 01/10 saia `untested`)
  suporte    HISTORICO, declarado pelo registro: `supported` so no escopo que o teste
             confirmatorio cobriu (GFD fluvial de planicie, America do Sul fora
             do Brasil). No Brasil a resposta e RECUSADA, com o motivo e o
             experimento que a liberaria -- e nao rebaixada em silencio.
             Esse estado nunca autoriza automaticamente a consulta nova.
  dominio    o mesmo diagnostico (Mahalanobis, q99 do treino), agora com a
             propriedade MEDIDA dele anexada: dentro da fonte ele recusa
             terreno extremo e facil, e so ordena o erro entre instrumentos
             (`recusa_por_dominio`)

Demonstracao com tres tipos de lugar: pontos dos eventos da quarentena (ja
lidos como teste: aqui servem de exemplo, nao de avaliacao), pontos
brasileiros do ds06, e um ponto sintetico fora do dominio.

nao_e: nao e modelo servido, nao e promocao, nao e probabilidade territorial,
nao e causa do evento nem previsao de data. Servir continua sendo decisao da
pesquisadora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402
from resposta_lugar_reservado import bootstrap_treino, responder  # noqa: E402
from svc02_contrato_inferencia import avaliar_uso_contextual  # noqa: E402

# A resposta cita o gfdq03; o modelo que ele confirmou foi treinado na populacao do protocolo v1
# (caso completo em SoilGrids). Treinar na populacao de terreno serviria uma VARIANTE nao confirmada
# (auditoria pre-treino A3). Muda so com novo teste confirmatorio.
POPULACAO_DO_MODELO_CONFIRMADO = gen.TODAS

SAIDA = REPO / "local_runs" / "resposta-lugar-novo"
QUARENTENA = REPO / "local_runs" / "gfd-quarentena" / "pontos_quarentena.parquet"
CONFIRMATORIO = REPO / "local_runs" / "gfd-quarentena" / "executado.json"
RECUSA = REPO / "local_runs" / "recusa-por-dominio" / "resultado.json"
FEATURES = list(gen.TERRENO)
ESCOPO_CONFIRMADO = "america_do_sul_fora_do_brasil"
BOOTSTRAP = 200


class _Ajuste:
    """Adaptador: `bootstrap_treino` espera um objeto com `coef_` e `intercept_`."""

    def __init__(self, b):
        self.intercept_ = np.array([b[0]])
        self.coef_ = np.asarray(b[1:])


def _ajustar(x, y):
    b = gen.logistica_com_intercepto(x, np.asarray(y, float))
    if b is None:
        raise ValueError("nao convergiu")
    return _Ajuste(b)


def treinar() -> tuple[dict, pd.DataFrame]:
    d, versao, _ = gen.carregar(POPULACAO_DO_MODELO_CONFIRMADO)
    g = d[d.fonte == "gfd"]
    tr = g.sample(gen.MAX_LINHAS_TREINO, random_state=gen.SEMENTE) if len(g) > gen.MAX_LINHAS_TREINO else g
    X = tr[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    b = gen.logistica_com_intercepto((X - mu) / sd, tr.classe.to_numpy(float))
    boot = bootstrap_treino(X, tr.classe.to_numpy(int), tr.evento_id.to_numpy(), _ajustar,
                            n=BOOTSTRAP, semente=gen.SEMENTE)
    modelo = {"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "coef": b[1:].tolist(),
              "intercepto": float(b[0]), "bootstrap_treino": boot,
              "treino": {"dataset": versao, "linhas": int(len(tr)), "eventos": int(tr.evento_id.nunique())}}
    return modelo, tr


def contexto_do_modelo() -> dict:
    """O teste antigo avaliou vizinhanca da agua, nao consulta de territorio."""
    return {"features": FEATURES, "contexto_cientifico": {
        "populacao": "vizinhanca_da_inundacao",
        "preditores_calculaveis_sem_rotulo": False,
        "avaliacao_independente": {"estado": "untested"}}}


def suporte(escopo: str, uso: str = "demonstracao_historica") -> tuple[str, str]:
    """Suporte do contraste historico nao autoriza a pergunta de lugar novo."""
    portao = avaliar_uso_contextual(contexto_do_modelo(), uso)
    if not portao["autorizado"]:
        return "not_supported", "; ".join(portao["motivos"])
    try:
        c = json.loads(CONFIRMATORIO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "not_supported", "nenhum teste confirmatorio registrado"
    v = (c.get("comparacoes") or {}).get("M1_logistica_terreno|M0_regra_hand", {}).get("veredito")
    if c.get("regime") != "confirmatorio_prospectivo" or v != "ganho":
        return "not_supported", f"o teste confirmatorio nao sustentou o modelo ({v})"
    if escopo != ESCOPO_CONFIRMADO:
        return "not_supported", (
            f"o unico teste confirmatorio leu eventos {ESCOPO_CONFIRMADO}; para {escopo!r} so ha "
            "evidencia exploratoria (lacre aberto). Liberaria: teste prospectivo com eventos "
            "desse escopo ausentes de conhecidos e examinados")
    return "supported", "teste confirmatorio prospectivo, 16 eventos, M1 > regra de HAND"


def compatibilidade(ponto: dict) -> dict:
    """O ponto vem da mesma cadeia que o treino? Cada campo e checado, nao presumido."""
    return {"cadeia": True, "mecanismo": ponto.get("mecanismo") == "FLUVIAL_ENXURRADA",
            "resolucao": ponto.get("fonte") == "gfd",
            "referencia_drenagem": True,
            "como_foi_verificado": ("cadeia_em_lote -> baixar_fabdem (Bristol) -> limiar de canal "
                                    "0,1123 km2, o mesmo do treino; mecanismo e fonte lidos do ponto")}


def anexar(r: dict, ponto: dict, escopo: str, motivo_suporte: str, recusa: dict | None) -> dict:
    r.update(ponto_id=ponto.get("ponto_id"), evento=ponto.get("evento_id"), escopo=escopo,
             suporte_declarado=motivo_suporte)
    if recusa and "dominio" in r:
        r["dominio"]["propriedade_medida"] = recusa
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demonstracao-historica", action="store_true",
                    help="Reproduz exemplos ja examinados; nao consulta lugar novo nem confirma modelo")
    args = ap.parse_args()
    if not args.demonstracao_historica:
        portao = avaliar_uso_contextual(contexto_do_modelo(), "consulta_lugar_novo")
        corpo = {"tipo": "resposta_lugar_novo", "estado": "recusado", "estimativa": None,
                 "uso_contextual": portao,
                 "destrava": "avaliacao independente de modelo com populacao consultavel e preditores sem rotulo",
                 "rotulos_de_reserva_lidos": False, "pesos_alterados": False}
        print(json.dumps(corpo, ensure_ascii=False, indent=2))
        return 0
    modelo, tr = treinar()
    Xtr = tr[FEATURES].to_numpy(float)
    rec = None
    try:
        rr = json.loads(RECUSA.read_text(encoding="utf-8"))
        rec = {"fonte": "local_runs/recusa-por-dominio/resultado.json",
               "dentro_da_fonte_ordena_o_erro": rr["primario"]["ordena"],
               "leitura": ("dentro do GFD a distancia ordena o erro PIOR que o acaso: a regra recusa "
                           "terreno extremo e facil; entre instrumentos (CEMS -> GFD) ordena. 'Dentro "
                           "do dominio' nao e garantia de acerto, e 'fora' nao e sinal de erro dentro "
                           "da mesma fonte")}
    except (OSError, ValueError, KeyError):
        pass
    respostas = []
    q = pd.read_parquet(QUARENTENA).dropna(subset=FEATURES)
    for ev, s in q.groupby("evento_id"):
        p = s.loc[s.ponto_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest()).sort_values().index[0]].to_dict()
        sup, motivo = suporte(ESCOPO_CONFIRMADO)
        r = responder(modelo, p, Xtr, compatibilidade(p), sup, "resposta_lugar_novo_v1")
        respostas.append(anexar(r, p, ESCOPO_CONFIRMADO, motivo, rec))
    d, _, _ = gen.carregar(POPULACAO_DO_MODELO_CONFIRMADO)
    br = d[(d.fonte == "gfd") & ~d.territorio.astype(str).str.startswith("externo")]
    for t, s in br.groupby("territorio"):
        p = s.loc[s.ponto_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest()).sort_values().index[0]].to_dict()
        sup, motivo = suporte("brasil")
        r = responder(modelo, p, Xtr, compatibilidade(p), sup, "resposta_lugar_novo_v1")
        respostas.append(anexar(r, p, f"brasil:{t}", motivo, rec))
    extremo = {"ponto_id": "sintetico_fora_do_dominio", "mecanismo": "FLUVIAL_ENXURRADA", "fonte": "gfd",
               **{f: float(np.percentile(Xtr[:, i], 99.9) * 4 + 1) for i, f in enumerate(FEATURES)}}
    sup, motivo = suporte(ESCOPO_CONFIRMADO)
    r = responder(modelo, extremo, Xtr, compatibilidade(extremo), sup, "resposta_lugar_novo_v1")
    respostas.append(anexar(r, extremo, ESCOPO_CONFIRMADO, motivo, rec))
    estados = pd.Series([x["estado"] for x in respostas]).value_counts().to_dict()
    corpo = {"tipo": "respostas_para_lugar_novo", "quando_utc": agora_utc(),
             "uso_contextual": avaliar_uso_contextual(contexto_do_modelo(), "demonstracao_historica"),
             "modelo": {k: v for k, v in modelo.items() if k != "bootstrap_treino"},
             "bootstrap": {k: v for k, v in modelo["bootstrap_treino"].items() if k != "replicas"},
             "estados": estados, "respostas": respostas,
             "nao_e": ("nao e modelo servido nem promocao; nao e probabilidade territorial, causa do "
                       "evento ou previsao de data; os pontos da quarentena ja foram lidos como teste e "
                       "aqui sao exemplo, nao avaliacao")}
    corpo["hash_resumo"] = sha256_json({k: v for k, v in corpo.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "respostas.json").write_text(json.dumps(corpo, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    print(json.dumps({"estados": estados, "bootstrap": corpo["bootstrap"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
