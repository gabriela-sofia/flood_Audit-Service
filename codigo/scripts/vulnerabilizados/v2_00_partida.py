"""Linhagem v2, etapa 1: linha de partida reproduzivel e tabela de particoes/dependencias.

Nao altera v1. Confere hashes do manifesto v1, recalcula os numeros primarios a partir dos
artefatos (sem reajustar nada), identifica a implementacao que gera cada escore e separa T1
(absoluto) de T2 (comparativo) como DESDOBRAMENTO INTERPRETATIVO de subcriterios ja existentes
no protocolo v1 - o veredito conjunto de H-T continua 'nao_sustentada'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import OUT as V1, RUNS, agora, auc_mann_whitney, gravar, ler, sha

V2 = RUNS / "transferencia-vulnerabilizados" / "v2"


def g2(p, valor):
    return gravar(p, valor, raiz=V2)


def congelar2(p, valor: dict) -> dict:
    """Como comum_tv.congelar, dentro da v2 (o de v1 so escreve na v1, e seu codigo tem hash registrado)."""
    import json
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use outra versao")
        return existente
    g2(p, {**valor, "quando_utc": agora()})
    return ler(p)


def main():
    man = ler(V1 / "manifesto.json")
    diverge = [k for k, h in man["arquivos"].items() if (V1 / k).exists() and sha(V1 / k) != h]
    ausentes = [k for k in man["arquivos"] if not (V1 / k).exists()]
    rep = {}
    for nome, pasta, cand, dom in [("rmpa", "confirmacao_rmpa", "geral_estrita", "dominio_estrita"),
                                   ("manaus", "confirmacao_manaus", "geral_sulamericano", "dominio_sulamericano")]:
        c = V1 / pasta
        g = pd.read_parquet(c / "grade.parquet", columns=["ponto_id", "classe_territorial", "terreno_valido"])
        d = g.merge(pd.read_parquet(c / "predicoes.parquet"), on="ponto_id").merge(
            pd.read_parquet(c / "referencia.parquet", columns=["ponto_id", "estado_primaria"]), on="ponto_id")
        d = d[d.terreno_valido & d[dom].fillna(False).astype(bool) & d.estado_primaria.isin(["positivo", "negativo"])]
        f = d[d.classe_territorial.eq("fcu")]
        y = f.estado_primaria.eq("positivo").astype(int)
        rep[nome] = {"auc_candidato": auc_mann_whitney(y, f[cand]), "auc_hand": auc_mann_whitney(y, f.regra_hand),
                     "n": int(len(f)), "positivos": int(y.sum()), "candidato": cand}
    av = ler(V1 / "confirmacao_rmpa/avaliacao/resultado.json")
    am = ler(V1 / "confirmacao_manaus/avaliacao/resultado.json")
    t1t2 = pd.DataFrame([
        {"estudo": "RMPA v1", "subcriterio": "T1 absoluto (AUC > 0,60)", "estimativa": av["H_T"]["T1_auc_candidato"]["estimativa"],
         "intervalo": av["H_T"]["T1_auc_candidato"]["intervalo"], "atendido": av["H_T"]["T1_sustentado"]},
        {"estudo": "RMPA v1", "subcriterio": "T2 comparativo (aprendido - HAND > 0)", "estimativa": av["H_T"]["T2_candidato_menos_hand"]["estimativa"],
         "intervalo": av["H_T"]["T2_candidato_menos_hand"]["intervalo"], "atendido": av["H_T"]["T2_superioridade"]},
        {"estudo": "RMPA v1", "subcriterio": "H-T conjunto (T1 e T2)", "estimativa": None, "intervalo": None, "atendido": False},
        {"estudo": "Manaus v1", "subcriterio": "T1 absoluto", "estimativa": am["H_T_linha"]["T1"]["estimativa"],
         "intervalo": am["H_T_linha"]["T1"]["intervalo"], "atendido": am["H_T_linha"]["T1"]["intervalo"][0] > .6},
        {"estudo": "Manaus v1", "subcriterio": "T2 comparativo", "estimativa": am["H_T_linha"]["T2"]["estimativa"],
         "intervalo": am["H_T_linha"]["T2"]["intervalo"], "atendido": am["H_T_linha"]["T2"]["intervalo"][0] > 0},
        {"estudo": "Manaus v1", "subcriterio": "H-T' conjunto", "estimativa": None, "intervalo": None, "atendido": None}])
    t1t2["nota"] = "desdobramento interpretativo de subcriterio previamente existente; veredito conjunto preservado"
    g2(V2 / "partida/t1_t2_separados.csv", t1t2)
    part = pd.DataFrame([
        {"conjunto": "doadores CEMS internacionais (ds-06 v8)", "papel_v2": "treino externo (A1i)", "rotulos_vistos_antes": "sim, em MT/curvas",
         "condicao": "historico/treino", "dependencia": "componentes do grafo comum EAL2"},
        {"conjunto": "doadores GFD sul-americanos + EMSR720 (componente gigante)", "papel_v2": "treino externo (A1)", "rotulos_vistos_antes": "sim",
         "condicao": "historico/treino", "dependencia": "ligados transitivamente; retencao por MAIN_BAS do destino + 50 km"},
        {"conjunto": "retencao internacional: 5 componentes CEMS por hash", "papel_v2": "teste de esquecimento", "rotulos_vistos_antes": "sim (MT)",
         "condicao": "retencao de dados ja vistos, nao protegidos", "dependencia": "fora do ajuste A1/A3/A4 em v2"},
        {"conjunto": "Taquari EMSR720 (pontos CEMS)", "papel_v2": "desenvolvimento brasileiro", "rotulos_vistos_antes": "sim",
         "condicao": "desenvolvimento", "dependencia": "mesmo episodio da RMPA; mesmo MAIN_BAS"},
        {"conjunto": "RMPA S2 06/05/2024 (v1)", "papel_v2": "desenvolvimento brasileiro e orcamento B entre episodios",
         "rotulos_vistos_antes": "sim, avaliacao unica v1 em 20/10/2026", "condicao": "desenvolvimento (nao confirmacao)", "dependencia": "episodio maio/2024"},
        {"conjunto": "Manaus S2 21/06/2021 (v1)", "papel_v2": "desenvolvimento (contraste de dominio)",
         "rotulos_vistos_antes": "sim, avaliacao unica v1", "condicao": "desenvolvimento", "dependencia": "pulso amazonico"},
        {"conjunto": "Curitiba S2 05/11/2023 (novo)", "papel_v2": "particao adaptacao (40% dos bairros, hash) + particao teste (60%)",
         "rotulos_vistos_antes": "nao", "condicao": "confirmacao, congelada antes da referencia", "dependencia": "cheia do Iguacu out-nov/2023; nao esta no acervo"},
        {"conjunto": "reservas EMSR868/871, populacao lacrada MT12", "papel_v2": "nao usadas", "rotulos_vistos_antes": "nao",
         "condicao": "reservadas a outros protocolos", "dependencia": "-"}])
    g2(V2 / "partida/particoes_dependencias.csv", part)
    g2(V2 / "partida/resultado.json", {
        "quando_utc": agora(), "manifesto_v1_sha": sha(V1 / "manifesto.json"),
        "arquivos_v1_conferidos": len(man["arquivos"]), "divergentes": diverge, "ausentes": ausentes,
        "reproducao": rep,
        "implementacao_dos_escores": {
            "A0/regra_hand": "-HAND do FABDEM/WBT 30 m (tv05_predicoes_rmpa.py, tv09_manaus.py)",
            "geral_*": "tv_transferencia.ajustar_bracos: StandardScaler + LogisticRegression(C=1, balanced), media de 3 sementes; preditores log1p HAND, log1p declive, TWI",
            "avaliacao": "comum_tv.auc_mann_whitney; bootstrap de pesos por comunidade (AucPonderavel)"},
        "historico_de_acesso_a_rotulos": "RMPA e Manaus abertos em 20/10/2026 (v1); Curitiba 2023 nunca aberto ate o congelamento v2",
        "regimes": {"transferencia_sem_rotulo": "B=0; entradas locais sem rotulo usadas so para retencao por bacia/distancia e dominio",
                    "adaptacao": "B>0 pontos rotulados brasileiros, com origem declarada (outro episodio ou particao de adaptacao do mesmo episodio)",
                    "continua": "A4: atualizacao de pesos a partir de A3 com e sem replay; retencao medida"}})
    print("divergentes", len(diverge), "ausentes", len(ausentes), rep)


if __name__ == "__main__":
    main()
