"""Casos territoriais em texto, gerados da candidata e das tabelas (nada escrito a mao).

Morro da Oficina, Vila Sao Francisco A e Sargento Boening como pilotos; Paraiso A/B
com a sensibilidade de borda; e a lista das comunidades sem registro observado.
Escolha fixada no prompt da campanha, nao pelo desempenho do modelo.
"""
from __future__ import annotations

import pandas as pd

from pa_comum import OUT, Etapa, ler_json

TER = OUT / "territorio"
CASOS = ["33039060003", "33039060014", "33039060021", "33039060022", "33039060052"]


def pct(x):
    return "n/d" if x is None or pd.isna(x) else f"{x:.2f}"


def main() -> int:
    e = Etapa("pa_06b_casos")
    cand = ler_json(TER / "candidata_resposta_territorial.json")
    por = {c["CD_FCU"]: c for c in cand["comunidades"]}
    t = pd.read_csv(TER / "tabela_42_fcu.csv", dtype={"CD_FCU": str}).set_index("CD_FCU")
    z = pd.read_csv(OUT / "resultados/emenda1/por_comunidade_zona_teste.csv", dtype={"CD_FCU": str}).set_index("CD_FCU")
    L = ["# Casos territoriais — Petrópolis (gerado de `candidata_resposta_territorial.json`)", "",
         "Percentis são ordenações relativas ao município por terreno anterior ao evento, não probabilidades. "
         "População é o universo censitário da comunidade em 2022, não pessoas atingidas.", ""]
    for cod in CASOS:
        c, r = por[cod], t.loc[cod]
        enc = c["encosta"]
        L += [f"## {c['NM_FCU']} ({cod})", "",
              f"- **Contexto social (observado, Censo 2022):** {r.populacao_censo2022} moradores, {r.domicilios_censo2022} domicílios; "
              f"renda média do responsável R$ {round(r.renda_resp_media_ponderada) if pd.notna(r.renda_resp_media_ponderada) else 'n/d'} "
              f"({c['contexto_social']['cobertura_renda']})."]
        for o in enc["observado"]:
            if o["tipo"].startswith("cicatriz"):
                L.append(f"- **Encosta observada (INPE/Charter 2022):** {o['poligonos']} polígono(s) de deslizamento, "
                         f"{o['area_m2']:.0f} m² dentro da FCU; {o['sensibilidade_borda']}.")
            elif o["tipo"].startswith("fluxo"):
                L.append(f"- **Fluxo de detritos observado (2022):** {o['poligonos']} polígono(s).")
            else:
                L.append(f"- **Relatos históricos (UFRJ/GEOCART, {o['periodo']}):** {o['relatos_dentro']} dentro da FCU e "
                         f"{o['relatos_ate_100m']} até 100 m; relato de dano por endereço, posição aproximada.")
        ct = enc["contexto_tecnico"]
        L.append(f"- **Contexto técnico (SIG municipal 2025, não ocorrência):** {ct['fracao_area']:.0%} da área da FCU sobreposta; classes {ct['classes']}.")
        inf = enc["inferido"]
        a, h, e1 = inf["antes_transferido_A1"], inf["atualizado_relatos_historicos_A3"], inf["atualizado_cicatrizes_2022_E1"]
        L.append(f"- **Encosta inferida:** transferida do Idai, percentil mediano {pct(a['percentil_mediano'])} ({a['estado']}); "
                 f"após atualização com cicatrizes de outros bairros (E1) {pct(e1['percentil_mediano'])} ({e1['estado']}); "
                 f"a versão atualizada com relatos históricos ({pct(h['percentil_mediano'])}) foi recusada pela avaliação. "
                 f"Domínio do doador: {inf['dominio_frac_dentro_doador']:.0%} das células.")
        if cod in z.index:
            zz = z.loc[cod]
            if zz.cicatrizes_2022_tocando_fcu:
                L.append(f"- **Teste nas cicatrizes de 2022 que tocam a FCU ({int(zz.cicatrizes_2022_tocando_fcu)}):** percentil médio "
                         f"declive {pct(zz.A0_regra_pct_medio_cicatrizes)}, Idai {pct(zz.A1_log_pct_medio_cicatrizes)}, "
                         f"Idai+cicatrizes fora do bairro {pct(zz.E1_A3_log_l25_oof_pct_medio_cicatrizes)}; recuperadas no quintil superior: "
                         f"{int(zz.A0_regra_cicatrizes_recuperadas_q20)} / {int(zz.A1_log_cicatrizes_recuperadas_q20)} / "
                         f"{int(zz.E1_A3_log_l25_oof_cicatrizes_recuperadas_q20)}.")
        ag = c["agua"]
        L.append(f"- **Água (inferida, {ag['maturidade']}):** escore médio {pct(ag['escore_medio'])}, máximo {pct(ag['escore_max'])}. {ag['nota']}.")
        cx = c["conexao"]
        L.append(f"- **Conexão (inferida):** {cx['area_montante_km2']:.3f} km² drenam para a FCU a até 1 km, "
                 f"{cx['area_montante_quintil_superior_encosta_km2']:.3f} km² no quintil superior de encosta; "
                 f"{cx['feicoes_charter2022_a_montante']} feição(ões) de 2022 a montante. {cx['nota']}.")
        if enc["desconhecido"]:
            L.append("- **Desconhecido:** " + "; ".join(enc["desconhecido"]) + ".")
        L.append("- **Observação necessária:** " + "; ".join(c["observacao_necessaria"]) + ".")
        L.append("")
    sem = t[(t.charter2022_deslizamentos == 0) & (t.ufrj_relatos_100m == 0)]
    L += ["## Comunidades sem registro observado de encosta nas fontes disponíveis", "",
          "Ausência de registro não é ausência do fenômeno. Para estas, a ferramenta oferece só inferência e contexto técnico:", ""]
    for cod, r in sem.sort_values("encosta_E1_pct_mediano", ascending=False).iterrows():
        L.append(f"- {r.NM_FCU} ({r.populacao_censo2022} moradores): percentil E1 {pct(r.encosta_E1_pct_mediano)}, "
                 f"zoneamento 2025 {r.zoneamento2025_fracao_area:.0%}, domínio {r.dominio_frac_dentro_doador:.0%}.")
    (TER / "casos_territoriais.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    e.saida(TER / "casos_territoriais.md")
    e.fechar(casos=len(CASOS), sem_registro=len(sem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
