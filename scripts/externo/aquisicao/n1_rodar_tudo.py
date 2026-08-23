"""
NIVEL 1 -- orquestrador: executa as quatro aquisicoes e consolida o relatorio.

O Nivel 1 e o conjunto de fontes que resolve a lacuna L1 do projeto --
NEGATIVO POR OBSERVACAO GARANTIDA -- que e o gargalo real documentado no gate
C4_BLOCKED_NO_FORMAL_NEGATIVES. Cada uma cobre uma faixa diferente do
problema, e por isso as quatro sao necessarias e nenhuma substitui a outra:

  A. Sen1Floods11  10 m,  global,  11 eventos   -> rotulo manual 3 estados
  B. UFO            3 m,  urbano,  14 eventos   -> unica em contexto urbano
  C. Copernicus EMS metrico, mundial, 490+ ativ -> AOI declarada = negativo
  D. GFD          250 m,  global, 913 eventos   -> negativo em escala

Ordem de execucao: da menor para a maior, para que uma falha de rede numa
base grande nao impeca as pequenas de completarem.

Este script NAO promove nada a negativo, NAO extrai variavel e NAO treina.
Ele adquire, inventaria e registra proveniencia.

Uso:
    python scripts/externo/n1_rodar_tudo.py
    python scripts/externo/n1_rodar_tudo.py --gfd-completo
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
RELATORIO = REPO / "local_runs" / "n1-relatorio-consolidado.md"


def rodar(script: str, args: list[str]) -> tuple[int, str]:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(AQUI / script)] + args,
        capture_output=True, text=True, cwd=str(REPO),
    )
    dt = time.time() - t0
    saida = proc.stdout or ""
    # so a parte sintetizada entre os marcadores
    linhas = [ln for ln in saida.splitlines() if ln.strip()]
    bloco = []
    dentro = False
    for ln in linhas:
        if ln.startswith("==="):
            if ln.strip() == "===END===":
                bloco.append(ln)
                dentro = False
                continue
            dentro = True
        if dentro:
            bloco.append(ln)
    texto = "\n".join(bloco) if bloco else saida[-2000:]
    if proc.returncode != 0 and proc.stderr:
        texto += "\n[STDERR] " + proc.stderr.strip()[-500:]
    return proc.returncode, f"{texto}\n[tempo] {dt/60:.1f} min"


def main() -> int:
    gfd_completo = "--gfd-completo" in sys.argv

    etapas = [
        ("n1b_ufo_zenodo.py", [], "B. Urban Flood Observations (3 m, urbano)"),
        ("n1c_cems_ativacoes.py", [], "C. Copernicus EMS (AOI declarada)"),
        ("n1a_sen1floods11.py", [], "A. Sen1Floods11 (10 m, rotulo 3 estados)"),
        ("n1d_global_flood_database.py",
         ["--baixar"] if gfd_completo else [],
         "D. Global Flood Database (250 m, 913 eventos)"),
    ]

    partes = [
        "# Nivel 1 -- Negativo por observacao garantida",
        "",
        f"Execucao: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Estas quatro fontes existem para resolver a lacuna L1 do REV-P: ate",
        "aqui o negativo era construido por AUSENCIA de registro (Protocolo C),",
        "o que e fragil por construcao e sustenta o gate",
        "C4_BLOCKED_NO_FORMAL_NEGATIVES. Todas as fontes abaixo trazem, no",
        "proprio dado, a afirmacao de que uma area foi observada e nao estava",
        "inundada.",
        "",
        "Nenhum negativo foi promovido nesta etapa. Nenhuma variavel foi",
        "extraida. Nenhum modelo foi treinado.",
        "",
    ]

    status = []
    for script, args, titulo in etapas:
        print(f"\n########## {titulo} ##########", flush=True)
        rc, texto = rodar(script, args)
        print(texto, flush=True)
        status.append((titulo, rc))
        partes += [f"## {titulo}", "", "```", texto, "```", ""]

    partes += ["## Status", ""]
    for titulo, rc in status:
        partes.append(f"- {'OK ' if rc == 0 else 'FALHA'} -- {titulo} (exit={rc})")
    partes += [
        "",
        "## Proxima tarefa (uma so)",
        "",
        "Adjudicacao do negativo: definir, com criterio escrito e auditavel,",
        "como um pixel/ponto observado-e-nao-inundado destas fontes vira",
        "negativo do dataset SUSC. Isso e decisao metodologica, nao aquisicao,",
        "e por isso nao foi feita aqui.",
        "",
    ]

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO.write_text("\n".join(partes), encoding="utf-8")

    print("\n===N1_CONSOLIDADO===")
    for titulo, rc in status:
        print(f"{'OK' if rc == 0 else 'FALHA'} exit={rc} :: {titulo}")
    print(f"RELATORIO={RELATORIO}")
    print("===END===")
    return 0 if all(rc == 0 for _, rc in status) else 2


if __name__ == "__main__":
    raise SystemExit(main())
