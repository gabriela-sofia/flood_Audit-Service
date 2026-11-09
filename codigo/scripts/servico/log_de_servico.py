"""
Redireciona a saida do processo para um arquivo, quando ele roda como servico.

POR QUE ISTO EXISTE, e por que nao e `>> arquivo.log` no lancador:

o Agendador de Tarefas desta maquina nao consegue lancar `cmd.exe`. Uma acao
apontada para `cmd /c ...` -- direto ou por dentro de um `.vbs` -- fica
pendurada sem executar nada e sem escrever linha nenhuma, e a tarefa aparece
como "Running" para sempre com nenhum processo por baixo. Quem sobe de pe
naquele contexto e o executavel chamado direto, e chamada direta nao tem para
onde redirecionar saida.

Entao o redirecionamento passa a ser do processo, e nao do shell que o chamou.
Isso e melhor por uma razao que independe do defeito: servico que sobe sozinho
e servico que ninguem ve falhar. Sem log, o unico sintoma de um `svc05` que
morreu no startup e uma porta que nao responde.

O QUE ELE NAO FAZ:

nao rotaciona por tempo, nao comprime, nao manda para lugar nenhum. Corta o
arquivo quando ele passa do teto e recomeca -- o suficiente para o disco nao
encher com um servico esquecido no ar, e pouco o bastante para nao virar uma
peca de infraestrutura que precise de manutencao propria.
"""

from __future__ import annotations

import sys
from pathlib import Path

TETO_BYTES = 5 * 1024 * 1024


def caminho_do_argumento(argv: list[str] | None = None) -> Path | None:
    """Le `--log <caminho>` da linha de comando, se estiver la."""
    argv = sys.argv if argv is None else argv
    if "--log" not in argv:
        return None
    i = argv.index("--log") + 1
    if i >= len(argv):
        return None
    return Path(argv[i])


def redirecionar(caminho: Path, teto: int = TETO_BYTES) -> None:
    """Manda `stdout` e `stderr` para o arquivo, em modo append.

    Sem buffer, de proposito: o valor de um log de servico esta em ler o que
    aconteceu ANTES do processo morrer, e saida ainda em buffer no momento da
    morte e saida perdida.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)

    # Corta antes de abrir. Cortar com o arquivo aberto deixaria o descritor
    # apontando para um deslocamento que nao existe mais, e o log seguiria
    # crescendo com um bloco de zeros na frente.
    if caminho.exists() and caminho.stat().st_size > teto:
        guardado = caminho.read_bytes()[-(teto // 2):]
        caminho.write_bytes(b"[log cortado ao passar do teto]\n" + guardado)

    destino = open(caminho, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = destino
    sys.stderr = destino


def redirecionar_se_pedido(argv: list[str] | None = None) -> Path | None:
    """O que os `main()` chamam: age so quando `--log` foi passado."""
    caminho = caminho_do_argumento(argv)
    if caminho is None:
        return None
    redirecionar(caminho)
    return caminho
