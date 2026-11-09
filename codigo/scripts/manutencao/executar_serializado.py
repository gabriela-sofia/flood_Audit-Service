"""Um trabalho pesado por vez nesta maquina, com trava em arquivo.

-------------------------------------------------------------------------
POR QUE ESTE ARQUIVO EXISTE
-------------------------------------------------------------------------

Em 18/09/2026 rodaram ao mesmo tempo: a fila de terreno (rede + WhiteboxTools), a
campanha de treino (`cems07` sozinho passa de uma hora), o retreino do encoder
(6 threads, 2,2 GB) e a suite inteira. A maquina tem 15,5 GB e, com o navegador,
o Drive e o resto abertos, sobrava menos de 1 GB. O sintoma nao foi erro: foi
tudo ficando lento ao mesmo tempo, e a epoca do encoder passando de 22 para 60
minutos.

A trava resolve a parte que e do projeto. Ela nao acelera nada -- ela impede que
dois trabalhos pesados disputem RAM, disco e rede, o que em maquina apertada
custa mais do que esperar a vez.

-------------------------------------------------------------------------
POR QUE ARQUIVO, E NAO UM SERVICO DE FILA
-------------------------------------------------------------------------

O projeto ja espera por MARCADOR em log (`campanha01 --esperar-log`), e nao por
processo, pela mesma razao: processo morto por falta de memoria some, e arquivo
fica. A trava aqui grava quem a tomou, quando e com que comando, entao uma trava
orfa (a maquina reiniciou no meio, como em 18/09) e diagnosticavel e removivel
com `--forcar`, em vez de virar mistério.

-------------------------------------------------------------------------
E A BATERIA, QUE E O CUSTO QUE DECIDIU TUDO ISTO
-------------------------------------------------------------------------

Declarado pela Gabriela em 22/09/2026: o que pesa nao e o disco, e a bateria do
notebook. Derivar terreno e treinar mantem CPU e disco em carga por horas, e num
portatil isso e a diferenca entre trabalhar e ficar preso na tomada.

Entao o padrao inverteu: trabalho pesado so comeca NA TOMADA. Fora dela o
comando recusa com codigo 3 e diz por que, e `--mesmo-na-bateria` existe para
quando a decisao for consciente (uma fatia curta, um teste). A trava nao
interrompe o que ja esta rodando se o cabo cair -- ela decide na largada, que e
onde a decisao e barata.

Nao mata processo, nao reordena fila e nao decide prioridade: quem chega
primeiro roda, e quem chega depois espera ou desiste.

Uso:
    python scripts/manutencao/executar_serializado.py -- python scripts/...py
    python scripts/manutencao/executar_serializado.py --esperar-min 0 -- <cmd>
    python scripts/manutencao/executar_serializado.py --estado
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAVA = REPO / "local_runs" / ".trava-pesada.json"
INTERVALO_S = 30


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def energia() -> dict:
    """Esta na tomada? Lido do Windows por `GetSystemPowerStatus`.

    Sem subprocesso: chamar o PowerShell a cada verificacao custaria mais do que
    a propria decisao. Onde a API nao existe (outro sistema, desktop sem
    bateria), devolve `na_tomada=True` -- a guarda existe para proteger bateria,
    e maquina sem bateria nao tem o que proteger.
    """
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte),
                    ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong)]

    if os.name != "nt":
        return {"na_tomada": True, "por_que": "fora do Windows, sem leitura"}
    st = _Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return {"na_tomada": True, "por_que": "GetSystemPowerStatus falhou"}
    if st.ACLineStatus == 255:
        return {"na_tomada": True, "por_que": "estado de energia desconhecido"}
    carga = (None if st.BatteryLifePercent in (255,)
             else int(st.BatteryLifePercent))
    return {"na_tomada": st.ACLineStatus == 1, "carga_pct": carga,
            "por_que": "tomada" if st.ACLineStatus == 1 else "bateria"}


def _viva(pid: int) -> bool:
    """O dono da trava ainda existe? Sem isso, reinicio deixa trava eterna.

    No Windows pergunta ao proprio sistema (OpenProcess + GetExitCodeProcess), sem
    subprocesso. Historia: (1) a versao original procurava o PID como SUBSTRING da
    saida do tasklist -- pid 300 casava com "6.300 K"; (2) a versao de 07/10 lia a
    coluna do CSV do tasklist e, as 16:37:20 UTC de 07/10, devolveu "morto" para o
    pid 22608 que ainda rodava (ele mesmo registrou a liberacao 28 s depois): a trava
    viva foi afastada como orfa e dois trabalhos pesados rodaram juntos.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            # ERROR_INVALID_PARAMETER (87): nao existe. Qualquer outro erro (ex.: acesso
            # negado) significa que existe e nao podemos olhar: na duvida, vivo.
            return ctypes.get_last_error() != 87
        try:
            codigo = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(codigo)):
                return True
            return codigo.value == 259                     # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _morta(pid: int, confirmacoes: int = 2, pausa_s: float = 2.0) -> bool:
    """Orfa so com morte CONFIRMADA: um falso "morto" afasta trava viva, e isso e pior que esperar."""
    for k in range(confirmacoes):
        if _viva(pid):
            return False
        if k + 1 < confirmacoes:
            time.sleep(pausa_s)
    return True


def _diario(evento: str, trava: Path, **dados) -> None:
    """Append-only ao lado da trava: quem tomou, esperou, forcou, liberou. Em 06/10/2026
    dois trabalhos pesados rodaram juntos e nada registrava como a trava trocou de dono."""
    try:
        with trava.with_suffix(".log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"quando": _agora(), "evento": evento, "pid": os.getpid(), **dados},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass


def estado(trava: Path = None) -> dict | None:
    trava = trava or TRAVA
    if not trava.exists():
        return None
    try:
        d = json.loads(trava.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return {"pid": -1, "comando": "trava ilegivel", "viva": False}
    d["viva"] = _viva(int(d.get("pid", -1)))
    return d


def _criar(trava: Path, conteudo: dict) -> bool:
    """Criacao ATOMICA: O_EXCL falha se outro processo criou no mesmo instante.

    A versao anterior lia o estado e depois escrevia: dois processos que chegassem
    juntos veriam "livre" e os dois rodariam.
    """
    trava.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(trava, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(conteudo, ensure_ascii=False))
    return True


def _afastar(trava: Path, motivo: str) -> None:
    """Tira a trava atual do caminho sem apagar: renomeia (atomico) para o historico."""
    destino = trava.with_name(f"{trava.stem}.{motivo}.{os.getpid()}.{int(time.time() * 1000)}.json")
    try:
        os.replace(trava, destino)
    except FileNotFoundError:
        pass


def tomar(comando: list, esperar_min: float, forcar: bool, trava: Path = None,
          intervalo_s: float = None) -> dict | None:
    """Devolve o conteudo gravado (com `token`) se tomou; None se desistiu."""
    trava = trava or TRAVA
    intervalo_s = INTERVALO_S if intervalo_s is None else intervalo_s
    limite = time.time() + esperar_min * 60
    conteudo = {"pid": os.getpid(), "quando": _agora(), "comando": " ".join(comando),
                "token": f"{os.getpid()}-{time.time_ns()}"}
    while True:
        if _criar(trava, conteudo):
            _diario("tomou", trava, comando=conteudo["comando"])
            return conteudo
        atual = estado(trava)
        if atual is None:
            continue  # liberada entre a tentativa e a leitura: tenta de novo
        if not atual["viva"] and _morta(int(atual.get("pid", -1))):
            print(f"trava orfa de {atual.get('quando')} (pid {atual.get('pid')}) afastada", flush=True)
            _diario("orfa_afastada", trava, dono=atual.get("pid"), comando=atual.get("comando"))
            _afastar(trava, "orfa")
            continue
        if forcar:
            # O dono anterior CONTINUA rodando: --forcar nao mata nada, so deixa os dois juntos.
            print(f"FORCADO: pid {atual['pid']} continua rodando ({atual.get('comando', '')[:80]})", flush=True)
            _diario("forcou", trava, dono=atual.get("pid"), comando=conteudo["comando"])
            _afastar(trava, "forcada")
            continue
        print(f"esperando: pid {atual['pid']} roda desde {atual['quando']} "
              f"({atual['comando'][:80]})", flush=True)
        if time.time() >= limite:
            _diario("desistiu", trava, dono=atual.get("pid"), comando=conteudo["comando"])
            return None
        time.sleep(intervalo_s)


def liberar(conteudo: dict, trava: Path = None) -> bool:
    """So o dono libera. A versao anterior apagava a trava que estivesse la: em 06/10/2026 o
    fim de um trabalho apagou a trava de outro que ainda rodava, e ele ficou sem trava."""
    trava = trava or TRAVA
    atual = estado(trava)
    if atual is None:
        _diario("liberar_sem_trava", trava)
        return False
    if atual.get("token") != conteudo.get("token"):
        _diario("nao_liberou_alheia", trava, dono=atual.get("pid"))
        return False
    _afastar(trava, "liberada")
    _diario("liberou", trava)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--esperar-min", type=float, default=240.0,
                   help="quanto esperar pela vez; 0 desiste na hora")
    p.add_argument("--forcar", action="store_true",
                   help="toma a trava mesmo ocupada; use so quando souber por que")
    p.add_argument("--estado", action="store_true")
    p.add_argument("--mesmo-na-bateria", action="store_true",
                   help="roda fora da tomada; use quando a fatia for curta")
    p.add_argument("comando", nargs=argparse.REMAINDER)
    a = p.parse_args()

    if a.estado:
        print(json.dumps(estado(), indent=2, ensure_ascii=False))
        return 0

    comando = [c for c in a.comando if c != "--"]
    if not comando:
        p.print_help()
        return 0

    e = energia()
    if not e["na_tomada"] and not a.mesmo_na_bateria:
        carga = e.get("carga_pct")
        print(f"RECUSADO: fora da tomada ({carga}% de carga). Trabalho pesado "
              "gasta bateria por horas; ligue o cabo ou use "
              "--mesmo-na-bateria.")
        return 3

    dono = tomar(comando, a.esperar_min, a.forcar)
    if dono is None:
        print("DESISTIU: a maquina ja tem um trabalho pesado rodando")
        return 2
    t0 = time.time()
    try:
        r = subprocess.run(comando, cwd=REPO)
    finally:
        liberar(dono)
    print(f"[serializado] rc={r.returncode} em {time.time() - t0:.0f}s")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
