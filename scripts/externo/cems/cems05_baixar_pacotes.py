"""
CEMS-05 -- Baixa os pacotes vetoriais das ativacoes. Sem clique manual.

POR QUE ISTO EXISTE:

o `n1e_cems_vetores.py` tentou seis padroes de URL, todos falharam, e o projeto
concluiu que o download do pacote era manual. A conclusao estava errada, e o
caminho foi encontrado em 11/08/2026 lendo o bundle do proprio portal:

    dashboard-api/public-activations/?code=EMSR847
        -> campo `productsPath`
        -> https://rapidmapping.emergency.copernicus.eu/backend/EMSR847/EMSR847_products.zip

Sem autenticacao. O parametro que o portal acrescenta na URL (`?${Pa(!1)}`) e
credencial de usuario logado e e OPCIONAL -- o arquivo baixa sem ele.

DUAS ARMADILHAS REGISTRADAS:

  1. HEAD responde erro nesta rota. Se o teste for por HEAD, a conclusao e
     "nao existe" -- e foi exatamente esse o erro que custou 163 minutos no
     incidente do CPL_VSIL_CURL_USE_HEAD. A verificacao correta e GET com
     `Range: bytes=0-200`, que devolve 206 + Content-Range com o tamanho total.

  2. `?activation=` e `?code=` sao aceitos e IGNORADOS pelo endpoint `aois/`.
     Aqui, no `public-activations/`, `?code=` funciona. Sao endpoints
     diferentes com convencoes diferentes; nao presuma.

INTEGRIDADE, e por que e obrigatoria: um zip truncado por queda de rede abre
sem erro em muitas ferramentas e so falha la na frente, no `cems02`, com uma
mensagem que nao aponta para a causa. Aqui todo arquivo e testado com
`zipfile.testzip()` antes de ser aceito, e o tamanho e conferido contra o
Content-Length declarado. Arquivo que nao passa e removido, nao renomeado.

RESUMIVEL: baixa em `.parcial` e retoma por `Range`. Arquivo ja completo e
verificado e pulado.

NAO faz: nao extrai, nao processa, nao promove rotulo. Baixa e verifica.

Uso:
    python scripts/externo/cems05_baixar_pacotes.py --listar
    python scripts/externo/cems05_baixar_pacotes.py --codigos EMSR847,EMSR796
    python scripts/externo/cems05_baixar_pacotes.py --prioridade-petropolis
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "cems-05-pacotes"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
UA = {"User-Agent": "Mozilla/5.0"}

# Ativacoes com AOI de serra verificada pelo cems04 (relevo>=400m e >=25% acima
# de 15 graus), tropicais, ordenadas por numero de AOIs que passaram.
PRIORIDADE_PETROPOLIS = [
    ("EMSR847", "Haiti, Cuba, Jamaica", 7),
    ("EMSR796", "Equador", 6),
    ("EMSR702", "Vanuatu", 5),
    ("EMSR778", "Honduras", 3),
    ("EMSR734", "Caribe", 2),
    ("EMSR789", "Equador", 2),
    ("EMSR805", "Mexico", 2),
    ("EMSR813", "Equador", 1),
]

TAMANHO_MINIMO_ACEITAVEL = 200_000   # abaixo disto o pacote nao tem vetor util
BLOCO = 1 << 20


def _json(url: str, tentativas: int = 4):
    ultimo = None
    for k in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001
            ultimo = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url} -> {type(ultimo).__name__}: {ultimo}")


def info(codigo: str) -> dict:
    d = _json(f"{API}/public-activations/?code={codigo}")
    r = d.get("results") or []
    if not r:
        raise RuntimeError(f"{codigo}: ativacao nao encontrada na API")
    a = r[0]
    if a.get("code") != codigo:
        raise RuntimeError(f"{codigo}: API devolveu {a.get('code')} -- filtro ignorado")
    # ATENCAO: `countries` vem como lista de DICTS aqui, e como lista de STRINGS
    # em `public-activations-info/`. Mesmo nome de campo, formato diferente,
    # nos dois endpoints da mesma API.
    def _pais(x):
        return x.get("name") or x.get("code") or str(x) if isinstance(x, dict) else str(x)

    return {"code": codigo, "nome": a.get("name"),
            "paises": ", ".join(_pais(c) for c in (a.get("countries") or [])),
            "n_aois": len(a.get("aois") or []),
            "data": str(a.get("eventTime"))[:10],
            "url": a.get("productsPath")}


def tamanho_remoto(url: str) -> int | None:
    """GET com Range. NAO use HEAD nesta rota -- responde erro mesmo existindo."""
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-200"})
        x = urllib.request.urlopen(req, timeout=45)
        cr = x.headers.get("Content-Range") or ""
        return int(cr.split("/")[-1]) if "/" in cr else None
    except Exception:  # noqa: BLE001
        return None


def zip_valido(p: Path) -> bool:
    try:
        with zipfile.ZipFile(p) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except Exception:  # noqa: BLE001
        return False


def baixar(codigo: str) -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        meta = info(codigo)
    except Exception as exc:  # noqa: BLE001
        print(f"[{codigo}] FALHA_META {exc}")
        return False
    url = meta["url"]
    if not url:
        print(f"[{codigo}] SEM productsPath na API")
        return False

    destino = OUT / f"{codigo}_products.zip"
    parcial = OUT / f"{codigo}_products.zip.parcial"

    if destino.exists() and zip_valido(destino):
        print(f"[{codigo}] JA_BAIXADO {destino.stat().st_size/1024**2:.1f}MB")
        return True

    total = tamanho_remoto(url)
    if total is None:
        print(f"[{codigo}] FALHA: servidor nao devolveu tamanho. Nao vou baixar as cegas.")
        return False
    if total < TAMANHO_MINIMO_ACEITAVEL:
        print(f"[{codigo}] RECUSADO: pacote de {total}B e pequeno demais para ter vetor util")
        return False

    ja = parcial.stat().st_size if parcial.exists() else 0
    if ja >= total:
        ja = 0
        parcial.unlink(missing_ok=True)

    print(f"[{codigo}] {meta['paises']} | {meta['n_aois']} AOIs | "
          f"{total/1024**2:.1f}MB" + (f" (retomando de {ja/1024**2:.1f}MB)" if ja else ""))

    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={ja}-"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(parcial, "ab") as f:
            lido = ja
            while True:
                b = r.read(BLOCO)
                if not b:
                    break
                f.write(b)
                lido += len(b)
                if lido % (32 * BLOCO) < BLOCO:
                    pct = 100 * lido / total
                    vel = (lido - ja) / max(time.time() - t0, 1e-9) / 1024**2
                    print(f"    {pct:5.1f}%  {lido/1024**2:7.1f}/{total/1024**2:.1f}MB  "
                          f"{vel:.1f}MB/s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{codigo}] INTERROMPIDO {type(exc).__name__} -- rode de novo para retomar")
        return False

    obtido = parcial.stat().st_size
    if obtido != total:
        print(f"[{codigo}] TRUNCADO {obtido} != {total} -- rode de novo para retomar")
        return False
    if not zip_valido(parcial):
        parcial.unlink(missing_ok=True)
        print(f"[{codigo}] ZIP_CORROMPIDO -- removido. Rode de novo.")
        return False

    parcial.replace(destino)
    print(f"[{codigo}] OK {total/1024**2:.1f}MB em {(time.time()-t0)/60:.1f}min -> {destino}")
    (OUT / f"{codigo}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def listar() -> int:
    print("===CEMS05_LISTAR===")
    print(f"{'code':9s} {'MB':>8s} {'AOIs':>5s}  pais / motivo")
    tot = 0
    for c, desc, n_ing in PRIORIDADE_PETROPOLIS:
        try:
            m = info(c)
            t = tamanho_remoto(m["url"]) or 0
        except Exception as exc:  # noqa: BLE001
            print(f"{c:9s} FALHA {exc}")
            continue
        tot += t
        print(f"{c:9s} {t/1024**2:8.1f} {m['n_aois']:5d}  {desc} "
              f"({n_ing} AOIs de serra)")
        print(f"          {m['url']}")
    print(f"\nTOTAL={tot/1024**3:.2f}GB")
    print("===END===")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if "--listar" in args:
        return listar()
    if "--codigos" in args:
        codigos = args[args.index("--codigos") + 1].split(",")
    elif "--prioridade-petropolis" in args:
        codigos = [c for c, _, _ in PRIORIDADE_PETROPOLIS]
    else:
        print(__doc__)
        return 0

    print(f"===CEMS05_BAIXAR=== alvos={len(codigos)}")
    ok = [c for c in codigos if baixar(c.strip())]
    print(f"\nBAIXADOS={len(ok)}/{len(codigos)}")
    if ok:
        print("\nProximo passo, para cada pacote:")
        for c in ok:
            print(f"  python scripts/externo/cems02_pipeline_ativacao.py "
                  f"--zip local_runs/cems-05-pacotes/{c}_products.zip --codigo {c}")
    print("===END===")
    return 0 if len(ok) == len(codigos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
