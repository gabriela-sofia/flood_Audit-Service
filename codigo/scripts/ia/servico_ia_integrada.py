"""Transporte local da IA integrada. Política científica permanece no SVC-02.

python scripts/ia/servico_ia_integrada.py --porta 8131
GET /ia/estado; POST /ia/consulta. Não inicia treinos por requisição HTTP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[2] / "local_runs/ia-integrada"


def motor():
    # Carregar a dependência neural só quando esta rota for usada. O serviço
    # legado conserva o ambiente de inferência sem obrigar PyTorch no startup.
    from ia_integrada import carregar, consultar
    return carregar, consultar


class Pedido(BaseModel):
    uso: str = "consulta_lugar_novo"
    quadro: str | None = None
    pontos: list[dict[str, float | None]] = Field(default_factory=list, max_length=5000)
    chip: str | None = None
    data_alvo: str | None = None
    profundidade: bool = False
    territorio: str | None = None
    municipio_id: str | None = None


def criar_router(pasta=RAIZ):
    pasta = Path(pasta)
    router = APIRouter()

    @router.get("/ia/estado")
    async def estado():
        try:
            carregar, _ = motor()
            _, meta = carregar(pasta)
        except (ImportError, OSError, ValueError, KeyError) as e:
            raise HTTPException(503, str(e)) from e
        return {"versao": meta["versao"], "peso": meta["peso"]["sha256"],
                "promovido": False, "aprendizado": meta["aprendizado"], "limites": meta["limites"]}

    @router.post("/ia/consulta")
    async def consulta(pedido: Pedido):
        try:
            _, consultar = motor()
            return consultar(pasta, pedido.model_dump())
        except (ValueError, KeyError, AttributeError) as e:
            raise HTTPException(422, str(e)) from e
        except (ImportError, OSError) as e:
            raise HTTPException(503, str(e)) from e

    return router


def criar_app(pasta=RAIZ):
    app = FastAPI(title="REV-P — IA integrada em desenvolvimento", version="v1")
    app.include_router(criar_router(pasta))
    return app


app = criar_app()


if __name__ == "__main__":
    import torch
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", type=Path, default=RAIZ)
    p.add_argument("--porta", type=int, default=8131)
    a = p.parse_args()
    torch.set_num_threads(2)
    uvicorn.run(criar_app(a.modelo), host="127.0.0.1", port=a.porta)
