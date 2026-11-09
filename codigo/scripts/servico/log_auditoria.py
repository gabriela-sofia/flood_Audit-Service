"""
Log de auditoria por requisicao -- JSON Lines, append-only, local.

=========================================================================
A REGRA QUE VALE ACIMA DE TODAS AS OUTRAS NESTE ARQUIVO:

    ISTO E LOG DE USO DO PRODUTO, NUNCA DADO DE TREINO.

Nenhuma linha gravada aqui pode um dia virar linha de uma tabela que alimenta
ajuste de modelo. Se isso acontecer, o proprio uso do servico passa a
realimentar o que o servico prediz, e vira exatamente a contaminacao que o
projeto audita em toda fonte externa -- so que vinda de dentro, e mais dificil
de enxergar depois.

A regra e repetida no cabecalho de cada arquivo de log gravado, e nao apenas
aqui, porque quem for varrer `logs/` daqui a um ano pode nunca abrir este
modulo.
=========================================================================

-------------------------------------------------------------------------
O QUE ENTRA NA LINHA, e por que so isso
-------------------------------------------------------------------------

timestamp UTC, geometria e CRS como vieram, regiao declarada ou deduzida,
status, e -- conforme o desfecho -- ou o portao que nao fechou com o detalhe,
ou o escore com IC e maturidade. Sempre a versao do contrato.

O que NAO entra: `model_card` e `explicacao` completos. Os dois sao funcao do
artefato servido mais a requisicao, ambos ja registrados; regravar por
requisicao e redundancia sem ganho de auditoria, e infla o arquivo com o texto
que menos varia.

-------------------------------------------------------------------------
ONDE MORA, e por que nao em local_runs/
-------------------------------------------------------------------------

`logs/` na raiz, git-ignored. `local_runs/` tambem e git-ignored, mas esta
semanticamente reservado a saida de execucao da linha cientifica -- ha scripts
que varrem aquele diretorio inteiro procurando artefato de rodada. Log de uso
do produto nao e artefato de rodada, e misturar os dois e o primeiro passo para
alguem tratar um como o outro. A separacao e por convencao, e a convencao esta
escrita aqui para poder ser cobrada.

-------------------------------------------------------------------------
PRIVACIDADE -- diferente da regra do cartao
-------------------------------------------------------------------------

O SVC-04 publica BAIRRO e nunca lat/lon, porque o cartao e material publico. O
log e o oposto: ele PRECISA da geometria recebida para servir de auditoria --
sem a coordenada nao da para reconstruir por que aquela resposta saiu.

A consequencia e que o log e dado sensivel, e nao pode ser publicado nem
exposto do jeito que o cartao e. Enquanto o servico roda em localhost as
coordenadas sao da propria autora. No dia em que sair disso, coordenada de
ocorrencia e dado pessoal (LGPD), e retencao e acesso passam a exigir decisao
-- que nao e urgente agora, mas nao pode ser lembrada depois do fato.
`ALERTA_PRIVACIDADE` fica no cabecalho de todo arquivo para isso.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"

NAO_E_DADO_DE_TREINO = (
    "log de uso do produto. NUNCA use nenhuma linha deste arquivo como dado de "
    "treino, rotulo ou feature: o servico passaria a realimentar o que prediz")
ALERTA_PRIVACIDADE = (
    "contem a geometria recebida, incluindo coordenada -- necessaria para "
    "auditoria. Dado sensivel: nao publicar, nao versionar, nao expor")


def agora_iso() -> str:
    """UTC em ISO 8601 com Z, o mesmo formato do prototipo de julho."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LogDeAuditoria:
    """Uma linha JSON por requisicao, append-only.

    Nao e infraestrutura de observabilidade, e nao pretende ser: e o minimo que
    ainda permite reconstruir, depois, por que uma resposta saiu como saiu.
    """

    def __init__(self, diretorio: Path = LOGS, nome: str | None = None):
        self.diretorio = Path(diretorio)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        dia = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.caminho = self.diretorio / (nome or f"inferencia_{dia}.jsonl")
        self._garantir_cabecalho()

    def _garantir_cabecalho(self) -> None:
        """Primeira linha do arquivo declara o que ele e -- e o que nao e.

        Quem abrir este arquivo sem ter lido este modulo precisa encontrar a
        regra antes do primeiro registro.
        """
        if self.caminho.exists() and self.caminho.stat().st_size > 0:
            return
        cabecalho = {
            "_cabecalho": True,
            "gerado_por": "scripts/servico/log_auditoria.py",
            "criado_em": agora_iso(),
            "nao_e_dado_de_treino": NAO_E_DADO_DE_TREINO,
            "privacidade": ALERTA_PRIVACIDADE,
            "formato": "JSON Lines, append-only, uma linha por requisicao",
        }
        self._escrever(cabecalho)

    def _escrever(self, linha: dict) -> None:
        with self.caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def registrar(self, requisicao: dict, resposta: dict,
                  cliente: str | None = None) -> dict:
        """Monta e grava a linha. Devolve o que gravou, para o teste conferir."""
        geo = requisicao.get("geometria") or {}
        pontos = geo.get("pontos") or []
        linha: dict = {
            "quando": agora_iso(),
            "cliente": cliente,
            "contrato": resposta.get("contrato"),
            "crs": geo.get("crs"),
            "tipo_de_geometria": geo.get("tipo"),
            "n_pontos": len(pontos),
            # a geometria como veio, sem arredondar: auditoria precisa do que
            # foi recebido, e nao de uma versao limpa dele
            "geometria_recebida": [{"lat": p.get("lat"), "lon": p.get("lon")}
                                   for p in pontos],
            "camadas_vieram_na_requisicao": any(
                p.get("camadas") for p in pontos),
            "cenario_de_chuva": requisicao.get("cenario_de_chuva"),
            "regiao_declarada": requisicao.get("regiao"),
            "regiao_resolvida": resposta.get("regiao"),
            "status": resposta.get("status"),
        }
        if resposta.get("status") == "ok":
            linha.update({
                "escore": resposta.get("escore"),
                "ic95": resposta.get("ic95"),
                "maturidade": resposta.get("maturidade"),
                "variaveis_em_extrapolacao": (
                    resposta.get("dominio") or {}).get("variaveis_em_extrapolacao"),
            })
        else:
            linha.update({
                "gate_que_nao_fechou": resposta.get("gate_que_nao_fechou"),
                "detalhe": resposta.get("detalhe"),
            })
        self._escrever(linha)
        return linha

    def ler(self) -> list[dict]:
        """Registros gravados, sem o cabecalho. Existe para o teste."""
        if not self.caminho.exists():
            return []
        linhas = [json.loads(l) for l in
                  self.caminho.read_text(encoding="utf-8").splitlines() if l]
        return [l for l in linhas if not l.get("_cabecalho")]
