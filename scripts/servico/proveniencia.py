"""
Camada comum de proveniencia do REV-P.

POR QUE ESTE ARQUIVO EXISTE:

antes dele, `hashlib.sha256` aparecia solto na camada de corpus, nos scripts de
modelo servido e em meia duzia de pipelines cientificos, cada um com uma
convencao propria de leitura (bytes? texto? JSON com que separador?). Dois
hashes do MESMO conteudo saiam diferentes, e um manifesto que nao reproduz o
hash de outro manifesto nao prova nada -- e pior que nao ter hash, porque
parece prova.

O que este modulo da e uma convencao unica e tres coisas:

  ARTEFATO   qualquer coisa com identidade e conteudo: um arquivo servido, um
             chunk do corpus, um payload de resposta, um relatorio.
  RELACAO    aresta tipada entre artefatos, no vocabulario fechado abaixo.
  MANIFESTO  o conjunto dos dois, serializado em JSON/JSONL deterministico.

Isto e PROV-lite: pega emprestado o VOCABULARIO do W3C PROV (entidade,
atividade, derivacao) sem carregar a ontologia inteira, que resolveria um
problema que o REV-P nao tem. Nao ha triplestore, nao ha SPARQL, nao ha
namespace. Ha um JSONL que um humano le no terminal e um teste compara.

-------------------------------------------------------------------------
A REGRA DO TIMESTAMP, que e a unica regra dificil aqui
-------------------------------------------------------------------------

`hash` NUNCA cobre o instante em que o registro foi escrito. Se cobrisse, o
mesmo conteudo teria hash diferente a cada rodada, e o manifesto deixaria de
servir para o que foi feito: dizer se algo MUDOU. O instante vive em
`timestamp_utc`, fora do conteudo hasheado, e e opcional -- so entra quando
alguem precisa saber QUANDO, e nunca quando alguem precisa saber SE.

O mesmo vale para caminho absoluto: o diretorio do usuario nao entra em hash
nenhum. Um corpus construido em duas maquinas tem de bater.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------ hashes
#
# Uma convencao, e o motivo de cada escolha:
#
#   sort_keys      dict em Python preserva ordem de insercao, entao o mesmo
#                  conteudo montado em ordem diferente daria hash diferente.
#   separators     sem espaco: `json.dumps` por padrao poe ", " e ": ", e a
#                  diferenca de whitespace nao e diferenca de conteudo.
#   ensure_ascii   False, para que a mesma palavra acentuada nao vire dois
#                  conteudos. UTF-8 na serializacao final.
#   default=str    numero numpy, Path e datetime viram texto em vez de
#                  estourar. Perde-se tipo, ganha-se que o manifesto sempre
#                  fecha -- e o tipo ja esta no campo `tipo`.

PREFIXO = "sha256:"


def sha256_bytes(b: bytes) -> str:
    return PREFIXO + hashlib.sha256(b).hexdigest()


def sha256_texto(s: str) -> str:
    """Texto e hasheado em UTF-8 com quebra de linha normalizada para LF.

    O repositorio ja trava LF no `.gitattributes`, mas um arquivo lido em
    Windows pode chegar aqui com CRLF dependendo de quem o escreveu. Hash que
    muda porque o editor mudou nao mede conteudo, mede editor.
    """
    return sha256_bytes(s.replace("\r\n", "\n").encode("utf-8"))


def sha256_arquivo(path: Path) -> str | None:
    """`None` quando o arquivo nao existe -- ausencia nao e hash de vazio."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return PREFIXO + h.hexdigest()


def canonicalizar(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha256_json(obj) -> str:
    return sha256_texto(canonicalizar(obj))


def agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


REPO = Path(__file__).resolve().parents[2]


def caminho_no_repo(caminho) -> str:
    """Caminho relativo ao repositorio quando ele esta dentro; absoluto senao.

    `Path.relative_to(REPO)` cru levanta `ValueError` para qualquer caminho
    fora do repositorio -- diretorio temporario de teste, disco externo, saida
    passada por parametro. Este e o MESMO defeito que ja quebrou o coletor de
    conhecimento e o ciclo de aquisicao, e nas duas vezes eu corrigi so a
    ocorrencia. Ele mora aqui agora para nao ter uma quarta.
    """
    caminho = Path(caminho)
    try:
        return caminho.relative_to(REPO).as_posix()
    except ValueError:
        return caminho.as_posix()


# --------------------------------------------------------------- vocabulario
#
# Fechado de proposito. Relacao nova entra por decisao, com um caso de uso
# concreto -- e nao porque quem escreveu a linha achou o nome bonito. Um
# vocabulario aberto vira, em tres meses, quinze sinonimos da mesma aresta.

RELACOES = {
    "DERIVADO_DE",     # artefato <- dataset/artefato anterior
    "GERADO_POR",      # resultado <- modelo/script/atividade
    "TREINADO_COM",    # modelo <- dataset
    "EXTRAIDO_DE",     # chunk <- documento
    "SUPORTADA_POR",   # resposta <- claim
    "SUPORTADO_POR",   # claim <- payload/chunk/artefato
}

TIPOS = {
    "arquivo", "modelo", "dataset", "documento", "chunk", "payload",
    "resposta", "claim", "relatorio", "grade", "corpus", "atividade",
}


@dataclass
class Artefato:
    """Uma coisa com identidade, conteudo e origem declarada.

    `hash` e do CONTEUDO. `id` e o nome estavel pelo qual outras arestas o
    chamam -- normalmente o caminho relativo ao repositorio, nunca absoluto.
    """

    id: str
    tipo: str
    hash: str | None = None
    versao: str | None = None
    origem: str | None = None
    gerado_por: str | None = None
    derivado_de: list = field(default_factory=list)
    timestamp_utc: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tipo not in TIPOS:
            raise ValueError(f"tipo de artefato desconhecido: {self.tipo!r}; "
                             f"conhecidos: {sorted(TIPOS)}")

    def conteudo_hasheavel(self) -> dict:
        """O que identifica o artefato, SEM o instante em que foi registrado."""
        return {"id": self.id, "tipo": self.tipo, "hash": self.hash,
                "versao": self.versao, "origem": self.origem,
                "gerado_por": self.gerado_por,
                "derivado_de": sorted(self.derivado_de),
                "extra": self.extra}

    def para_dict(self) -> dict:
        d = asdict(self)
        d["derivado_de"] = sorted(self.derivado_de)
        return d


@dataclass
class Relacao:
    origem: str
    relacao: str
    destino: str
    nota: str | None = None

    def __post_init__(self):
        if self.relacao not in RELACOES:
            raise ValueError(f"relacao desconhecida: {self.relacao!r}; "
                             f"conhecidas: {sorted(RELACOES)}")

    def para_dict(self) -> dict:
        return asdict(self)


class Grafo:
    """Registro de artefatos e arestas, com de-duplicacao por id.

    Nao e um banco: e uma lista com serializacao ordenada. O tamanho esperado
    e dezenas de nos, nao milhoes -- um grafo de proveniencia que precisasse
    de indice seria sinal de que o produto passou a registrar ruido em vez de
    cadeia.
    """

    def __init__(self, rotulo: str):
        self.rotulo = rotulo
        self._artefatos: dict = {}
        self._relacoes: list = []

    def registrar(self, artefato: Artefato) -> Artefato:
        anterior = self._artefatos.get(artefato.id)
        if anterior is not None and anterior.hash != artefato.hash:
            # Dois conteudos diferentes com o mesmo nome dentro de um mesmo
            # manifesto e ambiguidade, e ambiguidade em proveniencia e a
            # forma silenciosa de mentir. Falha alto.
            raise ValueError(
                f"artefato {artefato.id!r} ja registrado com hash diferente: "
                f"{anterior.hash} != {artefato.hash}")
        self._artefatos[artefato.id] = artefato
        return artefato

    def relacionar(self, origem: str, relacao: str, destino: str,
                   nota: str | None = None) -> Relacao:
        r = Relacao(origem=origem, relacao=relacao, destino=destino, nota=nota)
        if r.para_dict() not in [x.para_dict() for x in self._relacoes]:
            self._relacoes.append(r)
        return r

    @property
    def artefatos(self) -> list:
        return [self._artefatos[k] for k in sorted(self._artefatos)]

    @property
    def relacoes(self) -> list:
        return sorted(self._relacoes,
                      key=lambda r: (r.origem, r.relacao, r.destino))

    def orfaos(self) -> list:
        """Ids citados por uma aresta que nunca foram registrados.

        Aresta apontando para nada e o defeito mais comum de manifesto escrito
        a mao, e o unico que nao aparece na leitura casual.
        """
        conhecidos = set(self._artefatos)
        citados = {x for r in self._relacoes for x in (r.origem, r.destino)}
        return sorted(citados - conhecidos)

    def para_dict(self, com_timestamp: bool = True) -> dict:
        corpo = {
            "tipo": "manifesto_proveniencia_revp",
            "rotulo": self.rotulo,
            "artefatos": [a.para_dict() for a in self.artefatos],
            "relacoes": [r.para_dict() for r in self.relacoes],
            "orfaos": self.orfaos(),
            "vocabulario": sorted(RELACOES),
        }
        # O hash do grafo cobre o CONTEUDO dos nos e as arestas -- nunca o
        # instante. Ver o cabecalho: e o que permite comparar duas rodadas.
        corpo["hash_conteudo"] = sha256_json({
            "artefatos": [a.conteudo_hasheavel() for a in self.artefatos],
            "relacoes": [r.para_dict() for r in self.relacoes],
        })
        if com_timestamp:
            corpo["quando_utc"] = agora_utc()
        return corpo

    def gravar(self, destino: Path, com_timestamp: bool = True) -> Path:
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(self.para_dict(com_timestamp), indent=2,
                       ensure_ascii=False),
            encoding="utf-8")
        jsonl = destino.with_suffix(".jsonl")
        linhas = [canonicalizar({"registro": "artefato", **a.para_dict()})
                  for a in self.artefatos]
        linhas += [canonicalizar({"registro": "relacao", **r.para_dict()})
                   for r in self.relacoes]
        jsonl.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        return destino


# ------------------------------------------------- proveniencia de resposta

def artefato_de_arquivo(caminho: Path, tipo: str, raiz: Path | None = None,
                        **kw) -> Artefato:
    """Artefato a partir de um arquivo do repositorio, com id relativo."""
    caminho = Path(caminho)
    ident = caminho.name
    if raiz is not None:
        try:
            ident = str(caminho.resolve().relative_to(Path(raiz).resolve()))
        except ValueError:
            ident = caminho.name
    return Artefato(id=ident.replace("\\", "/"), tipo=tipo,
                    hash=sha256_arquivo(caminho),
                    origem="repositorio_local", **kw)


# Campos que a camada de comunicacao acrescenta DEPOIS da decisao cientifica.
# Eles nao podem entrar no hash do payload: senao o hash da resposta mudaria
# porque a frase de apresentacao mudou, e a pergunta que este hash responde e
# justamente "a CIENCIA mudou?".
CAMPOS_POSTERIORES = ("sintese", "auditoria_sintese", "proveniencia",
                      "fontes", "evidencia_visual")


def hash_do_payload_cientifico(resposta: dict) -> str:
    return sha256_json({k: v for k, v in resposta.items()
                        if k not in CAMPOS_POSTERIORES})


def bloco_de_proveniencia(resposta: dict, modelos_dir: Path | None = None,
                          extras: list | None = None) -> dict:
    """O bloco `proveniencia` opcional que a API pode anexar a uma resposta.

    ADITIVO POR CONTRATO: nao le nem reescreve nenhum campo cientifico. Se
    esta funcao levantar, quem chama devolve a resposta sem o bloco -- a
    cadeia de proveniencia e informacao SOBRE a resposta, nunca condicao para
    ela existir.
    """
    grafo = Grafo(rotulo="resposta_inferencia")
    mc = resposta.get("model_card") or {}
    nome_modelo = mc.get("modelo")

    payload = Artefato(
        id="payload_da_resposta", tipo="payload",
        hash=hash_do_payload_cientifico(resposta),
        versao=resposta.get("contrato"),
        origem="svc02_contrato_inferencia.inferir")
    grafo.registrar(payload)

    if nome_modelo:
        art_modelo = Artefato(
            id=f"modelo/{nome_modelo}", tipo="modelo",
            versao=mc.get("maturidade"),
            origem="svc01_construir_modelos_servidos",
            extra={"estimador": mc.get("estimador"),
                   "mecanismo": mc.get("mecanismo")})
        if modelos_dir is not None:
            art_modelo.hash = sha256_arquivo(
                Path(modelos_dir) / f"{nome_modelo}.json")
        grafo.registrar(art_modelo)
        grafo.relacionar(payload.id, "GERADO_POR", art_modelo.id,
                         "escore e IC vem do artefato servido, sem reajuste")

        ajuste = mc.get("ajustado_em") or {}
        if ajuste:
            art_dataset = Artefato(
                id="dataset/tabela_unica_admitidos", tipo="dataset",
                hash=sha256_json(ajuste), origem="ds05_tabela_unica",
                extra={"n": ajuste.get("n"), "grupos": ajuste.get("grupos")})
            grafo.registrar(art_dataset)
            grafo.relacionar(art_modelo.id, "TREINADO_COM", art_dataset.id)

    for extra in extras or []:
        grafo.registrar(extra)
        grafo.relacionar(payload.id, "SUPORTADA_POR", extra.id)

    d = grafo.para_dict(com_timestamp=False)
    d["nao_e"] = ("registro descritivo da cadeia; nenhum campo cientifico da "
                  "resposta depende deste bloco")
    return d
