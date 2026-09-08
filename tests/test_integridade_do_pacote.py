"""O pacote publico tem de ser internamente coerente.

POR QUE ESTE TESTE EXISTE:

Uma revisao encontrou quatro defeitos que nenhum teste pegava, porque nenhum deles e
um defeito de logica -- todos sao defeitos de CURADORIA:

  1. `commands.txt` de uma execucao citava um script que nao tinha sido incluido;
  2. tres testes citados em `commands.txt` tinham ficado para tras, e as frentes que
     eles cobriam foram a publico sem cobertura nenhuma;
  3. `scripts/servico/proveniencia.py` tinha sido substituido por um esboco de 33
     linhas sem `sha256_arquivo`, e com isso `dimensao_temporal.py` quebrava no import
     -- o pacote publicava um script que nao rodava;
  4. o mesmo conjunto de arquivos existia duplicado em dois diretorios, com 19 MB
     repetidos e duas copias para manter em sincronia.

Nenhum deles aparece rodando o codigo: aparecem olhando o pacote como pacote. E o que
este arquivo faz.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
EXECUCOES = RAIZ / "modelo" / "execucoes"


def comandos() -> list[tuple[Path, str]]:
    fora = []
    for arq in sorted(EXECUCOES.rglob("commands.txt")):
        for linha in arq.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                fora.append((arq, linha.strip()))
    return fora


CAMINHO = re.compile(r"(?:scripts|tests)/[\w./-]+\.py")


FORA = "fora do pacote:"


@pytest.mark.parametrize("arq,linha", comandos(),
                         ids=lambda v: v.name if isinstance(v, Path) else v)
def test_todo_comando_declarado_aponta_para_arquivo_que_existe(arq, linha):
    """Reproduzir e a promessa; comando que cita arquivo ausente e promessa falsa.

    Uma etapa pode legitimamente ficar fora deste pacote -- treino em `torch`,
    corpus documental. Nesse caso a linha comeca por `fora do pacote:` e o
    diretorio precisa explicar por que, em `fora_do_pacote.md`. Ausencia
    declarada e informacao; ausencia silenciosa e o defeito.
    """
    if linha.startswith(FORA):
        explicacao = arq.parent / "fora_do_pacote.md"
        assert explicacao.exists(), (
            f"{arq.relative_to(RAIZ).as_posix()} declara etapa fora do pacote "
            "sem o fora_do_pacote.md que explica por que")
        return
    for alvo in CAMINHO.findall(linha):
        assert (RAIZ / alvo).exists(), (
            f"{arq.relative_to(RAIZ).as_posix()} manda rodar '{alvo}', "
            "que nao esta no pacote")


CAMPOS_DE_PROVENIENCIA = ("gerado_por", "produzida_por", "gerado_por_script",
                          "estimador_importado_de", "mesmo_codigo_de")


def proveniencias() -> list[tuple[str, str]]:
    fora: list[tuple[str, str]] = []

    def anda(obj, origem):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in CAMPOS_DE_PROVENIENCIA and isinstance(v, str) \
                        and v.endswith(".py"):
                    fora.append((origem, v))
                anda(v, origem)
        elif isinstance(obj, list):
            for v in obj:
                anda(v, origem)

    for arq in sorted(EXECUCOES.rglob("*.json")):
        try:
            d = json.loads(arq.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        anda(d, arq.relative_to(RAIZ).as_posix())
    return sorted(set(fora))


@pytest.mark.parametrize("origem,script", proveniencias(),
                         ids=lambda v: v.rsplit("/", 1)[-1])
def test_todo_campo_de_proveniencia_aponta_para_script_que_existe(origem, script):
    """`gerado_por` que nao resolve deixa o artefato sem como ser refeito."""
    assert (RAIZ / script).exists(), f"{origem} declara '{script}', ausente do pacote"


def modulos_python() -> list[Path]:
    return [p for p in sorted((RAIZ / "scripts").rglob("*.py"))
            if "__pycache__" not in p.parts]


@pytest.mark.parametrize("modulo", modulos_python(), ids=lambda p: p.name)
def test_todo_script_publicado_e_sintaticamente_valido(modulo):
    ast.parse(modulo.read_text(encoding="utf-8"), filename=str(modulo))


def imports_internos() -> list[tuple[str, str]]:
    """Import de modulo do proprio repositorio, pelo nome do arquivo.

    Os scripts usam `sys.path.insert` em vez de pacote instalado, entao o import
    e pelo nome do modulo. Aqui basta que exista um arquivo com aquele nome em
    algum lugar de `scripts/`: quem cuida do caminho e o proprio script.
    """
    nomes = {p.stem for p in modulos_python()}
    fora = []
    for p in modulos_python():
        try:
            arvore = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for no in ast.walk(arvore):
            alvo = None
            if isinstance(no, ast.ImportFrom) and no.module:
                alvo = no.module.split(".")[0]
            elif isinstance(no, ast.Import):
                alvo = no.names[0].name.split(".")[0]
            if alvo and alvo in nomes:
                fora.append((p.relative_to(RAIZ).as_posix(), alvo))
    return sorted(set(fora))


@pytest.mark.parametrize("origem,modulo", imports_internos(),
                         ids=lambda v: v.rsplit("/", 1)[-1])
def test_import_interno_resolve_dentro_do_pacote(origem, modulo):
    achou = any(p.stem == modulo for p in modulos_python())
    assert achou, f"{origem} importa '{modulo}', que nao esta no pacote"


def test_nenhum_arquivo_duplicado_entre_diretorios_de_artefato():
    """Duas copias do mesmo byte sao duas copias para manter em sincronia."""
    import hashlib

    vistos: dict[str, list[str]] = {}
    for p in sorted(EXECUCOES.rglob("*")):
        if not p.is_file() or p.stat().st_size < 4096:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        vistos.setdefault(h, []).append(p.relative_to(RAIZ).as_posix())
    duplicados = {h: c for h, c in vistos.items() if len(c) > 1}
    assert not duplicados, f"conteudo repetido em: {duplicados}"


def test_toda_execucao_versionada_esta_no_inventario_de_artefatos():
    """Diretorio que existe e nao esta documentado e lacuna de curadoria."""
    inventario = (RAIZ / "docs" / "ARTEFATOS_CURADOS.md").read_text(encoding="utf-8")
    faltando = [d.name for d in sorted(EXECUCOES.iterdir())
                if d.is_dir() and d.name not in inventario]
    assert not faltando, f"fora de docs/ARTEFATOS_CURADOS.md: {faltando}"


def test_toda_execucao_versionada_declara_como_foi_produzida():
    faltando = [d.name for d in sorted(EXECUCOES.iterdir())
                if d.is_dir() and not (d / "commands.txt").exists()]
    assert not faltando, f"sem commands.txt: {faltando}"


VOCABULARIO_FORA_DO_REGISTRO = re.compile(
    r"(?i)\b(chatgpt|openai|copilot|codex|fine-?tuning|review-only)\b")

# Estes dois so sao cobrados na DOCUMENTACAO. No codigo eles aparecem como
# identificador -- nome de diretorio de execucao, rotulo de tipo de origem --
# e renomear identificador para atender a registro de texto quebraria o
# contrato com o dado ja gravado. O registro que importa e o que a pessoa le.
VOCABULARIO_SO_NA_DOCUMENTACAO = re.compile(r"(?i)\b(llm|rag)\b")


def arquivos_de_texto() -> list[Path]:
    fora = []
    for padrao in ("*.py", "*.md", "*.txt", "*.yml"):
        for p in RAIZ.rglob(padrao):
            if any(x in p.parts for x in (".git", "__pycache__", "outputs_public")):
                continue
            if p.name == Path(__file__).name:
                continue  # este arquivo E a lista de termos; citar nao e usar
            fora.append(p)
    return sorted(fora)


@pytest.mark.parametrize("arquivo", arquivos_de_texto(), ids=lambda p: p.name)
def test_o_texto_publico_fica_no_registro_do_repositorio(arquivo):
    achados = VOCABULARIO_FORA_DO_REGISTRO.findall(
        arquivo.read_text(encoding="utf-8", errors="ignore"))
    assert not achados, f"{arquivo.name}: {sorted(set(achados))}"


CONTATO_PESSOAL = re.compile(r"[\w.+-]+@(?:gmail|hotmail|outlook|yahoo)\.[a-z.]+")


def todo_arquivo_de_texto() -> list[Path]:
    """Inclui `outputs_public/`: endereco pessoal vaza igual no material historico."""
    fora = []
    for padrao in ("*.py", "*.md", "*.txt", "*.yml", "*.json", "*.csv"):
        for p in RAIZ.rglob(padrao):
            if any(x in p.parts for x in (".git", "__pycache__")):
                continue
            if p.name == Path(__file__).name:
                continue
            fora.append(p)
    return sorted(fora)


@pytest.mark.parametrize("arquivo", todo_arquivo_de_texto(), ids=lambda p: p.name)
def test_nenhum_endereco_pessoal_vai_a_publico(arquivo):
    achados = CONTATO_PESSOAL.findall(
        arquivo.read_text(encoding="utf-8", errors="ignore"))
    assert not achados, f"{arquivo.name}: {achados}"


@pytest.mark.parametrize(
    "arquivo", [p for p in arquivos_de_texto() if p.suffix == ".md"],
    ids=lambda p: p.name)
def test_a_documentacao_publica_nao_usa_vocabulario_de_ferramenta(arquivo):
    achados = VOCABULARIO_SO_NA_DOCUMENTACAO.findall(
        arquivo.read_text(encoding="utf-8", errors="ignore"))
    assert not achados, f"{arquivo.name}: {sorted(set(achados))}"


# =========================================================================
# a documentacao tem de fechar consigo mesma
# =========================================================================

def paginas_de_documentacao() -> list[Path]:
    return sorted(list((RAIZ / "docs").glob("*.md"))
                  + list((RAIZ / "modelo").glob("*.md"))
                  + [RAIZ / "README.md"])


@pytest.mark.parametrize("pagina", paginas_de_documentacao(), ids=lambda p: p.name)
def test_toda_pagina_esta_no_inventario_de_artefatos(pagina):
    """Pagina que existe e nao esta inventariada e a mesma lacuna dos artefatos."""
    inventario = (RAIZ / "docs" / "ARTEFATOS_CURADOS.md").read_text(encoding="utf-8")
    nome = pagina.relative_to(RAIZ).as_posix()
    assert nome in inventario, f"{nome} fora de docs/ARTEFATOS_CURADOS.md"


LINK_MD = re.compile(r"\]\(([^)#][^)]*)\)")


@pytest.mark.parametrize("pagina", paginas_de_documentacao(), ids=lambda p: p.name)
def test_todo_link_relativo_da_documentacao_resolve(pagina):
    """Link quebrado num documento publico e promessa de leitura que nao existe."""
    quebrados = []
    for alvo in LINK_MD.findall(pagina.read_text(encoding="utf-8")):
        if alvo.startswith(("http://", "https://", "mailto:")):
            continue
        if not (pagina.parent / alvo.split("#")[0]).resolve().exists():
            quebrados.append(alvo)
    assert not quebrados, f"{pagina.name}: {quebrados}"


def test_o_readme_indexa_toda_pagina_de_documentacao():
    """Documento que ninguem alcanca pelo README e documento que ninguem le."""
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    faltando = [p.name for p in sorted((RAIZ / "docs").glob("*.md"))
                if p.relative_to(RAIZ).as_posix() not in readme]
    assert not faltando, f"nao indexados no README: {faltando}"
