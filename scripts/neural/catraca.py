"""A catraca: o campeao so troca para melhor, e "nao aprendeu" vira registro.

-------------------------------------------------------------------------
O QUE ELA CONSERTA
-------------------------------------------------------------------------

A queixa que originou esta cadeia foi "a cada treino ele nao aprende mais". Ela
descrevia duas coisas diferentes, e so uma era sobre o modelo:

  1. rodadas nao acumulavam        -> resolvido pela peca A (retomada)
  2. rodadas podiam PIORAR o que   -> e este arquivo
     ja estava servindo

O caso 2 aconteceu de verdade e esta medido: a corrida de 30 epocas gravou o
peso da ULTIMA epoca, a perda de validacao tinha passado por 0,151 na epoca 25
e voltado a 0,196 na 30, e o resultado foi 30 epocas medirem PIOR que 12. A
leitura sobre o teto se inverteu por causa disso.

Com catraca esse desfecho deixa de ser possivel. O campeao so e substituido por
quem prova ser melhor; quem nao prova e RECUSADO e fica registrado como
recusado. A curva do que esta servindo passa a ser monotona POR CONSTRUCAO, e
"nao aprendeu nesta rodada" vira um fato anotado em vez de uma regressao
silenciosa.

E e isso que transforma insistencia em acumulo: a rodada seguinte parte do
CAMPEAO, nao da ultima corrida. Rodada ruim custa o tempo dela e nada mais.

-------------------------------------------------------------------------
AS TRES TRAVAS, e o que cada uma impede
-------------------------------------------------------------------------

1. A REGUA NUNCA E O TESTE. `PORTAO_DE_REGUA` recusa arena montada sobre
   `teste` ou `bolivia`. Escolher campeao pelo conjunto onde a afirmacao final
   e feita e escolher quem vai bem no juiz -- e o `treino.py` ja tomou essa
   decisao para a epoca, pela mesma razao. A catraca decide em `validacao`; o
   teste continua sendo gasto uma vez, no fim, para a afirmacao.

2. O PAREAMENTO E OBRIGATORIO. Desafiante e campeao tem de ter sido medidos nas
   MESMAS sementes. Comparar medias de sementes diferentes devolve ao numero
   exatamente a variancia que o pareamento existe para tirar -- e na frente A
   essa variancia e de 0,45 de IoU, muito maior que qualquer efeito ja medido
   aqui.

3. O GANHO PRECISA SAIR DO RUIDO. Quem decide e `regua.analisar_pareado`, e o
   teste primario dela e de POSTOS -- nao o t. A razao esta medida e registrada
   em `docs/PRE_REGISTRO_TESTE_DE_POSTOS_2026_09_04.md`: uma linhagem que
   diverge (perda 4,56 contra um nivel de 0,9) leva o desvio do t a 1,49 e
   apaga o sinal de todas as outras, mesmo quando 6 de 6 sementes melhoram.

   Promover no sinal da media faria a catraca subir por sorteio, e uma catraca
   que sobe por sorteio nao e melhor que nao ter catraca -- e pior, porque
   parece garantia. E como o teste de sinal tem piso 2*0,5^n, disputa com menos
   de 6 sementes nunca promove: e limitacao do desenho, nao do desafiante.

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO NAO E
-------------------------------------------------------------------------

Nao e caminho de promocao para modelo servido de suscetibilidade. Ele governa
peso de rede neural em `local_runs/neural-catraca/`, e `PORTAO_DE_ESCRITA`
recusa qualquer destino dentro de `local_runs/svc-01-modelos/`. A regra do
`harness_comparacao` continua valendo: nada aqui desloca escore publicado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from proveniencia import agora_utc  # noqa: E402
from proveniencia_de_peso import commit_atual, selar  # noqa: E402
from regua import analisar_pareado  # noqa: E402

SAIDA = REPO / "local_runs" / "neural-catraca"

# Conjuntos onde a afirmacao final e feita. Escolher campeao neles gastaria o
# holdout na selecao, e a medida final passaria a responder sobre o vencedor de
# uma busca em vez de sobre o modelo.
SPLITS_PROIBIDOS_NA_REGUA = ("teste", "bolivia")

# Diretorios servidos. A catraca nao escreve neles em hipotese nenhuma.
DESTINOS_PROIBIDOS = ("svc-01-modelos", "svc-03-grade")

PRIMEIRO_CAMPEAO = "PRIMEIRO_CAMPEAO"
PROMOVIDO = "PROMOVIDO"
RECUSADO_DENTRO_DO_RUIDO = "RECUSADO_DENTRO_DO_RUIDO"
RECUSADO_REGRESSAO = "RECUSADO_REGRESSAO"
RECUSADO_SEM_PAREAMENTO = "RECUSADO_SEM_PAREAMENTO"
VEREDITOS = (PRIMEIRO_CAMPEAO, PROMOVIDO, RECUSADO_DENTRO_DO_RUIDO,
             RECUSADO_REGRESSAO, RECUSADO_SEM_PAREAMENTO)
PROMOCOES = (PRIMEIRO_CAMPEAO, PROMOVIDO)


class PortaoDaCatraca(Exception):
    """Uso que quebraria a garantia da catraca. Falha dura, sempre."""


def arena(nome: str, metrica: str, split: str,
          maior_e_melhor: bool = True) -> dict:
    """A identidade da disputa: tarefa, metrica e conjunto onde se mede.

    Duas disputas com reguas diferentes nao podem compartilhar campeao, e por
    isso a arena entra no nome do arquivo. Sem isso, um encoder promovido por
    perda de reconstrucao e outro por IoU disputariam o mesmo posto medindo
    coisas incomparaveis.
    """
    if split in SPLITS_PROIBIDOS_NA_REGUA:
        raise PortaoDaCatraca(
            f"arena sobre '{split}' e proibida: escolher campeao no conjunto "
            f"onde a afirmacao final e feita transforma a medida final na "
            f"medida do vencedor de uma busca; use 'validacao'")
    return {"nome": nome, "metrica": metrica, "split": split,
            "maior_e_melhor": bool(maior_e_melhor),
            "id": f"{nome}__{metrica}__{split}"}


def _portao_de_escrita(caminho: Path) -> Path:
    partes = set(Path(caminho).parts)
    if partes & set(DESTINOS_PROIBIDOS):
        raise PortaoDaCatraca(
            f"a catraca nao escreve em diretorio servido: {caminho}")
    return caminho


def _arquivo_do_campeao(a: dict, saida: Path) -> Path:
    return _portao_de_escrita(saida / f"campeao_{a['id']}.json")


def _arquivo_do_historico(a: dict, saida: Path) -> Path:
    return _portao_de_escrita(saida / f"historico_{a['id']}.jsonl")


def campeao(a: dict, saida: Path = SAIDA) -> dict | None:
    """O campeao corrente, ou `None` se a arena ainda nao teve um."""
    arquivo = _arquivo_do_campeao(a, saida)
    if not arquivo.exists():
        return None
    return json.loads(arquivo.read_text(encoding="utf-8"))


def historico(a: dict, saida: Path = SAIDA) -> list[dict]:
    """Toda disputa ja julgada nesta arena, promovida ou nao.

    A recusa fica gravada de proposito. "Esta rodada nao aprendeu" e resultado,
    e apagar as recusas faria a linhagem parecer uma sequencia de acertos.
    """
    arquivo = _arquivo_do_historico(a, saida)
    if not arquivo.exists():
        return []
    return [json.loads(linha) for linha in
            arquivo.read_text(encoding="utf-8").splitlines() if linha.strip()]


def _ganho(medidas_desafiante: dict, medidas_campeao: dict,
           maior_e_melhor: bool) -> list[float]:
    """Diferencas pareadas por semente, no sentido "positivo e melhor"."""
    sementes = sorted(medidas_desafiante)
    sinal = 1.0 if maior_e_melhor else -1.0
    return [sinal * (float(medidas_desafiante[s]) - float(medidas_campeao[s]))
            for s in sementes]


def _regrediu(regressao: dict | None, regressao_do_campeao: dict | None,
              tolerancia: float) -> list[str]:
    """As guardas que pioraram alem da tolerancia.

    Existe porque retomar sobre dado novo degrada o antigo, e o esquecimento
    catastrofico e silencioso na metrica principal: a rodada pode subir na
    regua e ter perdido o que ja sabia.
    """
    if not regressao or not regressao_do_campeao:
        return []
    piores = []
    for chave, valor in regressao.items():
        antes = regressao_do_campeao.get(chave)
        if antes is None:
            continue
        # guardas sao perdas: menor e melhor.
        if float(valor) > float(antes) + tolerancia:
            piores.append(chave)
    return piores


def disputar(a: dict, rotulo: str, medidas: dict, proveniencia: dict,
             regressao: dict | None = None, tolerancia_regressao: float = 0.0,
             alfa: float = 0.05, saida: Path = SAIDA) -> dict:
    """Julga um desafiante contra o campeao corrente e grava o veredito.

    `medidas` e `{semente: valor}` -- nunca uma media pronta. A catraca precisa
    dos pares para poder usar o teste pareado, e receber a media ja calculada
    tornaria impossivel distinguir um ganho real de um sorteio favoravel.
    """
    atual = campeao(a, saida)
    medidas = {str(k): float(v) for k, v in medidas.items()}
    if not medidas:
        raise PortaoDaCatraca("desafiante sem medida nao disputa")

    registro = {
        "tipo": "disputa_de_catraca",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "arena": a,
        "desafiante": rotulo,
        "medidas": medidas,
        "media": round(sum(medidas.values()) / len(medidas), 6),
        "proveniencia": proveniencia,
        "regressao": regressao,
        "campeao_anterior": atual["desafiante"] if atual else None,
        "nao_e": ("promocao de modelo servido: esta arena governa peso de rede "
                  "neural e nao desloca escore de suscetibilidade publicado. E "
                  "PROMOVIDO nao e afirmacao de que o modelo ficou bom -- e de "
                  "que ficou melhor que o campeao anterior nesta regua"),
    }

    if atual is None:
        registro.update(veredito=PRIMEIRO_CAMPEAO, analise=None,
                        por_que="arena sem campeao; este vira a referencia")
    elif set(medidas) != set(atual["medidas"]):
        registro.update(
            veredito=RECUSADO_SEM_PAREAMENTO, analise=None,
            por_que=(f"sementes diferentes do campeao "
                     f"({sorted(medidas)} vs {sorted(atual['medidas'])}); "
                     f"sem pareamento a comparacao devolve ao numero a "
                     f"variancia entre sementes"))
    else:
        piores = _regrediu(regressao, atual.get("regressao"),
                           tolerancia_regressao)
        analise = analisar_pareado(
            _ganho(medidas, atual["medidas"], a["maior_e_melhor"]), alfa=alfa)
        registro["analise"] = analise
        if piores:
            registro.update(
                veredito=RECUSADO_REGRESSAO,
                por_que=f"guardas pioraram alem da tolerancia: {piores}")
        elif analise["veredito"] == "EFEITO_DETECTADO" and analise["media"] > 0:
            registro.update(
                veredito=PROMOVIDO,
                por_que=(f"o IC95 do ganho pareado exclui zero "
                         f"({analise['ic95']})"))
        else:
            registro.update(
                veredito=RECUSADO_DENTRO_DO_RUIDO,
                por_que=(f"ganho de {analise['media']} nao sai do ruido; o "
                         f"menor detectavel com {analise['n']} sementes e "
                         f"{analise['mde']}"))

    selar(registro)
    saida.mkdir(parents=True, exist_ok=True)
    with _arquivo_do_historico(a, saida).open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    if registro["veredito"] in PROMOCOES:
        _arquivo_do_campeao(a, saida).write_text(
            json.dumps(registro, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return registro


def curva_do_campeao(a: dict, saida: Path = SAIDA) -> list[dict]:
    """A curva do que esteve servindo -- monotona por construcao.

    Diferente de `historico`, que traz toda tentativa. Esta e a serie que se
    pode mostrar como "o que o sistema sabe hoje", e a garantia de que ela nao
    desce esta em `test_neural_catraca`, nao na prosa daqui.
    """
    serie, melhor = [], None
    for r in historico(a, saida):
        if r["veredito"] not in PROMOCOES:
            continue
        melhor = r["media"] if melhor is None else (
            max(melhor, r["media"]) if a["maior_e_melhor"]
            else min(melhor, r["media"]))
        serie.append({"quando_utc": r["quando_utc"],
                      "campeao": r["desafiante"], "valor": melhor})
    return serie
