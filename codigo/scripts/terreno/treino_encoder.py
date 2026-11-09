"""B2 -- treino do encoder de terreno, sem rotulo nenhum.

-------------------------------------------------------------------------
A PERDA SO OLHA ONDE ESCONDEU
-------------------------------------------------------------------------

Se a perda cobrisse o recorte inteiro, o pedaco visivel dominaria: copiar o que
esta na entrada e trivial e ha muito mais pixel visivel que escondido. A rede
minimizaria a perda sem nunca resolver a parte dificil, e a curva de treino
cairia bonito enquanto nada de estrutural fosse aprendido.

E o mesmo modo de falha que a frente A encontrou na medida de confianca: o
facil e numeroso, o dificil e raro, e a media esconde o que interessa. La
custou uma calibracao inteira para descobrir; aqui ja entra corrigido.

-------------------------------------------------------------------------
NENHUM ROTULO ENTRA AQUI
-------------------------------------------------------------------------

`LabelHand` nao e lido neste arquivo. O encoder ve terreno e so terreno, e a
fracao de agua so aparece depois, na avaliacao, como alvo externo. Se o rotulo
entrasse no treino, a comparacao com o baseline descritivo -- que tambem nao o
usa -- deixaria de ser justa.

-------------------------------------------------------------------------
NORMALIZACAO SO COM O TREINO
-------------------------------------------------------------------------

Como na frente A. Media e desvio por camada saem exclusivamente dos chips de
treino e vao gravados no checkpoint, porque quem for extrair embedding depois
precisa aplicar exatamente a mesma transformacao.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "terreno"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from dados_terreno import (  # noqa: E402
    CAMADAS, TETO_DE_RECORTES, aois_derivadas, chips_sem_rotulo, ler_terreno,
    nomes_com_terreno, recortes_por_chip,
)
from modelo_encoder import (  # noqa: E402
    AutoencoderMascarado, contar_parametros, mascarar,
)
from proveniencia import agora_utc  # noqa: E402
from proveniencia_de_peso import (  # noqa: E402
    commit_atual, hash_do_estado, selar, semear,
)

SAIDA = REPO / "local_runs" / "terreno-encoder"

# Teto do peso de UM chip na normalizacao. 262.144 e exatamente 512x512,
# o tamanho de um chip do Sen1Floods11: com o teto, uma AOI grande vale o
# mesmo que um chip, e nao onze.
PIXELS_POR_CHIP_NA_ESTATISTICA = 262144

# Quantos chips sem rotulo ficam de fora do treino so para monitorar a epoca.
CHIPS_DE_MONITORAMENTO = 24


def _reservados_para_monitorar(n: int = CHIPS_DE_MONITORAMENTO) -> tuple:
    """Sufixo estavel dos chips sem rotulo. Fora do treino e fora do LOCO.

    Sufixo, e nao amostra aleatoria: `chips_sem_rotulo` devolve em ordem
    estavel, e um sufixo e reproduzivel sem depender de semente.
    """
    sem_rotulo = chips_sem_rotulo()
    return tuple(sem_rotulo[-n:]) if len(sem_rotulo) > n else ()


def estatisticas_do_treino(nomes: list[str],
                           pixels_por_chip: int | None = None) -> dict:
    """Media e desvio por camada, so em pixel valido, so no treino.

    `pixels_por_chip` limita quanto CADA chip pesa. Sem o limite, a soma e
    sobre todos os pixels validos, e isso so era inofensivo enquanto todo chip
    tinha 512x512: as AOIs derivadas nao tem. Petropolis tem 1676x1688 -- 11
    vezes um chip --, e sem limite quatro regioes grandes puxariam a
    normalizacao inteira para a forma do terreno delas.

    O padrao e `None`, que reproduz exatamente o comportamento anterior. O
    limite entra quando AOI entra, e vai declarado no checkpoint.
    """
    n = len(CAMADAS)
    somas = np.zeros(n, dtype=np.float64)
    somas_q = np.zeros(n, dtype=np.float64)
    contagem = np.zeros(n, dtype=np.int64)
    sorteio = np.random.default_rng(0)
    for nome in nomes:
        chip = ler_terreno(nome)
        for i in range(n):
            vals = chip.pilha[i][chip.valido].astype(np.float64)
            if pixels_por_chip and vals.size > pixels_por_chip:
                # amostra sem reposicao, semente fixa: a estatistica tem de ser
                # a mesma em duas corridas do mesmo conjunto.
                vals = vals[sorteio.choice(vals.size, pixels_por_chip,
                                           replace=False)]
            somas[i] += vals.sum()
            somas_q[i] += (vals ** 2).sum()
            contagem[i] += vals.size
    contagem[contagem == 0] = 1
    media = somas / contagem
    var = np.maximum(somas_q / contagem - media ** 2, 1e-12)
    return {"media": [round(float(v), 6) for v in media],
            "desvio": [round(float(v), 6) for v in np.sqrt(var)],
            "camadas": list(CAMADAS)}


def preparar(nome: str, estat: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Devolve `(x, valido)` normalizado.

    Pixel invalido vira zero -- a media do treino apos normalizar -- e sai na
    mascara. Preencher sem marcar ensinaria a rede que ausencia de dado e
    terreno medio, que e a versao desta frente do erro que a frente A evitou.
    """
    chip = ler_terreno(nome)
    media = np.asarray(estat["media"], dtype=np.float32).reshape(-1, 1, 1)
    desvio = np.asarray(estat["desvio"], dtype=np.float32).reshape(-1, 1, 1)
    x = (chip.pilha - media) / desvio
    x[:, ~chip.valido] = 0.0
    return (torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(chip.valido))


TAMANHO_DO_BUFFER = 256


def preparar_janela(nome: str, estat: dict, lado: int,
                    gerador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Um recorte uniforme de `lado` px lido SO na janela, com a transformacao de `preparar`.

    Mesma distribuicao do `_recorte` sobre o raster inteiro (posicao uniforme),
    sem ler o raster inteiro: num tile GFD de 4423x4453 em float64 a leitura
    completa custava ~11 s por recorte. Validade, log1p e normalizacao sao as
    de `ler_terreno`/`preparar`.
    """
    import rasterio
    from rasterio.windows import Window
    from dados_terreno import CAMADAS_EM_LOG, SEM_DADO, diretorio
    d = diretorio(nome)
    with rasterio.open(d / f"{CAMADAS[0]}.tif") as r0:
        h, w = r0.height, r0.width
    i = int(torch.randint(0, max(h - lado, 0) + 1, (1,), generator=gerador).item())
    j = int(torch.randint(0, max(w - lado, 0) + 1, (1,), generator=gerador).item())
    janela = Window(j, i, min(lado, w), min(lado, h))
    pilha, valido = [], None
    for camada in CAMADAS:
        with rasterio.open(d / f"{camada}.tif") as r:
            a = r.read(1, window=janela).astype(np.float32)
        v = np.isfinite(a) & (a != SEM_DADO)
        valido = v if valido is None else (valido & v)
        pilha.append(a)
    pilha = np.stack(pilha)
    for k, camada in enumerate(CAMADAS):
        if camada in CAMADAS_EM_LOG:
            pilha[k] = np.log1p(np.where(valido & (pilha[k] > 0), pilha[k], 0.0))
    media = np.asarray(estat["media"], dtype=np.float32).reshape(-1, 1, 1)
    desvio = np.asarray(estat["desvio"], dtype=np.float32).reshape(-1, 1, 1)
    x = (pilha - media) / desvio
    x[:, ~valido] = 0.0
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(valido)


def amostras_da_epoca(nomes: list[str], estat: dict, lado: int,
                      gerador: torch.Generator, k_por_chip: dict,
                      buffer: int = TAMANHO_DO_BUFFER, por_janela: bool = False):
    """Recortes de uma epoca, misturados, com UMA leitura por raster.

    -------------------------------------------------------------------------
    POR QUE UM BUFFER, E NAO EMBARALHAR TUDO
    -------------------------------------------------------------------------

    Embaralhar a lista inteira de recortes releria `EMSR778_AOI01` -- 4136x6883
    -- dezenas de vezes por epoca. Ler o raster por amostra transformaria o
    treino num benchmark de disco.

    Ler o raster por chip e emitir os recortes em sequencia resolveria o I/O e
    criaria outro problema: todo lote sairia do mesmo lugar, e o gradiente
    passaria a medir um pedaco de mundo por vez.

    O buffer resolve os dois. Le uma vez por chip, acumula os recortes daquele
    chip junto com os de varios outros, e sorteia do buffer -- I/O de uma
    passagem, mistura de muitos lugares.
    """
    ordem = torch.randperm(len(nomes), generator=gerador).tolist()
    pote: list = []

    def sortear():
        i = int(torch.randint(0, len(pote), (1,), generator=gerador).item())
        pote[i], pote[-1] = pote[-1], pote[i]
        return pote.pop()

    for idx in ordem:
        nome = nomes[idx]
        if por_janela:
            for _ in range(max(1, int(k_por_chip.get(nome, 1)))):
                try:
                    recorte, valido = preparar_janela(nome, estat, lado, gerador)
                except Exception:
                    continue
                if recorte.shape[1] == lado and recorte.shape[2] == lado:
                    pote.append((recorte, valido))
            while len(pote) >= buffer:
                yield sortear()
            continue
        try:
            x, v = preparar(nome, estat)
        except Exception:
            # Chip ilegivel nao derruba a epoca; ele some do conjunto daquela
            # passagem e reaparece na proxima se voltar a ser legivel.
            continue
        for _ in range(max(1, int(k_por_chip.get(nome, 1)))):
            recorte, valido = _recorte(x, v, lado, gerador)
            if recorte.shape[1] == lado and recorte.shape[2] == lado:
                pote.append((recorte, valido))
        while len(pote) >= buffer:
            yield sortear()
    while pote:
        yield sortear()


def _recorte(x: torch.Tensor, valido: torch.Tensor, lado: int,
             gerador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = x.shape
    if lado >= min(h, w):
        return x, valido
    i = int(torch.randint(0, h - lado + 1, (1,), generator=gerador).item())
    j = int(torch.randint(0, w - lado + 1, (1,), generator=gerador).item())
    return x[:, i:i + lado, j:j + lado], valido[i:i + lado, j:j + lado]


def perda_mascarada(saida: torch.Tensor, alvo: torch.Tensor,
                    escondido: torch.Tensor,
                    valido: torch.Tensor) -> torch.Tensor:
    """Erro quadratico so onde escondeu E onde ha dado.

    A interseccao das duas mascaras e o ponto: esconder um pedaco que ja era
    `nodata` nao cria tarefa nenhuma, e cobrar a reconstrucao dele ensinaria a
    rede a prever o valor de preenchimento.
    """
    alvo_da_perda = escondido & valido.unsqueeze(1)
    if not alvo_da_perda.any():
        return torch.zeros((), requires_grad=True)
    alvo_expandido = alvo_da_perda.expand_as(saida)
    return ((saida - alvo) ** 2)[alvo_expandido].mean()


@torch.no_grad()
def perda_validacao(modelo, nomes: list[str], estat: dict, lado: int,
                    fracao: float, lado_bloco: int,
                    seed: int) -> float:
    """Mesma tarefa, com mascara fixa por seed -- comparavel entre epocas."""
    modelo.eval()
    gerador = torch.Generator().manual_seed(seed)
    total, n = 0.0, 0
    for nome in nomes:
        x, valido = preparar(nome, estat)
        x, valido = _recorte(x, valido, lado, gerador)
        entrada = x.unsqueeze(0)
        mascarada, escondido = mascarar(entrada, fracao, lado_bloco, gerador)
        perda = perda_mascarada(modelo(mascarada), entrada, escondido,
                                valido.unsqueeze(0))
        if torch.isfinite(perda) and float(perda) > 0:
            total += float(perda.detach())
            n += 1
    return round(total / n, 6) if n else float("nan")


def treinar(epocas: int = 20, base: int = 16, lado: int = 128,
            lote: int = 4, taxa: float = 1e-3, seed: int = 20260902,
            fracao_mascara: float = 0.5, lado_bloco: int = 16,
            limite_treino: int | None = None,
            limite_validacao: int | None = None,
            threads: int | None = None,
            usar_sem_rotulo: bool = False,
            fracao_sem_rotulo: float = 1.0,
            usar_aois: bool = False,
            por_area: bool = False,
            marcos: tuple | None = None,
            ao_marcar=None,
            teto_de_recortes: int | None = TETO_DE_RECORTES,
            passos_alvo: int | None = None,
            estado_inicial: dict | None = None,
            estatisticas_fixas: dict | None = None,
            nomes_extra: list | None = None,
            linhagem: dict | None = None,
            leitura_por_janela: bool = False) -> dict:
    """`usar_sem_rotulo` acrescenta os 215 chips que nao tem rotulo nenhum.

    E a propriedade que separa auto-supervisao de todo o resto: dado sem rotulo
    e dado UTIL aqui. O encoder aprende forma de terreno reconstruindo o que foi
    escondido, e para isso o rotulo nunca fez falta -- entao cada raster novo
    que entrar no `ter-01` melhora a representacao sem que ninguem rotule nada.

    Os chips sem rotulo nao aparecem em nenhum split do Sen1Floods11 (colisao
    verificada: zero) e nao entram na avaliacao, porque nao ha alvo contra o
    que medir. Eles entram so no treino, que e onde nao precisam de alvo.

    `usar_aois` acrescenta as 123 AOIs que a MESMA cadeia derivou e que
    estavam invisiveis so por causa do prefixo do diretorio -- ativacoes do
    CEMS e as regioes servidas. Duas coisas mudam com elas, e as duas vao
    declaradas no manifesto:

      1. o encoder passa a ver terreno BRASILEIRO, que o Sen1Floods11 nao tem;
      2. as AOIs nao sao 512x512 (Petropolis tem 1676x1688), entao a
         normalizacao ganha um teto por chip -- sem ele, quatro regioes grandes
         puxariam a estatistica inteira.

    O que NAO muda: nenhuma delas entra na avaliacao, e o LOCO continua sendo
    sobre os paises do Sen1Floods11. AOI derivada e dado de treino, nunca alvo.
    """
    if threads:
        torch.set_num_threads(threads)
    semear(seed)

    nomes_treino = nomes_com_terreno("treino")[:limite_treino]
    n_rotulados = len(nomes_treino)
    if usar_sem_rotulo:
        extras = chips_sem_rotulo()
        if fracao_sem_rotulo < 1.0:
            # Prefixo, e nao amostra aleatoria: `chips_sem_rotulo` devolve em
            # ordem estavel, e um prefixo e reproduzivel sem depender de seed.
            extras = extras[:int(round(len(extras) * fracao_sem_rotulo))]
        nomes_treino = nomes_treino + extras
    n_aois = 0
    if usar_aois:
        aois = aois_derivadas()
        n_aois = len(aois)
        nomes_treino = nomes_treino + aois
    # O QUARTO CONJUNTO, e por que ele nao custa nada.
    #
    # `nomes_com_terreno("validacao")` servia a DUAS funcoes: monitorar o
    # treino e compor o LOCO da avaliacao. Escolher epoca por uma perda medida
    # no mesmo conjunto que depois julga o embedding e escolher a epoca que vai
    # bem no juiz.
    #
    # Os chips SEM ROTULO nao tem `fracao_de_agua`, entao eles NUNCA podem
    # entrar no LOCO -- nao ha alvo contra o que medir. Reservar uma fatia
    # deles para monitoramento remove a dupla funcao de graca. A fatia e um
    # sufixo em ordem estavel, e ela sai do treino.
    # CONTINUIDADE (03/10/2026, campanha integrada N2). `nomes_extra` acrescenta
    # rasters escolhidos por quem chama (ex.: so os tiles GFD da metade de
    # pre-treino); `estado_inicial` continua de um checkpoint; e
    # `estatisticas_fixas` mantem a normalizacao do checkpoint -- recalcular
    # com o conjunto novo mudaria a escala da entrada que os pesos esperam.
    # Os tres sao opcionais: sem eles o treino e o mesmo de antes.
    if nomes_extra:
        nomes_treino = nomes_treino + [n for n in nomes_extra if n not in nomes_treino]
    reservados = _reservados_para_monitorar()
    nomes_treino = [n for n in nomes_treino if n not in reservados]
    nomes_valid = (list(reservados) if reservados
                   else nomes_com_terreno("validacao"))[:limite_validacao]

    inicio_estat = time.time()
    # O limite por chip so entra quando AOI entra: sem AOI todos os chips tem
    # 512x512 e o limite nao mudaria nada, mas mudaria o numero, e um numero
    # que muda sem que a decisao mude quebra a comparacao com as corridas ja
    # medidas da frente B.
    pixels_por_chip = PIXELS_POR_CHIP_NA_ESTATISTICA if usar_aois else None
    estat = (estatisticas_fixas if estatisticas_fixas is not None
             else estatisticas_do_treino(nomes_treino, pixels_por_chip))
    segundos_estat = round(time.time() - inicio_estat, 2)

    modelo = AutoencoderMascarado(canais=len(CAMADAS), base=base)
    if estado_inicial is not None:
        modelo.load_state_dict(estado_inicial)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    gerador = torch.Generator().manual_seed(seed)

    # ORCAMENTO DE PASSOS, e nao de epocas.
    #
    # Comparar conjuntos de tamanhos diferentes com o mesmo numero de EPOCAS
    # confunde duas coisas: mais dado e mais atualizacao de gradiente. Um
    # conjunto 1,85x maior recebe 1,85x mais passos por epoca, e o ganho pode
    # vir de ter treinado mais, nao de ter visto mais lugares.
    #
    # Com `passos_alvo` o numero de atualizacoes fica fixo, e a unica coisa que
    # muda entre as corridas e a DIVERSIDADE do que a rede viu. E essa a
    # pergunta da curva de escala.
    # AMOSTRAGEM POR AREA, e o que ela conserta.
    #
    # `por_area=False` reproduz o comportamento medido ate aqui: um recorte por
    # diretorio por epoca, independente do tamanho do raster. Isso amostra
    # `EMSR778_AOI01` 108 vezes menos densamente por pixel que um chip 512x512,
    # e Petropolis 10,8 vezes -- as AOIs carregam 84% da area e recebiam 21%
    # das amostras.
    #
    # Com `por_area=True` cada chip contribui em proporcao a area, com teto.
    # O orcamento de passos NAO muda: o que muda e de onde as amostras vem.
    k_por_chip = (recortes_por_chip(nomes_treino, lado, teto_de_recortes)
                  if por_area else {n: 1 for n in nomes_treino})
    amostras_por_epoca = sum(k_por_chip.values())
    passos_por_epoca = max(amostras_por_epoca // lote, 1)
    if passos_alvo:
        # TETO, e nao arredondamento. Com `round`, um conjunto cujo tamanho de
        # epoca fica logo acima do orcamento recebe UMA epoca e para antes da
        # conta: `por_area` treinou 1.117 passos contra os 1.200 de `por_chip`,
        # e a comparacao que se dizia de orcamento fixo tinha 7% de diferenca.
        # Com teto o laco sempre ultrapassa o alvo e o corte por
        # `passos_totais >= passos_alvo` faz o resto.
        epocas = max(-(-passos_alvo // passos_por_epoca), 1)

    historico = []
    passos_totais = 0
    inicio = time.time()
    for epoca in range(1, epocas + 1):
        modelo.train()
        acumulada, passos = 0.0, 0
        fonte = amostras_da_epoca(nomes_treino, estat, lado, gerador,
                                  k_por_chip, por_janela=leitura_por_janela)
        while True:
            xs, vs = [], []
            for _ in range(lote):
                try:
                    x, v = next(fonte)
                except StopIteration:
                    break
                xs.append(x)
                vs.append(v)
            if len(xs) < lote:
                break
            entrada = torch.stack(xs)
            valido = torch.stack(vs)
            mascarada, escondido = mascarar(entrada, fracao_mascara,
                                            lado_bloco, gerador)
            otimizador.zero_grad()
            perda = perda_mascarada(modelo(mascarada), entrada, escondido,
                                    valido)
            if not torch.isfinite(perda) or float(perda) == 0:
                continue
            perda.backward()
            otimizador.step()
            acumulada += float(perda.detach())
            passos += 1
            passos_totais += 1
            # CHECKPOINT NO MARCO, e nao so no fim.
            #
            # O treino gravava peso apenas na ultima epoca. Isso custava duas
            # coisas: travar no fim perdia a corrida inteira, e medir a curva
            # de compute exigia UMA CORRIDA POR PONTO -- quatro corridas para
            # quatro pontos, quando uma corrida longa ja passa por todos eles.
            if marcos and ao_marcar and passos_totais in marcos:
                ao_marcar(passos_totais, modelo, estat)
            if passos_alvo and passos_totais >= passos_alvo:
                break

        registro = {
            "epoca": epoca,
            "perda_treino": round(acumulada / passos, 6) if passos else None,
            "passos": passos,
            "perda_validacao": perda_validacao(
                modelo, nomes_valid, estat, lado, fracao_mascara,
                lado_bloco, seed),
            "segundos_acumulados": round(time.time() - inicio, 2),
        }
        historico.append(registro)
        if passos_alvo and passos_totais >= passos_alvo:
            break
        print(f"epoca {epoca}/{epocas} treino={registro['perda_treino']} "
              f"validacao={registro['perda_validacao']} "
              f"({registro['segundos_acumulados']}s)",
              file=sys.stderr, flush=True)

    manifesto = {
        "tipo": "encoder_mascarado_terreno",
        "etapa": "B2",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "torch": torch.__version__,
        "dispositivo": "cpu",
        "seed": seed,
        "hiperparametros": {"epocas": epocas, "base": base, "lado": lado,
                            "lote": lote, "taxa": taxa,
                            "fracao_mascara": fracao_mascara,
                            "lado_bloco": lado_bloco,
                            "threads": threads or torch.get_num_threads()},
        "monitoramento": ("chips sem rotulo reservados, fora do treino e "
                          "fora do LOCO" if reservados else
                          "split de validacao (dupla funcao)"),
        "split": {"treino": len(nomes_treino), "validacao": len(nomes_valid),
                  "rotulados_no_treino": n_rotulados,
                  "sem_rotulo_no_treino":
                      len(nomes_treino) - n_rotulados - n_aois,
                  "aois_derivadas_no_treino": n_aois},
        "usou_sem_rotulo": usar_sem_rotulo,
        "fracao_sem_rotulo": fracao_sem_rotulo,
        "usou_aois_derivadas": usar_aois,
        "pixels_por_chip_na_estatistica": pixels_por_chip,
        "amostragem": "por_area" if por_area else "por_chip",
        "teto_de_recortes": teto_de_recortes if por_area else None,
        "amostras_por_epoca": amostras_por_epoca,
        "marcos": sorted(marcos) if marcos else None,
        "passos_alvo": passos_alvo,
        "passos_totais": passos_totais,
        "camadas": list(CAMADAS),
        "parametros": contar_parametros(modelo),
        "dimensao_embedding": modelo.dimensao_embedding,
        "normalizacao": estat,
        "segundos_estatisticas": segundos_estat,
        "historico": historico,
        "hash_estado": hash_do_estado(modelo.state_dict()),
        "usou_rotulo": False,
        "linhagem": linhagem or {"continuado_de": None},
        "estatisticas_fixas": estatisticas_fixas is not None,
        "nomes_extra": len(nomes_extra or []),
        "leitura_por_janela": leitura_por_janela,
        "nao_e": ("modelo de suscetibilidade: e uma representacao de forma de "
                  "terreno, aprendida sem nenhum rotulo, que so vale o que "
                  "provar valer na avaliacao externa"),
    }
    selar(manifesto)
    return {"manifesto": manifesto, "estado": modelo.state_dict()}


def gravar(resultado: dict, rotulo: str, saida: Path = SAIDA) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    torch.save(resultado["estado"], saida / f"encoder_{rotulo}.pt")
    destino = saida / f"encoder_{rotulo}_manifesto.json"
    destino.write_text(
        json.dumps(resultado["manifesto"], indent=2, ensure_ascii=False),
        encoding="utf-8")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lado", type=int, default=128)
    parser.add_argument("--lote", type=int, default=4)
    parser.add_argument("--taxa", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--fracao-mascara", type=float, default=0.5)
    parser.add_argument("--lado-bloco", type=int, default=16)
    parser.add_argument("--limite-treino", type=int, default=None)
    parser.add_argument("--limite-validacao", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--fracao-sem-rotulo", type=float, default=1.0)
    parser.add_argument("--passos-alvo", type=int, default=None,
                        help="orcamento fixo de passos de gradiente")
    parser.add_argument("--sem-rotulo", action="store_true",
                        help="acrescenta os 215 chips sem rotulo ao treino")
    parser.add_argument("--aois", action="store_true",
                        help="acrescenta as AOIs derivadas pela cadeia canonica")
    parser.add_argument("--por-area", action="store_true",
                        help="amostra recortes em proporcao a area, com teto")
    parser.add_argument("--teto-de-recortes", type=int,
                        default=TETO_DE_RECORTES)
    parser.add_argument("--rotulo", default="fumaca")
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    parser.add_argument("--gravar", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--nao-gravar", dest="gravar", action="store_false",
                        help="nao escreve artefato; so imprime o resumo")
    args = parser.parse_args()

    resultado = treinar(
        epocas=args.epocas, base=args.base, lado=args.lado, lote=args.lote,
        taxa=args.taxa, seed=args.seed, fracao_mascara=args.fracao_mascara,
        lado_bloco=args.lado_bloco, limite_treino=args.limite_treino,
        limite_validacao=args.limite_validacao, threads=args.threads,
        usar_sem_rotulo=args.sem_rotulo,
        fracao_sem_rotulo=args.fracao_sem_rotulo,
        usar_aois=args.aois, por_area=args.por_area,
        teto_de_recortes=args.teto_de_recortes,
        passos_alvo=args.passos_alvo)
    if args.gravar:
        destino = gravar(resultado, args.rotulo)
        resultado["manifesto"]["gravado_em"] = str(destino)
    print(json.dumps(resultado["manifesto"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
