"""Discriminacao, escala e decisao a partir dos histogramas gravados por evento.

Cada medida guarda, por evento, contagens de pixels positivos e negativos em
1000 faixas de probabilidade. Daqui saem: AP; IoU/precisao/recall em qualquer
limiar; Brier, NLL e confiabilidade (ECE) pelo centro da faixa; e temperature
scaling (Guo et al. 2017) ajustado num grupo e aplicado a outro, remapeando as
faixas monotonicamente. Resolucao de 0,001: aproximacao declarada, nao inferencia
pixel a pixel. Ajustes so no grupo de calibracao; aplicar a outro grupo nunca reajusta.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

BINS = 1000
CENTROS = (np.arange(BINS) + .5) / BINS
LIMIARES = np.round(np.arange(.01, 1.0, .01), 2)


def arrays(e):
    pos, neg = np.asarray(e["histograma"]["positivo"], float), np.asarray(e["histograma"]["negativo"], float)
    if len(pos) != BINS or len(neg) != BINS:
        raise ValueError("histograma fora do contrato de 1000 faixas")
    return pos, neg


def ap(pos, neg):
    if pos.sum() == 0:
        return None
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    return float(((tp / np.maximum(tp+fp, 1)) * pos[::-1]).sum() / pos.sum())


def no_limiar(pos, neg, t):
    k = int(np.ceil(t*BINS - 1e-9))
    tp, fp = pos[k:].sum(), neg[k:].sum()
    fn = pos.sum() - tp
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "iou": float(tp/(tp+fp+fn)) if tp+fp+fn else None,
            "precisao": float(tp/(tp+fp)) if tp+fp else None, "recall": float(tp/(tp+fn)) if tp+fn else None,
            "fracao_predita_agua": float((tp+fp)/(pos.sum()+neg.sum())) if pos.sum()+neg.sum() else None}


def probabilisticas(pos, neg, centros=CENTROS):
    n = pos.sum() + neg.sum()
    c = np.clip(centros, 1e-6, 1-1e-6)
    brier = float(((pos*(1-c)**2) + (neg*c**2)).sum()/n)
    nll = float(-(pos*np.log(c) + neg*np.log(1-c)).sum()/n)
    ece, conf = 0., []
    for i in range(15):
        sel = (centros >= i/15) & (centros < (i+1)/15)
        m = pos[sel].sum() + neg[sel].sum()
        if m:
            prev = float(pos[sel].sum()/m)
            media = float(((pos[sel]+neg[sel])*centros[sel]).sum()/m)
            ece += m/n * abs(prev-media)
            conf.append({"faixa": [i/15, (i+1)/15], "n": int(m), "predito": media, "observado": prev})
    return {"brier": brier, "nll": nll, "ece": float(ece), "confiabilidade": conf,
            "prevalencia": float(pos.sum()/n), "media_predita": float(((pos+neg)*centros).sum()/n)}


def remapear(pos, neg, T):
    """Aplica p -> sigma(logit(p)/T) faixa a faixa, preservando contagens."""
    novo = expit(logit(np.clip(CENTROS, 1e-6, 1-1e-6))/T)
    idx = np.minimum((novo*BINS).astype(int), BINS-1)
    return np.bincount(idx, pos, BINS), np.bincount(idx, neg, BINS)


def ajustar_temperatura(eventos):
    """T que minimiza a NLL somada no grupo de calibracao (grade log + refinamento)."""
    pos = sum(arrays(e)[0] for e in eventos)
    neg = sum(arrays(e)[1] for e in eventos)
    z = logit(np.clip(CENTROS, 1e-6, 1-1e-6))

    def nll(T):
        p = np.clip(expit(z/T), 1e-6, 1-1e-6)
        return float(-(pos*np.log(p) + neg*np.log(1-p)).sum())
    grade = np.exp(np.linspace(np.log(.05), np.log(20), 121))
    T = float(grade[int(np.argmin([nll(t) for t in grade]))])
    for passo in (.9, .97, .99):
        cand = [T*passo**k for k in range(-10, 11)]
        T = float(min(cand, key=nll))
    return T


def escolher_limiar(eventos, T=1.):
    """Limiar que maximiza a IoU macro dos eventos do grupo de calibracao, no espaco calibrado."""
    melhores = []
    for t in LIMIARES:
        ious = []
        for e in eventos:
            pos, neg = remapear(*arrays(e), T) if T != 1 else arrays(e)
            m = no_limiar(pos, neg, t)
            if m["iou"] is not None and pos.sum() > 0:
                ious.append(m["iou"])
        if ious:
            melhores.append((float(np.mean(ious)), -abs(t-.5), float(t)))
    if not melhores:
        raise ValueError("grupo de calibracao sem positivos")
    return max(melhores)[2]


def medir_evento(e, limiar, T=1., legado=.2):
    pos, neg = arrays(e)
    cpos, cneg = remapear(pos, neg, T) if T != 1 else (pos, neg)
    return {"ap": ap(pos, neg), "prevalencia": float(pos.sum()/max(pos.sum()+neg.sum(), 1)),
            "positivos": int(pos.sum()), "negativos": int(neg.sum()),
            "legado": no_limiar(pos, neg, legado), "calibrado": no_limiar(cpos, cneg, limiar),
            "prob_sem_calibrar": probabilisticas(pos, neg), "prob_calibrada": probabilisticas(cpos, cneg),
            "mediana_p_positivos": float(CENTROS[np.searchsorted(np.cumsum(pos), pos.sum()/2)]) if pos.sum() else None,
            "mediana_p_negativos": float(CENTROS[np.searchsorted(np.cumsum(neg), neg.sum()/2)]) if neg.sum() else None}


def classificar_falha(m, competente=.1):
    """Separa competencia, falha de decisao (ordena mas o limiar nao converte) e de discriminacao."""
    if m["positivos"] == 0:
        return "sem_positivo"
    iou = m["legado"]["iou"] or 0.
    if iou >= competente:
        return "competente_no_legado"
    piso = max(.2, 3*m["prevalencia"])
    if m["ap"] is not None and m["ap"] >= piso:
        return "falha_de_decisao" if m["legado"]["tp"] + m["legado"]["fp"] == 0 or iou < competente else "outro"
    return "falha_de_discriminacao"
