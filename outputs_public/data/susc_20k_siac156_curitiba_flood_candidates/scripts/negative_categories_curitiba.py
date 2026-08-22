"""Lista de categorias do SIAC 156 tratadas como negativo causalmente independente de chuva, em
Curitiba especificamente -- NÃO herdada da lista de Recife por conveniência (regra V3 de
`docs/metodologia_cientifica/revp_criterio_ponto_negativo_recife_e_replicacao_curitiba_petropolis.md`
seção 4.1: "não copiar a lista de Recife -- cada categoria precisa de justificativa causal
própria para a cidade").

Critério (mesmo das 5 condições do documento, seção 3): presença de OUTRO fenômeno real,
datado e localizado, categoricamente distinto de chuva/drenagem/enchente -- nunca ausência de
registro de enchente.

Cada entrada documenta o motivo. Formato: allowlist explícita (Assunto, Subdivisao) -- mais
conservador que uma blocklist, porque qualquer categoria não listada fica de fora por padrão,
em vez de entrar por omissão.
"""

from __future__ import annotations

# (Assunto, Subdivisao, justificativa_causal)
CURITIBA_NEG_CATEGORIES: list[tuple[str, str, str]] = [
    ("Iluminação Pública", "Manutenção de Luminárias",
     "Falha elétrica/manutenção de lâmpada -- mesma categoria usada como negativo em Recife "
     "(EMLURB ILUMINACAO PUBLICA). Falha de luminária não é desencadeada por chuva."),
    ("Semáforo", "Lâmpada apagada",
     "Falha elétrica/mecânica pontual, mesma lógica de iluminação pública."),
    ("Trânsito", "Fiscalização de estacionamento irregular",
     "Fiscalização administrativa de rotina, sem relação com chuva."),
    ("Trânsito", "Fiscalização imediata",
     "Fiscalização administrativa de rotina."),
    ("Trânsito", "Fiscalização programada",
     "Fiscalização administrativa agendada, sem relação com chuva."),
    ("Trânsito", "Fiscalização de bloqueio de pista",
     "Bloqueio por veículo/obstáculo -- não é registro de água na via (esse já está coberto "
     "pela categoria de enchente); risco de contaminação seria só se o bloqueio fosse por "
     "alagamento, o que teria entrado como Drenagem/Alagamentos, categoria já tratada à parte."),
    ("Trânsito", "Veículo abandonado",
     "Fiscalização de veículo parado, sem relação com chuva."),
    ("Cartão transporte", "Geral",
     "Atendimento administrativo (cartão-transporte municipal), sem relação com chuva."),
    ("Cartão transporte", "Cartão débito/crédito",
     "Atendimento administrativo, sem relação com chuva."),
    ("IPTU", "Informação",
     "Atendimento tributário administrativo, sem relação com chuva."),
    ("Disque Solidariedade", "Doações",
     "Atendimento social administrativo, sem relação com chuva."),
    ("Animais", "Maus tratos - Animais domésticos",
     "Denúncia de bem-estar animal, sem relação causal com chuva."),
    ("Animais domésticos", "Remoção de caninos mortos",
     "Serviço de remoção de animal morto -- causa é acidente/doença, não chuva; risco de "
     "confundimento seria só se a causa registrada fosse afogamento em enchente, o que não é "
     "o padrão desta subdivisão (recolhimento rotineiro de via pública)."),
    ("Animais domésticos", "Remoção de felinos mortos",
     "Mesma justificativa do item anterior."),
    ("Poluição", "Sonora - Noturna",
     "Denúncia de ruído, sem relação com chuva."),
    ("Poluição", "Sonora - Diurna",
     "Denúncia de ruído, sem relação com chuva."),
    ("Fiscalização do comércio estabelecido", "Comércio diurno",
     "Fiscalização administrativa de comércio, sem relação com chuva."),
    ("Coleta", "Resíduos vegetais de jardim",
     "Coleta de resíduo doméstico de rotina, sem relação causal com chuva (volume pode variar "
     "sazonalmente por poda, não por evento de chuva pontual)."),
    ("Coleta", "Entulhos diversos (pequena quantidade)",
     "Coleta de resíduo de rotina, sem relação com chuva."),
    ("Coleta", "Caliças até 5 carrinhos de mão",
     "Coleta de resíduo de obra de rotina, sem relação com chuva."),
    ("Praças", "Conservação",
     "Manutenção de rotina de praça -- distinto de dano por temporal, que teria assunto "
     "próprio de árvore/queda; conservação de rotina não é gatilho de chuva."),
    ("Praças", "Manutenção",
     "Mesma justificativa do item anterior."),
    ("Lombada física", "Implantação",
     "Obra de infraestrutura viária planejada, sem relação com chuva."),
    ("Sinalização horizontal", "Repintura",
     "Manutenção viária de rotina, sem relação com chuva."),
    ("Fiscalização de obras", "Alvará de construção",
     "Trâmite administrativo de licenciamento, sem relação com chuva."),
    ("Abordagem Social - Situação de Rua", "Pessoas/famílias em desabrigo na rua",
     "Atendimento social, sem relação causal direta com chuva (ainda que o clima possa "
     "agravar a situação de rua, o REGISTRO em si não é desencadeado por chuva do mesmo jeito "
     "que um alagamento -- mantido por ser o padrão de atendimento social contínuo, não "
     "reativo a evento climático)."),
    ("Unidade de Saúde - UMS", "Agendamento - Consulta com Especialista",
     "Atendimento administrativo de saúde, sem relação com chuva."),
    ("Árvore", "Poda de manutenção em árvores",
     "Manutenção agendada de rotina -- distinto de queda de árvore por temporal (excluída "
     "à parte, ver EXCLUDED abaixo)."),
    ("Árvore", "Remoção de tocos / destoca",
     "Manutenção de rotina, não é reação a queda por temporal."),
]

# Categorias explicitamente EXCLUÍDAS por dependerem de chuva/drenagem em Curitiba -- mesma
# lógica causal de Recife (EXCLUDE_NEG_CATEGORIES), mas reavaliada categoria a categoria pro
# vocabulário real do SIAC 156, não copiada.
CURITIBA_EXCLUDED_RAIN_LINKED: list[tuple[str, str, str]] = [
    ("Drenagem", "*", "Categoria é o próprio sistema causal da enchente (limpeza/desobstrução "
     "de caixa de captação, erosão em galeria pluvial, reposição de grelha) -- mesma exclusão "
     "de Recife (grupo DRENAGEM do 156)."),
    ("Pavimentação", "*", "Dano de pavimento se agrava com chuva/erosão -- mesma exclusão de "
     "Recife (grupo PAVIMENTAÇÃO)."),
    ("Árvore", "Árvore ou galho caído – com bloqueio de passagem",
     "Queda de árvore/galho é frequentemente desencadeada por temporal/vento -- mesma lógica "
     "de Recife excluindo 'arvore_em_risco'."),
    ("Árvore", "Árvore ou galho caído – sem bloqueio de passagem",
     "Mesma justificativa do item anterior."),
    ("Vigilância de Zoonoses", "Risco para Leptospirose/ Roedores em bueiro",
     "Risco de leptospirose está causalmente ligado a água parada/enchente -- contaminaria o "
     "contraste com o próprio fenômeno-alvo."),
    ("Fiscalização de terrenos baldios ou edificados", "Limpeza – dengue",
     "Combate a foco de dengue está ligado a água parada acumulada, frequentemente pós-chuva "
     "-- excluído por relação causal com chuva."),
    ("Fiscalização", "Cabos danificados",
     "Dano de cabo aéreo pode ser causado por queda de galho em temporal -- ambíguo o "
     "suficiente pra excluir por precaução."),
]


_NEG_CATEGORY_SET: set[tuple[str, str]] = {(a, s) for a, s, _ in CURITIBA_NEG_CATEGORIES}


def is_curitiba_negative_category(assunto: str, subdivisao: str) -> bool:
    return ((assunto or "").strip(), (subdivisao or "").strip()) in _NEG_CATEGORY_SET
