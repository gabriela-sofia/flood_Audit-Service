from pathlib import Path
import importlib.util
import json
import sys
import pytest

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ/'reproducao'))
from verificar_pacote import verificar
from conferir_resultados import conferir
from materializar_evidencias import materializar

@pytest.fixture(scope='session',autouse=True)
def preparar_arquivos_incluidos():
    materializar()

def test_arquivos_codigo_links_e_rastreabilidade():
    r=verificar()
    assert r['ok'],r['erros']

def test_transcricao_vem_dos_artefatos():
    r=conferir()
    assert r['ok'],r['divergencias']

def test_recorte_contem_apenas_um_artigo_atual():
    assert len(list((RAIZ/'artigo').rglob('*.pdf')))==1
    assert not (RAIZ/'historico').exists()

def test_exportacao_preserva_os_dois_estados_ajustados():
    from acervo import ler_json
    fonte=ler_json('local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json')
    for nome,modelo in [('completa','Q_sel_ext'),('com_reserva','Qsep_sel_fis')]:
        ps=json.loads((RAIZ/f'modelos/fisico/pesos_{nome}.json').read_text(encoding='utf-8'))
        assert ps['modelo']==modelo
        assert ps['parametros']==fonte[modelo]

def carregar_teste(nome):
    p=RAIZ/'codigo/tests'/nome
    spec=importlib.util.spec_from_file_location('portavel_'+p.stem,p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_metrica_de_recife_por_bairro():
    # Reutiliza a prova numérica original; a função usa só dados sintéticos.
    m=carregar_teste('test_recife_aprendizado_v1.py')
    m.test_percentil_dentro_do_bairro_ignora_outros_bairros()

def test_prior_de_recife_preserva_ou_atualiza_conforme_regularizacao():
    m=carregar_teste('test_recife_aprendizado_v1.py')
    m.test_prior_forte_preserva_q_e_fraco_aprende()

def test_metrica_por_cicatriz_confere_por_outro_caminho():
    m=carregar_teste('test_petropolis_aprendizado_v1.py')
    m.test_auc_obj_igual_a_forca_bruta()

def test_prior_de_encosta():
    m=carregar_teste('test_petropolis_aprendizado_v1.py')
    m.test_prior_forte_prende_coeficientes_e_prior_fraco_aprende()
