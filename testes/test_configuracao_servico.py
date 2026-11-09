"""A integração HTTP opcional exige configuração e permissão explícitas."""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'codigo/scripts/servico'))
from svc06_explicacao_llm import MotorHTTP


def test_configuracao_nao_autoriza_egresso(monkeypatch):
    monkeypatch.setenv('REV_P_MODELO_HTTP', 'modelo_de_teste')
    monkeypatch.setenv('REV_P_MODELO_API_KEY', 'chave_ficticia')
    assert not MotorHTTP().disponivel()[0]
    assert MotorHTTP(permitir_egresso=True).disponivel() == (True, '')


def test_permissao_sem_configuracao_mantem_servico_indisponivel(monkeypatch):
    monkeypatch.delenv('REV_P_MODELO_HTTP', raising=False)
    monkeypatch.setenv('REV_P_MODELO_API_KEY', 'chave_ficticia')
    assert not MotorHTTP(permitir_egresso=True).disponivel()[0]
    monkeypatch.delenv('REV_P_MODELO_API_KEY')
    assert not MotorHTTP(modelo='modelo_de_teste', permitir_egresso=True).disponivel()[0]
