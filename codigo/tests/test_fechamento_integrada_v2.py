from pathlib import Path
import sys
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/ia'))
import ia_integrada_v2 as ia
import retomar_avaliacao_v2 as resume
import revisao_humana_v2 as review

def meta_sintetico(tmp_path,monkeypatch):
    ab=tmp_path/'ablacao';cont=tmp_path/'protocolo.json'
    names=[ab/n for n in ('combinado_s7/melhor.pt','combinado_s7/treino.json','protocolo.json','resultado.json')]+[cont,tmp_path/'inventario.json',tmp_path/'cartoes_cientificos.json']
    for p in names:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('sintetico')
    monkeypatch.setattr(resume.old,'AB',ab);monkeypatch.setattr(resume.old,'CONT',cont)
    return {'fontes':[{'arquivo':str(p),'sha256':ia.sha(p)} for p in names]},names

def test_contexto_mutavel_nao_e_dependencia_sar(tmp_path,monkeypatch):
    meta,paths=meta_sintetico(tmp_path,monkeypatch)
    paths[-1].write_text('novo contexto, não é dado neural')
    selected=resume.fontes_sar(meta)
    assert len(selected)==6 and not any(Path(f['arquivo']).name=='cartoes_cientificos.json' for f in selected)

def test_peso_sar_alterado_continua_bloqueado(tmp_path,monkeypatch):
    meta,paths=meta_sintetico(tmp_path,monkeypatch);paths[0].write_text('peso alterado')
    with pytest.raises(ValueError,match='alterada'):resume.fontes_sar(meta)

def test_revisao_declarada_preserva_incerto_e_nao_promove(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'DEST',tmp_path)
    ia.gravar(tmp_path/'manifesto.json',{'ids':['synthetic_a','synthetic_b']})
    csv=tmp_path/'respostas.csv';decl=tmp_path/'declaracao.json'
    pd.DataFrame([{'id':key,'resposta':'incerto','confianca':'1','revisor':'revisor_teste','tipo_revisor':'humano'} for key in ('synthetic_a','synthetic_b')]).to_csv(csv,index=False)
    ia.gravar(decl,{'tipo_revisor':'humano','revisor_id':'revisor_teste','formulario_sha256':ia.sha(csv)})
    got=review.validar(csv,decl,tmp_path)
    assert got['contagens']=={'incerto':2} and not got['rotulo_promovido_automaticamente']

def test_revisao_identidade_ia_e_hash_adulterado_recusados(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'DEST',tmp_path);csv=tmp_path/'respostas.csv';csv.write_text('synthetic')
    decl=tmp_path/'declaracao.json'
    ia.gravar(decl,{'tipo_revisor':'humano','revisor_id':'revisor_ia_visual','formulario_sha256':ia.sha(csv)})
    with pytest.raises(ValueError,match='identidade'):review.validar(csv,decl,tmp_path)
    ia.gravar(decl,{'tipo_revisor':'humano','revisor_id':'revisor_teste','formulario_sha256':'sha256:errado'})
    with pytest.raises(ValueError,match='hash'):review.validar(csv,decl,tmp_path)

