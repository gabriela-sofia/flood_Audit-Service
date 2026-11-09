"""Interface cega funcional dos 49 itens; nenhuma resposta é preenchida pela IA."""
from pathlib import Path
import argparse,base64,csv,html,json,re
import pandas as pd
from ia_integrada_v2 import REPO,OUT,ler,sha,gravar

SOURCE=REPO/'local_runs/confirmacao-rodada3/v1/revisao/ampliada_v1'
DEST=OUT/'revisao_humana'
TOKENS_IA={'ia','ai','codex','claude','chatgpt','gpt','gemini','llm','openai'}

def preparar():
    form=pd.read_csv(SOURCE/'formulario_ampliado.csv',dtype=str).fillna('')
    if form.id.duplicated().any() or len(form)!=49:raise ValueError('identidade do pacote mudou')
    DEST.mkdir(parents=True,exist_ok=True);cards=[];sources=[SOURCE/'formulario_ampliado.csv',SOURCE/'instrucoes.json']
    for row in form.itertuples():
        images=[]
        for suffix,label in [('falsacor_30m','Falsa-cor de 30 m'),('rgb_10m','RGB de 10 m')]:
            file=SOURCE/'imagens'/f'{row.id}_{suffix}.png';sources.append(file)
            image=base64.b64encode(file.read_bytes()).decode()
            images.append(f'<figure><img src="data:image/png;base64,{image}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        cards.append(f'<article data-id="{html.escape(row.id)}"><h2>Item {len(cards)+1} · {html.escape(row.id)}</h2><div class="images">{"".join(images)}</div><label>O pixel no quadrado magenta contém água visível? <select class="resposta"><option value="">Escolher</option><option value="agua">Água</option><option value="seco">Seco</option><option value="incerto">Não consigo determinar</option></select></label><label>Confiança <select class="confianca"><option value="">Escolher</option><option>1</option><option>2</option><option>3</option></select></label></article>')
    template=r'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>REV-P · revisão humana cega</title>
<style>body{font:17px system-ui;background:#eef3f6;color:#172e3b;max-width:1000px;margin:auto;padding:24px}header{position:sticky;top:0;background:#eef3f6;padding:12px;box-shadow:0 2px 6px #bbc;z-index:2}article{background:white;padding:20px;margin:22px 0;border-radius:12px}.images{display:flex;gap:12px}figure{margin:0;flex:1}img{width:100%;image-rendering:auto}figcaption{font-size:14px}label{display:block;margin-top:12px}select,input,button{font:inherit;padding:8px}button{cursor:pointer;background:#176784;color:white;border:0;border-radius:6px;margin:8px}h2{font-size:19px}</style>
<header><h1>Revisão humana de água observável</h1><p>Examine o pixel indicado. Nuvem ou dúvida: “Não consigo determinar”. Nenhuma classe automática, predição ou motivo de seleção é mostrado.</p><label>Seu nome ou identificador <input id="revisor" autocomplete="name"></label><p id="progresso"></p></header>
<main>CARDS</main><footer><label><input id="declaracao" type="checkbox"> Declaro que sou a pessoa que examinou as imagens e registrou estas respostas.</label><button id="csv">Baixar respostas CSV</button><button id="json">Baixar declaração JSON</button><p id="erro" role="alert"></p><p>São itens dirigidos para adjudicação, não amostra para estimar acurácia populacional. A declaração registra autoria informada e integridade; não é autenticação independente.</p></footer>
<script>
const cards=[...document.querySelectorAll('article')],key='revp-revisao-humana-v2-49';
function save(){localStorage.setItem(key,JSON.stringify({revisor:document.querySelector('#revisor').value,rows:cards.map(c=>({id:c.dataset.id,resposta:c.querySelector('.resposta').value,confianca:c.querySelector('.confianca').value}))}));document.querySelector('#progresso').textContent=cards.filter(c=>c.querySelector('.resposta').value&&c.querySelector('.confianca').value).length+' / '+cards.length+' respondidos';}
try{const z=JSON.parse(localStorage.getItem(key)||'{}');document.querySelector('#revisor').value=z.revisor||'';for(const c of cards){const row=(z.rows||[]).find(r=>r.id===c.dataset.id);if(row){c.querySelector('.resposta').value=row.resposta;c.querySelector('.confianca').value=row.confianca;}}}catch(e){}
document.addEventListener('change',save);save();
const quote=x=>'"'+String(x).replaceAll('"','""')+'"';
function data(){const rid=document.querySelector('#revisor').value.trim();if(!rid||!document.querySelector('#declaracao').checked)throw Error('Informe sua identidade e marque a declaração.');if(cards.some(c=>!c.querySelector('.resposta').value||!c.querySelector('.confianca').value))throw Error('Responda os 49 itens; use “Não consigo determinar” quando necessário.');return {rid,csv:'id,resposta,confianca,revisor,tipo_revisor\n'+cards.map(c=>[c.dataset.id,c.querySelector('.resposta').value,c.querySelector('.confianca').value,rid,'humano'].map(quote).join(',')).join('\n')+'\n'};}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.querySelector('#csv').onclick=()=>{try{const z=data();download('respostas_humanas.csv',z.csv,'text/csv;charset=utf-8');document.querySelector('#erro').textContent='';}catch(e){document.querySelector('#erro').textContent=e.message;}};
document.querySelector('#json').onclick=async()=>{try{const z=data(),digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(z.csv));const hash=[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');download('revisor_humano.json',JSON.stringify({tipo_revisor:'humano',revisor_id:z.rid,formulario_sha256:'sha256:'+hash,pacote:'ampliada_v1_49',declaracao:'Examinei as imagens e registrei pessoalmente as respostas.'},null,2),'application/json');document.querySelector('#erro').textContent='';}catch(e){document.querySelector('#erro').textContent=e.message;}};
</script></html>'''
    (DEST/'revisao.html').write_text(template.replace('CARDS',''.join(cards)),encoding='utf-8')
    gravar(DEST/'manifesto.json',{'tipo':'interface_humana_cega','itens':len(form),'ids':form.id.tolist(),'html_sha256':sha(DEST/'revisao.html'),
      'fontes':[{'arquivo':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sources],
      'respostas_preenchidas_pela_ia':0,'pendencia':'aguarda revisor humano','selecionada_para':'adjudicação dirigida; não estima acurácia'})
    return {'itens':len(form),'html':str(DEST/'revisao.html'),'respostas_humanas':0}

def validar(csv_path,decl_path,destino=DEST):
    decl=ler(decl_path);rid=str(decl.get('revisor_id','')).strip()
    if decl.get('tipo_revisor')!='humano' or not rid or TOKENS_IA.intersection(re.findall(r'[^\W_]+',rid.lower())):raise ValueError('identidade humana declarada inválida')
    if decl.get('formulario_sha256')!=sha(csv_path):raise ValueError('hash da declaração não confere')
    d=pd.read_csv(csv_path,dtype=str).fillna('');manifest=ler(DEST/'manifesto.json')
    if d.id.duplicated().any() or set(d.id)!=set(manifest['ids']):raise ValueError('ids divergentes')
    if not d.resposta.isin(['agua','seco','incerto']).all() or not d.confianca.isin(['1','2','3']).all():raise ValueError('resposta/confiança inválida')
    if set(d.revisor.str.strip())!={rid} or set(d.tipo_revisor)!={'humano'}:raise ValueError('autoria divergente')
    result={'tipo':'recebimento_revisao_humana_declarada','revisor_id':rid,'itens':len(d),'contagens':d.resposta.value_counts().to_dict(),
       'csv_sha256':sha(csv_path),'declaracao_sha256':sha(decl_path),'rotulo_promovido_automaticamente':False,
       'limite':'autoria autodeclarada; adjudicação dirigida não mede acurácia populacional'}
    path=Path(destino)/('recebimento_'+sha(csv_path)[7:19]+'.json')
    if path.exists() and ler(path)!=result:raise ValueError('registro anterior divergente')
    gravar(path,result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('acao',choices=['preparar','validar']);ap.add_argument('--csv',type=Path);ap.add_argument('--declaracao',type=Path);a=ap.parse_args()
    print(json.dumps(preparar() if a.acao=='preparar' else validar(a.csv,a.declaracao),ensure_ascii=False))

