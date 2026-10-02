"""Combina OOF existentes; não importa nem retreina classificadores.
Pesos/desempates: F1 macro OOF dos outros quatro folds. Ver README sobre
 a dependência indireta dos modelos-base (não é validação aninhada).
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

MODELS = ('LinearSVC', 'XGBoost', 'Random Forest')
FIXED_ORDER = (0, 1, 2)  # Igualdade exata de F1: ordem fixa, anterior aos resultados.


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(values, name):
    n = pd.to_numeric(pd.Series(values), errors='raise')
    check(n.notna().all() and np.isfinite(n).all() and (n == np.floor(n)).all(), f'{name}: inteiro inválido')
    return n.to_numpy(dtype=np.int64)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_table(path):
    if path.suffix == '.parquet':
        print(f'Lendo {path}', flush=True)
        return pd.read_parquet(path)
    header = pd.read_csv(path, nrows=0)
    print(f'{path.name}: colunas reais = {header.columns.tolist()}', flush=True)
    print('Contando registros para a barra de leitura...', flush=True)
    total = sum(len(c) for c in pd.read_csv(path, usecols=[0], chunksize=200000))
    chunks = []
    with tqdm(total=total, desc=f'Lendo {path.parent.name}', unit='doc', dynamic_ncols=True) as bar:
        for chunk in pd.read_csv(path, chunksize=100000):
            chunks.append(chunk); bar.update(len(chunk))
    check(bool(chunks), f'Arquivo vazio: {path}')
    return pd.concat(chunks, ignore_index=True)


def load(source, name, manifest):
    source = Path(source)
    if source.is_dir():
        candidates = [source / f'predicoes_oof{ext}' for ext in ('.csv', '.parquet')]
        candidates = [p for p in candidates if p.exists()]
        check(len(candidates) == 1, f'{name}: esperado exatamente um predicoes_oof.csv/parquet em {source}')
        source = candidates[0]
    check(source.is_file(), f'Arquivo ausente: {source}')
    df = read_table(source)
    required = {'y_true_id', 'y_pred_id', 'y_true', 'y_pred'}
    check(required <= set(df), f'{name}: esquema não reconhecido; faltam {required-set(df)}. Não inferimos colunas.')
    used = [source]
    # Nos exports XGBoost inspecionados, ambos os vetores têm a ordem de y/OOF.
    # Não usar arange nem os idx_*_parquet: o export antigo grava ali posições filtradas.
    if 'indice_parquet' not in df:
        path = source.parent / 'indices_parquet.npy'
        check(path.exists(), f'{name}: precisa de {path}; não é seguro inventar IDs pela ordem das linhas')
        values = np.load(path, allow_pickle=False)
        check(values.ndim == 1 and len(values) == len(df), f'{name}: indices_parquet.npy incompatível')
        df['indice_parquet'] = values; used.append(path)
    if 'fold' not in df:
        path = source.parent / 'folds.npy'
        check(path.exists(), f'{name}: precisa do vetor folds.npy da mesma execução')
        values = np.load(path, allow_pickle=False)
        check(values.ndim == 1 and len(values) == len(df), f'{name}: folds.npy incompatível')
        df['fold'] = values; used.append(path)
    for col in ['indice_parquet', 'fold', 'y_true_id', 'y_pred_id']:
        df[col] = integer(df[col], f'{name}/{col}')
    check(len(df)>0 and not df.indice_parquet.duplicated().any(), f'{name}: IDs duplicados ou arquivo vazio')
    check((df.indice_parquet >= 0).all(), f'{name}: ID negativo')
    check(set(df.fold) == {1,2,3,4,5}, f'{name}: esperados folds 1..5')
    for col in ['y_true', 'y_pred']:
        check(df[col].notna().all() and df[col].astype(str).str.strip().ne('').all(), f'{name}: nomes ausentes em {col}')
        df[col] = df[col].astype(str)
    manifest[name] = {'columns': df.columns.tolist(), 'n': len(df), 'files': [{'path':str(p.resolve()), 'sha256':digest(p)} for p in used]}
    return df.sort_values('indice_parquet').reset_index(drop=True)


def align(frames, nclasses, canonical=None):
    ref = frames[0]
    for frame, name in zip(frames[1:], MODELS[1:]):
        check(len(frame)==len(ref), f'{name}: número de predições diferente')
        for col in ['indice_parquet','fold','y_true_id','y_true']:
            check(np.array_equal(ref[col].to_numpy(),frame[col].to_numpy()), f'{name}: divergência em {col}')
    pairs = ref[['y_true_id','y_true']].drop_duplicates().sort_values('y_true_id')
    check(not pairs.y_true_id.duplicated().any() and not pairs.y_true.duplicated().any(), 'Mapa de classes ambíguo')
    check(pairs.y_true_id.tolist() == list(range(nclasses)), f'Esperadas {nclasses} classes com IDs 0..{nclasses-1}')
    names = pairs.y_true.tolist()
    for frame, name in zip(frames,MODELS):
        check(frame.y_pred_id.between(0,nclasses-1).all(), f'{name}: predição fora do espaço de classes')
        check(np.array_equal(np.asarray(names)[frame.y_pred_id],frame.y_pred), f'{name}: nomes/IDs preditos inconsistentes')
    if canonical is not None:
        c=read_table(canonical)
        check({'indice_parquet','fold','id_classe','classe'} <= set(c), 'Folds canônicos: esquema incompatível')
        for col in ['indice_parquet','fold','id_classe']: c[col]=integer(c[col], col)
        check(not c.indice_parquet.duplicated().any(), 'Folds canônicos têm IDs duplicados')
        c=c.set_index('indice_parquet')
        check(ref.indice_parquet.isin(c.index).all(), 'Há documentos fora dos folds canônicos')
        selected=c.loc[ref.indice_parquet]
        for a,b in [('fold','fold'),('y_true_id','id_classe'),('y_true','classe')]:
            check(np.array_equal(ref[a].to_numpy(),selected[b].to_numpy()), f'Folds canônicos divergem em {a}')
    return ref, names, np.column_stack([f.y_pred_id.to_numpy() for f in frames])


def metrics_from_cm(cm):
    tp=cm.diagonal().astype(float); support=cm.sum(axis=1); predicted=cm.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp),where=(support+predicted)>0)
    return {'accuracy':float(tp.sum()/cm.sum()), 'precision_macro':float(precision.mean()),
        'recall_macro':float(recall.mean()),'f1_macro':float(f1.mean()),
        'precision_weighted':float(np.average(precision,weights=support)),
        'recall_weighted':float(np.average(recall,weights=support)),
        'f1_weighted':float(np.average(f1,weights=support)), 'n_documentos':int(cm.sum()),
        'auc_roc':None,'auc_status':'não calculada: entrada baseada em rótulos'}, (precision,recall,f1,support)


def get_weights(y_other, pred_other, nclasses):
    raw=np.array([metrics_from_cm(confusion_matrix(y_other,pred_other[:,j],labels=np.arange(nclasses)))[0]['f1_macro'] for j in FIXED_ORDER])
    weights=raw/raw.sum() if raw.sum()>0 else np.full(3,1/3)
    check(np.isclose(weights.sum(),1), 'Pesos não somam 1')
    ranking=sorted(FIXED_ORDER,key=lambda j:(-raw[j],j))
    return raw,weights,ranking


def vote(pred, weights, ranking, batch_size=100000, progress=True):
    """Não recebe y_true; cada candidato é uma classe proposta por algum modelo.
    Pontua classes somando os pesos de todos os modelos que a propuseram.
    Empate de scores: atol=1e-12, rtol=0; depois ranking de F1 e ordem fixa.
    """
    n=len(pred); output=np.empty(n,dtype=np.int64); owner=np.full(n,-1,dtype=np.int8)
    tied=np.zeros(n,dtype=bool)
    for start in tqdm(range(0,n,batch_size),desc=f'Votação ({n:,} documentos)',unit='lote',disable=not progress,dynamic_ncols=True):
        block=pred[start:start+batch_size]
        scores=np.zeros((len(block),3),dtype=float)
        for candidate in FIXED_ORDER:
            scores[:,candidate]=((block == block[:,candidate,None])*weights).sum(axis=1)
        winners=np.isclose(scores,scores.max(axis=1,keepdims=True),rtol=0,atol=1e-12)
        distinct=np.zeros(len(block),dtype=np.int8)
        for j in FIXED_ORDER:
            first=np.ones(len(block),dtype=bool)
            for earlier in range(j): first &= block[:,j]!=block[:,earlier]
            distinct += (winners[:,j]&first)
        ties=distinct>1
        chosen=np.full(len(block),-1,dtype=np.int8)
        for j in ranking:
            take=(chosen<0)&winners[:,j]; chosen[take]=j
        check((chosen>=0).all(),'Documento sem vencedor')
        output[start:start+len(block)]=block[np.arange(len(block)),chosen]
        owner[start:start+len(block)]=np.where(ties,chosen,-1)
        tied[start:start+len(block)]=ties
    return output,tied,owner


def agreement(pred):
    same=(pred[:,0]==pred[:,1])&(pred[:,1]==pred[:,2])
    different=(pred[:,0]!=pred[:,1])&(pred[:,0]!=pred[:,2])&(pred[:,1]!=pred[:,2])
    return np.where(same,'tres_concordam',np.where(different,'tres_discordam','exatamente_dois_concordam'))


def save_evaluation(y,pred,names,out,model,fold):
    cm=confusion_matrix(y,pred,labels=np.arange(len(names)))
    metrics,per_class=metrics_from_cm(cm)
    folder=out/'matrizes_confusao'/model/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(cm,index=names,columns=names).to_csv(folder/'bruta.csv')
    pd.DataFrame(cm/np.maximum(cm.sum(axis=1,keepdims=True),1),index=names,columns=names).to_csv(folder/'normalizada.csv')
    pd.DataFrame(dict(zip(['precision','recall','f1','support'],per_class)),index=names).to_csv(folder/'metricas_por_classe.csv')
    return {'modelo':model,'fold':fold,**metrics}


def save_csv(df, path, batch_size=100000):
    """Barra em linhas efetivamente gravadas; arquivos novos somente."""
    with open(path, 'x', encoding='utf-8', newline='') as stream:
        with tqdm(total=len(df), desc=f"Gravando {Path(path).name}", unit="linha", dynamic_ncols=True) as bar:
            for start in range(0,len(df),batch_size):
                block=df.iloc[start:start+batch_size]
                block.to_csv(stream,index=False,header=start==0)
                bar.update(len(block))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['svc','xgb','rf']: p.add_argument('--'+flag,required=True,type=Path,help='Pasta do modelo ou CSV/parquet OOF')
    p.add_argument('--folds',type=Path,help='CSV canônico opcional; recomendado')
    p.add_argument('--out',type=Path,default=Path('resultados_comites_oof')/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--n-classes',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=100000)
    args=p.parse_args()
    check(args.n_classes>1 and args.batch_size>0,'Argumentos numéricos inválidos')
    check(not args.out.exists(),'Pasta de saída já existe; escolha outra. Nada será sobrescrito.')
    etapas=tqdm(total=5,desc='ETAPAS GERAIS',unit='etapa',dynamic_ncols=True)
    print('1/5 Inspeção das entradas; nenhum treinamento será executado.',flush=True)
    manifest={}
    frames=[load(src,name,manifest) for src,name in zip([args.svc,args.xgb,args.rf],MODELS)]
    etapas.update(1)
    print('2/5 Alinhamento por indice_parquet e validação de rótulos/folds.',flush=True)
    ref,names,preds=align(frames,args.n_classes,args.folds)
    del frames
    folds=ref.fold.to_numpy(); y=ref.y_true_id.to_numpy(); ids=ref.indice_parquet.to_numpy()
    args.out.mkdir(parents=True,exist_ok=False)
    audit={'inputs':manifest,'classes':names,'python':platform.python_version(),'numpy':np.__version__,
        'pandas':pd.__version__,'sklearn':sklearn.__version__,'regra_residual':list(MODELS),
        'tolerancia_score':{'atol':1e-12,'rtol':0},'n_documentos':len(y),
        'limitacao':'Pesos de OOF nos demais folds têm dependência indireta do fold avaliado via treinamento dos modelos-base; não é nested CV.'}
    if args.folds: audit['folds_canonicos']={'path':str(args.folds.resolve()),'sha256':digest(args.folds)}
    (args.out/'auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    outputs={key:np.full(len(y),-1,dtype=np.int64) for key in ['Hard Voting','Hard Voting Ponderado']}
    ties={key:np.zeros(len(y),dtype=bool) for key in outputs}
    owners={key:np.full(len(y),-1,dtype=np.int8) for key in outputs}
    weights_rows=[]; masks_audit=[]
    etapas.update(1)
    print('3/5 Pesos excluindo diretamente cada fold; definição de TODAS as previsões.',flush=True)
    for f in tqdm(range(1,6),desc='Folds do comitê'):
        test=np.flatnonzero(folds==f); other=np.flatnonzero(folds!=f)
        check(len(test)>0 and len(other)>0,'Fold vazio')
        check(np.intersect1d(ids[test],ids[other]).size==0,'Documento do fold avaliado entrou nos pesos')
        check(not np.any(folds[other]==f),'Fold avaliado entrou nos pesos')
        raw,w,ranking=get_weights(y[other],preds[other],args.n_classes)
        masks_audit.append({'fold_avaliado':f,'folds_pesos':','.join(map(str,sorted(set(folds[other])))),
            'n_avaliados':len(test),'n_calculo_pesos':len(other),'intersecao_ids':0})
        for j,name in enumerate(MODELS):
            weights_rows.append({'fold':f,'modelo':name,'f1_macro_outros_folds':raw[j],'peso_normalizado':w[j],
                'prioridade_desempate':ranking.index(j)+1,'fallback_pesos_uniformes':bool(raw.sum()==0)})
        for key,weights in [('Hard Voting',np.ones(3)),('Hard Voting Ponderado',w)]:
            outputs[key][test],ties[key][test],owners[key][test]=vote(preds[test],weights,ranking,args.batch_size)
    check(all((v>=0).all() for v in outputs.values()),'OOF incompleta')
    # Só após definir todas as previsões, avaliar cada comitê em seus próprios folds.
    etapas.update(1)
    print('4/5 Métricas por fold, globais e matrizes.',flush=True)
    models={name:preds[:,j] for j,name in enumerate(MODELS)};models.update(outputs)
    global_rows=[];fold_rows=[]
    for name,pred in tqdm(models.items(),desc='Avaliação dos cinco métodos'):
        global_rows.append(save_evaluation(y,pred,names,args.out,name,'global'))
        for f in range(1,6):
            mask=folds==f
            fold_rows.append(save_evaluation(y[mask],pred[mask],names,args.out,name,f))
    etapas.update(1)
    print('5/5 Exportação de OOF e diagnósticos.',flush=True)
    types=agreement(preds)
    for key,filename in [('Hard Voting','hard_voting'),('Hard Voting Ponderado','hard_voting_ponderado')]:
        df=ref[['indice_parquet','fold','y_true_id','y_true']].copy()
        for j,name in enumerate(['svc','xgb','rf']):df['pred_'+name]=preds[:,j]
        df['y_pred_id']=outputs[key];df['y_pred']=np.asarray(names)[outputs[key]]
        df['concordancia']=types;df['empate_de_score']=ties[key]
        df['modelo_desempate']=[MODELS[j] if j>=0 else '' for j in owners[key]]
        save_csv(df,args.out/f'predicoes_oof_{filename}.csv',args.batch_size)
    agreement_rows=[];tie_rows=[];pair_rows=[]
    for f in ['global',1,2,3,4,5]:
        mask=np.ones(len(y),dtype=bool) if f=='global' else folds==f
        for kind in ['tres_concordam','exatamente_dois_concordam','tres_discordam']:
            count=int(((types==kind)&mask).sum())
            agreement_rows.append({'fold':f,'tipo':kind,'quantidade':count,'percentual':100*count/mask.sum()})
        for key in outputs:
            for j,name in enumerate(MODELS):
                count=int(((owners[key]==j)&mask).sum())
                tie_rows.append({'fold':f,'comite':key,'modelo_decisor':name,'n_desempates':count})
        for a,b in [(0,1),(0,2),(1,2)]:
            ca=preds[mask,a]==y[mask];cb=preds[mask,b]==y[mask]
            pair_rows.append({'fold':f,'modelo_a':MODELS[a],'modelo_b':MODELS[b],
                'ambos_acertam':int((ca&cb).sum()),'ambos_erram':int((~ca&~cb).sum()),
                'somente_a_acerta':int((ca&~cb).sum()),'somente_b_acerta':int((~ca&cb).sum())})
    # ------------------------------------------------------------------
    # Resumo das métricas dos 5 folds: média e desvio-padrão amostral
    # ------------------------------------------------------------------
    fold_df = pd.DataFrame(fold_rows)

    metric_cols = [
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'f1_weighted',
    ]

    resumo_folds = (
        fold_df
        .groupby('modelo', sort=False)[metric_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # Achata os nomes das colunas:
    # ('f1_macro', 'mean') -> 'f1_macro_media'
    # ('f1_macro', 'std')  -> 'f1_macro_desvio'
    resumo_folds.columns = [
        'modelo' if col[0] == 'modelo'
        else f"{col[0]}_{'media' if col[1] == 'mean' else 'desvio'}"
        for col in resumo_folds.columns
    ]

    # O pandas usa ddof=1 em GroupBy.std(), portanto este é o
    # desvio-padrão AMOSTRAL entre os cinco folds.
    resumo_folds['n_folds'] = 5

    for filename,rows in [
        ('metricas_globais', global_rows),
        ('tabela_comparativa_final', global_rows),
        ('metricas_por_fold', fold_rows),
        ('pesos_por_fold', weights_rows),
        ('auditoria_exclusao_folds', masks_audit),
        ('concordancia_desacordo', agreement_rows),
        ('desempates_por_modelo', tie_rows),
        ('complementaridade_pares', pair_rows),
    ]:
        save_csv(pd.DataFrame(rows), args.out/f'{filename}.csv', args.batch_size)

    # Novo arquivo: média ± desvio-padrão das métricas entre os 5 folds
    save_csv(
        resumo_folds,
        args.out/'resumo_metricas_folds.csv',
        args.batch_size
    )

    print('\nMÉTRICAS GLOBAIS')
    print(
        pd.DataFrame(global_rows)[
            ['modelo','accuracy','precision_macro','recall_macro','f1_macro','f1_weighted']
        ].to_string(index=False)
    )

    print('\nMÉDIA ± DESVIO-PADRÃO ENTRE OS 5 FOLDS')
    for _, row in resumo_folds.iterrows():
        print(
            f"{row['modelo']}: "
            f"accuracy={row['accuracy_media']:.4f} ± {row['accuracy_desvio']:.4f} | "
            f"precision_macro={row['precision_macro_media']:.4f} ± {row['precision_macro_desvio']:.4f} | "
            f"recall_macro={row['recall_macro_media']:.4f} ± {row['recall_macro_desvio']:.4f} | "
            f"f1_macro={row['f1_macro_media']:.4f} ± {row['f1_macro_desvio']:.4f} | "
            f"f1_weighted={row['f1_weighted_media']:.4f} ± {row['f1_weighted_desvio']:.4f}"
        )
    etapas.update(1); etapas.close()
    print(f'Concluído: {args.out.resolve()}',flush=True)

if __name__=='__main__':
    main()
