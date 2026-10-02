# Instruções para Reprodução do Trabalho

> **Visão Geral:** Este documento descreve a estrutura de diretórios, os arquivos de código necessários e onde encontrar os resultados detalhados das avaliações dos modelos.

---

## Como rodar o trabalho

Todos os códigos e scripts necessários para realizar o treinamento dos modelos estão concentrados na pasta principal **`reproduzir/`**. 

Abaixo está a relação dos arquivos que executam o fluxo de trabalho:

| Arquivo de Código | Descrição |
| :--- | :--- |
| `treinar_random_forest_50_kfold` | Script para treinar o modelo Random Forest usando 50 folds. |
| `treinar_svc_50_kfold` | Script para treinar o modelo SVC usando 50 folds. |
| `treinar_xgboost_50_com_roc` | Script para treinar o XGBoost, incluindo cálculo da curva ROC. |
| `comites_oof_50_com_desvio` | Script para montar o comitê (Ensemble) utilizando as previsões *Out-of-Fold* (OOF). |

---

## Resultados e Avaliações

Após a execução dos arquivos acima, os resultados de cada modelo — contendo as **avaliações por fold** e as **métricas gerais** — são salvos automaticamente no seguinte subdiretório:
**`reproduzir/modelo_treinado/`**

Os arquivos de resultados gerados recebem os seguintes nomes:

- `random_forest_50_kfold`
- `svc_50_kfold`
- `xgboost_50_kfold`
- `comite_3_modelos`

---

## Resumo da Estrutura de Diretórios

Para facilitar a navegação, sua estrutura de pastas final (após executar os códigos) ficará com este formato:

```text
reproduzir/
├── treinar_random_forest_50_kfold
├── treinar_svc_50_kfold
├── treinar_xgboost_50_com_roc
├── comites_oof_50_com_desvio
└── modelo_treinado/
    ├── random_forest_50_kfold
    ├── svc_50_kfold
    ├── xgboost_50_kfold
    └── comite_3_modelos
```

> **Atenção:** Certifique-se de iniciar a execução dos scripts de treinamento estando dentro da pasta raiz do projeto ou diretamente dentro da pasta `reproduzir`. Isso garante que os caminhos relativos apontem corretamente para a pasta `modelo_treinado`.
