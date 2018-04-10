Estrutura dos arquivos do repositório

* modules
    * face_model.py - Gera o modelo da face usando a análise das componentes principais.
    * pca.py - Funções úteis para o cálculo da PCA.
    * procrustes.py - Procedimentos para normalização das formas do dataset para que a PCA possa ser aplicada.
    * regression_tree.py - Classe que contém o modelo das árvores de regressão.
    * util.py - Funções úteis para cálculos menores em geral.
* build_dataset.py - Script para processamento do dataset, com leitura das imagens e serialização para facilitar o carregamento para a memória.
* model_train.py - Script para treinar o modelo.
* webcam_test.py - Script para testar o modelo usando a webcam.