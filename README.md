# Detecção de Pontos Faciais em Bases de Dados Defeituosas com Árvores de Regressão e Análise de Componentes Principais

Algoritmo em Python para detecção de pontos faciais rápida em imagens otimizado para treino em bases de dados defeituosas.

## Requisitos

Os softwares abaixo (ou versões compatíveis) são necessários para executar os scripts.

- [Python 3.6](https://www.python.org/);
- [OpenCV 3.4.2](https://github.com/opencv/opencv/releases/tag/3.4.2);

## Estrutura dos arquivos do repositório

- modules
  - face_model.py - Gera o modelo da face usando a análise das componentes principais.
  - pca.py - Funções úteis para o cálculo da PCA.
  - procrustes.py - Procedimentos para normalização das formas do dataset para que a PCA possa ser aplicada.
  - regression_tree.py - Classe que contém o modelo das árvores de regressão.
  - util.py - Funções úteis para cálculos menores em geral.
- build_dataset.py - Script para processamento do dataset, com leitura das imagens e serialização para facilitar o carregamento para a memória.
- model_train.py - Script para treinar o modelo.
- webcam_test.py - Script para testar o modelo usando a webcam.

# Documentação

- [Introdução](#introducao)
- [Metodologia](#metodologia)
  - [Tratamento de pontos faltantes](#tratamento-de-pontos-faltantes)
  - [Análise de componentes principais](#analise-de-componentes-principais)
  - [Aplicação das árvores de regressão](#aplicacao-das-arvores-de-regressao)
  - [Estimativa inicial](#estimativa-inicial)
  - [Processamento das amostras](#processamento-das-amostras)
  - [Treino e predição](#treino-e-predicao)
  - [Avaliação da estimativa](#avaliacao-da-estimativa)
- [Resultados](#resultados)
  - [Desempenho em bases com falhas de demarcação](#desempenho-em-bases-com-falhas-de-demarcacao)
  - [Discussões](#discussoes)
- [Conclusão](#conclusao)
- [Referências](#referencias)

## Introdução

No âmbito da comunicação humana não-verbalizada, a leitura visual de traços faciais representa papel de crítica importância. A expressividade da face torna a externalização de aspectos relacionados ao estado emocional, físico e mental do interlocutor um processo natural e inconsciente, que nem sempre pode ocorrer com a mesma eficiência em outras formas de comunicação (Frith 2009). Nesse contexto, elementos como a posição das sobrancelhas, lábios e olhos atuam como entidades comunicativas cujo potencial interativo pode se estender também às interfaces digitais. Assim, o uso da face como mecanismo de interação é capaz de prover experiências de uso ricas, que, além de promover intuitividade, dinamicidade e inclusividade, ampliam o espectro de interações oferecidos por modais tradicionais, como teclado e _mouse_ (Jaimes and Sebe 2007).

Em essência, a detecção de traços faciais em imagens consiste em determinar a posição de pontos característicos, fixamente demarcados sobre regiões específicas do rosto. Metodologias estatísticas baseadas em modelos paramétricos foram exploradas na literatura para a detecção de pontos característicos de estruturas orgânicas em imagens (Cootes et al. 1994). Nessas abordagens, os perfis de níveis de cinza ao longo de linhas normais ao contorno das formas foram utilizados para calcular o conjunto de parâmetros ótimos. Alternativamente, o cálculo com base na textura da região de interesse também foi estudado em trabalhos posteriores (Cootes, Edwards, and Taylor 2001).

Parametrizações baseadas nos graus de liberdade determinantes na posição dos pontos da forma também foram adotadas para a detecção de poucos elementos característicos da face (Dollár, Welinder, and Perona 2010). Para isso, foram empregados técnicas de regressão, nos quais os pontos alinham-se em suas respectivas localizações após sucessivos ajustes realizados sobre uma estimativa inicial para as variáveis envolvidas.

A estratégia de alinhamento progressivo também foi utilizada em modelos não paramétricos, _i_._e_., algoritmos nos quais o ajuste é feito diretamente sobre a posição dos pontos em vez de parâmetros descritivos (Cao et al. 2012). O ajuste gradual das formas em cada estágio do processo pode ser obtido por meio do trabalho conjunto de árvores de regressão binárias que, embora tenham desempenho pouco satisfatório se aplicadas individualmente, podem alcançar boa acurácia quando usadas em sequência (Kazemi and Sullivan 2014).

Em geral, as técnicas baseadas em ajustes granulares alcançam performances superiores às de modelos paramétricos, uma vez que são capazes de considerar com maior rigor as sutilezas das formas, além de implicarem em custo computacional menor para treino e aplicação. Entretanto, o desempenho de tais abordagens tende a sofrer impacto negativo significativo quando os dados usados para treino apresentam falhas de demarcações, visto que sua capacidade de ajuste depende do correto posicionamento de todos os pontos. Demarcações faltantes podem advir de aspectos comuns à construção de bases para aplicações no mundo real, tais como falha humana, oclusão, ou mesmo imprecisões causadas pela presença de ruído. Nesse cenário, o uso da parametrização, aliado às estratégias de ajuste progressivo das técnicas de regressão, é capaz de produzir modelos menos sensíveis à presença de falhas.

## Metodologia

Seja $\boldsymbol{I}$ uma imagem representada como uma matriz bidimensional em que cada elemento contém a intensidade em escala de cinza de um pixel. Para um conjunto de $n$ pontos de interesse de $\boldsymbol{I}$, dados por $(x_1, y_1), (x_2, y_2), \cdots , (x_n, y_n)$, e distribuídos ao longo dos traços faciais delimitantes, pode-se definir uma forma $\boldsymbol{S}$ conforme a matriz abaixo. Deseja-se, pois, obter a matriz estimativa $\boldsymbol{\hat{S}}$, de modo que os elementos de $\boldsymbol{\hat{S}}$ se aproximem dos respectivos elementos de $\boldsymbol{S}$ tanto quanto possível.

$\boldsymbol{S} =\begin{bmatrix}x_1 & y_1 \x_2 & y_2 \vdots & \vdots \x_n & y_n\end{bmatrix}.$

Em síntese, o processo inicia-se pela correção de possíveis falhas na base de dados através da heurística de interpolação linear. Em seguida, o modelo paramétrico é construído com auxílio da análise de componentes principais e uma estimativa inicial é calculada para cada amostra. Por fim, uma cadeia de regressores é treinada para realizar o ajuste dos parâmetros das estimativas iniciais, até que se alcance a convergência.

### Tratamento de pontos faltantes

Quando a base de dados utilizada para a construção do modelo apresenta falhas na demarcação dos pontos, para cada forma do conjunto total existe uma probabilidade $\rho$, com $0 \leq \rho \leq 1$, de não se conhecer o ponto $(x_i, y_i)$. Nesses casos, faz-se necessário aplicar heurísticas corretivas de interpolação para preencher as lacunas existentes. Como a análise de componentes principais incorpora variações gerais da forma, em vez de posições específicas dos pontos que a constituem, métodos simples podem ser utilizados, como a interpolação linear.

Seja $K = \langle (x_i, y_i), \cdots, (x_j, y_j) \rangle$ uma lista contígua de pontos faltantes entre os índices $i$ e $j$, ao longo forma $\boldsymbol{S}$, de modo que $(x_{i - 1}, y_{i-1})$ e $(x_{j + 1}, y_{j + 1})$ sejam pontos de extremidade com localizações conhecidas. É possível interpolar os valores dos pontos $(x_k, y_k)$, com $i \leq k \leq j$, por meio da equação:

![(x_k, y_k) = (x_{i - 1}, y_{i - 1}) + \frac{j - k + 1}{j - i + 1}[(x_{j + 1}, y_{j + 1}) - (x_{i - 1}, y_{i-1})].](https://s3.amazonaws.com/abnersn/github/facial-landmarks/equations/eq_21.png)

### Análise de componentes principais

Se todas as formas em um dado conjunto $W = {\boldsymbol{S}_1, \boldsymbol{S}_2, \cdots, \boldsymbol{S}_k}$ representam consistentemente um mesmo objeto, então seus pontos devem seguir uma distribuição estatística no espaço bidimensional (Cootes et al. 1994). Ao se conhecer tal distribuição, obtém-se um modelo parametrizado capaz de descrever as formas do conjunto, bem como produzir novas variações com aspecto plausível e fiel ao original. A análise de componentes principais – do inglês, _principal component analysis_, ou PCA– é uma abordagem eficiente para determinar essa distribuição. Uma forma $\boldsymbol{S}$ pode ser reconstruída a partir dos parâmetros da PCA conforme a equação a seguir.

$\boldsymbol{S} = \boldsymbol{\bar{S}} + \boldsymbol{B}\boldsymbol{p}.$

$\boldsymbol{\bar{S}}$ se refere à média das $k$ formas de $W$, $\boldsymbol{B}$ é uma matriz composta pelos $t$ primeiros autovetores da matriz de covariância das formas e $\boldsymbol{p}$ contém $t$ parâmetros de deformação. Ao aplicar-se variações nos elementos do vetor $\boldsymbol{p}$, é possível produzir formas que não existiam no conjunto original.

Antes do cálculo do modelo é preciso normalizar as formas de $W$. Para normalizar uma forma $\boldsymbol{S}$, com $n$ pontos $(x_i, y_i)$, efetua-se a translação, escalonamento e rotação de $\boldsymbol{S}$ pelos respectivos fatores $\boldsymbol{\tau}$, $\lambda$ e $\theta$ descritos a seguir (Kendall 1989). Os termos $(z_i, w_i)$ são pontos de $\boldsymbol{S_1}$, após sua normalização por $\boldsymbol{\tau}$ e $\lambda$.

![\boldsymbol{\tau} = \sum_{i = 1}^{n}\frac{(x_i, y_i)}{n};
\qquad
\lambda = \sqrt{\sum_{i = 1}^{n} \frac{x_i^2 + y_i^2}{n}};
\qquad
\theta = \arctan{\left ( \frac{\sum_{i = 1}^{n} y_iz_i - x_iw_i}{\sum_{i = 1}^{n} x_iz_i + y_iw_i} \right )}.](https://s3.amazonaws.com/abnersn/github/facial-landmarks/equations/eq_45.png)

### Aplicação das árvores de regressão

O cálculo da estimativa $\boldsymbol{\hat{S}}$ de uma forma $\boldsymbol{S}$ é feito com base em uma cadeia de regressores constituídos por múltiplas árvores de regressão. Nos regressores, cada árvore atua como um estimador fraco, capaz de utilizar o resíduo da árvore anterior para fazer uma predição rudimentar do ajuste que precisa ser realizado. Assim, a aplicação sucessiva de estimadores fracos reduz gradativamente o resíduo, até que se alcance a convergência próxima do resultado esperado (Hill and Lewicki 2006).

A atuação dos regressores consiste, pois, essencialmente, em aplicar sucessivas correções ao posicionamento dos pontos de uma estimativa inicial. Portanto, o processo deve iniciar-se pelo cálculo de tal estimativa. Posteriormente, cada árvore do regressor deve dividir o conjunto de treino em agrupamentos, de modo que cada grupo contenha formas que sofrerão ajustes similares. Por conseguinte, a predição de novas amostras deve ser realizada pela aplicação dos mesmos critérios de divisão empregados na construção da árvore.

### Estimativa inicial

Para aplicar as árvores de regressão, é preciso definir, a princípio, uma estimativa inicial para $\boldsymbol{S}$. A forma média $\boldsymbol{\bar{S}}$ serve bem a essa finalidade, porque permite alcançar, já de início, maior proximidade com o resultado final do que uma estimativa aleatória, por exemplo. No segundo termo, a matriz $\boldsymbol{B}$ provém do cálculo da análise de componentes principais e o vetor de parâmetros $\boldsymbol{p}$ deverá ser fornecido pela cadeia de regressores. A forma média é, então, posicionada na região central da face, detectada com o auxílio do método de histograma de gradientes orientados (Dalal and Triggs 2005).

Após o cálculo da estimativa inicial, tanto $\boldsymbol{\hat{S}}$ quanto $\boldsymbol{S}$ são normalizadas pelos fatores $\tau$, $\lambda$ e $\theta$. Assim, é possível aplicar sobre elas a análise de componentes principais, para calcular os parâmetros $\boldsymbol{\hat{p}}$ e $\boldsymbol{p}$. Logo, a mudança necessária sobre $\boldsymbol{\hat{p}}$ para obter $\boldsymbol{p}$, _i_._e_., o resíduo da estimativa inicial, é dada por $\Delta \boldsymbol{p} = \boldsymbol{p} - \boldsymbol{\hat{p}}$. Deseja-se, pois, que as árvores de regressão sejam capazes de fornecer um ajuste aproximado $\Delta \boldsymbol{\hat{p}}$ para diminuir o resíduo tanto quanto possível.

### Processamento das amostras

Estruturalmente, as árvores de regressão são árvores binárias nas quais os nós internos dividem o conjunto de amostras com base em critérios específicos aplicados sobre os dados de treino. Nós de um mesmo nível aplicam o mesmo critério. Para incorporar a informação dos pixels da imagem $\boldsymbol{I}$ no processo de treino, um conjunto $M = {\boldsymbol{m}_1, \boldsymbol{m}_2, \cdots, \boldsymbol{m}_q}$ de pontos de referência aleatórios da região da face é associado a cada regressor. A partir deles, os critérios de divisão das árvores são definidos como tuplas $\mu = (\boldsymbol{m}_i, \boldsymbol{m}_j, \alpha)$, com $1 \leq i,j \leq q$ e $i \neq j$. O valor $\alpha$ é um fator limiar de diferença entre as intensidades dos pixels em $\boldsymbol{m}_i$ e $\boldsymbol{m}_j$. Tanto $\alpha$ quanto o primeiro ponto, $\boldsymbol{m}_i$, são sorteados aleatoriamente com distribuição de probabilidade uniforme. Entretanto, para a escolha de $\boldsymbol{m}_j$, são favorecidos pontos com distância euclideana mais próxima de $\boldsymbol{m}_i$.

Portanto, em cada nó $\beta$ da árvore, para cada imagem dos dados de treino, se a intensidade do pixel de coordenada $\boldsymbol{m_i}$ subtraída da intensidade do pixel de coordenada $\boldsymbol{m_j}$ superar o limiar $\alpha$, a imagem é encaminhada para a subárvore direita $\beta_\text{dir}$. Caso contrário, é encaminhada à subárvore esquerda $\beta_\text{esq}$. Ao fim das sucessivas divisões, a predição final $\Delta \boldsymbol{\hat{p}}$ da árvore é determinada pela média dos vetores $\Delta \boldsymbol{p}$ das amostras remanescentes em cada folha.

A predição de uma árvore pode ser melhorada significativamente se os critérios de divisão para cada nível forem escolhidos dentre os melhores de um conjunto $C = {\mu_1, \mu_2, \cdots, \mu_c}$, gerado de forma randômica. Um critério pode ser avaliado quanto à sua qualidade pela função $f: C \to \mathbb{R}$, definida conforme a equação abaixo, na qual $\beta_{\mu, \text{dir}}$ e $\beta_{\mu, \text{esq}}$ referem-se ao conjunto de imagens que foram encaminhadas para a subárvore direita e esquerda pelo critério $\mu$.

$f(\mu) = \left\lVert\beta_{\mu, \text{dir}}\right\rVert  \sum_{\forall , \Delta \boldsymbol{p}, \in , \beta_{\mu, \text{dir}}} \Delta \boldsymbol{p} + \left\lVert\beta_{\mu, \text{esq}}\right\rVert   \sum_{\forall , \Delta \boldsymbol{p} , \in , \beta_{\mu, \text{esq}}} \Delta \boldsymbol{p}.$

### Treino e predição

No processo de treino, cada árvore recebe uma lista de tuplas dada por $D = \langle(\boldsymbol{I}_1, \boldsymbol{p}_1, \boldsymbol{\hat{p}}_1, \Delta\boldsymbol{p}, M_1), \cdots, (\boldsymbol{I}_k, \boldsymbol{p}_k, \boldsymbol{\hat{p}}_k, \Delta\boldsymbol{p}, M_k) \rangle$, cujos termos correspondem a cada uma das $k$ amostras de imagens com formas $\boldsymbol{S}$ demarcadas. Uma vez que o treinamento de uma árvore é concluído, o processo para efetuar a predição de uma nova amostra consiste em aplicar os critérios $\mu$ à imagem em cada nível até que se atinja um nó folha. Em seguida, a predição $\Delta \boldsymbol{\hat{p}}$ que foi associada ao nó durante a aprendizagem é multiplicada a um fator de amortecimento $\phi$, e somada aos parâmetros $\boldsymbol p$ da amostra correspondente, de modo a obter um novo vetor de parâmetros atualizado $\boldsymbol{p}'$. Pode-se, então atualizar a forma da estimativa $\boldsymbol{\hat{S}}$.

Após a aplicação de sucessivas árvores de regressão, a estimativa $\boldsymbol{\hat{S}}$ deve convergir para uma configuração que, embora mais próxima de $\boldsymbol{S}$ do que a princípio, ainda requer ajustes. Faz-se necessário, portanto, atualizar o conjunto de pontos de referência $M$ das futuras árvores, para que reflitam a mudança de aspecto de $\boldsymbol{\hat{S}}$ e uma nova cadeia de ajustes possa ocorrer. Seja $\boldsymbol{\hat{S}}'$ a forma $\boldsymbol{\hat{S}}$ antes do ajuste pelas árvores. Para que sejam reposicionados, os pontos $\boldsymbol{m}$ são rotacionados por um ângulo $\gamma$ em torno do ponto mais próximo de $\boldsymbol{\hat{S}}'$ com o raio da rotação escalonado conforme um fator de escalonamento $\psi$. Os fatores $\gamma$ e $\psi$ são, respectivamente, a rotação e escala que transformam $\boldsymbol{\hat{S}}'$ em $\boldsymbol{\hat{S}}$  (Kazemi and Sullivan 2014).

### Avaliação da estimativa

Para averiguar o grau de aproximação entre a predição e a forma real, o erro é calculado como a média das distâncias euclideanas entre os pontos de $\boldsymbol{S}$ e seus correspondentes em $\boldsymbol{\hat{S}}$, como descrito na equação abaixo. Para mitigar diferenças de resolução das imagens das amostras, as distâncias são normalizadas pela distância interocular $d$, calculada a partir da diferença entre os pontos médios dos trechos correspondentes aos olhos em $\boldsymbol{S}$ (Zhu and Ramanan 2012).

$E(\boldsymbol{S}, \boldsymbol{\hat{S}}) = \frac{1}{n}{\sum_{i = 1}^{n}\frac{\sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}}{d}}.$

## Resultados

A metodologia descrita foi aplicada sobre a base de dados Helen Dataset (Le et al. 2012). Na base, há 2330 imagens demarcadas com 194 pontos distribuídos ao longo dos olhos, nariz, lábios, sobrancelhas e maxilar. Do conjunto total de dados, 2000 amostras foram usadas para treino dos modelos, 315 para testes e 15 foram descartadas por apresentarem pontos demarcados fora do espaço definido pela região de coordenadas positivas. As imagens foram processadas com espaço de cores em escala de cinza.

A princípio, o método foi testado com todas as formas definidas integralmente, isto é, sem pontos faltantes. A fim de comparar a influência do número de parâmetros da análise de componentes principais no desempenho das árvores de regressão, diferentes quantidades de parâmetros foram testadas. O erro resultante para as configurações utilizadas está exposto na tabela a seguir. Na figura, é possível comparar o ajuste realizado pelos modelos treinados com 80, 100 e 120 parâmetros na PCA em relação à forma real, _i_._e_., o ajuste ótimo.

#### Comparativo entre os erros resultantes nas amostras de teste após a aplicação dos regressores treinados pelo método proposto.

| **80 parâmetros** | **100 parâmetros** | **120 parâmetros** |
| ----------------- | ------------------ | ------------------ |
| 0,1922            | 0,1098             | 0,1090             |

#### Resultado de predições feitas pelo método proposto sobre uma amostra do Helen Dataset, com diferentes quantidades de parâmetros da PCA, em comparação à demarcação real dos pontos.

<img src="https://s3.amazonaws.com/abnersn/github/facial-landmarks/figuras/fig1.jpg" width="600" />

### Desempenho em bases com falhas de demarcação

De modo a avaliar o desempenho da técnica quando aplicada em bases com falhas na demarcação dos pontos, foram construídas bases de treino deficitárias com 40%, 60% e 80% de pontos faltantes. Para isso, pontos selecionados aleatoriamente ao longo das curvas que compõem as formas de treino originais foram removidos, com exceção das extremidades. Assim, para correção das lacunas, a heurística de interpolação linear pode ser utilizada. O resultado produzido por esse procedimento sobre um exemplar das amostras encontra-se ilustrado abaixo. É possível observar que a interpolação ocasiona perda na suavidade das curvas, além de distorções que aumentam com a quantidade de pontos removidos.

#### Amostras de treino com pontos faltantes, corrigidos pela heurística de interpolação linear, em comparação com a forma original

<img src="https://s3.amazonaws.com/abnersn/github/facial-landmarks/figuras/fig2.jpg" width="600" />

Sobre as bases faltantes, foram treinadas cadeias de regressores a partir de modelos PCA com diferentes quantidades de parâmetros. As taxas obtidas permitem constatar a proximidade entre os resultados alcançados pelos modelos faltantes e os modelos de referência, treinados sobre a base de dados completa. Apesar das deformações resultantes da interpolação linear, os ajustes puderam conservar a suavidade das curvas e consistência das formas, mesmo na ausência de 80% dos pontos.

No intuito de verificar a redução progressiva do erro ao longo da cadeia de regressores, os modelos de 120 parâmetros treinados com as amostras faltantes foram avaliados em cada estágio do processo de teste. Os resultados estão expostos nos gráficos a seguir. Observa-se que a proximidade entre os modelos treinados com bases deficitárias e o modelo treinado com a base completa é consistente ao longo de todo o processo de treinamento.

#### Exemplos de resultados obtidos a partir de modelos treinados com pontos faltantes em comparação com as respectivas demarcações reais. Foram utilizados 120 parâmetros da PCA

<img src="https://s3.amazonaws.com/abnersn/github/facial-landmarks/figuras/fig3.jpg" width="600" />

#### Erro resultante nas amostras de teste após a aplicação dos regressores treinados em amostras com pontos faltantes corrigidas por interpolação linear

| **Pontos faltantes** | **80 parâmetros** | **100 parâmetros** | **120 parâmetros** |
| -------------------- | ----------------- | ------------------ | ------------------ |
| **40%**              | 0,1926            | 0,1132             | 0,1133             |
| **60%**              | 0,1929            | 0,1152             | 0,1140             |
| **80%**              | 0,1930            | 0,1138             | 0,1166             |

#### Diferença percentual entre os erros dos modelos treinados em amostras com pontos faltantes e os modelos treinados com amostras completas

| **Pontos faltantes** | **80 parâmetros** | **100 parâmetros** | **120 parâmetros** |
| -------------------- | ----------------- | ------------------ | ------------------ |
| **40%**              | +0,20%            | +3,08%             | +3,96%             |
| **60%**              | +0,38%            | +4,93%             | +4,60%             |
| **80%**              | +0,41%            | +3,63%             | +7,02%             |

#### Redução do erro em função do número de regressores para modelos treinados com parte dos pontos faltantes em comparação à mesma métrica para o modelo treinado integralmente. Foram usados 120 parâmetros da PCA

<img src="https://s3.amazonaws.com/abnersn/github/facial-landmarks/figuras/fig4.jpg" width="600" />

### Discussões

Por meio dos resultados obtidos após a aplicação da metodologia com variações na quantidade de parâmetros PCA, constata-se que o erro tende a diminuir quando a quantidade de parâmetros aumenta. De fato, mais parâmetros implicam em maior grau de variabilidade na representação dos dados pela análise de componentes principais. Consequentemente, o modelo torna-se capaz de se ajustar melhor às sutilezas das formas. Todavia, em bases com muitos pontos faltantes, o erro do modelo com 120 parâmetros mostrou-se ligeiramente maior. Essa diferença se deve ao superajuste do modelo com excesso de parâmetros às descontinuidades que foram imprecisamente corrigidas pela interpolação linear.

Ao longo de toda a cadeia de regressores, o desempenho alcançado pela metodologia proposta sobre os dados faltantes mostrou-se próximo do que foi obtido com as amostras íntegras. Desse modo, embora simples, a heurística corretiva de interpolação linear é suficiente para que o modelo generalizado pela PCA possa ajustar-se com níveis de precisão similares aos obtidos com as formas completas. Tal flexibilidade existe porque a análise de componentes principais permite explorar a distribuição estatística da posição dos pontos no espaço para realizar o ajuste. Assim, o efeito das correções imprecisas pode ser reduzido, desde que a quantidade de parâmetros utilizadas na PCA não seja excessiva.

Ademais, o uso de estratégias de interpolação para corrigir possíveis faltas permite maior proveito das amostras de treino, ainda que parte das demarcações não estejam bem determinadas. Mais imagens à disposição para o treino trazem maior diversidade para as amostras. Assim, as árvores de regressão tornam-se capazes de fornecer predições adequadas em um escopo maior de iluminações, cenários e outras variações circunstanciais das imagens sobre as quais serão aplicadas.

## Conclusão

A combinação da estratégia paramétrica baseada na análise de componentes principais e o ajuste sequencial da cadeia de árvores de regressão resulta em um modelo de detecção de pontos faciais menos sensível à ausência de demarcações nas amostras de treino. A simples interpolação linear dos pontos faltantes mostrou-se suficiente para que os regressores pudessem realizar o ajuste com desempenho próximo do que obtiveram quando treinados sobre as bases de dados com formas íntegras. Desse modo, pelo uso da heurística de correção, mesmo as amostras falhas podem ser aproveitadas no processo de treino. Assim, conforme proposto, a metodologia baseada na combinação das abordagens supracitadas permite localizar pontos faciais característicos com desempenho satisfatório, mesmo quando os dados disponíveis apresentam falhas.

## Referências

Cao, X., Y. Wei, F. Wen, and J. Sun. 2012. “Face Alignment by Explicit Shape Regression.” In _2012 Ieee Conference on Computer Vision and Pattern Recognition_, 2887–94. <https://doi.org/10.1109/CVPR.2012.6248015>.

Cootes, Timothy F, Gareth J Edwards, and Christopher J Taylor. 2001. “Active Appearance Models.” _IEEE Transactions on Pattern Analysis & Machine Intelligence_, no. 6: 681–85.

Cootes, Timothy F, Andrew Hill, Christopher J Taylor, and Jane Haslam. 1994. “Use of Active Shape Models for Locating Structures in Medical Images.” _Image and Vision Computing_ 12 (6): 355–66.

Dalal, Navneet, and Bill Triggs. 2005. “Histograms of Oriented Gradients for Human Detection.” In _Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on_, 1:886–93. IEEE.

Dollár, Piotr, Peter Welinder, and Pietro Perona. 2010. “Cascaded Pose Regression.” In _Computer Vision and Pattern Recognition (Cvpr), 2010 Ieee Conference on_, 1078–85. IEEE.

Frith, Chris. 2009. “Role of Facial Expressions in Social Interactions.” _Philosophical Transactions of the Royal Society of London B: Biological Sciences_ 364 (1535): 3453–8.

Hill, T, and P Lewicki. 2006. _Statistics: Methods and Applications: A Comprehensive Reference for Science, Industry, and Data Mining_.

Jaimes, Alejandro, and Nicu Sebe. 2007. “Multimodal Human–Computer Interaction: A Survey.” _Computer Vision and Image Understanding_ 108 (1-2): 116–34.

Kazemi, Vahid, and Josephine Sullivan. 2014. “One Millisecond Face Alignment with an Ensemble of Regression Trees.” In _Proceedings of the Ieee Conference on Computer Vision and Pattern Recognition_, 1867–74.

Kendall, David G. 1989. “A Survey of the Statistical Theory of Shape.” _Statistical Science_ 4 (2): 87–99.

Le, Vuong, Jonathan Brandt, Zhe Lin, Lubomir Bourdev, and Thomas S Huang. 2012. “Interactive Facial Feature Localization.” In _European Conference on Computer Vision_, 679–92. Springer.

Zhu, Xiangxin, and Deva Ramanan. 2012. “Face Detection, Pose Estimation, and Landmark Localization in the Wild.” In _Computer Vision and Pattern Recognition (Cvpr), 2012 Ieee Conference on_, 2879–86. IEEE.

## Autores

- Abner Nascimento - [Universidade Federal do Ceará](http://www.ec.ufc.br/).
- Danilo A. Oliveira - [Universidade Federal do Ceará](http://www.ec.ufc.br/).
- Maria Raquel L. de Couto - [Universidade Federal do Ceará](http://www.ec.ufc.br/).
- Iális C. de Paula Júnior - [Universidade Federal do Ceará](http://www.ec.ufc.br/).
