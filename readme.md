Disciplina de Inteligência Artificial - Professor Munif - Unicesumar 2026

Classificação Automatizada de Soquetes de CPU (AMD vs. Intel)

1. Identificação da Equipe

Augusto Palma Guglielmi - RA: 23021606-2

Fabrício Notoya Thomas - RA: 23082446-2

Guilherme Pitão Garcia - RA: 23000966-2

Pietro Pasqual Silva - RA: 23183509-2

Thiago Lopes Martins - RA: 23016365-2

2. Resumo do Projeto

2.1 Contextualização do Tema Escolhido

A triagem, identificação e destinação de componentes eletrônicos descartados representam um desafio de engenharia essencial para a viabilização de fluxos de logística reversa e reciclagem tecnológica sustentável de lixo eletrônico (e-waste). No processamento de placas-mãe, identificar fisicamente as gerações e fabricantes dos soquetes de CPU (Central Processing Unit) de forma automatizada permite triar o hardware para reaproveitamento ou refino metalúrgico especializado.

2.2 Problema Investigado

O projeto investiga a capacidade de algoritmos de Visão Computacional combinados com Inteligência Artificial para classificar imagens de soquetes de processadores. Inicialmente, o escopo pretendia diferenciar 32 classes específicas de soquetes. Diante de limites de aquisição física de imagens com qualidade consistente, o número de classes viáveis foi reduzido para 11.

Entretanto, o maior desafio residia na alta similaridade geométrica macroscópica dessas 11 classes e na variância espacial (pequenas variações no ângulo em que a foto da placa-mãe foi tirada). Isso submeteu os modelos preditivos ao erro posicional absoluto, limitando severamente a acurácia. A solução exigiu reestruturar o problema para uma classificação binária focada na distinção estrutural entre as duas arquiteturas dominantes no mercado de hardware:

AMD (Padrão físico PGA, caracterizado por uma matriz de furos e uma alavanca metálica lateral de retenção).

Intel (Padrão físico LGA, caracterizado por pinos de contato expostos diretamente no próprio soquete).

2.3 Hipótese da Equipe

Postulou-se que atributos brutos como canais de cor RGB e intensidade de luz global inserem apenas ruído e colinearidade matemática nos modelos. No entanto, aplicando filtros de extração de contornos em escala de cinza (gray_edges 32px) aliados a uma etapa de pré-processamento para alinhamento geométrico por rotação coordenada (garantindo conformidade vetorial às amostras) e uma junção de classes binária (AMD vs. Intel), elimina-se a variância espacial, permitindo que classificadores clássicos de ensemble e redes neurais MLP superem de forma estável o patamar de 90% de acurácia geral.

2.4 Descrição do Dataset Utilizado

A construção e evolução empírica da base de dados passou por quatro fases investigativas essenciais para validar o comportamento dos algoritmos:

Dataset V1.0 (Abordagem Cromática): Base construída com foco em canais de cor RGB e luminosidade global. A auditoria via regras do Apriori demonstrou que as cores variavam unicamente pela iluminação externa no momento da captura (regras com 98% de confiança ligando as variações), tornando o agrupamento ineficaz.

Dataset V2.0 (Bordas e Desalinhamento): Abandono das variáveis de cor pelo uso de detectores de bordas em escala de cinza (gray_edges 32px) sobre 11 classes originais de soquetes. Sem alinhamento, a acurácia dos modelos supervisionados ficou estagnada em apenas 52% de acerto (MLP).

Dataset V3.0 (Alinhamento por Rotação): Pipeline aplicado para rotacionar todas as fotos e alinhá-las no mesmo eixo, estabilizando as matrizes geométricas.

Dataset V3.0 Final (Fusão Binária): Dataset rotacionado com classes fundidas em duas categorias (AMD x Intel), submetido a escalonamento linear e validado por validação cruzada estratificada de 5 partições (5-fold cross-validation).

2.5 Métodos de IA Utilizados

Para cobrir plenamente os requisitos metodológicos solicitados ao longo da disciplina:

Métodos do 1º Bimestre (Machine Learning Clássico):

Não-Supervisionados: K-Means, AGNES e DIANA (usados para demonstrar que o agrupamento não-supervisionado puro sobre dados desalinhados é inviável, apresentando Silhouette próximo a zero).

Supervisionados: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Árvore de Decisão Clássica, Random Forest e AdaBoost.

Métodos do 2º Bimestre (Deep Learning / Avançados):

Rede Neural Multi-Layer Perceptron (MLP) com múltiplas camadas ocultas e ajuste dinâmico por retropropagação (backpropagation).

Extração de Padrões e Regras:

Algoritmo Apriori (auditoria de cores na V1.0) e algoritmo PRISM (geração de regras de decisão lógicas indutivas).

3. Avaliação dos Modelos e Comparação dos Resultados

3.1 A Evidência Matemática do PRISM

O impacto da engenharia de dados (rotação geométrica) foi comprovado pelas regras de indução geradas pelo algoritmo PRISM. No dataset desalinhado, as regras tentavam memorizar pixels estáticos irrelevantes. Após o alinhamento da V3, o PRISM gerou regras simples e altamente generalizáveis para produção:

SE dim0 = 0 e dim4 = 0 ENTÃO classe = AMD
SE dim1 = 0 e dim2 = 0 ENTÃO classe = INTEL


Isso prova que as regiões de relevância física das bordas das travas e pinos passaram a ocupar localizações idênticas e previsíveis nas matrizes de características.

3.2 Tabela de Comparação de Desempenho (Dataset V3.0 Final)

A tabela abaixo consolida os resultados alcançados pelos modelos no dataset final (com rotação e junção de classes):

Algoritmo Evaluated

Configuração / Dataset Usado

Acurácia Geral Obtida

Árvore de Decisão Clássica

V3.0 Final (Binário / Rotacionado)

~ 89%

Rede Neural MLP

V3.0 Final (Binário / Rotacionado)

92%

Support Vector Machine (SVM)

V3.0 Final (Binário / Rotacionado)

93%

Random Forest

V3.0 Final (Binário / Rotacionado)

93%

AdaBoost

V3.0 Final (Binário / Rotacionado)

Maior Acurácia (Top 1)

O Campeão do 1º Bimestre (Método Clássico): AdaBoost — Apresentou a maior performance preditiva, explorando de forma iterativa os contornos geométricos residuais das duas marcas.

O Campeão do 2º Bimestre (Método Avançado): Rede Neural MLP — Atingiu 92% de acurácia, demonstrando excelente capacidade de generalização e conformidade matemática.

3.3 Visualização e Análise das Métricas de Treinamento

Ineficácia dos Métodos Não-Supervisionados (Dataset V1.0)

A dispersão do PCA 2D e o Dendrograma gerados com a base desalinhada evidenciam uma sobreposição severa que inviabilizou o agrupamento direto por distância geométrica. A ausência de limites claros de clusters confirma a necessidade de um pipeline de Visão Computacional estruturado e baseado em alinhamento para que classificadores encontrem as fronteiras de decisão das classes de hardware.

Comparativo de Modelos (Dataset V3.0 Final)

Os resultados no dataset final consolidaram o AdaBoost no topo da precisão geral. Algoritmos robustos como o SVM, Random Forest e a Rede Neural MLP empataram estavelmente com 93% e 92% de acurácia respectivamente, demonstrando que as melhorias metodológicas no alinhamento das matrizes beneficiaram indistintamente todas as arquiteturas avaliadas.

Matriz de Confusão do Modelo Final

A análise de distribuição de erros do modelo AdaBoost aponta para uma diagonal principal com forte concentração de acertos, mantendo taxas de falsos positivos e falsos negativos residuais insignificantes. Isso demonstra que o sistema possui alto poder discriminativo mesmo em condições de validação cruzada rigorosa.

Curva de Perda (Loss Curve) da Rede Neural MLP

O comportamento do erro exponencial decrescente estabiliza-se de forma contínua abaixo da marca de 100 épocas de treinamento. A suavidade da descida atesta que a rede neural conseguiu otimizar seus pesos sem oscilações bruscas e sem sinais de overfitting (sobreajuste), confirmando a excelente preparação das variáveis preditivas.

4. Como Executar e Reproduzir o Treinamento

Toda a lógica de carregamento de dados, execução de pipelines e visualização de resultados de treinamento é controlada de forma interativa por meio de uma interface acessível via navegador web.

4.1 Passo 1: Instalar Dependências e Iniciar o Servidor Web

No terminal, navegue até a pasta raiz do projeto e instale as bibliotecas necessárias antes de inicializar o servidor de aplicação:

pip install -r requirements.txt
python app.py


4.2 Passo 2: Acessar a Interface Gráfica

Após inicializar o servidor, o terminal exibirá a URL de escuta local. Abra o seu navegador de preferência e acesse o endereço padrão exibido no terminal.

4.3 Passo 3: Reproduzir os Testes e Visualizar os Resultados

Dentro da interface web de controle, realize as seguintes ações para auditar o comportamento da inteligência artificial:

Seleção de Datasets: Selecione entre as versões da base de dados (sendo processadores a última versão refinada com os ajustes de rotação e junção) para analisar a evolução de acurácia.

Executar Treinamento: Clique no botão de treinamento interativo para que o servidor processe os algoritmos de classificação das duas partes em tempo real sobre a base escolhida.

Selecione Grupos: Para maior eficácia e reproduzir o resultado final, deve ser feito o agrupamento de sockets da amd VS sockets da intel.

Auditoria de Gráficos: A interface renderizará e atualizará dinamicamente na tela os gráficos de comparação de acurácia de modelos, as matrizes de confusão e o comportamento de perda (loss) da rede neural.
