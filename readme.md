Classificação Automatizada de Soquetes de CPU (AMD vs. Intel)

Disciplina de Inteligência Artificial — Professor Munif — Unicesumar 2026

Integrantes da Equipe

Augusto Palma Guglielmi — RA: 23021606-2

Fabrício Notoya Thomas — RA: 23082446-2

Guilherme Pitão Garcia — RA: 23000966-2

Pietro Pasqual Silva — RA: 23183509-2

Thiago Lopes Martins — RA: 23016365-2

Como Executar e Reproduzir o Treinamento

Toda a lógica de carregamento de dados, execução de pipelines e visualização de resultados de treinamento é controlada diretamente de forma interativa por meio de uma interface acessível via navegador web.

Passo 1: Iniciar o Servidor Web

No terminal, navegue até a pasta do projeto e inicialize o servidor executando o comando:

pip install -r requirements.txt
python app.py


Passo 2: Acessar a Interface Gráfica

Após inicializar o servidor, o terminal exibirá a URL local para acesso. Abra o seu navegador web de preferência e acesse o endereço padrão exibidos no terminal.

Passo 3: Reproduzir os Testes e Visualizar Resultados

Na interface web, você poderá realizar todo o processo exigido para auditoria de maneira visual e reproduzível:

Seleção de Datasets: Escolha entre as diferentes versões da base de dados (processadores é a última versão com ajuste) para comparar diretamente as mudanças no desempenho histórico descritas no relatório.

Executar Treinamento: Clique no botão de treinamento para iniciar os algoritmos das Partes 1 e 2 em tempo real sobre a base escolhida.

Auditoria de Gráficos: A interface carregará e atualizará dinamicamente na tela os gráficos de comparação de acurácia de modelos, as matrizes de confusão detalhadas e o comportamento da curva de perda da rede neural MLP.
