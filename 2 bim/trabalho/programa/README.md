# Curador de Imagens

App local pra montar **datasets de imagens de qualquer coisa**. Você cria categorias com
sua própria **keyword de busca**, baixa imagens (DuckDuckGo / Google / Google API), curatela,
recorta no que importa, gera variações (data augmentation) e exporta em ZIP.

## Rodar
```bash
pip install -r requirements.txt
python app.py
```
Abra **http://127.0.0.1:5000**.

## Categorias (keyword editável)
- **+ Categoria:** nome (vira a pasta), **keyword/busca** e **meta** de processadas.
- **✎** edita a keyword e a meta de uma categoria; **×** remove da lista.
- Cada categoria tem sua própria busca — ex.: `gato siamês`, `placa mãe am4`, `carro vermelho`.

## Buscadores
- **DuckDuckGo** (recomendado): agrega DDG + Bing, pagina, filtra ≥1000×1000 e relevância
  pelas palavras da sua keyword.
- **Google (scrape):** sem API key, best-effort (pouco volume / pode bloquear).
- **Google API:** Custom Search JSON API, confiável. Configure **API key + cx** em **⚙ Ajustes**.
  (Crie a key no Google Cloud e um Search Engine ID com busca de imagens ativada.)

## Curadoria
- Arraste o mouse no fundo da grade (caixa de seleção) ou use o quadradinho do card; `Del` exclui.
  (`Ctrl`+`A` = tudo, `Esc` = limpa.)
- **Clique numa imagem** → abre o visualizador. **Arraste sobre a imagem** p/ marcar a região →
  **Processar** (`Enter`) recorta, corrige orientação e salva em `processed/<cat>/`.
  Sem marcar = imagem inteira. Setas navegam, `Del` exclui, `Esc` fecha.

## Aumento de dados (gerar imagens novas)
No painel direito marque **Espelhar / Brilho / Hue** (com sliders). Essas opções:
- são aplicadas junto ao **Processar** (geram variações da imagem recortada), e/ou
- via **“Gerar variações nas processadas”** (aplica em todas as já processadas de uma vez).
Variações ganham sufixo `__m` (espelho), `__b` (brilho), `__h` (hue).

## Meta + aviso
Defina a **meta** de processadas por categoria. Ao atingir, aparece um aviso com duas opções:
**Continuar curando** ou **Interromper e limpar a galeria** (apaga os candidatos restantes;
o cache evita rebaixá-los depois).

## Exportar (ZIP) com limite
No painel direito: escolha **escopo** (categoria atual / todas) e **máx. por categoria**
(0 = sem limite). Ex.: achou só 10 boas? Coloque 10. Gera `dataset_*.zip` com uma pasta por categoria.

## Pastas
```
candidates/<cat>/   # baixadas, aguardando curadoria
processed/<cat>/    # recortes + variações (o dataset) — .jpg
cache/<cat>.json    # hashes já vistos (não rebaixa)
categories.json     # nome, keyword e meta de cada categoria
settings.json       # Google API key + cx
```
Nada sai da sua máquina (download direto dos hosts; Google API fala só com a API do Google).

## Novidades desta versão
- **Ctrl+Z** desfaz a última exclusão (vai pra lixeira interna e volta pros candidatos).
- **Exportar categorias específicas:** no escopo, escolha “Selecionadas” e clique em **Escolher…**
  para marcar quais categorias entram no ZIP.
- **🧠 Treinamento de IA** (usa as imagens de `processed/`, categoria = rótulo):
  - **Parte 1 – Supervisionado:** Árvore de Decisão, KNN, Naive Bayes, SVM, Rede Neural (MLP),
    Random Forest, AdaBoost e PRISM (ilustrativo). Métricas: **Acurácia, Precisão, Matriz de Confusão**
    (validação cruzada). Gráficos: comparação de modelos, matriz de confusão e curva de perda (MLP).
  - **Parte 2 – Não supervisionado:** K-Means, AGNES (aglomerativo/Ward) e DIANA (divisivo, proxy).
    Métricas: **Silhouette** e **Davies-Bouldin**. Gráficos: **dendrograma** e dispersão PCA 2D.
  - **Regras de associação:** Apriori sobre features discretizadas (ilustrativo).

> Observação: PRISM, DIANA e Apriori são versões ilustrativas (PRISM/DIANA não têm
> implementação canônica no scikit-learn; Apriori sobre pixels é demonstrativo). Os demais
> usam scikit-learn de verdade. Treinar exige `pip install -r requirements.txt` (agora inclui
> scikit-learn, numpy, scipy e matplotlib).

## v3 — Projetos, templates e correções
- **Correção do crash de treino** (`Failed to fetch / ERR_CONNECTION_RESET`): era o OpenMP
  abortando o processo ao carregar o scikit-learn. Resolvido setando `KMP_DUPLICATE_LIB_OK`
  e limitando threads antes dos imports, e rodando o servidor com `threaded=True` sem reloader.
- **Projetos (pasta principal + subpasta por categoria):** tudo agora vive em
  `projects/<projeto>/{candidates,processed,cache,_trash}/<categoria>/`. Há um seletor de projeto
  no topo, com **+ Projeto**, editar (renomeia a pasta) e remover.
- **Template de busca com `$cat$`:** cada projeto tem um template (ex.: `motherboard $cat$`).
  Qualquer texto entre `$...$` (`$cat$`, `$socket_name$`, etc.) é trocado pelo nome da categoria —
  então você não precisa digitar a query de cada categoria. Dá pra sobrescrever por categoria.
- **CRUD completo:** renomeia projeto, renomeia categoria (move as pastas), edita template,
  keyword e meta.
- **Auto-detecção:** pastas criadas manualmente dentro de `processed/` ou `candidates/` viram
  categorias automaticamente.
- **Treino sem viés:** “Máx. imagens/categoria” e “Balancear classes” (iguala a quantidade entre
  categorias) no painel de treino.
- **Dispositivo (CPU/CUDA/ROCm) com fallback:** seletor de dispositivo; detecta CUDA/ROCm via torch
  se instalado. Os modelos clássicos do scikit-learn rodam em CPU — se você escolher GPU sem ela
  disponível, faz fallback automático pra CPU e avisa.

### Atualize as dependências
```bash
pip install -r requirements.txt
python app.py
```
> Se você já tinha a versão antiga, as pastas `candidates/` e `processed/` na raiz não são mais
> usadas — agora tudo fica em `projects/`. Crie um projeto e mova suas imagens para
> `projects/<seu-projeto>/processed/<categoria>/` (a auto-detecção reconhece).

## v4 — Exportar resultados + métodos extensíveis
- **Exportar resultados:** no painel de treino há **⬇ Exportar resultados**, que baixa um
  **relatório HTML auto-contido** com os parâmetros usados (categorias, máx/cat, balanceamento,
  dispositivo), as tabelas de métricas (acurácia/precisão, silhouette/Davies-Bouldin, itemsets/regras)
  e **todos os gráficos embutidos** (comparação, matriz de confusão, curva de perda, dendrograma,
  dispersão). Abre em qualquer navegador e pode virar PDF via “Imprimir”.
- **Adicionar mais funções (técnicas) facilmente:** os métodos agora ficam num **registro central**
  no `trainer.py`. Para incluir uma nova técnica, basta UMA linha:
  ```python
  # supervisionado
  SUPERVISED["gb"] = ("Gradient Boosting", lambda min_c: GradientBoostingClassifier(random_state=0))
  # agrupamento
  CLUSTERING["spectral"] = ("Spectral", lambda k: SpectralClustering(n_clusters=k))
  ```
  Ela aparece **automaticamente** na interface (os checkboxes são carregados de `/api/train/methods`).
  Supervisionado precisa de `.fit/.predict`; agrupamento precisa de `.fit_predict`.

## v5 — Feedback de carregamento + treino mais profundo
- **Feedback visual no treino:** o painel de treino abre **na hora** (com spinners "carregando…")
  e cada execução mostra um **spinner + status** e desabilita o botão enquanto roda — não trava mais
  esperando todos os requests.
- **Representações de features:** RGB, tons de cinza, **bordas (Sobel)**, **cinza+bordas** (recomendado
  p/ peças) e **HOG** (usa scikit-image se instalado) — escolhe no dropdown “Features”.
- **Tamanho da imagem** (16–48 px), **escalonamento** (StandardScaler) e **nº de folds** configuráveis.
- **Validação à prova de vazamento:** se houver variações de augmentation (`__m/__b/__h`), elas ficam
  no **mesmo fold** (StratifiedGroupKFold) — evita métricas infladas por ter a base no treino e a
  variação no teste.
- **Diagnósticos:** mostra a **linha de base do acaso** (1/nº de classes) p/ contextualizar a acurácia,
  e no não-supervisionado compara os clusters com as classes reais via **ARI** e **NMI**.

## v6 — Modo revisão
- **🔁 Revisar:** botão na barra da galeria. Entra no modo revisão e mostra as imagens **já
  processadas** da categoria; clique numa, marque um **trecho mais específico** (ex.: a área dos
  pinos/encaixes) e **Processar** salva o recorte mais fino no lugar.
- A **versão anterior é arquivada** automaticamente em `categoria_old` (e `categoria_old2`,
  `categoria_old3`… se você revisar o mesmo arquivo de novo). Essas viram categorias próprias
  (auto-detectadas) — é só não selecioná-las no treino, ou usá-las p/ comparar.
- “Sair da revisão” recarrega o projeto p/ exibir as novas categorias `_old`.

## v7 — Rotação, fusão de classes e sugestões
- **Rotação no visualizador:** botões ⟲ ⟳ giram a imagem 90° (candidata ou, na revisão, a já
  processada). Use p/ **padronizar a orientação** antes de recortar (essencial p/ os entalhes de
  chaveamento ficarem sempre no mesmo lugar).
- **Fundir classes no treino:** no painel de treino, marque categorias, dê um nome ao grupo e clique
  em **Fundir** — elas viram **uma classe só** durante o treino, sem mexer nas pastas. Útil p/ unir
  sockets quase idênticos (ex.: LGA1150/1155/1156 → “LGA115x”).
- **Sugestões de fusão:** após o treino supervisionado, o app mostra os **pares de classes mais
  confundidos** entre si (pela matriz de confusão). Clique numa sugestão p/ fundir aquele par
  automaticamente e rode o treino de novo.

## v8 — Auto-detecção por projeto
- **Auto-detecção de projetos:** qualquer pasta criada dentro de `projects/` é reconhecida como um
  projeto (pastas ocultas, começando com `.`, são ignoradas).
- **Conteúdo sincronizado com o disco:** ao abrir um projeto, o `project.json` é sincronizado —
  pastas novas em `processed/`/`candidates/` viram categorias, e categorias cujas pastas foram
  apagadas por fora deixam de aparecer.
- **Atualização ao vivo:** botão **🔄** ao lado do seletor de projeto reescaneia projetos e pastas
  na hora; além disso a lista é reescaneada sozinha quando você volta o foco para a janela
  (ex.: depois de criar/mover pastas no explorador de arquivos).

## v9 — Busca automática de agrupamento
- **🔎 Testar agrupamentos:** botão no painel de treino (seção “Fundir classes”). Funde
  gulosamente as classes **mais confundidas entre si** (pela matriz de confusão) e mostra a
  trajetória: a cada junção, nº de classes, acurácia e **kappa de Cohen** (concordância acima do
  acaso — não premia fundir tudo à toa).
- **Melhor com o mínimo de junções:** recomenda (⭐) o agrupamento com **menor número de fusões**
  que já chega perto do melhor kappa (joelho da curva). Cada linha tem “aplicar” — clique e rode o
  treino supervisionado normal.
- **Balanceamento e máximo de imagens PÓS-junção:** ao testar, o limite por classe e o
  balanceamento são aplicados **depois** de cada fusão (no rótulo já agrupado), evitando viés.
- **Faixa de folds:** a avaliação roda a validação cruzada para cada nº de folds no intervalo
  escolhido (ex.: 4 a 6) e usa a **média**, deixando a comparação mais estável.

## v10 — Significância e benchmark com vários modelos
- **Estatística de significância na busca de agrupamento:** cada agrupamento mostra
  - **Acaso** (1/nº de classes) — para ver se a acurácia alta é só efeito de ter menos classes;
  - **Kappa de Cohen** com **média±desvio** entre os folds da faixa;
  - **p-valor** (teste binomial) de o melhor modelo **superar o acaso** (verde = significativo, “n.s.” = não);
  - **Δκ de cada junção** com marca **✓** quando o ganho está **acima do ruído** entre folds
    (vale a pena) ou **~** quando está dentro do ruído (provavelmente só o efeito de menos classes).
  - Recomendação ⭐ = menos junções que chega perto do melhor kappa **e** supera o acaso.
- **Benchmark com vários modelos:** a busca usa todos os **modelos marcados na Parte 1**
  (botão **todos/nenhum**), avaliando cada agrupamento × cada modelo × cada fold da faixa, e mostra
  o detalhamento por modelo no agrupamento recomendado. Útil p/ deixar rodando um teste pesado.

## v11 — Salvar resultados, melhor K, e revisão com original + rotação livre
- **Salvar/reabrir resultados (JSON):** no painel de treino, **💾 Salvar resultados** grava tudo
  (supervisionado, não supervisionado, regras e a busca de agrupamento) em
  `projects/<projeto>/results/<nome>.json`. O seletor **Resultados salvos → Reabrir** recarrega e
  **reanalisa** (re-renderiza tabelas/gráficos, dá p/ exportar o relatório HTML de novo).
- **Estatística individual por modelo no multi-teste:** na busca de agrupamento, cada linha tem
  **“modelos”**, que abre as métricas (acur., kappa, **p-valor próprio**) de **cada modelo** naquele
  caso específico — não só do melhor.
- **Melhor K (Parte 2):** botão **Calcular melhor K** testa K num intervalo e escolhe o de maior
  **silhouette**, com tabela + curva; **usar K** joga o valor no campo de clusters.
- **Imagem original + rotação livre na revisão:** ao processar, a imagem **inteira original** é
  guardada (indexada pelo nome). No modo revisão dá p/ marcar **“usar original (c/ margem)”** e
  **girar em qualquer ângulo** (não só 90°) com um slider; assim, se o recorte ficou apertado, a
  rotação não corta conteúdo (a original tem margem). O recorte/rotação vira a nova processada e a
  anterior vai p/ `categoria_old`.

## v12 — Rotação dinâmica, disco e 2 surpresas
- **Rotação dinâmica:** no modo revisão a imagem gira **ao vivo** enquanto você mexe no slider —
  o preview de alta fidelidade (com expansão, sem cortar conteúdo) é carregado ao soltar.
- **Disco de rotação:** um **disco arrastável em círculo** ao lado do slider; arraste a bolinha
  para girar a imagem como num botão giratório.
- 🎁 **Surpresa 1 — Detector de duplicatas:** botão **🔍 Duplicatas** na galeria acha imagens
  iguais/quase iguais (hash perceptual dHash) e **marca as redundantes** p/ você excluir num clique.
- 🎁 **Surpresa 2 — Mapa do dataset (PCA):** botão **🗺️ Mapa do dataset** no treino plota todas as
  imagens processadas em 2D, coloridas por categoria — você **vê** quais categorias se sobrepõem
  (logo, são difíceis de separar) e quais ficam isoladas.

## v13 — Busca avançada, presets, curva de aprendizado e mais
- **Busca avançada (⚙️ ao lado de Buscar):** filtro de resolução configurável em 3 modos —
  **flexível** (lado menor × lado maior, orientação livre: ex. 500×1000 onde o 500 pode ser X ou Y),
  **largura×altura fixo** (com mín/máx) ou **sem filtro**; **buscas OR** (queries alternativas, uma
  por linha); e **relevância** ligável/desligável.
- **Relatório de busca:** depois de baixar, mostra exatamente quantas imagens foram encontradas e
  **por que cada uma foi descartada** (relevância, resolução por metadados, falha no download,
  resolução após baixar, duplicadas) — responde "por que pedi 60 e vieram 20".
- **Rotação rápida + inverter:** botões de **+15° / +45° / +90°** e **inverter H/V** no modo revisão.
- **Curva de aprendizado:** acurácia × quantidade de imagens, com dica se mais dados ajudariam.
- **Classes efetivas (acur×k):** coluna na busca de agrupamento que cresce linearmente com o nº de
  classes — assim 80% com 6 classes não fica "pior" que 95% com 2 só por ter mais classes.
- **Relevância/pesos das categorias:** tabela tipo planilha (ordenada por peso) que vira
  `class_weight` no treino (DT/SVM/RF) — dá mais importância às classes que você escolher.
- **Presets de configuração:** salve/aplique conjuntos de ajustes (classes marcadas, features,
  modelos, fusões, pesos…) por nome.
- **Linha do tempo:** lista os resultados salvos por data com a melhor métrica de cada um.
