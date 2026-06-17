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
