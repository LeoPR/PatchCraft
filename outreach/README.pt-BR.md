<!-- l10n: doc_id=patchcraft-outreach-readme · lang=pt-BR · source_lang=en · translation_of=README.md -->
[English](README.md) · **Português**

# `outreach/`: material para apresentar o projeto

Peças para mostrar o PatchCraft publicamente. É material de apoio, não faz parte da
biblioteca nem da documentação dela, que fica em [`docs/`](../docs/). Não publica medição
nova: todo número vem de um documento datado do repositório, e cada um nomeia o comando que
o reproduz.

## Como está organizado

A **raiz** guarda a **fonte de notícia** datada: um arquivo por anúncio, com o estado, as
manchetes e os limites ditos por inteiro. As **subpastas** são os **canais**, e cada uma dá
àquela fonte o formato que o meio aceita.

A regra que mantém os dois alinhados: nenhum texto de canal muda sem a fonte datada mudar
antes.

| Caminho | O que é |
|---|---|
| [`2026-09-03-lancamento.md`](2026-09-03-lancamento.md) / [`2026-09-03-release.en.md`](2026-09-03-release.en.md) | a fonte de notícia atual (PT / EN) |
| [`linkedin/`](linkedin/) | LinkedIn: `post.*` (curto), `artigo.*` (longo técnico) |

Aqui o português é a língua canônica, ao contrário do resto do projeto, porque o público a
que estes textos se dirigem lê português primeiro. O inglês é a tradução.

## Limites de cada canal

- **Post do LinkedIn** (`linkedin/post.*`): cerca de 3.000 caracteres, e só as duas ou três
  primeiras linhas aparecem antes do "ver mais", então o gancho vem primeiro. Hashtags no
  fim e sem acento, porque hashtag acentuada quebra a busca do LinkedIn.
- **Artigo do LinkedIn** (`linkedin/artigo.*`): formato longo, com títulos e tabelas
  renderizando, bom para a versão que carrega os números. Termina com o link do repositório.

## Antes de publicar

**Todo número foi verificado**, e não estimado, e reconferido em 2026-09-03: os 3840 pixels
deixados em zero, o reshape que preserva a forma e perde os pixels, as 126.736 geometrias
enumeradas, os 3936 erros em 14969 da regra antiga contra 8 da nova, e a tabela inteira do
benchmark. As medições estão em `docs/PERFORMANCE.md` e nas entradas `0.5.0` e `0.5.1` do
changelog.

**O que estes textos evitam de propósito:**

- comparar com outras bibliotecas. A linha de base de toda tabela é o próprio caminho em
  torch puro do PatchCraft, e dizer isso mantém a conversa sobre a medição em vez de sobre
  se a comparação foi justa;
- dizer "mais rápido" sem uma geometria, uma máquina e uma versão de torch junto;
- superlativo. O gancho é a retratação, não a vantagem.

**O risco que vale conhecer.** Estes textos abrem com o fato de que a biblioteca publicou
uma afirmação falsa sobre a própria numérica. É a coisa mais interessante deles e também a
mais fácil de ler como fraqueza. Eu acho que joga a favor do projeto, porque quem trabalha
com numérica reconhece o que significa um autor medir uma afirmação, descobrir que estava
errada e publicar a correção com a varredura por trás. Mas a escolha é sua, e os dois
textos ficam de pé sem isso: corte a seção da retratação e a peça vira um "aqui está uma
biblioteca pequena" comum, que é um texto mais fraco e mais seguro.

**Não suavize a seção de limites.** Ela é curta, é verdadeira, e é a parte que dá
credibilidade ao resto.
