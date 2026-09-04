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
| [`2026-09-04-lancamento.md`](2026-09-04-lancamento.md) / [`2026-09-04-release.en.md`](2026-09-04-release.en.md) | a fonte de notícia atual (PT / EN) |
| `2026-09-03-*.md` | a fonte anterior, registro da 0.5.1; datada, não se reescreve |
| [`linkedin/`](linkedin/) | LinkedIn: `post.*` (curto), `artigo.*` (longo técnico) |

Aqui o português é a língua canônica, ao contrário do resto do projeto, porque o público a
que estes textos se dirigem lê português primeiro. O inglês é a tradução.

## Limites de cada canal

- **Post do LinkedIn** (`linkedin/post.*`): cerca de 3.000 caracteres, e só as duas ou três
  primeiras linhas aparecem antes do "ver mais". Essas linhas não podem conter jargão: o
  público do LinkedIn é largo, e uma primeira frase que só fala com quem já conhece o
  assunto filtra em vez de convidar. O texto tem uma ideia que o atravessa do começo ao
  fim, e cada parágrafo a avança: aqui, a falha silenciosa, da qual os dois defeitos são
  instâncias e a retratação é a mesma falha uma camada acima. Contexto antes de jargão,
  densidade sem tom professoral, e um fecho que volta à abertura. Hashtags no fim e sem acento,
  porque hashtag acentuada quebra a busca do LinkedIn.
- **Artigo do LinkedIn** (`linkedin/artigo.*`): formato longo, com títulos e tabelas
  renderizando, bom para a versão que carrega os números. Termina com o link do repositório.

## Antes de publicar

**Todo número foi verificado**, e não estimado, e reconferido em 2026-09-04: os 3840 pixels
deixados em zero, o reshape que preserva a forma e perde os pixels, as 126.736 geometrias
enumeradas, os 3936 erros em 14969 da regra antiga contra 8 da nova, e a tabela inteira do
benchmark. As medições estão em `docs/PERFORMANCE.md` e nas entradas `0.5.0` e `0.5.1` do
changelog.

**O que estes textos evitam de propósito:**

- comparar com outras bibliotecas. A linha de base de toda tabela é o próprio caminho em
  torch puro do PatchCraft, e dizer isso mantém a conversa sobre a medição em vez de sobre
  se a comparação foi justa;
- dizer "mais rápido" sem uma geometria, uma máquina e uma versão de torch junto;
- superlativo. O gancho é o defeito silencioso, não a vantagem e não a retratação.

**Onde a retratação fica, e por que ela mudou de lugar.** Até 2026-09-04 estes textos
abriam pelo fato de a biblioteca ter publicado uma afirmação falsa sobre a própria numérica.
Era o lugar errado, por um motivo que não tem nada a ver com coragem: o leitor nunca viu a
afirmação antiga. Ele não tem o antes. Então a retratação descreve um estado que ele não
presenciou, e a única coisa que ela de fato transmite no primeiro contato é que a biblioteca
errou, entregue antes de o leitor saber para que a biblioteca serve.

Ela agora fica dentro da seção da suíte de falsificação, onde ganha o lugar como resposta a
"por que esse teste tem essa forma". O material não ficou mais macio e nenhum número saiu;
ele deixou de ser manchete. Quem abre é o defeito silencioso, que é o único item destes
textos que o leitor pode ter no código dele no momento em que lê.

**Não suavize a seção de limites.** Ela é curta, é verdadeira, e é a parte que dá
credibilidade ao resto.
