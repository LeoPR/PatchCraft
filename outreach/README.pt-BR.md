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
| [`linkedin/`](linkedin/) | LinkedIn: `post.*` (curto), `artigo.*` (longo técnico), e as figuras |

As figuras ficam em `linkedin/figuras/<língua>/`, uma subpasta por língua para o diretório
do canal não misturar texto com binário. O `tools/make_outreach_figures.py` gera todas, em
PNG para subir e SVG ao lado para editar, e rodar o script regenera tudo. Elas obedecem à
mesma regra dos números do texto: existe um comando que as reproduz.

Nenhuma é ilustração. Os painéis de imagem são os tensores que o `extract` e o `reconstruct`
devolvem de fato, o erro no canto do painel aproximado é a diferença real ampliada até
ficar visível, o texto da recusa é o que o `reconstruct` levanta, e os mapas de cobertura
saem do `fold` e do `unfold` do torch sobre um tensor de uns.

É uma página só por língua, `pagina.png` e `page.png`, em três blocos: o recorte e o
reshape que embaralha, a comparação por passo entre o `fold`/`unfold` escrito à mão e o
PatchCraft, e o caso típico num dígito do MNIST. Onde os dois caminhos dão o mesmo tensor, a
página diz isso em vez de repetir a imagem, porque a diferença ali não é a conta, é o
contrato.

O MNIST é baixado na primeira execução, e a página degrada sem ele: o terceiro bloco some e
o script avisa, em vez de falhar.

Aqui o português é a língua canônica, ao contrário do resto do projeto, porque o público a
que estes textos se dirigem lê português primeiro. O inglês é a tradução.

## Limites de cada canal

- **Post do LinkedIn** (`linkedin/post.*`): cerca de 3.000 caracteres, e só as duas ou três
  primeiras linhas aparecem antes do "ver mais". Essas linhas não podem conter jargão: o
  público do LinkedIn é largo, e uma primeira frase que só fala com quem já conhece o
  assunto filtra em vez de convidar. Contexto antes de jargão, densidade sem tom
  professoral, e um fecho que fecha em vez de parar. Hashtags no fim e sem acento,
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

**Superfície não carrega histórico de desenvolvimento.** Estes textos dizem o que a
biblioteca faz hoje. O caminho até aqui, incluindo a afirmação numérica que foi publicada,
medida e retratada, fica no CHANGELOG, no ADR 0003 e nos estudos datados, que é onde alguém
vai procurar de propósito.

A razão é do leitor, não de coragem: quem chega agora nunca viu a versão antiga. Contar a
correção não descreve nada que a pessoa tenha presenciado, e a única coisa que transmite no
primeiro contato é que a biblioteca errou, antes de ela saber para que a biblioteca serve.
Pior quando o defeito relatado era nos testes, porque aí o leitor termina sem saber nada
sobre o produto e com a dúvida de se a suíte funciona.

O teste disso, para o próximo texto: cada parágrafo tem que responder "o que isso me diz
sobre usar a biblioteca?". Uma medição antiga pode ficar, desde que entre como argumento
para a regra atual e não como relato do que houve. A comparação entre a regra do máximo e a
da potência de dois é o exemplo: ela explica por que o contrato é o que é.

**Não suavize a seção de limites.** Ela é curta, é verdadeira, e é a parte que dá
credibilidade ao resto.
