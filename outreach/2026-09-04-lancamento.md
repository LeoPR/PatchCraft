<!-- l10n: doc_id=patchcraft-outreach-2026-09-04 · lang=pt-BR · canonical -->
**Português** · [English](2026-09-04-release.en.md)

# PatchCraft 0.5.4, fonte de notícia

Documento datado que serve de fonte para os textos de canal em [`linkedin/`](linkedin/).
Regra: nenhum texto de canal muda sem esta fonte mudar antes. Nada aqui é estimativa;
cada número tem um comando que o reproduz.

Substitui [`2026-09-03-lancamento.md`](2026-09-03-lancamento.md), que fica como registro
da 0.5.1 e não se reescreve. **A diferença entre as duas é que esta só traz o que a
biblioteca faz hoje.** O histórico de desenvolvimento saiu, e continua onde pertence: no
CHANGELOG, no ADR 0003 e nos estudos datados.

## O que a biblioteca é

Recorta uma imagem em patches e remonta. Uma imagem por chamada, um tensor entra e um
tensor sai. Não há eixo de lote, nem dataset, nem dataloader, nem modelo, e essa fronteira
está escrita como vinculante em `docs/THEORY.md` §0.

`pip install patchcraft` · MIT · pré-1.0 · Python 3.12 a 3.14 · torch 2.6 ou mais novo.

## As manchetes, na ordem em que interessam a quem não conhece o projeto

### 1. Os dois defeitos que não avisam

São a razão de a biblioteca existir, e o leitor pode estar com um dos dois agora.

O primeiro é o reshape. O `F.unfold` devolve `(1, C*ph*pw, L)`, e o reshape intuitivo para
`(L, C, ph, pw)` entrega a forma certa com os pixels errados. O `assert` de shape passa, o
treino roda, a perda desce um pouco menos, e nada reclama.

O segundo é o stride, a distância de um patch para o próximo, quando ele não cobre a imagem. Numa imagem 128 por 128 com patch 32 e stride
20, a grade para no pixel 112 e deixa 3840 dos 16384 pixels em zero. Um `fold` escrito à
mão devolve essa imagem parcialmente preta sem erro.

Nenhum dos dois é difícil de corrigir. Os dois são fáceis de não perceber, e é essa
diferença que justifica escrever uma vez com teste em volta.

### 2. Um contrato numérico que o chamador consegue avaliar antes de chamar

A ida e volta é exata bit a bit **se e somente se todo valor do mapa de cobertura for
potência de dois**. Fora disso o erro por pixel é limitado por `(k+1)·eps·|v|`, com `k` a
contagem de cobertura daquele pixel.

A razão é uma só: dividir um float por uma potência de dois é a única divisão que nunca
arredonda. `stride == patch_size` sempre satisfaz, porque toda contagem vale 1.

O que torna isso um contrato e não uma promessa é a segunda metade: a condição se calcula
a partir da geometria, sem rodar nada. Quem vai chamar sabe de antemão em que regime está.

### 3. Como o contrato é verificado

Um teste com a função explícita de falsificar a afirmação acima. Ele enumera as 126.736
geometrias legais do espaço **sem consultar o predicado**, para que o enumerador e a coisa
testada sejam independentes, e procura os dois contraexemplos: um caso dentro da regra que
não seja exato, e um caso fora que seja exato por sorte. A varredura completa fica atrás de
`PATCHCRAFT_SWEEP_FULL=1`.

**Por que a regra é essa e não uma mais frouxa.** A alternativa óbvia seria só manter a
sobreposição máxima pequena. Sobre 14.969 geometrias retangulares, a regra do máximo erra
3.936 casos; a da potência de dois erra 8, e os 8 erram prometendo menos do que entregam.
Um contrato pode prometer de menos, e não pode prometer demais, porque fora da regra o erro
cresce com a cobertura e chega a 19 ULP em float32 sem nada sinalizar.

**Um detalhe do gerador de dados**, transferível para quem testar patches em qualquer
lugar: as imagens são ruído de mantissa cheia sorteado direto no dtype alvo, nunca uma
rampa inteira. Dado inteiro, num float, fecha a ida e volta exato em geometrias onde dado
aleatório não fecha, então uma suíte construída com `torch.arange` passa sem verificar
nada.

### 4. O acelerador nativo, e por que o ganho não é contagem de núcleos

Cinco das seis wheels trazem um kernel Rust para o fold com sobreposição, compilado dentro
da própria wheel. Não existe extra para habilitar.

| Geometria | Chamada | Torch puro | Acelerado | Ganho |
|---|---|---|---|---|
| 3x512x512, patch 32, stride 16 | `reconstruct` | 16,3 ms | 2,3 ms | 7,1x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 453,7 ms | 32,1 ms | 14,1x |
| 3x2048x2048, patch 64, stride 32 | `stitch` hann | 460,9 ms | 37,9 ms | 12,2x |

A objeção óbvia é que o acelerador só usa mais núcleos. Ela foi medida: forçando o torch a
4, 8, 16 e 36 threads no caso maior, o caminho puro fica entre 365 ms e 465 ms, e com 36
threads não é melhor do que com 8. O `F.fold` não escala com lote 1, que é a razão de o
kernel existir.

O benchmark compara os dois caminhos com `torch.equal` antes de reportar qualquer tempo, e
sai com erro se eles diferirem.

## Onde isto se aplica hoje

São 1619 testes passando e 1656 coletados, com CI em {Ubuntu, Windows} x {Python 3.12,
3.13, 3.14} no caminho puro mais um job acelerado nos dois sistemas. A superfície pública
são 20 nomes, congelados por teste com `inspect.signature`.

**Em CPU.** O acelerador só aceita tensor na CPU e devolve o resto ao torch, então numa GPU
a biblioteca funciona e não acelera. Esse caminho nunca executou, então em CUDA vale tratar
tudo como o caminho em torch puro: correto, e sem o ganho da tabela.

**Em Linux e Windows x86-64** o kernel roda na CI a cada commit, e é onde os números foram
medidos. As wheels de macOS e aarch64 carregam o kernel construído e conferido, mas ele
nunca foi executado em CI, então nessas vale medir contra os próprios dados.

**Antes da 1.0**, e sem consumidor externo, que é o critério que o projeto escolheu para se
dizer estável: o dígito do meio ainda pode mexer no que sai, e cada mudança entra no
changelog com a medição atrás.

## Reprodução

```
pip install patchcraft
python tools/benchmark.py --markdown
PATCHCRAFT_SWEEP_FULL=1 pytest tests/test_exactness.py
```

## Ligações

- Repositório: https://github.com/LeoPR/PatchCraft
- Desempenho, com máquina, versões e data: `docs/PERFORMANCE.md`
- O contrato de exatidão: `docs/ADR/0003-reversibility-classes.md`, aceito em 2026-09-03
- Os limites, com as medições: `docs/GUIDE.md` seção 8
