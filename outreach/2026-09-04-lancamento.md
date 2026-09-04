<!-- l10n: doc_id=patchcraft-outreach-2026-09-04 · lang=pt-BR · canonical -->
**Português** · [English](2026-09-04-release.en.md)

# PatchCraft 0.5.4, fonte de notícia

Documento datado que serve de fonte para os textos de canal em [`linkedin/`](linkedin/).
Regra: nenhum texto de canal muda sem esta fonte mudar antes. Nada aqui é estimativa;
cada número tem um comando que o reproduz.

Substitui [`2026-09-03-lancamento.md`](2026-09-03-lancamento.md), que fica como registro
da 0.5.1. **A ordem das manchetes mudou, e essa é a diferença principal entre as duas.**
A fonte anterior abria pela retratação. Quem lê o anúncio nunca viu a afirmação antiga,
então a retratação não descreve nada que a pessoa tenha presenciado: ela só transmite que
a biblioteca errou, antes de dizer para que ela serve. O defeito silencioso vem primeiro
agora, porque é o único item da lista que o leitor pode ter no código dele neste momento.

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

O segundo é o stride que não cobre a imagem. Numa imagem 128 por 128 com patch 32 e stride
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

### 3. A suíte cujo trabalho é derrubar o contrato

Um teste com a função explícita de falsificar a afirmação acima. Ele enumera as 126.736
geometrias legais do espaço **sem consultar o predicado**, para que o enumerador e a coisa
testada sejam independentes, e procura os dois contraexemplos: um caso dentro da regra que
não seja exato, e um caso fora que seja exato por sorte. A varredura completa fica atrás de
`PATCHCRAFT_SWEEP_FULL=1`.

Ele existe nessa forma por um motivo concreto, e aqui é onde a retratação entra, como
procedência e não como manchete: uma versão anterior deste contrato foi publicada, medida,
encontrada falsa e corrigida em público. Dizia que o erro fora da regra ficava em torno de
1 ULP, quando chega a 19 ULP em float32. A regra antiga olhava o máximo do mapa de
cobertura; sobre 14.969 geometrias retangulares ela erra 3.936 casos, e a regra da potência
de dois erra 8, todos na direção segura de prometer menos do que entrega.

E a suíte não pegou o erro porque os testes montavam as imagens com `torch.arange`. Dado
inteiro fecha a ida e volta exato onde dado aleatório não fecha. O teste passava porque
estava fazendo a pergunta errada com muita confiança, e essa é a parte transferível para
qualquer outro projeto.

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

## Estado, dito por inteiro

São 1619 testes passando e 1656 coletados, com CI em {Ubuntu, Windows} x {Python 3.12,
3.13, 3.14} no caminho puro mais um job acelerado nos dois sistemas. A superfície pública
são 20 nomes, congelados por teste com `inspect.signature`.

Nenhum projeto externo consumiu a biblioteca ainda, e esse é o critério que ela própria
escolheu para se dizer estável. Nenhum caminho CUDA dela jamais executou, nem na CI nem
fora dela. Três das cinco wheels aceleradas nunca tiveram o kernel executado em CI: as de
macOS e aarch64 são construídas e conferidas, e só.

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
- O que o projeto se recusa a afirmar: `docs/GUIDE.md` seção 8
