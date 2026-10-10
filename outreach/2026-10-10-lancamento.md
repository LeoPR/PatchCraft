<!-- l10n: doc_id=patchcraft-outreach-2026-10-10 · lang=pt-BR · canonical -->
**Português** · [English](2026-10-10-release.en.md)

# PatchCraft 0.5.5, fonte de notícia

Documento datado que serve de fonte para os textos de canal em [`linkedin/`](linkedin/).
Regra: nenhum texto de canal muda sem esta fonte mudar antes. Nada aqui é estimativa;
cada número tem um comando que o reproduz.

Substitui [`2026-09-04-lancamento.md`](2026-09-04-lancamento.md), que fica como registro
da 0.5.4 e não se reescreve. **A diferença entre as duas é a tabela do acelerador**, medida
de novo no código da 0.5.5, e a seção de limites, que diz onde cada caminho foi executado.

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

Numa geometria legal isso quer dizer que toda contagem é 1, 2 ou 4, e nesses casos as duas
contas são exatas: somar duas ou quatro cópias de um valor não arredonda, e dividir por 2 ou
por 4 também não. `stride == patch_size` sempre satisfaz, porque toda contagem vale 1.

O que torna isso um contrato e não uma promessa é a segunda metade: a condição se calcula
a partir da geometria, sem rodar nada. Quem vai chamar sabe de antemão em que regime está.

### 3. Como o contrato é verificado

Um teste com a função explícita de falsificar a afirmação acima. Ele enumera as 126.736
geometrias legais do espaço **sem consultar o predicado**, para que o enumerador e a coisa
testada sejam independentes, e procura os dois contraexemplos: um caso dentro da regra que
não seja exato, e um caso fora que seja exato por sorte. A varredura completa fica atrás de
`PATCHCRAFT_SWEEP_FULL=1`.

**Por que a regra é essa e não uma mais frouxa.** A alternativa óbvia seria só manter a
contagem máxima de sobreposição em até 4. Sobre as 126.736 geometrias legais que a suíte
enumera, ela promete exatidão em 13.870 que não voltam exatas; a da potência de dois não
promete nenhuma que falhe nem deixa de prometer nenhuma que volte exata
(`python tools/compare_exactness_rules.py`). Um contrato pode prometer de menos, e não pode
prometer demais, porque fora da regra o erro cresce com a cobertura sem nada sinalizar: na
pior geometria medida, num pixel coberto 81 vezes, chega a 19 ULP em float32. O mesmo
comando mede esse pior caso.

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
| 3x512x512, patch 32, stride 16 | `reconstruct` | 13,3 ms | 2,3 ms | 5,8x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 470,0 ms | 28,5 ms | 16,5x |
| 3x2048x2048, patch 64, stride 32 | `stitch` hann | 453,3 ms | 31,0 ms | 14,6x |

A objeção óbvia é que o acelerador só usa mais núcleos. Ela foi medida: forçando o torch a
4, 8, 16 e 36 threads no caso maior, o caminho puro fica entre 390 ms e 448 ms, e com 36
threads não é melhor do que com 8. O `F.fold` não escala com lote 1, que é a razão de o
kernel existir.

O benchmark compara os dois caminhos com `torch.equal` antes de reportar qualquer tempo, e
sai com erro se eles diferirem.

## Onde isto se aplica hoje

A CI roda em {Ubuntu, Windows} x {Python 3.12, 3.13, 3.14} no caminho puro, mais um job
acelerado nos dois sistemas. A superfície pública
são 20 nomes, congelados por teste com `inspect.signature`.

**Em CPU.** O acelerador só aceita tensor na CPU e devolve o resto ao torch, então numa GPU
a biblioteca funciona e não acelera. A CI não tem GPU e nenhum teste rodou em CUDA, então
lá vale conferir a exatidão no próprio dispositivo antes de contar com ela.

**Em Linux e Windows x86-64** o kernel roda na CI a cada commit, e a tabela foi medida numa
estação Windows. As wheels de macOS e aarch64 carregam o kernel construído e conferido, mas
ele nunca foi executado em CI. Nessas, uma execução com `PATCHCRAFT_ACCEL=0` ao lado da
normal mostra se os dois caminhos concordam nos seus dados.

**Antes da 1.0**, e sem consumidor externo, que é o critério que o projeto escolheu para se
dizer estável: o dígito do meio ainda pode mexer no que sai, e cada mudança entra no
changelog com a medição atrás.

## Reprodução

```
git clone https://github.com/LeoPR/PatchCraft
cd PatchCraft
pip install patchcraft pytest
python tools/benchmark.py --markdown
PATCHCRAFT_SWEEP_FULL=1 pytest tests/test_exactness.py
```

## Ligações

- Repositório: https://github.com/LeoPR/PatchCraft
- Desempenho, com máquina, versões e data: `docs/PERFORMANCE.md`
- O contrato de exatidão: `docs/ADR/0003-reversibility-classes.md`, aceito em 2026-09-03
- Os limites, com as medições: `docs/GUIDE.md` seção 8
