<!-- l10n: doc_id=patchcraft-outreach-linkedin-artigo · lang=pt-BR · canonical -->
**Português** · [English](artigo.en.md)

# Recortar uma imagem em pedaços e remontar sem perder pixel no caminho

*Artigo técnico. Cada número aqui tem um comando que o reproduz no repositório. Onde a
biblioteca não ajuda, o texto diz que não ajuda.*

---

Recortar uma imagem em patches, rodar alguma coisa em cada pedaço e remontar parece uma
tarefa de vinte linhas. Eu escrevi essas vinte linhas mais vezes do que gostaria de admitir,
e as reescrevi errado o suficiente para valer a pena escrever uma vez com teste em volta.

Este texto começa pelos dois erros que essas vinte linhas cometem em silêncio, porque são
a razão de a biblioteca existir e porque você pode estar com um deles agora.

## Dois defeitos que não avisam

O primeiro mora no reshape. O `F.unfold` do torch devolve um tensor `(1, C*ph*pw, L)`, e o
reshape intuitivo para `(L, C, ph, pw)` entrega a forma certa com os pixels errados. O
`assert` de shape passa, o treino roda, a perda desce um pouco menos, e não há mensagem de
erro em lugar nenhum.

O segundo mora no stride. Numa imagem de 128 por 128, com patch 32 e stride 20, a grade
para no pixel 112 e deixa 3840 dos 16384 pixels em zero. Um `fold` escrito à mão devolve
essa imagem parcialmente preta sem reclamar.

Nenhum dos dois é difícil de corrigir. Os dois são fáceis de não perceber, e essa é a
diferença que justifica escrever uma vez, com teste em volta, em vez de reescrever a cada
projeto.

## O contrato numérico, e por que ele é avaliável antes da chamada

A biblioteca faz uma afirmação numérica: em que condições a ida e a volta devolvem o mesmo
tensor, bit a bit.

> A ida e volta é exata se e somente se **todo** valor do mapa de cobertura for potência de
> dois. Fora disso, o erro por pixel é limitado por `(k+1)·eps·|v|`, com `k` a contagem de
> cobertura daquele pixel.

A razão é uma só: dividir um float por uma potência de dois é a única divisão que nunca
arredonda. `stride == patch_size` sempre satisfaz isso, porque toda contagem vale 1.

O que faz disso um contrato, e não uma promessa, é a segunda metade: a condição se calcula a
partir da geometria, sem rodar nada. Quem vai chamar sabe de antemão em que regime está.

Essa formulação é a segunda. A primeira foi publicada, medida e encontrada falsa, e é por
isso que as duas seções seguintes existem. A versão antiga dependia de a contagem máxima de
sobreposição ser pequena, e dizia que fora dessa condição o erro ficava em torno de 1 ULP;
medindo, ele cresce com a cobertura de cada pixel e chega a 19 ULP em float32. Ela olhava
para o máximo do mapa, e a correta olha para todos os valores dele. Sobre uma varredura de
geometrias retangulares, a regra do máximo erra 3936 de 14969 casos e a da potência de dois
erra 8, todos na direção segura de prometer menos do que entregam. Um contrato pode prometer
de menos. Não pode prometer demais.

## Por que a suíte não pegou

Essa é a parte que eu levaria para qualquer outro projeto.

Os testes de round-trip construíam as imagens com `torch.arange`. Dado inteiro, num float,
fecha a ida e volta exato em geometrias onde dado aleatório não fecha, porque não há
mantissa suficiente sendo usada para o arredondamento aparecer. A suíte passava porque
estava fazendo a pergunta errada, com muita confiança.

A correção não foi só trocar a frase. Foi trocar o gerador de dados por um helper auditado,
que sorteia ruído de mantissa cheia direto no dtype alvo e é proibido de derivar float32 a
partir de float64.

## O teste que existe para derrubar a afirmação

Depois de corrigir, escrevi um teste com a função explícita de falsificar o contrato novo.

Ele enumera as 126.736 geometrias legais do espaço sem consultar o predicado, para que o
enumerador e a coisa testada sejam independentes. Sobre uma amostra semeada, procura os
dois contraexemplos que derrubariam o contrato: um caso dentro da regra que não seja exato,
e um caso fora da regra que seja exato em todas as sementes, o que indicaria que o predicado
está estreito demais. A varredura completa das 126.736 fica atrás de uma variável de
ambiente, porque leva pouco mais de um minuto.

Acho que uma biblioteca numérica vale menos pela garantia que anuncia e mais pelo teste que
mantém apontado contra a própria garantia.

## O acelerador, e a objeção que ele merece

O caminho quente é o fold com sobreposição, que é onde o tempo vai quando os patches se
sobrepõem. Cinco das seis wheels trazem um kernel Rust para ele, compilado dentro da própria
wheel, sem extra para habilitar. As demais plataformas recebem a wheel universal e rodam o
caminho em torch, que devolve os mesmos valores.

| Geometria | Chamada | Torch puro | Acelerado | Ganho |
|---|---|---|---|---|
| 3x512x512, patch 32, stride 16 | `reconstruct` | 16,3 ms | 2,3 ms | 7,1x |
| 3x1024x1024, patch 64, stride 32 | `reconstruct` | 53,5 ms | 7,8 ms | 6,9x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 453,7 ms | 32,1 ms | 14,1x |
| 3x2048x2048, patch 64, stride 32 | `stitch` hann | 460,9 ms | 37,9 ms | 12,2x |

A objeção óbvia é que o kernel apenas usa mais núcleos do que o torch usa. Ela merece uma
medição, não uma resposta. Forçando o torch a 4, 8, 16 e 36 threads no caso maior, o caminho
puro fica entre 365 ms e 465 ms, e com 36 threads não fica melhor do que com 8. O `F.fold`
não escala com lote 1, e é essa a razão de o kernel existir.

O benchmark roda cada caso duas vezes, uma com o acelerador e outra com ele desligado, e
compara os dois resultados com `torch.equal` **antes** de reportar qualquer tempo. Se
diferirem, ele imprime a tabela, diz que diferiram e sai com erro. Um benchmark de duas
contas diferentes não é um benchmark.

Uma armadilha que essa disciplina pegou: o install editável estava compilando o kernel em
debug, porque a ferramenta segue o comando de build a menos que você mande o contrário.
Isso transformava um ganho de 14x em 2,1x. Eu quase publiquei os números do binário errado.

## O que o projeto não afirma

Nenhum projeto externo consumiu a biblioteca ainda, e esse é o critério que ela própria
escolheu para se dizer estável.

Nenhum caminho CUDA dela jamais executou, nem na CI nem fora. O acelerador recusa qualquer
tensor que não esteja na CPU e devolve o trabalho ao torch.

Três das cinco wheels aceleradas nunca tiveram o kernel executado em CI. As de macOS e
aarch64 são construídas e têm o conteúdo conferido, e isso é tudo.

E a afirmação numérica desta página já esteve publicada em outra forma, como verdadeira. É
razoável supor que exista uma terceira que eu ainda não medi.

## Prático

Python 3.12 a 3.14, torch 2.6 ou mais novo, MIT, pré-1.0. São 1619 testes passando e 1656
coletados, com CI em {Ubuntu, Windows} x {3.12, 3.13, 3.14} e um job acelerado nos dois
sistemas. A superfície
pública são 20 nomes, congelados por teste.

```
pip install patchcraft
```

O código, as medições e a documentação do que não funciona estão abertos.

https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
