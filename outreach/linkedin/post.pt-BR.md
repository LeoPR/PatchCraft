<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=pt-BR · canonical -->
**Português** · [English](post.en.md)

# Post curto para o LinkedIn

> Pronto para publicar. Todo número foi medido e é reproduzível no repositório.
> Fonte: [`../2026-09-04-lancamento.md`](../2026-09-04-lancamento.md).
>
> **Estrutura, na ordem:** um parágrafo de contexto sem jargão, um que nomeia o assunto,
> a virada, os dois defeitos, a biblioteca, o contrato, o teste, o fecho. As duas
> primeiras linhas são as únicas que aparecem antes do "ver mais", então elas não podem
> conter nenhuma palavra que o leitor precise já saber.

---

**Toda imagem grande que entra numa rede neural é cortada em pedaços antes.**

Ela não cabe inteira na memória, e mesmo quando cabe, boa parte do trabalho sai melhor
pedaço a pedaço. Então você recorta, processa cada pedaço e cola tudo de volta no fim.
Esses pedaços têm nome: patches.

Parece um problema de vinte linhas. Eu escrevi essas vinte linhas mais vezes do que
gostaria de admitir, e errei o bastante para passar a desconfiar delas.

Erram assim.

No recorte, a função do torch que faz esse serviço devolve os pedaços num formato
empacotado. A reorganização intuitiva para o formato que você quer entrega a forma certa
com os pixels errados. O `assert` de shape passa. O treino roda. A perda desce um pouco
menos. Nada avisa.

Na colagem, o passo entre um pedaço e o seguinte pode não cobrir a imagem inteira. Numa
imagem de 128 por 128, com pedaço de 32 e passo de 20, a grade para no pixel 112 e deixa
3840 dos 16384 pixels em zero. Quase um quarto da imagem volta preto, e a função devolve
isso sem levantar erro nenhum.

Nenhum dos dois é difícil de corrigir. Os dois são fáceis de não perceber, e é essa
diferença que justifica escrever uma vez, com teste em volta, em vez de reescrever a cada
projeto.

Foi o que fiz, e chama-se PatchCraft.

O que ele afirma sobre a própria conta é uma condição que você avalia **antes** de chamar:
recortar e remontar devolve exatamente os mesmos bits se, e somente se, cada pixel for
coberto por um número de pedaços que seja potência de dois. Fora disso o erro tem limite
escrito. Você decide isso olhando a geometria, no papel, sem rodar nada.

E existe um teste cujo trabalho é derrubar essa afirmação. Ele varre as 126.736 geometrias
possíveis sem consultar a regra, e procura o caso que a contradiz.

Ele tem essa forma porque a primeira versão dessa afirmação foi publicada, medida e
encontrada falsa.

Acho que uma biblioteca numérica vale menos pela garantia que anuncia e mais pelo teste que
ela mantém apontado contra a própria garantia.

Python 3.12 a 3.14, MIT, pré-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
