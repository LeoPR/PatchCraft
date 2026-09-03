<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=pt-BR · canonical -->
**Português** · [English](post.en.md)

# Post curto para o LinkedIn

> Pronto para publicar. Todo número foi medido e é reproduzível no repositório.
> O gancho é a retratação, não a vantagem.

---

**Publiquei uma biblioteca e depois descobri que ela mentia na própria documentação.**

O PatchCraft recorta uma imagem em patches e remonta. As docstrings dele afirmavam, em
quinze lugares, quando essa ida e volta é exata bit a bit. A afirmação estava errada, e não
por pouco: dizia que fora da regra o erro fica em torno de 1 ULP, quando ele cresce com a
sobreposição e chega a 19 ULP em float32.

O contrato correto, medido: a ida e volta é exata se e somente se toda contagem do mapa de
cobertura for potência de dois. Fora disso o erro por pixel é limitado por `(k+1)·eps·|v|`.

O que mais me interessou foi entender por que a suíte não pegou isso. Os testes montavam as
imagens com `torch.arange`, e dado inteiro fecha exato onde dado aleatório não fecha. O
teste passava porque estava perguntando errado.

Então escrevi um teste com a função explícita de derrubar a afirmação nova. Ele enumera as
126.736 geometrias legais do espaço, sem consultar o predicado, e procura os dois
contraexemplos: um caso dentro da regra que não seja exato, e um caso fora que seja exato
por sorte.

Acho que uma biblioteca numérica vale menos pela garantia que anuncia e mais pelo teste que
ela mantém apontado contra a própria garantia.

Python 3.12 a 3.14, MIT, pré-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
