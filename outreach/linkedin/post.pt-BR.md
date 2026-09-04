<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=pt-BR · canonical -->
**Português** · [English](post.en.md)

# Post curto para o LinkedIn

> Pronto para publicar. Todo número foi medido e é reproduzível no repositório.
> O gancho é o defeito silencioso, que é a única coisa aqui que o leitor pode ter no
> código dele agora. Fonte: [`../2026-09-04-lancamento.md`](../2026-09-04-lancamento.md).

---

**Se você recorta imagens em patches com `F.unfold`, tem uma chance boa de estar
embaralhando os pixels sem receber erro nenhum.**

O `F.unfold` do torch devolve `(1, C*ph*pw, L)`. O reshape intuitivo para `(L, C, ph, pw)`
entrega a forma certa com os pixels errados. O `assert` de shape passa, o treino roda, a
perda desce um pouco menos, e não há mensagem em lugar nenhum.

O vizinho dele é o stride que não cobre a imagem. Numa imagem 128 por 128 com patch 32 e
stride 20, a grade para no pixel 112 e deixa 3840 dos 16384 pixels em zero. Um `fold`
escrito à mão devolve essa imagem parcialmente preta, também sem reclamar.

Escrevi essas vinte linhas mais vezes do que gostaria de admitir, e foi por isso que virou
biblioteca. O PatchCraft recorta uma imagem em patches e remonta, com teste em volta dos
dois defeitos acima.

O que ele afirma sobre a numérica é uma condição que você avalia **antes** de chamar: a ida
e volta é exata bit a bit se e somente se todo valor do mapa de cobertura for potência de
dois, e fora disso o erro por pixel é limitado por `(k+1)·eps·|v|`. Isso se calcula a
partir da geometria, sem rodar nada.

E existe um teste cuja função explícita é derrubar essa afirmação. Ele enumera as 126.736
geometrias legais do espaço sem consultar o predicado, e procura os dois contraexemplos: um
caso dentro da regra que não seja exato, e um caso fora que seja exato por sorte.

Ele tem essa forma porque uma versão anterior desse contrato foi publicada, medida e
encontrada falsa. Acho que uma biblioteca numérica vale menos pela garantia que anuncia e
mais pelo teste que mantém apontado contra a própria garantia.

Python 3.12 a 3.14, MIT, pré-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
