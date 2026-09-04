<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=pt-BR · canonical -->
**Português** · [English](post.en.md)

# Post curto para o LinkedIn

> Pronto para publicar. Todo número foi medido e é reproduzível no repositório.
> Fonte: [`../2026-09-04-lancamento.md`](../2026-09-04-lancamento.md).
>
> **A ideia que atravessa o texto:** falha silenciosa. Código que roda, devolve a forma
> certa e está errado sem avisar. Os dois defeitos são instâncias dela; o contrato é a
> resposta; a retratação é a mesma falha uma camada acima, na garantia em vez de no
> código. O fecho volta à abertura. Nenhum termo técnico antes de o leitor saber do que se
> trata.

---

**Existe um tipo de erro que nenhum teste pega: o código roda, a saída tem a forma certa,
e o número está um pouco errado.**

Ninguém fica sabendo. O modelo só aprende um pouco pior, e "um pouco pior" não dispara
alarme nenhum. Carreguei um caso desses por vários projetos sem perceber.

Uma imagem grande quase nunca entra inteira numa rede neural. Ela é cortada em pedaços,
cada pedaço é processado, e no fim tudo é colado de volta. Os pedaços se chamam patches, e
o recorte e a colagem parecem um problema de vinte linhas. Eu escrevi essas vinte linhas
em cada projeto, e em cada um elas erraram em silêncio.

No recorte, a função do torch devolve os pedaços empacotados numa ordem que não é a
intuitiva. A reorganização óbvia entrega a forma certa com os pixels embaralhados. O
`assert` de shape passa. O treino roda.

Na colagem, o passo entre um pedaço e o próximo pode não fechar a imagem. Com pedaço de 32
e passo de 20 numa imagem de 128, a grade para no pixel 112. Quase um quarto da imagem
volta preto, e a função devolve isso sem levantar nada.

Nenhum dos dois é difícil de corrigir. A parte difícil vem depois: corrigido, como você
sabe que está certo?

Foi essa pergunta que virou biblioteca. O PatchCraft recorta e remonta, e o que ele tem de
diferente não é o recorte. É dizer, antes de você chamar, em que condição o resultado é
exato bit a bit: quando cada pixel é coberto por um número de pedaços que seja potência de
dois. Fora disso, o erro tem um limite escrito. Você decide olhando a geometria, sem rodar
nada.

E existe um teste cujo único trabalho é derrubar essa afirmação. Ele varre as 126.736
geometrias possíveis sem consultar a regra, procurando o caso que a contradiz.

Ele existe porque a primeira versão da afirmação estava errada. Foi publicada, medida e
retratada. A suíte não tinha pegado porque os testes usavam dados inteiros, que fecham
exato onde dados reais não fecham. O teste passava porque perguntava errado.

É a mesma falha silenciosa do começo, uma camada acima. Não no código, mas na garantia
sobre o código.

Uma biblioteca numérica vale menos pela garantia que anuncia e mais pelo teste que mantém
apontado contra a própria garantia.

Python 3.12 a 3.14, MIT, pré-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
