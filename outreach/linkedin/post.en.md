<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=en · translation_of=post.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](post.pt-BR.md)

# Short LinkedIn post

> Ready to publish. Every number was measured and is reproducible in the repository.
> The hook is the retraction, not the advantage.

---

**I published a library and then found out it was lying in its own documentation.**

PatchCraft cuts an image into patches and puts it back together. Its docstrings stated, in
fifteen places, when that round trip is bit-exact. The statement was wrong, and not by a
little: it said the error outside the rule stays around 1 ULP, when it grows with the
overlap and reaches 19 ULP in float32.

The correct contract, measured: the round trip is exact if and only if every value in the
coverage count map is a power of two. Outside that, the per-pixel error is bounded by
`(k+1)·eps·|v|`.

What interested me more was working out why the suite never caught it. The tests built
their images with `torch.arange`, and integer data round-trips exactly where random data
does not. The test passed because it was asking the wrong question.

So I wrote a test whose explicit job is to break the new statement. It enumerates the
126,736 legal geometries of the space without consulting the predicate, and hunts for both
counterexamples: a case inside the rule that is not exact, and a case outside it that is
exact by luck.

I think a numerical library is worth less for the guarantee it announces and more for the
test it keeps pointed at that guarantee.

Python 3.12 to 3.14, MIT, pre-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
