<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=en · translation_of=post.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](post.pt-BR.md)

# Short LinkedIn post

> Ready to publish. Every number was measured and reproduces in the repository.
> Source: [`../2026-09-04-release.en.md`](../2026-09-04-release.en.md).
>
> **The idea that runs through the text:** silent failure. Code that runs, returns the
> right shape and is wrong without saying so. The two defects are instances of it; the
> contract is the answer; the retraction is the same failure one layer up, in the
> guarantee rather than in the code. The close returns to the opening. No technical term
> before the reader knows what the subject is.

---

**There is a kind of bug no test catches: the code runs, the output has the right shape,
and the number is slightly wrong.**

Nobody finds out. The model just learns a little worse, and "a little worse" sets off no
alarm. I carried one of these through several projects without noticing.

A large image almost never goes into a neural network whole. It is cut into pieces, each
piece is processed, and at the end everything is glued back together. The pieces are called
patches, and the cutting and glueing look like a twenty-line problem. I wrote those twenty
lines in every project, and in every one they went wrong in silence.

On the cutting side, the torch function hands the pieces back packed in an order that is
not the intuitive one. The obvious rearrangement gives you the right shape with the pixels
scrambled. The shape `assert` passes. Training runs.

On the glueing side, the step from one piece to the next may not close the image. With a
piece of 32 and a step of 20 on a 128 image, the grid stops at pixel 112. Almost a quarter
of the image comes back black, and the function returns it without raising anything.

Neither is hard to fix. The hard part comes after: once fixed, how do you know it is right?

That question is what became a library. PatchCraft cuts and reassembles, and what sets it
apart is not the cutting. It is telling you, before you call, under what condition the
result is bit-exact: when every pixel is covered by a number of pieces that is a power of
two. Outside that, the error has a written bound. You decide by looking at the geometry,
without running anything.

And there is a test whose only job is to bring that claim down. It sweeps the 126,736
possible geometries without consulting the rule, hunting for the case that contradicts it.

It exists because the first version of the claim was wrong. It was published, measured and
retracted. The suite had not caught it because the tests used integer data, which comes out
exact where real data does not. The test passed because it was asking the wrong question.

It is the same silent failure as the opening, one layer up. Not in the code, but in the
guarantee about the code.

A numerical library is worth less for the guarantee it announces and more for the test it
keeps pointed at its own guarantee.

Python 3.12 to 3.14, MIT, pre-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
