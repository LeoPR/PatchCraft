# References

Two groups, collected on 2026-10-09: the numerical analysis of sums and means,
including the integer midpoint literature, and the imaging and systems sources
that show how overlapping estimates are blended in practice. Each entry gives
the locator that was opened and what the paper uses it for.

Every row of [`sources.tsv`](sources.tsv) pairs a locator with a piece of
text the source must contain, a Crossref title for a DOI and a quoted passage
for a page or a PDF, and

    python docs/PAPER/experiments/check_sources.py

re-opens them all. Two entries were not opened and are marked: the facts cited
from them come from sources that were.

## Numerical analysis and integer averaging

- **[bloch2006]** Joshua Bloch. Extra, Extra - Read All About It: Nearly All Binary
  Searches and Mergesorts are Broken. Google Research Blog (2 June 2006).
  Locator:
  https://research.google/blog/extra-extra-read-all-about-it-nearly-all-binary-searches-and-mergesorts-are-broken/
  Used for: The canonical real case of an in-domain midpoint, (low+high)/2, whose
  intermediate overflowed after lying latent in the JDK for about nine years. It gives
  the fixes low + ((high - low)/2) and (low + high) >>> 1, and points to Ruggieri 2003.

- **[boldo2015]** Sylvie Boldo. Formal verification of programs computing the
  floating-point average. ICFEM 2015, LNCS 9407, pp. 17-32 (Springer).
  Locator: https://inria.hal.science/hal-01174892/document (DOI
  10.1007/978-3-319-25423-4_2)
  Used for: The closest primary source to the thesis for n = 2: (M+M)/2 overflows
  although the mean is M, and (x+y)/2 is correctly rounded when it does not overflow
  (Thm 1). (x/2)+(y/2) is correctly rounded when |x| >= 2^(Ei+2p+1) but returns 0 for x
  = y = 2^Ei (Thm 2), and 'division by the radix is an exact operation, provided the
  input is not subnormal'.

- **[chan1983]** Tony F. Chan, Gene H. Golub, Randall J. LeVeque. Algorithms for
  computing the sample variance: analysis and recommendations. The American Statistician
  37(3):242-247, 1983.
  Locator:
  https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf
  (copy of the publisher PDF; DOI 10.1080/00031305.1983.10483115)
  Used for: States the Welford/West/Hanson updating formulas (1.3a)-(1.3b) and the
  pairwise combination (1.5)-(1.6), and notes that pairwise summation cuts the relative
  error of the sum from O(N) to O(log N). Its 'textbook one-pass algorithm' is a
  precedent for naming a formula that is exact in theory but fails in floating point.

- **[chen2022]** Raymond Chen. On finding the average of two unsigned integers without
  overflow. The Old New Thing, Microsoft DevBlogs (7 Feb 2022).
  Locator: https://devblogs.microsoft.com/oldnewthing/20220207-00/?p=106223
  Used for: A practitioner's list of the four strategies: the difference form,
  pre-dividing plus the carry (a/2)+(b/2)+(a&b&1) ('pre-divide the values before
  adding'), SWAR, and casting to a larger type. It links the Wong patent.

- **[cppref-lerp]** cppreference.com contributors. std::lerp. cppreference.com, 2026
  (accessed 2026-10-09).
  Locator: https://en.cppreference.com/w/cpp/numeric/lerp
  Used for: Lists the exactness, finiteness on [0,1], consistency and monotonicity
  guarantees of the standard two-point weighted mean, which 'normalize the weights
  first' generalizes.

- **[cppref-midpoint]** cppreference.com contributors. std::midpoint. cppreference.com,
  2026 (accessed 2026-10-09).
  Locator: https://en.cppreference.com/w/cpp/numeric/midpoint
  Used for: The adopted C++20 semantics: 'Half the sum of a and b. No overflow occurs.'
  Integer results are rounded towards a, floats incur at most one inexact operation, and
  the page demonstrates a uint32 (a+b)/2 that wraps.

- **[cpython-sum]** Python Software Foundation (CPython contributors). CPython v3.13.0,
  Python/bltinmodule.c, builtin sum() float path. GitHub, python/cpython, 2024.
  Locator: https://raw.githubusercontent.com/python/cpython/v3.13.0/Python/bltinmodule.c
  (lines 2649-2661)
  Used for: Shows that industry adopted compensated summation: the float sum()
  implements Neumaier's improved Kahan-Babuska algorithm and cites the 1974 DOI.
  sum([1e100, 1, -1e100, 1]) returns 2.0, where a plain loop returns 1.0.

- **[croci2022]** Matteo Croci, Massimiliano Fasi, Nicholas J. Higham, Theo Mary, Mantas
  Mikaitis. Stochastic rounding: implementation, error analysis and applications. Royal
  Society Open Science 9(3):211631, 2022.
  Locator: https://pmc.ncbi.nlm.nih.gov/articles/PMC8905452/ (DOI 10.1098/rsos.211631)
  Used for: Makes the range premise explicit (the standard model holds 'if no overflow
  or underflow occurs') and restates gamma_n. Under stochastic rounding the expected sum
  is exact, a sqrt(n)u bound holds with high probability and stagnation is avoided; that
  is a probabilistic alternative, not needed for integer means, where the remainder
  carry is exact.

- **[cxx-draft-midpoint]** ISO/IEC JTC1/SC22/WG21. C++ working draft, 26.10.16 Midpoint
  [numeric.ops.midpoint]. eel.is C++ draft (commit c7015b48), 2026 (accessed
  2026-10-09).
  Locator: https://eel.is/c++draft/numeric.ops.midpoint
  Used for: The current normative wording, unchanged from P0811R3: 'No overflow occurs'
  and 'If T is a floating-point type, at most one inexact operation occurs.'

- **[dietz-magic]** Henry Gordon Dietz. The Aggregate Magic Algorithms (entry 'Average
  of Integers'). University of Kentucky, Aggregate.Org online technical report, undated
  (accessed 2026-10-09).
  Locator: https://aggregate.org/MAGIC/
  Used for: Explains x + y = (x XOR y) + 2(x AND y), so (x&y)+((x^y)>>1) 'cannot
  overflow'. It is probably the '[Dietz]' that Warren credits, but this is unconfirmed
  because his bibliography is not in the sample pages; it is not Will Dietz of ICSE
  2012.

- **[dietz2012]** Will Dietz, Peng Li, John Regehr, Vikram Adve. Understanding integer
  overflow in C/C++. Proc. 34th International Conference on Software Engineering (ICSE),
  pp. 760-770 (extended in ACM TOSEM 25(1), 2015, DOI 10.1145/2743019), 2012.
  Locator: https://www.cs.utah.edu/~regehr/papers/overflow12.pdf (DOI
  10.1109/ICSE.2012.6227142)
  Used for: Supplies the taxonomy of intentional or unintentional and well-defined or
  undefined overflow, and 'UINT_MAX+1 must evaluate to zero in every conforming C and
  C++ implementation'. That is why an unsigned accumulator, like numpy's uint16 mean,
  wraps silently.

- **[goldberg1991]** David Goldberg. What every computer scientist should know about
  floating-point arithmetic. ACM Computing Surveys 23(1):5-48, 1991.
  Locator: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html (edited
  reprint, Numerical Computation Guide App. D; DOI 10.1145/103162.103163)
  Used for: Provides Theorem 11 (Sterbenz: y/2 <= x <= 2y gives x - y exactly), Theorem
  7 ((m/n)*n = m for n = 2^i + 2^j), the remark that scaling by a power of two is
  harmless, and Theorem 8 (the Kahan summation bound). The reprint's section
  'Differences Among IEEE 754 Implementations' is not by Goldberg.

- **[goualard2014]** Frederic Goualard. How do you compute the midpoint of an interval?.
  ACM Transactions on Mathematical Software 40(2), article 11, 2014.
  Locator: https://hal.science/hal-00576641/file/computing-midpoint.pdf (HAL v2; DOI
  10.1145/2493882)
  Used for: Reviews midpoint formulas in interval libraries: Intlab V5.5 'avoids an
  overflow by first dividing the operands by two before adding the results, instead of
  the other way round', and Pitfall 2 is the textbook overflow. It also gives Hauser's
  exact-addition theorems, which hold under underflow, and the containment requirement
  m(I) in I.

- **[herring2019-p0811r3]** S. Davis Herring. P0811R3: Well-behaved interpolation for
  numbers and pointers. ISO/IEC JTC1/SC22/WG21 (C++ committee paper), 2019.
  Locator: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0811r3.html
  Used for: The standards-track statement of the problem: (a+b)/2 overflows for integers
  and floats, a+(b-a)/2 overflows for mixed signs, and a/2+b/2 prevents overflow but
  rounds subnormals badly. The lerp form a+t(b-a) 'can overflow if a and b have the
  largest exponent and opposite signs', so the reference lerp uses t*b+(1-t)*a when ab
  <= 0; the paper also remarks that 'mean' is the better name for floats.

- **[higham1993]** Nicholas J. Higham. The accuracy of floating point summation. SIAM
  Journal on Scientific Computing 14(4):783-799, 1993.
  Locator: https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf (published version
  on the author's site; DOI 10.1137/0914050)
  Used for: Primary source for the standard model (1.2), gamma_n, the
  recursive-summation bound (2.6), the partial-sum form (2.7)-(2.8), the pairwise bound
  (3.6) and the compensated bound (3.10)-(3.11). It names means among the sums it covers
  and assumes no underflow, so range is outside its scope.

- **[higham2002asna]** Nicholas J. Higham. Accuracy and Stability of Numerical
  Algorithms, 2nd ed., chapter 4 'Summation' (pp. 79-92). SIAM, Philadelphia, 2002.
  Locator: https://doi.org/10.1137/1.9780898718027 and
  https://doi.org/10.1137/1.9780898718027.ch4 (Crossref records); author page
  https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/
  Used for: [Metadata only: the SIAM chapter returned HTTP 403 and the Manchester
  table-of-contents host was unreachable.] This is the textbook version of Higham 1993,
  and Croci et al. 2022 cite its Lemma 3.1 for gamma_n; take exact equations from Higham
  1993, which I read.

- **[higham2019squeeze]** Nicholas J. Higham, Srikara Pranesh, Mawussi Zounon. Squeezing
  a matrix into half precision, with an application to solving linear systems. SIAM
  Journal on Scientific Computing 41(4):A2536-A2551, 2019.
  Locator: https://perso.ens-lyon.fr/elisa.riccietti/doc/papers_M2/squeezing.pdf
  (published version under CC BY 4.0; DOI 10.1137/18M1229511)
  Used for: A numerical-analysis precedent for scaling before computing in fp16: bring
  the largest element within theta <= 1 of the overflow level and leave 'headroom for
  subsequent computations', because 'Overflow is unrecoverable'. Its Table 1.1 gives
  fp16 u = 4.88e-4, smallest normal 6.10e-5 and largest 6.55e4.

- **[ieee754-2019]** IEEE Computer Society (Microprocessor Standards Committee). IEEE
  Standard for Floating-Point Arithmetic (IEEE Std 754-2019). IEEE Standards
  Association, 2019.
  Locator: https://standards.ieee.org/ieee/754/6210/ (IEEE SA page: active, published
  2019-07-22; DOI 10.1109/IEEESTD.2019.8766229 via Crossref)
  Used for: [Metadata only: the standard's text is behind a paywall.] It is the
  normative source for correctly rounded operations; the exactness of power-of-two
  scaling is taken from Goldberg, Boldo and P0811R3 and checked exhaustively in float16.

- **[kahan1965]** W. Kahan. Pracniques: further remarks on reducing truncation errors.
  Communications of the ACM 8(1):40, 1965.
  Locator: https://doi.org/10.1145/363707.363723 (Crossref record; ACM full text
  returned HTTP 403)
  Used for: [Metadata only.] The origin of compensated summation; cite it for priority
  and take the algorithm and bound from Higham 1993 (3.10)-(3.11) and Goldberg 1991
  Theorem 8, which I read.

- **[knuth-taocp2]** Donald E. Knuth. The Art of Computer Programming, Vol. 2:
  Seminumerical Algorithms (2nd ed. 1981; 3rd ed. 1997), sec. 4.2.2. Addison-Wesley,
  1981/1997.
  Locator: not opened (no legitimate open copy; I deliberately did not use the
  unauthorized archive.org scans)
  Not opened. The facts cited from it are taken from sources that were.
  Used for: NOT OPENED; do not cite without checking a physical copy. Higham 1993 cites
  the 2nd ed., Ex. 19, pp. 229 and 572-573, for the compensated-summation proof; the
  attribution of Welford's recurrence to sec. 4.2.2 rests only on secondary web pages.

- **[neumaier1974]** A. Neumaier. Rundungsfehleranalyse einiger Verfahren zur Summation
  endlicher Summen. ZAMM - Zeitschrift fuer Angewandte Mathematik und Mechanik
  54(1):39-51, 1974.
  Locator: https://doi.org/10.1002/zamm.19740540106 (Crossref record)
  Used for: [Metadata only.] The improved Kahan-Babuska summation; its bound is quoted
  in Ogita-Rump-Oishi 2005 eq. (1.3), and CPython 3.13's built-in sum() implements it
  with this DOI in a source comment.


- **[ogita2005]** Takeshi Ogita, Siegfried M. Rump, Shin'ichi Oishi. Accurate sum and
  dot product. SIAM Journal on Scientific Computing 26(6):1955-1988, 2005.
  Locator: https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf (DOI 10.1137/030601818)
  Used for: Covers error-free transformations (TwoSum) and sums as accurate as twice the
  working precision, and quotes Neumaier's bound (1.3). It states 'We assume that no
  overflow occurs, but allow underflow', another case of the precision literature taking
  range for granted.

- **[ruggieri2003]** Salvatore Ruggieri. On computing the semi-sum of two integers.
  Information Processing Letters 87(2):67-71, 2003.
  Locator: https://pages.di.unipi.it/ruggieri/Papers/semisum.pdf (author preprint; DOI
  10.1016/S0020-0190(03)00263-1)
  Used for: States both halves of the thesis for n = 2: the result is representable
  because min <= floor((a+b)/2) <= max, and the usual fix casts 'up to a larger numeric
  data type' and back, which 'may not be available'. It then derives an overflow-free
  program with Hoare logic.

- **[sterbenz1974]** Pat H. Sterbenz. Floating-Point Computation. Prentice-Hall,
  Englewood Cliffs, NJ, 1974.
  Locator: not opened (bibliographic entry seen in Goldberg 1991's reference list)
  Not opened. The facts cited from it are taken from sources that were.
  Used for: NOT OPENED. The source of the Sterbenz lemma and of the carefully written
  average of two floats; cite it through Goldberg Thm 11, Goualard's footnote 10 and
  Boldo 2015, which reproduces Sterbenz's averaging formulas.

- **[warren2013]** Henry S. Warren, Jr.. Hacker's Delight, 2nd ed., sec. 2-5 'Average of
  Two Integers' (p. 19). Addison-Wesley Professional (ISBN 978-0-321-84268-8), 2013
  (published Sep 2012).
  Locator:
  https://www.informit.com/content/images/9780321842688/samplepages/0321842685.pdf
  (publisher sample pages, which contain sec. 2-5);
  https://www.informit.com/store/hackers-delight-9780321842688
  Used for: Gives the floor average (x & y) + ((x XOR y) >>u 1) 'without causing
  overflow [Dietz]', the ceiling average (x | y) - ((x XOR y) >>u 1), signed variants
  and x + ((y - x) >>u 1) for x <= y. All are verified exhaustively for uint8, uint16
  and int8.

- **[welford1962]** B. P. Welford. Note on a method for calculating corrected sums of
  squares and products. Technometrics 4(3):419-420, 1962.
  Locator: https://doi.org/10.1080/00401706.1962.10490022 (Crossref; also DOI
  10.2307/1266577; the publisher page returned HTTP 403)
  Used for: [Metadata only.] The running-mean update M_j = M_{j-1} + (x_j - M_{j-1})/j;
  quote the formula from Chan-Golub-LeVeque 1983 eq. (1.3a), which attributes it to
  Welford, West and Hanson.

- **[west1979]** D. H. D. West. Updating mean and variance estimates: an improved
  method. Communications of the ACM 22(9):532-535, 1979.
  Locator: https://doi.org/10.1145/359146.359153 (Crossref record; ACM full text
  returned HTTP 403)
  Used for: [Metadata only.] The weighted incremental mean, a precedent for normalizing
  the weights and then accumulating; Chan et al. 1983 cite it among the updating
  algorithms.

- **[wong1999]** Roney S. Wong (assignee Samsung Electronics). Calculating the average
  of two integer numbers rounded towards zero in a single instruction cycle (US Patent
  6,007,232). US Patent and Trademark Office (filed 1996-11-14, granted 1999-12-28).
  Locator: https://patents.google.com/patent/US6007232A/en
  Used for: The n = 2 case of divide-first with a remainder carry, done in hardware:
  shift each operand right, add, and increment when both shifted-out bits are 1
  (unsigned case).

## Imaging and systems

- **[cmsis-dsp-cfft-q15]** Arm Limited (CMSIS-DSP). arm_cfft_q15 (Complex FFT Q15):
  input and output formats. CMSIS-DSP documentation and source (commit
  83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b), accessed 2026-10-09.
  Locator: https://arm-software.github.io/CMSIS-DSP/latest/group__ComplexFFTQ15.html ;
  https://raw.githubusercontent.com/ARM-software/CMSIS-DSP/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b/Source/TransformFunctions/arm_cfft_q15.c
  Used for: Divide-first where widening is costly. The fixed-point FFT scales inside
  ('Each stage in middle stages provides two down scaling of the input'). The output
  loses log2(N) fractional bits: input 1.15, output 5.11 at N=16 and 11.5 at N=1024,
  with 'Number of bits to upscale' 4 and 10. This is the precision cost of keeping every
  stage inside 16 bits.

- **[cmsis-dsp-mean]** Arm Limited (CMSIS-DSP). arm_mean_q15 / arm_mean_q7 /
  arm_mean_q31: Scaling and Overflow Behavior. CMSIS-DSP documentation and source
  (commit 83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b), accessed 2026-10-09.
  Locator: https://arm-software.github.io/CMSIS-DSP/latest/group__mean.html ;
  https://raw.githubusercontent.com/ARM-software/CMSIS-DSP/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b/Source/StatisticsFunctions/arm_mean_q15.c
  Used for: Embedded DSP widens too. q15 input is 'accumulated in a 32-bit accumulator
  in 17.15 format' (q31 into 64-bit 33.31), with the claim 'There is no risk of internal
  overflow with this approach'. The code holds the sum in q31_t and divides by
  blockSize. By arithmetic (not measured), 17.15 holds at most 2^16 q15 samples while
  blockSize is uint32_t, so widening moves the cliff rather than removing it.

- **[dabov-2007]** Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, Karen Egiazarian.
  Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering. IEEE
  Transactions on Image Processing 16(8): 2080-2095, 2007.
  Locator: doi:10.1109/TIP.2007.901238 (opened: author preprint
  https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf and the Crossref record; IEEE
  Xplore empty)
  Used for: BM3D aggregation is the sum-then-divide form: block estimates are 'weighted
  and accumulating in a buffer', the weights are accumulated 'in another buffer', and
  aggregation is 'a simple element-wise division between the two buffers'. A Kaiser
  window is part of the weights.

- **[elad-aharon-2006]** Michael Elad, Michal Aharon. Image Denoising Via Sparse and
  Redundant Representations Over Learned Dictionaries. IEEE Transactions on Image
  Processing 15(12): 3736-3745, 2006.
  Locator: doi:10.1109/TIP.2006.881969 (opened:
  https://legacy.sites.fas.harvard.edu/~cs278/papers/sparse.pdf , DOI printed on p. 1;
  Crossref record)
  Used for: Shows that patch-based restoration ends in an overlap average: of eq. (8)
  the authors write 'all it says is that averaging of the denoised patches is to be
  done', computable 'on a pixel-by-pixel basis'.

- **[glaccum-manpage]** Khronos Group. glAccum (OpenGL 2.1 reference page). OpenGL
  Reference Pages, registry.khronos.org, accessed 2026-10-09.
  Locator: https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glAccum.xml
  Used for: The accumulation buffer is 'an extended-range color buffer', yet each
  component stores values in [-1, 1], and a result outside that range leaves the pixel
  'undefined'. GL_ACCUM adds value times the normalized color. This range is why the
  documented idiom divides first (redbook-accpersp).

- **[griffin-lim-1984]** Daniel W. Griffin, Jae S. Lim. Signal estimation from modified
  short-time Fourier transform. IEEE Transactions on Acoustics, Speech, and Signal
  Processing 32(2): 236-243, 1984.
  Locator: doi:10.1109/TASSP.1984.1164317 (opened: Crossref record only; IEEE full text
  is paywalled and was not read)
  Used for: Classical source of the least-squares overlap-add estimate that divides by
  the summed window envelope. Cite only as the reference torch.istft names, because the
  equations were not read here.

- **[haeberli-akeley-1990]** Paul Haeberli, Kurt Akeley. The accumulation buffer:
  hardware support for high-quality rendering. SIGGRAPH '90 Proceedings (also ACM
  SIGGRAPH Computer Graphics 24(4), doi:10.1145/97880.97913), pp. 309-318, 1990.
  Locator: doi:10.1145/97879.97913 (opened:
  https://www.microsoft.com/en-us/research/publication/the-accumulation-buffer-hardware-support-for-high-quality-rendering/
  , https://history.siggraph.org/?p=122075 and Crossref records for both DOIs; abstract
  only, full text not read)
  Used for: Hardware precedent for widening: the architecture adds 'an additional
  high-precision image buffer' that 'is used to integrate images that are rendered into
  the framebuffer' for antialiasing, motion blur and depth of field.

- **[hasinoff-2016]** Samuel W. Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams,
  Jonathan T. Barron, Florian Kainz, Jiawen Chen, Marc Levoy. Burst photography for high
  dynamic range and low-light imaging on mobile cameras. ACM Transactions on Graphics
  35(6) (Proc. SIGGRAPH Asia 2016), pp. 1-12.
  Locator: doi:10.1145/2980179.2980254 (opened: PDF
  https://www.hdrplusdata.org/hdrplus.pdf with the DOI printed on p. 1,
  https://hdrplusdata.org/ , the research.google publication page and the Crossref
  record; dl.acm.org refused)
  Used for: Camera burst merging. Eq. (5) is 'equivalent to a temporal average'. The
  paper widens on purpose: 'our input is 10-bit raw and we merge to 14 bits to preserve
  the precision gained from merging'. It also normalizes first: tiles overlap by half,
  and a modified raised cosine is used whose 'total contribution from all tiles sum to
  one at every position', so no division is needed.

- **[hernandez-mier-2025]** Yahir Hernández-Mier, Marco Aurelio Nuño-Maganda, Said
  Polanco-Martagón, Guadalupe Acosta-Villarreal, Rubén Posada-Gómez. Unsupervised
  Optical Mark Recognition on Answer Sheets for Massive Printed Multiple-Choice Tests.
  Journal of Imaging 11(9): 308, 2025.
  Locator: doi:10.3390/jimaging11090308 (opened:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12470569/ and the Crossref record; mdpi.com
  refused)
  Used for: Optional OMR context. The mark decision compares 'the mean of the grayscale
  values for each option circle' with a reference taken from a clean sheet, together
  with a pixel count. The input is a scanner, one image per sheet, with no frame
  averaging; smartphone and webcam capture is described as another OMR generation. Of
  limited weight for the paper.

- **[intel-sdm-pavgb]** Intel Corporation. Intel 64 and IA-32 Architectures Software
  Developer's Manual, Vol. 2 (2A-2D), PAVGB/PAVGW - Average Packed Integers. Order
  Number 325383-093US, 2026 (September 2026 edition).
  Locator: https://cdrdv2.intel.com/v1/dl/getContent/671110 (redirects to
  https://cdrdv2-public.intel.com/929352/325383-093-sdm-vol-2abcd.pdf)
  Used for: Hardware widening for the uint8 mean of two: elements 'are added together, a
  1 is added to the temporary sum, and that result is shifted right one bit position',
  with the pseudo-code comment 'Temp sum before shifting is 9 bits'.

- **[knutsson-westin-1993]** Hans Knutsson, Carl-Fredrik Westin. Normalized and
  Differential Convolution. Proc. IEEE Conference on Computer Vision and Pattern
  Recognition (CVPR), pp. 515-523, 1993.
  Locator: doi:10.1109/CVPR.1993.341081 (opened:
  https://mlanthology.org/cvpr/1993/knutsson1993cvpr-normalized/ and the Crossref
  record; IEEE Xplore returned an empty page, full text not read)
  Used for: Origin of the signal/certainty split: data and operators are separated into
  signal and certainty, and missing samples get certainty zero. PatchCraft's fold
  divided by fold(ones) is the case with a box applicability and unit certainty. The
  division itself is cited from piroddi-petrou-2003, because only the abstract was read
  here.

- **[micikevicius-2018]** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory
  Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev,
  Ganesh Venkatesh, Hao Wu. Mixed Precision Training. ICLR 2018 (arXiv:1710.03740v3).
  Locator: doi:10.48550/arXiv.1710.03740 (opened: https://arxiv.org/abs/1710.03740 and
  the PDF https://arxiv.org/pdf/1710.03740v3)
  Used for: The ML-systems statement of the widening answer: FP16 dot products
  accumulate 'the partial products into an FP32 value, which is converted to FP16 before
  writing to memory', and 'Large reductions (sums across elements of a vector) should be
  carried out in FP32.'

- **[numpy-mean-docs]** NumPy Developers. numpy.mean. NumPy v2.5 Manual, accessed
  2026-10-09 (v2.5).
  Locator: https://numpy.org/doc/2.5/reference/generated/numpy.mean.html
  Used for: dtype is the 'Type to use in computing the mean. For integer inputs, the
  default is float64'. The Notes recommend 'a higher-precision accumulator using the
  dtype keyword' for float32, and say float16 is computed with float32 intermediates.
  The page does not say what happens when dtype is an integer type; the source shows it
  (numpy-methods-src).

- **[numpy-methods-src]** NumPy Developers. numpy/_core/_methods.py, function _mean.
  NumPy source, tag v2.5.2 (identical to the installed numpy 2.5.2), accessed 2026-10-09
  (v2.5.2).
  Locator: https://raw.githubusercontent.com/numpy/numpy/v2.5.2/numpy/_core/_methods.py
  Used for: Explains the reader's cliff. Integer and bool inputs default to 'f8' and
  float16 to 'f4'. umr_sum then accumulates in the given dtype, and um.true_divide(ret,
  rcount, out=ret, casting='unsafe') casts the quotient back into it. So dtype=np.uint16
  sums in uint16, wraps, and divides the wrapped value.

- **[numpy-sum-docs]** NumPy Developers. numpy.sum. NumPy v2.5 Manual, accessed
  2026-10-09 (v2.5).
  Locator: https://numpy.org/doc/2.5/reference/generated/numpy.sum.html
  Used for: dtype is 'the accumulator in which the elements are summed'. Small unsigned
  inputs default to a platform-width unsigned accumulator, which is a widening default.
  'Arithmetic is modular when using integer types, and no error is raised on overflow'
  (example: np.ones(128, int8).sum(dtype=int8) gives -128).

- **[numpy-types-overflow]** NumPy Developers. Data types (section 'Overflow errors').
  NumPy v2.5 Manual, user guide, accessed 2026-10-09 (v2.5).
  Locator: https://numpy.org/doc/2.5/user/basics.types.html
  Used for: General statement that fixed-size NumPy integers overflow silently, unlike
  Python int (np.power(100, 9, dtype=np.int32) gives -1486618624), and that iinfo/finfo
  give the bounds. Useful background for 'the domain is a fixed range'.

- **[opencv-addweighted-docs]** OpenCV team. cv::addWeighted and cv::add (Operations on
  arrays). OpenCV 4.13.0 documentation, accessed 2026-10-09 (4.13.0).
  Locator: https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html
  Used for: dst = saturate(src1*alpha + src2*beta + gamma). With alpha + beta = 1 this
  is a convex combination, evaluated first and then saturated. 'Saturation is not
  applied when the output array has the depth CV_32S. You may even get result of an
  incorrect sign in the case of overflow.'

- **[opencv-intro]** OpenCV team. Introduction, section 'Saturation Arithmetics'. OpenCV
  4.13.0 documentation, accessed 2026-10-09 (4.13.0).
  Locator: https://docs.opencv.org/4.13.0/d1/dfb/intro.html
  Used for: The imaging library's policy for 8- and 16-bit pixels: storing 'the lowest 8
  (16) bits of the result' causes 'visual artifacts', so 'the so-called saturation
  arithmetics is used' (I = min(max(round(r), 0), 255)), and 'This semantics is used
  everywhere in the library.' This is saturate instead of wrap; it is not divide-first.

- **[opencv-motion-docs]** OpenCV team. cv::accumulate and cv::accumulateWeighted
  (Motion Analysis). OpenCV 4.13.0 documentation, accessed 2026-10-09 (4.13.0).
  Locator: https://docs.opencv.org/4.13.0/d7/df3/group__imgproc__motion.html
  Used for: Frame accumulation in OpenCV requires a widened accumulator: dst must have
  'a depth of CV_32F or CV_64F' for 8-bit input. accumulateWeighted 'Updates a running
  average' as dst = (1 - alpha) dst + alpha src, an online convex combination held in a
  float accumulator.

- **[opencv-saturate-docs]** OpenCV team. cv::saturate_cast (Utility and system
  functions and macros). OpenCV 4.13.0 documentation, accessed 2026-10-09 (4.13.0).
  Locator: https://docs.opencv.org/4.13.0/db/de0/group__core__utils.html
  Used for: An out-of-range value 'is not formed just by taking low bits of the input,
  but instead the value is clipped' for 8- and 16-bit targets. Floats are rounded and
  then clipped. 'For 32-bit integers, no clipping is done.' The saturation guarantee
  stops at 32 bits.

- **[pielawski-2020]** Nicolas Pielawski, Carolina Wählby. Introducing Hann windows for
  reducing edge-effects in patch-based image segmentation. PLOS ONE 15(3): e0229839,
  2020.
  Locator: doi:10.1371/journal.pone.0229839 (opened:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7067425/ , https://arxiv.org/abs/1910.07831
  and the Crossref record)
  Used for: Peer-reviewed overlap blending for inference that normalizes the weights
  first. Windows 'integrate to one (with a stride of half a window)', and special border
  and corner windows make 'the sum of all overlapping windows over the full image' equal
  1. The stated motive: 'We also avoid additional floating point errors by using windows
  that integrate to one'. Plain averaging yields 'four times as many artefacts'.
  window_sum_check.py tests the floating-point side of this claim.

- **[piroddi-petrou-2003]** Roberta Piroddi, Maria Petrou. Normalized Convolution: A
  Tutorial. CVonline, University of Edinburgh (School of Informatics), 2003 (page date).
  Locator:
  https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html
  Used for: Secondary source that states the normalized-convolution division in words:
  convolve the zero-filled signal and the certainty map with the same filter, then
  'divide the first convolution by the second one'. This is the sum-then-divide form of
  reconstruct and stitch.

- **[pytorch-accumulatetype]** PyTorch contributors. aten/src/ATen/AccumulateType.h.
  PyTorch source, tag v2.14.0, accessed 2026-10-09 (v2.14.0).
  Locator:
  https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/AccumulateType.h
  Used for: The widening answer, written down: intermediate computations run 'at a
  higher precision than the input and output precision', and acc_type 'is especially
  important for reduced precision operations like float16 and bfloat16, where relatively
  benign looking inputs can easily end up overflowing/underflowing'. Table: Half and
  BFloat16 accumulate in float; float in double on CPU and in float on CUDA; uint8 and
  the other integers in int64.

- **[pytorch-col2im]** PyTorch contributors. aten/src/ATen/native/im2col.h (col2im) and
  aten/src/ATen/native/Col2Im.cpp. PyTorch source, tag v2.14.0, accessed 2026-10-09
  (v2.14.0; identical to the headers in the installed 2.14.0+cpu wheel apart from a
  packaging guard).
  Locator:
  https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/im2col.h
  ;
  https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/Col2Im.cpp
  Used for: The CPU kernel behind F.fold adds patches into an output buffer of the
  tensor's own scalar_t ('data_im[...] += data_col[...]'), dispatched for kHalf and
  kBFloat16, with no acc_type. This explains why fold overflows in float16 (144 of 256
  pixels) while reductions such as torch.mean do not.

- **[pytorch-fold-docs]** PyTorch contributors. torch.nn.Fold. PyTorch 2.14
  documentation, accessed 2026-10-09 (v2.14.0).
  Locator: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html
  Used for: States that Fold combines overlapping values 'by summing all values from all
  containing blocks', and gives the normalization recipe: divisor =
  fold(unfold(input_ones)), with fold and unfold inverses 'up to constant divisor' when
  the divisor has no zero. This is the textbook sum-then-divide that PatchCraft's
  reconstruct implements.

- **[pytorch-istft-docs]** PyTorch contributors. torch.istft. PyTorch 2.14
  documentation, accessed 2026-10-09 (v2.14.0).
  Locator: https://docs.pytorch.org/docs/2.14/generated/torch.istft.html
  Used for: PyTorch's own overlap-add inverse checks the NOLA condition on the summed
  squared window envelope and names Griffin and Lim (1984) as its source. It is the
  audio counterpart of fold-then-divide.

- **[pytorch-numerical-accuracy]** PyTorch contributors. Numerical accuracy. PyTorch
  2.14 documentation (Notes), accessed 2026-10-09 (v2.14.0).
  Locator: https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html
  Used for: The 'Extremal values' section states the paper's premise: 'intermediate
  results may overflow the range of the used datatype, the end result may overflow too,
  even though it is representable in the original datatype'. Its example (norm of [1e20,
  1e20] gives inf, while double gives 1.4142e+20) reproduces on 2.14.0+cpu. The same
  page says half-precision GEMMs accumulate in single precision for 'improved resilience
  to overflow'.

- **[pytorch-opmathtype]** PyTorch contributors. aten/src/ATen/OpMathType.h. PyTorch
  source, tag v2.14.0, accessed 2026-10-09 (v2.14.0).
  Locator:
  https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/OpMathType.h
  Used for: One-line policy: 'For FP16 or BFloat16 inputs, ops should perform internal
  math in FP32.' Half, BFloat16 and the Float8 types map to float.

- **[pytorch-sum-docs]** PyTorch contributors. torch.sum. PyTorch 2.14 documentation,
  accessed 2026-10-09 (v2.14.0).
  Locator: https://docs.pytorch.org/docs/2.14/generated/torch.sum.html
  Used for: The documented user-level remedy is to widen: with dtype, 'the input tensor
  is casted to dtype before the operation is performed. This is useful for preventing
  data type overflows.'

- **[redbook-accpersp]** Silicon Graphics, Inc. (OpenGL Programming Guide example
  programs, as kept in the Mesa demos repository). accpersp.c ('Use the accumulation
  buffer to do full-scene antialiasing'). gitlab.freedesktop.org/mesa/demos,
  src/redbook, commit 649baedafcb90313ade69909fdef1ee156ab5f8d, 1993-1997 (copyright
  line).
  Locator:
  https://gitlab.freedesktop.org/mesa/demos/-/raw/649baedafcb90313ade69909fdef1ee156ab5f8d/src/redbook/accpersp.c
  Used for: The canonical graphics idiom is divide-first: inside a loop of ACSIZE (8)
  jittered renders, glAccum(GL_ACCUM, 1.0/ACSIZE), then glAccum(GL_RETURN, 1.0). Each
  pass is scaled by 1/n before it is added, so the accumulated value stays in [0, 1].
  The original Red Book site (glprogramming.com) is now a parked domain.

- **[scipy-check-nola]** SciPy developers. scipy.signal.check_NOLA. SciPy v1.18.0
  Manual, accessed 2026-10-09 (v1.18.0).
  Locator:
  https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.signal.check_NOLA.html
  Used for: Weighted overlap-add inversion divides by sum_t w^2[n - tH]. The NOLA
  constraint 'ensures that the normalization factors in the denominator of the
  overlap-add inversion equation are not zero'. This is the same positive-denominator
  condition stitch needs from its folded kernel.

- **[swinir-2021]** Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool,
  Radu Timofte. SwinIR: Image Restoration Using Swin Transformer. arXiv:2108.10257,
  2021.
  Locator: doi:10.48550/arXiv.2108.10257 (opened: https://arxiv.org/abs/2108.10257)
  Used for: Identifies the restoration model whose released test script does tiled
  inference. The abstract does not describe tiling, so the tiling is cited from
  swinir-code.

- **[swinir-code]** Jingyun Liang et al. (repository JingyunLiang/SwinIR).
  main_test_swinir.py, function test(). GitHub, commit
  6545850fbf8df298df73d81f3e8cba638787c8bd, accessed 2026-10-09.
  Locator:
  https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/main_test_swinir.py
  Used for: Overlapping-tile inference with uniform weights: E.add_(out_patch),
  W.add_(ones_like(out_patch)), output = E.div_(W), with E created by type_as(img_lq).
  The README says to use '--tile 400 if you run out-of-memory'. For the same geometry W
  equals fold(unfold(ones)), and the E/W round trip is not bit-exact when counts 3, 6 or
  9 occur (check_sources.py).

- **[szeliski-2006]** Richard Szeliski. Image Alignment and Stitching: A Tutorial.
  Foundations and Trends in Computer Graphics and Vision 2(1): 1-104 (opened as
  Microsoft Research Technical Report MSR-TR-2004-92, version of 10 Dec 2006), 2006 (TR
  version opened; Crossref dates the FnT issue 2007).
  Locator: doi:10.1561/0600000009 (opened:
  https://people.cs.umass.edu/~elm/Teaching/Docs/Image%20Align%20Szel.pdf and the
  Crossref record; the microsoft.com PDF returned 403)
  Used for: Panorama feathering is the same weighted average, C(x) = sum w_k I_k / sum
  w_k (TR eq. 153), and 'On computer graphics hardware, this kind of summation can be
  performed in an accumulation buffer (using the A channel as the weight)'. This links
  mosaicking to the accumulation buffer.

- **[wronski-2019]** Bartlomiej Wronski, Ignacio Garcia-Dorado, Manfred Ernst, Damien
  Kelly, Michael Krainin, Chia-Kai Liang, Marc Levoy, Peyman Milanfar. Handheld
  Multi-Frame Super-Resolution. ACM Transactions on Graphics 38(4), Article 28, 2019.
  Locator: doi:10.1145/3306346.3323024 (opened:
  https://storage.googleapis.com/gweb-research2023-media/pubtools/5211.pdf with the DOI
  printed, the research.google publication page and the Crossref record)
  Used for: A shipped online burst merge in the normalized-convolution form. Frames are
  processed sequentially; 'we accumulate weighted contributions and weights' and 'divide
  the accumulated color contributions by the accumulated weights' at the end (eq. 1: C =
  sum c w R / sum w R).
