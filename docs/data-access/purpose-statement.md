# Research purpose statement

Paste this into the "purpose of your research / proposed duration" box on both
forms. It is deliberately specific: the request is evaluated by a person, and a
vague statement ("machine learning research") is the usual reason these get
queried or refused.

---

**Purpose.** Development and evaluation of compact semantic segmentation models
for the ocular region in visible-light imagery, as the perception layer of a
non-commercial research project on webcam-based vision screening
(https://github.com/aaryanpatil2007/vision-screen).

Two specific uses:

1. **Benchmarking under the SSBC protocol.** We have trained a 284,000-parameter
   encoder–decoder that matches a published DeepLabV3-ResNet101 baseline on the
   periorbital segmentation dataset (Nahass et al., Ophthalmology Science 2025)
   at 0.47% of its parameter count. We wish to evaluate the same architecture on
   MOBIUS under the SSBC 2020 / SSRBC 2023 sclera protocol, and on SBVPI, so the
   result can be compared against a benchmark with published competitive
   entries rather than against a single baseline.

2. **A four-class benchmark on MOBIUS.** MOBIUS carries sclera / iris / pupil /
   periocular annotations for 3,559 images, but to our knowledge no paper
   publishes a clean four-class mIoU on it. We intend to establish and publish
   that figure, with code and evaluation protocol released so it is reproducible.

Our interest is specifically in **sclera**, which is the weakest class in every
eye-segmentation benchmark we have surveyed (0.674 IoU for the OpenEDS 2020
baseline; 0.074 for zero-shot SAM 2), because its boundary with the eyelid is a
shadow rather than an edge.

**Duration of processing.** 12 months from the date of access, after which the
data will be deleted unless an extension is requested.

**Non-commercial.** This is an unfunded personal research project. No commercial
use is intended or made. Any change in that status would be brought back to the
provider for prior authorisation, per clause 4.

**Safeguards.** Data held on encrypted local storage on a single machine, not
copied to any shared or cloud location, not committed to the project's public
repository (the repository's .gitignore excludes all dataset directories), and
not redistributed in whole or in part. For MOBIUS specifically, any figure in a
publication will use scaled-down or watermarked images such that individuals
cannot be identified, per clause 3.
