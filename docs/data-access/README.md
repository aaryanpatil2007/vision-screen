# Getting SBVPI and MOBIUS

Both are held by the Computer Vision Laboratory at the University of Ljubljana.
Neither has a download link: access requires a **hand-signed** form emailed to
the maintainer, so this step cannot be automated and has to be done by a person.

**Forms** (Google Docs — open, File → Download):
- SBVPI: https://docs.google.com/document/d/1HhR0T5qhzipRxUeDspZmqWiR5OGGT10yTnPDfnfQeBw
- MOBIUS: https://docs.google.com/document/d/17-wLf8cPXKfBIS2gRRTJJ0XjZUi_1CkoEW3yhi_KlFc

Local copies of the form text are in `sbvpi-form.txt` and `mobius-form.txt` so
you can see what they ask before opening anything.

**Steps**
1. Fill in name, organisation, address, country, e-mail, signatory.
2. Paste `purpose-statement.md` into the purpose/duration box.
3. Hand-sign and date both.
4. Email both as PDFs to matej.vitek@fri.uni-lj.si — draft in `EMAIL-DRAFT.md`.
5. For MOBIUS, ask for the **segmentation subset (~700 MB)**; the full 3.5 GB
   set adds recognition imagery this project has no use for.

**Terms that affect how the code must be written**, not just what you promise:
- non-commercial research only, for the stated purpose
- no redistribution of the data, its parts, or the download link — so these stay
  under `data/`, which `.gitignore` already excludes
- MOBIUS figures in any publication must be scaled down or watermarked so
  individuals cannot be identified
- three papers per dataset must be cited, and the provider gets copies of any
  publication

**Why bother.** The periorbital result beats a single off-the-shelf baseline.
These two carry published competitive entries — Sclera-TransFuse at 93.59 mIoU
/ 96.66 F1 on SBVPI, and the SSBC 2020 winner at F1 0.868 on MOBIUS — so a
result there is a comparison against a field rather than against one paper.
