# Third-Party Notices

Path-AI Scientist is source-available research software built around third-party code,
datasets, services, and document assets. This file is attribution information, not a replacement
for the applicable license texts.

## AI-Scientist-v2

- Project: The AI Scientist-v2, Sakana AI
- Source: https://github.com/SakanaAI/AI-Scientist-v2
- Local snapshot: `vendor/AI-Scientist-v2`
- Original baseline identity: `UPSTREAM_MANIFEST.sha256`
- Existing local patches: see `docs/SOURCE_PROVENANCE.md`; the release commit identifies the distributed tree
- License: The AI Scientist Source Code License v1.0
- Retained license: `vendor/AI-Scientist-v2/LICENSE`

The upstream license requires its complete license to accompany distributions, carries restricted
use provisions, prohibits medical diagnosis without human oversight, and requires prominent
disclosure when scientific manuscripts or reports are machine-generated or produced using The AI
Scientist. The repository root `LICENSE` applies those terms to this distributed derivative work.

The upstream README states that its tree-search component is built on the AIDE project:
https://github.com/WecoAI/aideml. Users redistributing or substantially modifying the vendored
tree-search code should also review AIDE's current notices and license at the source revision they
choose to use.

## MedMNIST and PathMNIST

- Project: MedMNIST / MedMNIST+
- Source: https://github.com/MedMNIST/MedMNIST
- Official data distribution: https://doi.org/10.5281/zenodo.10519652
- Dataset license: CC BY 4.0, except DermaMNIST, which is CC BY-NC 4.0
- MedMNIST code license: Apache-2.0

PathMNIST is covered by the MedMNIST CC BY 4.0 dataset terms. Dataset files are not distributed in
this source repository. Users must obtain them from the official distribution, retain attribution,
and cite MedMNIST and the PathMNIST source-data publications requested by the MedMNIST project.
MedMNIST states that the dataset is not intended for clinical use.

## LaTeX and example-paper assets

The vendored upstream tree contains ICML/ICLR templates, BibTeX styles, LaTeX style files, and
few-shot example papers. Copyright notices embedded in those files remain in effect. In
particular, `algorithm.sty`, `algorithmic.sty`, `natbib.sty`, the conference templates, and the
included example papers may have terms distinct from the repository-level software license.
Do not remove their embedded notices when redistributing the vendored snapshot.

## Hosted APIs and generated content

Optional model gateways and literature services are external services, not bundled components.
Their provider terms, model terms, rate limits, and content policies apply separately. API keys
must be supplied through the process environment and must never be committed. Generated papers,
code, and experiment outputs require human review before dissemination.
