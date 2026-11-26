# Domain context

ToOp's domain language (N-1, contingency, outage group, relevant substation, split, action set, …) is defined once,
for humans and agents, in the published glossary: [`docs/glossary.md`](docs/glossary.md).

This file deliberately contains no definitions, so there is only one place to keep up to date.

Rules:

- Use the glossary terms and the code identifiers it lists. Do not introduce synonyms (e.g. `outage` vs `failure`
  vs `contingency`) without checking the glossary first.
- If you introduce, rename or sharpen a domain term, update `docs/glossary.md` in the same change.
- For how the pieces fit together, see the architecture model in `docs/architecture/README.md`.
