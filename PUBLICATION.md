# Publication and release preparation

## F1000Research software-availability statement (draft)

> The CLASP source code corresponding to this article is available from
> https://github.com/Sabya2/ClASP/releases/tag/v1.0.0. The repository cites the
> archived release at https://doi.org/10.5281/zenodo.23976795. Synthetic
> validation data and automated tests are included in the repository. The
> software licence is **[LICENCE — author/rightsholder decision required]**.

Before submission, verify that the DOI resolves to the exact tagged source used
for the article and replace the remaining licence placeholder after the
author/rightsholder decision.

## Release procedure

1. Resolve the licence/rightsholder decision and update `CITATION.cff`, the README,
   and the statement above. Do not infer a licence from repository visibility.
2. Replace the development version in `srcs/version.py`, `CITATION.cff`, and
   `CHANGELOG.md` with the same release version and date.
3. Run `python -m pip install -r requirements-dev.txt`, `pytest -q`, and
   `python -m compileall -q public_app.py pages srcs` in a clean supported Python
   environment. Validate citation metadata in an isolated environment installed
   from `requirements-metadata.txt` with `cffconvert --validate`.
4. Verify that the existing `v1.0.0` tag points to the exact source archived
   for the publication record. Do not move an archived tag.
5. Verify that the Zenodo record is the version-specific archive for `v1.0.0`;
   use the version DOI rather than the concept DOI when citing the exact software.
6. Add the confirmed licence and final citation metadata, then verify that every
   link resolves to the same released source.

GitHub release `v1.0.0` and DOI `10.5281/zenodo.23976795` exist. Licence
approval remains an author/rightsholder decision.

## Metadata inventory and decisions still required

- Repository history identifies Sabyasachi Chakrabarty as the author. Co-author,
  affiliation, ORCID, and preferred contact details require author confirmation.
- GitHub release `v1.0.0` was published on 5 November 2026.
- No current licence file was present. Distribution rights and licence choice must
  be confirmed by the author/rightsholder before a licence file is added.
- Version `1.0.0` identifies the public release used for the publication record.
