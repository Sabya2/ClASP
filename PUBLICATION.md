# Publication and release preparation

## F1000Research software-availability statement (draft)

> The CLASP source code corresponding to this article will be available from
> **[RELEASE URL — insert after the tagged GitHub release exists]**. The exact
> release will be archived in Zenodo at **[ZENODO DOI — insert after archiving]**.
> Synthetic validation data and automated tests are included in the repository.
> The software licence is **[LICENCE — author/rightsholder decision required]**.

The placeholders must not be replaced until the corresponding release, archive,
and rights decisions actually exist.

## Release procedure

1. Resolve the licence/rightsholder decision and update `CITATION.cff`, the README,
   and the statement above. Do not infer a licence from repository visibility.
2. Replace the development version in `srcs/version.py`, `CITATION.cff`, and
   `CHANGELOG.md` with the same release version and date.
3. Run `python -m pip install -r requirements-dev.txt`, `pytest -q`, and
   `python -m compileall -q public_app.py pages srcs` in a clean supported Python
   environment. Validate citation metadata in an isolated environment installed
   from `requirements-metadata.txt` with `cffconvert --validate`.
4. Commit the reviewed state, create an annotated tag (for example `v0.1.0`) on
   that exact commit, and publish a GitHub release from the tag.
5. Connect the GitHub repository to Zenodo, archive that exact tagged release,
   and record the version-specific DOI. Do not use the concept DOI where an exact
   software-version citation is required.
6. Add the real release URL and DOI to `CITATION.cff`, this statement, and the
   manuscript, then verify that they resolve to the tagged source.

No release, DOI, licence approval, repository-visibility change, or Zenodo upload
has been performed as part of this preparation.

## Metadata inventory and decisions still required

- Repository history identifies Sabyasachi Chakrabarty as the author. Co-author,
  affiliation, ORCID, and preferred contact details require author confirmation.
- No Git tag or existing release was present during this review.
- No current licence file was present. Distribution rights and licence choice must
  be confirmed by the author/rightsholder before a licence file is added.
- Version `0.1.0-dev` identifies the reviewed development state; the release number
  remains an author/maintainer decision.
