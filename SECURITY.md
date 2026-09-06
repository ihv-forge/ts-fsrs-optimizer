# Security

This package computes numbers. It reads no files, opens no sockets, spawns no
processes and has no runtime dependencies, so the realistic attack surface is
small.

Worth reporting privately anyway:

- anything that makes it hang, allocate without bound, or crash the host on
  attacker-influenced input;
- a supply-chain problem with the published artifact — a tarball whose contents
  do not match this repository, or a failing provenance check.

Report through
[GitHub's private advisory form](https://github.com/ihv-forge/ts-fsrs-optimizer/security/advisories/new)
rather than a public issue.

Wrong _results_ are bugs, not vulnerabilities — open a normal issue with the
input that produced them.

Only the latest release is supported.
