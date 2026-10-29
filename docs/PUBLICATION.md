# Source publication

This public repository starts from a reviewed source snapshot with no
historical parent commit. The source archive has its own explicit boundary,
file inventory and checksum receipt. Earlier device restoration history and
embedded runtime archives are outside this distribution.

The original device-bundle repository remains private. This repository has
no ancestry from that private history. Public releases are built from the
reviewed commits here, with model weights and device runtime inputs supplied
separately.

Before distributing a generated source archive, verify it with:

```sh
python3 scripts/release_source.py --verify /path/to/pixelforge-source.tar.gz
```

The checker verifies the member boundary, source hashes, inventory and file
types before extraction. The adjacent `.sha256` file identifies the exact
archive bytes. Hardware evidence remains scoped to the recorded phone and
placement; another device needs its own runtime and model validation.
