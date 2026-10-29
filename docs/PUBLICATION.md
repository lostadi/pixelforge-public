# Source publication

The publication candidate starts from a reviewed source snapshot with no
historical parent commit. The source archive has its own explicit boundary,
file inventory and checksum receipt. Earlier device restoration history and
embedded runtime archives are outside this distribution.

Publish the clean candidate into an empty repository. Keep the original
device-bundle repository private: changing its visibility would also expose
its older branches, tags, references and release assets. A clean branch alone
does not remove that history.

Before distributing a generated source archive, verify it with:

```sh
python3 scripts/release_source.py --verify /path/to/pixelforge-source.tar.gz
```

The checker verifies the member boundary, source hashes, inventory and file
types before extraction. The adjacent `.sha256` file identifies the exact
archive bytes. Hardware evidence remains scoped to the recorded phone and
placement; another device needs its own runtime and model validation.
