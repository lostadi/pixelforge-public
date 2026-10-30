"""Versioned, lossless-at-authoring gate/up stacking and bounded result reuse.

Only projections with exactly the same activation can share a fused result.
The GGUF is never modified; source row values and ordering remain explicit in
the separately pinned plan. No activation, gate function or CPU matmul is used
to synthesize a model result here.
"""
from copy import deepcopy
import hashlib
import math
from types import SimpleNamespace

import numpy as np


class FusedFFNWeights:
    def __init__(self, weights, plan, *, bank_rows=4096, selected=None):
        if type(bank_rows) is not int or bank_rows < 1:
            raise ValueError("Fused bank rows must be positive")
        self.base = weights
        self.tensors = dict(weights.tensors)
        self.fused_sources = {}
        original = {item['name']: item for item in plan['dense_tensors']}
        dense = []
        for gate, item in sorted(original.items()):
            if not gate.endswith('.ffn_gate.weight'):
                continue
            up = gate[:-len('ffn_gate.weight')] + 'ffn_up.weight'
            other = original.get(up)
            if other is None or (item['rows'], item['columns']) != (other['rows'], other['columns']):
                continue
            name = gate[:-len('ffn_gate.weight')] + 'ffn_gate_up.weight'
            if name in self.tensors or len(name.encode()) > 63:
                raise ValueError("Fused tensor name collides or exceeds protocol limit")
            rows, columns = item['rows'] * 2, item['columns']
            self.fused_sources[name] = (gate, up)
            self.tensors[name] = SimpleNamespace(shape=(rows, columns), type_id=0)
            dense.append({'name': name, 'rows': rows, 'columns': columns,
                          'bank_rows': min(bank_rows, rows), 'graphs': math.ceil(rows/bank_rows),
                          'type_id': 0, 'role': 'stacked-gate-up'})
            down = gate[:-len('ffn_gate.weight')] + 'ffn_down.weight'
            if down in original:
                direct = dict(original[down])
                direct.update(bank_rows=min(bank_rows, direct['rows']),
                              graphs=math.ceil(direct['rows']/bank_rows), role='direct-ffn-down')
                dense.append(direct)
        if selected is not None and (not isinstance(selected, list) or not selected
                or any(name not in self.fused_sources for name in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Fused selection must name distinct planned gate/up tensors")
        chosen = set(self.fused_sources if selected is None else selected)
        self.runtime_fused_names = chosen | {
            name[:-len('ffn_gate_up.weight')] + 'ffn_down.weight' for name in chosen}
        self.plan = deepcopy(plan)
        self.plan.update(format='pixelforge.gguf.fused-ffn-plan.v1', dense_tensors=dense,
                         fused_source_tensors={k: list(v) for k, v in self.fused_sources.items()},
                         dense_graphs=sum(x['graphs'] for x in dense),
                         dense_parameters=sum(x['rows']*x['columns'] for x in dense))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def numpy_rows(self, name, start, stop):
        if name not in self.fused_sources:
            return self.base.numpy_rows(name, start, stop)
        gate, up = self.fused_sources[name]
        rows = self.base.tensors[gate].shape[0]
        if not 0 <= start < stop <= rows * 2:
            raise ValueError("Invalid fused row range")
        parts = []
        if start < rows:
            parts.append(self.base.numpy_rows(gate, start, min(stop, rows)))
        if stop > rows:
            parts.append(self.base.numpy_rows(up, max(start-rows, 0), stop-rows))
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


class FusedFFNDispatcher:
    """Reuse only the sibling projection of the most recent fused operation."""
    def __init__(self, native, weights, *, cache_mib=64, selected=None, prefill_only=False):
        if type(cache_mib) is not int or not 0 <= cache_mib <= 256:
            raise ValueError("Fused result cache must be from 0 to 256 MiB")
        self.native = native
        if type(prefill_only) is not bool:
            raise ValueError("prefill_only must be a boolean")
        self.prefill_only = prefill_only
        if selected is not None and (not isinstance(selected, list) or not selected
                or any(name not in weights.fused_sources for name in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Fused selection must name distinct planned gate/up tensors")
        self.pairs = {member: (name, index, members, weights.base.tensors[member].shape[0])
                      for name, members in weights.fused_sources.items()
                      if selected is None or name in selected
                      for index, member in enumerate(members)}
        self.budget = cache_mib * 1024**2
        self.pending = None
        self.computed = self.reused = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, name, activation, *, prefill=False):
        checker = getattr(self.native.weights, 'check_source_unchanged', None)
        if checker is not None:
            checker(check_path=True)
        item = self.pairs.get(name)
        if item is None or (self.prefill_only and not prefill):
            self.pending = None
            return self.native(name, activation, prefill=prefill)
        data = np.ascontiguousarray(activation, dtype=np.float32)
        identity = (data.shape, hashlib.sha256(memoryview(data).cast('B')).digest(), prefill)
        if self.pending is not None:
            other, expected, result = self.pending
            self.pending = None
            if other == name and expected == identity:
                self.reused += 1
                return result
        virtual, index, members, rows = item
        self.native._make_room(estimate=0,
                              transient=data.shape[0] * (data.shape[1]*12 + rows*28))
        output = self.native(virtual, data, prefill=prefill)
        self.computed += 1
        if output.shape != (data.shape[0], rows * 2):
            raise ValueError("Invalid fused output dimensions")
        result = output[:, index*rows:(index+1)*rows].copy()
        sibling = output[:, (1-index)*rows:(2-index)*rows]
        if sibling.nbytes <= self.budget:
            self.pending = (members[1-index], identity, sibling.copy())
        return result

    def stats(self):
        return dict(self.native.stats(), fused_gate_up_operations=self.computed,
                    fused_sibling_reuses=self.reused,
                    fused_pending_bytes=0 if self.pending is None else self.pending[2].nbytes)
