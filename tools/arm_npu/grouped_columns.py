"""Original quantized column tiles with NPU partials and ordered F32 sums.

The original GGUF and native allowlist stay unchanged. Only internal aliases
name disjoint 3072-column block slices. Only a split tail is zero-extended to
the proven 3072-column geometry; unsplit narrow matrices remain unpadded.
"""
import numpy as np

from author_grouped_quant import unpack_groups
from compiled_model import canonical
from dynamic_fc import DynamicProgramCache
from grouped_fc import GroupedProgramCache, GroupedWeightExecutor, GroupedPhaseExecutor


def column_ranges(columns):
    if type(columns) is not int or columns < 256 or columns % 256:
        raise ValueError('Column banking requires whole original 256-value blocks')
    return [(start,min(start+3072,columns)) for start in range(0,columns,3072)]


class ColumnLayout:
    """Canonical aliases derived exclusively from the pinned original plan."""
    def __init__(self, plan):
        self.originals, self.aliases, self.bindings, self.layers = {}, {}, {}, {}
        for item in plan.get('dense_tensors', []):
            name = item.get('name')
            if not isinstance(name, str) or not name or name in self.originals:
                raise ValueError('Invalid or duplicate original column source')
            self.originals[name] = dict(item)
        for name, item in self.originals.items():
            if item.get('type_id') not in (13, 14):
                continue
            rows, columns, bank = item.get('rows'), item.get('columns'), item.get('bank_rows')
            if any(type(value) is not int or value < 1 for value in (rows, columns, bank)) or bank > rows:
                raise ValueError('Invalid original column source dimensions')
            spans = column_ranges(columns)
            aliases = []
            for start, stop in spans:
                alias = name if len(spans) == 1 else f'{name}::cols[{start}:{stop}]'
                if alias in self.bindings or (alias != name and alias in self.originals):
                    raise ValueError('Column alias collides with another source tensor')
                self.bindings[alias] = (name, start, stop)
                physical = 3072 if len(spans)>1 else stop-start
                self.layers[alias] = dict(item, name=alias, columns=physical,logical_columns=stop-start,
                    source_tensor=name, column_start=start, column_stop=stop)
                aliases.append(alias)
            self.aliases[name] = aliases

    def aliases_for(self, name):
        try:
            return list(self.aliases[name])
        except (KeyError, TypeError) as error:
            raise ValueError('Column source is outside the pinned original Q5_K/Q6_K plan') from error

    def binding(self, alias):
        try:
            return self.bindings[alias]
        except (KeyError, TypeError) as error:
            raise ValueError('Unknown or noncanonical original column alias') from error

    def layer(self, alias):
        self.binding(alias)
        return dict(self.layers[alias])


class ColumnProgramCache(GroupedProgramCache):
    def __init__(self, directory, source_gguf=None, **kwargs):
        DynamicProgramCache.__init__(self, directory, source_gguf,
            _program_kind='grouped-columns-v1', **kwargs)
        self.layout = ColumnLayout(self.source_plan)

    def aliases_for(self, name):return self.layout.aliases_for(name)
    def binding(self, alias):return self.layout.binding(alias)
    def layer(self, alias):return self.layout.layer(alias)

    def mark_complete(self, weight_dtype='original_groups'):
        # The manifest deliberately retains original names/dimensions; validate
        # every internal tile instead of the original full-width matrix key.
        if not self.layout.aliases:
            raise ValueError('Cannot mark an empty original column plan complete')
        for name in self.layout.aliases:
            for alias in self.aliases_for(name):
                layer = self.layer(alias)
                for start, stop in self.ranges(alias):
                    if self.get(alias, start, stop, weight_dtype, layer['columns'], load_code=False) is None:
                        raise ValueError('Cannot mark incomplete column geometry complete')
        def complete(value):
            dtypes = set(value.get('completed_weight_dtypes', []))
            dtypes.add(weight_dtype)
            value.update(complete=True, completed_weight_dtypes=sorted(dtypes))
        self._update(complete)


class ColumnWeightExecutor(GroupedWeightExecutor):
    mode_name = 'original_columns'

    def _weight_arrays(self, alias, start, stop):
        name, first, last = self.cache.binding(alias)
        item = self.cache.layout.originals[name]
        tensor = self.weights.tensors[name]
        kind = self.cache.group_kind(alias)
        if tensor.shape != (item['rows'], item['columns']) or tensor.type_id != kind:
            raise ValueError('Original column source differs from its pinned matrix')
        if (start, stop) not in self.cache.ranges(alias):
            raise ValueError('Noncanonical original column row bank')
        block_bytes = 176 if kind == 13 else 210
        # Slice both dimensions BEFORE the only contiguous raw copy. Never
        # decode a full-width F32 matrix or create a second full-width DMA bank.
        raw = self.weights.raw(name).reshape(tensor.shape[0], tensor.shape[1]//256, block_bytes)
        raw = np.ascontiguousarray(raw[start:stop, first//256:last//256]).reshape(-1, block_bytes)
        levels, scale, offset = unpack_groups(raw, kind)
        rows, columns, group = stop-start, last-first, 32 if kind == 13 else 16
        result = [np.ascontiguousarray(levels.reshape(rows,columns//group,group),dtype=np.int8),
                  np.ascontiguousarray(scale.reshape(rows,columns//group,1),dtype='<f4')]
        if offset is not None:
            result.append(np.ascontiguousarray(offset.reshape(rows,columns//group,1),dtype='<f4'))
        physical=self.cache.layer(alias)['columns']
        if physical>columns:
            padded=[]
            for array in result:
                value=np.zeros((rows,physical//group,array.shape[2]),dtype=array.dtype)
                value[:,:columns//group,:]=array
                padded.append(value)
            result=padded
        return result


class ColumnPhaseExecutor(GroupedPhaseExecutor):
    def __init__(self, native, weights, prefill_cache, decode_cache):
        if (prefill_cache.source_sha256 != decode_cache.source_sha256 or
                canonical(prefill_cache.source_plan) != canonical(decode_cache.source_plan) or
                prefill_cache.source_identity != decode_cache.source_identity or
                tuple(weights.source_identity) != prefill_cache.source_identity):
            raise ValueError('Column phase caches must share original source and row/column layout')
        self.native = native
        self.prefill = ColumnWeightExecutor(native, weights, prefill_cache)
        self.decode = self.prefill if decode_cache is prefill_cache else ColumnWeightExecutor(native,weights,decode_cache)
        self.prefill_single_vector_operations = 0
        self.logical_projection_operations = self.column_partial_matmuls = self.cpu_partial_additions = 0
        self.column_padded_partial_matmuls = self.column_padded_activation_values = 0

    def __call__(self, name, activation, *, prefill=False):
        try:
            aliases = self.prefill.cache.aliases_for(name)
            original = self.prefill.cache.layout.originals[name]
            x = np.asarray(activation,dtype=np.float32)
            if x.ndim != 2 or len(x)<1 or x.shape[1]!=original['columns'] or not np.isfinite(x).all():
                raise ValueError('Invalid original column projection activation')
            accumulator = None
            for alias in aliases:
                source, first, last = self.prefill.cache.binding(alias)
                if source != name:raise ValueError('Column alias belongs to another original tensor')
                physical=self.prefill.cache.layer(alias)['columns']
                tile=x[:,first:last]
                if physical>last-first:
                    tile=np.zeros((len(x),physical),dtype=np.float32)
                    tile[:,:last-first]=x[:,first:last]
                partial = GroupedPhaseExecutor.__call__(self, alias, tile,prefill=prefill)
                if partial.dtype!=np.float32 or partial.shape!=(len(x),original['rows']) or not np.isfinite(partial).all():
                    raise ValueError('Invalid NPU column partial result')
                self.column_partial_matmuls += 1
                if physical>last-first:
                    self.column_padded_partial_matmuls += 1
                    self.column_padded_activation_values += len(x)*(physical-(last-first))
                if accumulator is None:
                    accumulator = partial
                else:
                    np.add(accumulator,partial,out=accumulator)
                    self.cpu_partial_additions += 1
                    if not np.isfinite(accumulator).all():raise ValueError('Nonfinite F32 column partial sum')
            self.logical_projection_operations += 1
            return accumulator
        except BaseException:
            self.native.close()
            raise

    def stats(self):
        return dict(super().stats(),dynamic_weight_mode='original_columns',column_bank_maximum=3072,
            logical_projection_operations=self.logical_projection_operations,
            column_partial_matmuls=self.column_partial_matmuls,cpu_partial_additions=self.cpu_partial_additions,
            column_padded_partial_matmuls=self.column_padded_partial_matmuls,
            column_padded_activation_values=self.column_padded_activation_values,
            column_zero_padding='split tails only: zero activation, original-code and effective-coefficient extension to 3072',
            cpu_partial_sum='ordered float32 additions of NPU matrix partials; no CPU weight matmul')
