import operator
import numpy as np
import xarray as xr
from ..common.hashing import md5encode


ATTR_COORDS_HASH = "coords_hash"


class LazyDataArray:
  def __init__(self, values, dims, coords, attrs=None, data=None):
    self.__values = values
    self.__variable = None
    self.__dims = dims
    self.__coords = coords
    self.__attrs = attrs or {}
    self.__data = data
    self.__coord_to_index = {}

  @property
  def dims(self):
    return self.__dims

  @property
  def coords(self):
    return self.__coords

  @property
  def attrs(self):
    return self.__attrs

  @property
  def values(self):
    return self.__values

  @property
  def variable(self):
    if self.__variable is None:
      if self.__data is not None:
        self.__variable = self.__data.variable
      else:
        self.__variable = xr.Variable(self.dims, self.values, attrs=self.attrs)
    return self.__variable

  def isel(self, query=None, drop=False, **more_query):
    if query is None:
      query = {}
    query = dict(query, **more_query)
    new_coords = {}
    new_dims = list(self.dims)
    index_query = [slice(None, None, None)] * len(self.dims)
    for key, value in query.items():
      if key not in self.coords:
        raise KeyError(f"No coord named `{key}`")
      coord = self.coords[key]
      if coord.dims != (key, ):
        raise ValueError("unsupported selection")
      dim = key
      dim_idx = self.dims.index(key)
      if isinstance(value, slice):
        if value.step is not None:
          raise ValueError("unsupported step")
        index_query[dim_idx] = value
        new_coords[dim] = self.coords.variables[dim][value]
      elif isinstance(value, list):
        index_query[dim_idx] = value
        new_coords[dim] = xr.Variable((dim, ), value)
      else:
        index_query[dim_idx] = value
        if not drop:
          new_coords[dim] = xr.Variable((), self.coords.variables[dim][value])
        new_dims[dim_idx] = None
    values = self.values.__getitem__(tuple(index_query))
    return LazyDataArray(values, [dim for dim in new_dims if dim is not None], coords=new_coords, attrs=self.attrs)

  def sel(self, query=None, drop=False, **more_query):
    if query is None:
      query = {}
    query = dict(query, **more_query)
    new_coords = {}
    new_dims = list(self.dims)
    index_query = [slice(None, None, None)] * len(self.dims)
    for key, value in query.items():
      if key not in self.coords:
        raise KeyError(f"No coord named `{key}`")
      coord = self.coords[key]
      if coord.dims != (key, ):
        raise ValueError("unsupported selection")
      dim = key
      dim_idx = self.dims.index(key)
      if isinstance(value, slice):
        start = self._get_coord_idx(dim, value.start) if value.start is not None else None
        stop = self._get_coord_idx(dim, value.stop) if value.stop is not None else None
        if value.step is not None:
          raise ValueError("unsupported step")
        index_query[dim_idx] = slice(start, stop)
        new_coords[dim] = self.coords.variables[dim][slice(start, stop)]
      elif isinstance(value, list):
        index_query[dim_idx] = [self._get_coord_idx(dim, item) for item in value]
        new_coords[dim] = xr.Variable((dim, ), value)
      else:
        index_query[dim_idx] = self._get_coord_idx(dim, value)
        if not drop:
          new_coords[dim] = xr.Variable((), value)
        new_dims[dim_idx] = None
    values = self.values.__getitem__(tuple(index_query))
    return LazyDataArray(values, [dim for dim in new_dims if dim is not None], coords=new_coords, attrs=self.attrs)

  def mean(self, dim=None):
    return self._reduce(np.nanmean, dim)

  def sum(self, dim=None):
    return self._reduce(np.nansum, dim)

  def std(self, dim=None):
    return self._reduce(np.nanstd, dim)

  def var(self, dim=None):
    return self._reduce(np.nanvar, dim)

  def min(self, dim=None):
    return self._reduce(np.nanmin, dim)

  def max(self, dim=None):
    return self._reduce(np.nanmax, dim)

  def median(self, dim=None):
    return self._reduce(np.nanmedian, dim)

  def argmax(self, dim=None):
    return self._reduce(np.nanargmax, dim)

  def argmin(self, dim=None):
    return self._reduce(np.nanargmin, dim)

  def all(self, dim=None):
    return self._reduce(np.all, dim)

  def any(self, dim=None):
    return self._reduce(np.any, dim)

  def cumsum(self, dim=None):
    # TODO: not reduce
    return self._reduce(np.nancumsum, dim)

  def cumprod(self, dim=None):
    # TODO: not reduce
    return self._reduce(np.nancumprod, dim)

  def rolling(self, *args, **kwargs):
    return self.to_xarray().rolling(*args, **kwargs)

  def rolling_exp(self, *args, **kwargs):
    return self.to_xarray().rolling_exp(*args, **kwargs)

  def __add__(self, other):
    return _binary(operator.add, self, other)

  def __radd__(self, other):
    return _binary(operator.add, other, self)

  def __sub__(self, other):
    return _binary(operator.sub, self, other)

  def __rsub__(self, other):
    return _binary(operator.sub, other, self)

  def __mul__(self, other):
    return _binary(operator.mul, self, other)

  def __rmul__(self, other):
    return _binary(operator.mul, other, self)

  def __truediv__(self, other):
    return _binary(operator.truediv, self, other)

  def __rtruediv__(self, other):
    return _binary(operator.truediv, other, self)

  def __pow__(self, power, modulo=None):
    return _binary(operator.pow, self, power)

  def __rpow__(self, other):
    return _binary(operator.pow, other, self)

  def __and__(self, other):
    return _binary(operator.and_, self, other)

  def __rand__(self, other):
    return _binary(operator.and_, other, self)

  def __or__(self, other):
    return _binary(operator.or_, self, other)

  def __ror__(self, other):
    return _binary(operator.or_, other, self)

  def __abs__(self):
    return _unary(operator.abs, self)

  def __pos__(self):
    return _unary(operator.pos, self)

  def __neg__(self):
    return _unary(operator.neg, self)

  def __invert__(self):
    return _unary(operator.invert, self)

  def to_xarray(self):
    if self.__data is None:
      from xarray.core.merge import _create_indexes_from_coords
      variable = self.variable
      indexes, coords = _create_indexes_from_coords(self.coords)
      self.__data = xr.DataArray(variable, coords=coords, indexes=indexes, fastpath=True)
    return self.__data

  def _get_coord_idx(self, dim, value):
    if dim not in self.__coord_to_index:
      self.__coord_to_index[dim] = {value: i for i, value in enumerate(self.coords[dim].data)}
    return self.__coord_to_index[dim][value]

  def _reduce(self, reducer, dim=None):
    if dim is None:
      values = reducer(self.values)
      dims = []
      coords = {}
      attrs = dict(self.attrs, **{ATTR_COORDS_HASH: {}})
    else:
      dim_i = self.dims.index(dim)
      values = reducer(self.values, axis=dim_i)
      dims = [d for d in self.dims if d != dim]
      coords = {}
      for key, value in self.coords.items():
        if dim not in value.dims:
          coords[key] = value
      if ATTR_COORDS_HASH in self.attrs:
        attr_hash = {key: value for key, value in self.attrs[ATTR_COORDS_HASH].items() if key != dim}
        attrs = dict(self.attrs, **{ATTR_COORDS_HASH: attr_hash})
      else:
        attrs = self.attrs
    return LazyDataArray(values, dims, coords, attrs=attrs)

  @classmethod
  def from_dataarray(cls, data):
    attrs = data.attrs
    if ATTR_COORDS_HASH not in attrs:
      attrs = dict(attrs, **{ATTR_COORDS_HASH: {key: md5encode(value.data.tobytes()) for key, value in data.coords.items()}})
    return cls(data.values, data.dims, data._coords, attrs, data=data)


def eager(x):
  if isinstance(x, LazyDataArray):
    return x.to_xarray()
  return x


def lazy(x):
  if isinstance(x, xr.DataArray):
    return LazyDataArray.from_dataarray(x)
  return x


def _binary(op, left, right):
  left_is_array = isinstance(left, (xr.DataArray, LazyDataArray))
  right_is_array = isinstance(right, (xr.DataArray, LazyDataArray))
  if left_is_array and right_is_array:
    left_attr_hash = left.attrs.get(ATTR_COORDS_HASH, {})
    right_attr_hash = right.attrs.get(ATTR_COORDS_HASH, {})
    if left.dims == right.dims and left_attr_hash and left_attr_hash == right_attr_hash:
      return LazyDataArray(op(left.values, right.values), left.dims, left.coords, left.attrs)
    else:
      common_dims = set(left.dims) & set(right.dims)
      left_attr_hash = {key: value for key, value in left_attr_hash.items() if key in common_dims}
      right_attr_hash = {key: value for key, value in right_attr_hash.items() if key in common_dims}
      if left_attr_hash == right_attr_hash:
        result_variable = op(left.variable, right.variable)
        attrs = {ATTR_COORDS_HASH: dict(left_attr_hash, **right_attr_hash)}
        return LazyDataArray(result_variable.values, result_variable.dims, dict(left.attrs, **right.attrs), attrs)
  elif left_is_array:
    return LazyDataArray(op(left.values, right), left.dims, left.coords, left.attrs)
  elif right_is_array:
    return LazyDataArray(op(left, right.values), right.dims, right.coords, right.attrs)
  return op(eager(left), eager(right))


def _unary(op, value):
  return LazyDataArray(op(value.values), value.dims, value.coords, value.attrs)
