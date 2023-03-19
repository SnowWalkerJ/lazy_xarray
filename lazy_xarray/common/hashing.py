import hashlib


__all__ = ["md5encode"]


def md5encode(data):
  if isinstance(data, str):
    data = data.encode()
  return hashlib.md5(data).hexdigest()
