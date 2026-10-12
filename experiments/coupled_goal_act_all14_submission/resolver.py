"""Byte-preserving historical-reference transport without path aliases on disk."""
import hashlib
from pathlib import Path

_VERIFIED={}


class Resolver:
    def __init__(self,root,bindings):
        self.root=Path(root).resolve(strict=True)
        self.bindings={};self.checked=_VERIFIED.setdefault(str(self.root),{})
        for row in bindings:
            original=row['original'];relative=Path(row['installed'])
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('Bounded relative installed path required')
            key=self.key(original)
            if key in self.bindings and self.bindings[key]!=str(relative):
                raise ValueError('Conflicting immutable historical reference')
            self.bindings[key]=str(relative)

    @staticmethod
    def key(row):
        if not {'path','bytes','sha256'}<=set(row) or type(row['bytes']) is not int or row['bytes']<0:
            raise ValueError('Exact original path/bytes/SHA reference required')
        if len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Literal SHA256 required')
        return (row['path'],row['bytes'],row['sha256'])

    def verify_ref(self,row):
        key=self.key(row)
        if key not in self.bindings:raise ValueError('Unbound original reference')
        path=self.root/self.bindings[key]
        for part in [path,*path.parents]:
            if part==self.root:break
            if part.is_symlink():raise ValueError('Symlink forbidden in installed asset path')
        if not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise ValueError('Missing or escaping installed asset')
        st=path.stat()
        if st.st_size!=key[1]:raise ValueError('Installed byte count differs')
        fingerprint=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if self.checked.get(key)!=fingerprint:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
            if h.hexdigest()!=key[2]:raise ValueError('Installed SHA256 differs')
            self.checked[key]=fingerprint
        return path

    def operational_ref(self,row):
        return dict(row,path=str(self.verify_ref(row)))
