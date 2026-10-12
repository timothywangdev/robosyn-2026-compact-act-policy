"""Transport literal Git-LFS pointer bytes without treating them as LFS objects."""
import base64
import hashlib
import json
from pathlib import Path

PREFIX=b'version https://git-lfs.github.com/spec/v1\n'
MANIFEST='.transport/git-lfs-pointer-manifest-v1.json'


def raw_ref(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def json_bytes(body):
    return (json.dumps(body,sort_keys=True,indent=2)+'\n').encode()


def safe_path(root,name):
    root=Path(root).resolve(strict=True);name=Path(name);target=root/name
    if name.is_absolute() or '..' in name.parts or target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Unsafe transport or restoration path')
    return target


def find_pointers(root,spec):
    rows={}
    for name,row in spec['files'].items():
        if row['bytes']>1024:continue
        raw=safe_path(root,name).read_bytes()
        if raw.startswith(PREFIX):
            if raw_ref(raw)!=row: raise ValueError('Frozen pointer fixture differs')
            rows[name]=raw
    return rows


def encode(root,stage,spec):
    pointers=find_pointers(root,spec);entries={};expected={}
    for name,raw in pointers.items():
        encoded_name='.transport/git-lfs-pointer-json/'+hashlib.sha256(name.encode()).hexdigest()+'.json'
        body=dict(schema='byte_exact_original_Git_LFS_pointer_transport_v1',original_path=name,
            original=raw_ref(raw),payload_base64=base64.b64encode(raw).decode('ascii'))
        encoded=json_bytes(body);target=safe_path(stage,encoded_name);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(encoded)
        entries[name]=dict(original=raw_ref(raw),transport_path=encoded_name,transport=raw_ref(encoded))
        expected[encoded_name]=raw_ref(encoded)
    manifest=json_bytes(dict(schema='selected_immutable_package_Git_LFS_pointer_transport_v1',entries=entries,
        original_files_rewritten=False,original_payloads_are_pointer_text_not_downloaded_LFS_objects=True))
    target=safe_path(stage,MANIFEST);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(manifest)
    expected[MANIFEST]=raw_ref(manifest)
    return entries,expected


def restore(root,manifest_raw,read_encoded,spec):
    body=json.loads(manifest_raw)
    if body['schema']!='selected_immutable_package_Git_LFS_pointer_transport_v1':raise ValueError('Transport schema differs')
    rows=body['entries']
    for name,row in rows.items():
        target=safe_path(root,name)
        if name not in spec['files'] or row['original']!=spec['files'][name]:raise ValueError('Transport is not bound to frozen original inventory')
        transport=row['transport_path']
        if not transport.startswith('.transport/git-lfs-pointer-json/'):raise ValueError('Unexpected encoded transport prefix')
        safe_path(root,transport)
        encoded=read_encoded(transport)
        if raw_ref(encoded)!=row['transport']:raise ValueError('Encoded transport bytes differ')
        value=json.loads(encoded)
        if value['schema']!='byte_exact_original_Git_LFS_pointer_transport_v1' or value['original_path']!=name or value['original']!=row['original']:
            raise ValueError('Transport original binding differs')
        raw=base64.b64decode(value['payload_base64'],validate=True)
        if not raw.startswith(PREFIX) or raw_ref(raw)!=row['original']:raise ValueError('Restored original bytes differ')
        if target.exists():
            if not target.is_file() or target.read_bytes()!=raw:raise ValueError('Refuse overwriting a changed original')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(raw)
    return len(rows)
