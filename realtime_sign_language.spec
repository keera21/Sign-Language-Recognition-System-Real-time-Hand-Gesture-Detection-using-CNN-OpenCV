# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['realtime_sign_language.py'],
    pathex=[],
    binaries=[],
    datas=[('model', 'model')],
    hiddenimports=['sklearn.utils._cython_blas', 'sklearn.utils._cyutility', 'sklearn.utils._isfinite'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='realtime_sign_language',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
