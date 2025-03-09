# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['chatbot.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('seq2seq_model_98000.h5', '.'), 
        ('seq2seq_model_90000.h5', '.'),  # 90000 모델 추가
        ('seq2seq_model_50000.h5', '.'),  # 50000 모델 추가
        ('tokenizer.pkl', '.')
    ],
    hiddenimports=['tensorflow', 'keras', 'h5py', 'numpy', 'PySide6'],
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
    name='chatbot',
    debug=True,  # debug를 True로 설정
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 콘솔 창 열기
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'],
)
