# ANPR Windows Installer

Builds a one-click `ANPR-Setup-x64.exe` with [Inno Setup](https://jrsoftware.org/isinfo.php)
that installs the pipeline (`anpr.exe` + runtime DLLs + models + config) and the
self-contained **ANPR Studio** GUI.

## Prerequisites

- Inno Setup 6 (`iscc.exe` on PATH, or use the IDE)
- .NET 10 SDK (to publish the studio)
- The pipeline built in Release and the models fetched (see below)

## Release steps

```powershell
# 1. Build the C++ pipeline (Release)
cd cmake/anpr
cmake --preset x64-release
cmake --build out/build/x64-release

# 2. Fetch / export the models
cd models
python prepare_models.py --vehicle
cd ..

# 3. Stage everything the installer needs
cd ../../installer
./package.ps1

# 4. Compile the installer  ->  installer/dist/ANPR-Setup-x64.exe
iscc anpr.iss
```

## What gets installed

```
{Program Files}\ANPR\
├── pipeline\
│   ├── anpr.exe + *.dll          (OpenCV, FFmpeg, …)
│   ├── mock_server.exe
│   ├── config\anpr.json          (from the sanitized example)
│   └── models\*.onnx, charset*.txt
└── studio\
    └── AnprStudio.exe            (self-contained, no .NET needed on the target)
```

Start-menu shortcuts: **ANPR Studio**, **ANPR (pipeline)**, **Edit config**.

## Versioning

Bump `#define AppVersion` in `anpr.iss` for each release and tag the git commit
(`v1.0.0`). The GitHub Actions release workflow (`.github/workflows/`) can attach
the produced installer to the release.

## Notes

- The bundled `config\anpr.json` is the **sanitized template** — set your real
  camera URL/credentials and API keys after installing.
- Models are downloaded, not committed; if `package.ps1` warns that none were
  found, run `prepare_models.py` first.
- The installer is unsigned. For distribution without SmartScreen warnings, sign
  `ANPR-Setup-x64.exe` and `AnprStudio.exe` with a code-signing certificate.
