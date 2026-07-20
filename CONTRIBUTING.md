# Contributing to ANPR

Thanks for your interest! This project is layered on purpose — most changes fit
into one module without touching the others.

## Development setup

```powershell
# Pipeline
cd cmake/anpr
cmake --preset x64-debug
cmake --build out/build/x64-debug
cd models && python prepare_models.py --vehicle

# Studio
cd proj/AnprStudio && dotnet build
```

## Project conventions

- **Language:** C++20 for the pipeline, C#/.NET (WPF) for the studio, Python for
  training. Code and comments in English; user-facing docs may be Turkish.
- **Style:** modern C++ (RAII, smart pointers, no raw `new`/`delete`), Doxygen-style
  comments on public interfaces. Match the surrounding code.
- **Architecture:** add capabilities behind the existing interfaces
  (`IFrameSource`, `IPlateRecognizer`, `INetworkTransport`) and wire them in via the
  factories. New tunables go in `core/Config.h` + `config/anpr.example.json`, never
  hard-coded.
- **Keep the Studio in sync:** any new config key, log line format, or API endpoint
  must be reflected in `proj/AnprStudio` in the same change.

## Before opening a PR

- Build both the pipeline (Release) and the Studio with no warnings.
- Run a quick end-to-end check (`anpr.exe --image=...` or a short video).
- Never commit real credentials, models (`*.onnx`), datasets, or the vendor SDK —
  `.gitignore` covers these; double-check `git status`.
- Update `cmake/anpr/docs/ARCHITECTURE.md` and the README if behavior changed.

## Reporting issues

Include your config (with secrets redacted), the relevant log lines, OS, and how
you built (CMake preset / compiler).
