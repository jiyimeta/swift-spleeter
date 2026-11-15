# Changelog

Releases before 0.4.1 are described on the [GitHub releases page](https://github.com/jiyimeta/swift-spleeter/releases).

## 0.4.1

Windows.

- **OrtInference passes the model path to ONNX Runtime as UTF-16 on Windows.** `OrtCreateSession` takes an
  `ORTCHAR_T *`, which is `wchar_t` on Windows and `char` elsewhere; 0.4.0 passed UTF-8 everywhere, so on Windows the
  session could not be created. Other platforms are unchanged.
- **The `Spleeter` target builds on platforms without Core ML.** It stays declared on every platform, so a consumer's
  dependency on the `Spleeter` product (conditional on Apple platforms or not) still resolves on Windows, Linux and
  Android. Its sources and its tests are wrapped in `#if canImport(CoreML)` instead of the manifest declaring the
  target for Apple only: a manifest condition is evaluated on the host, and a product that disappears there breaks
  that dependency. Off Apple platforms the target is SpleeterCore's re-export alone, and `swift test` now runs there
  (it used to stop at `no such module 'Accelerate'`). Apple platforms are unchanged.
