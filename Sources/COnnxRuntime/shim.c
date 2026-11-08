// ONNX Runtime's C API headers, vendored from the onnxruntime 1.31.0 release (MIT, see LICENSE-onnxruntime), for
// OrtInference to call into. The library itself is linked, not vendored: libonnxruntime from the Android AAR or the
// desktop release, found through the linker search path (see README.md, "ONNX Runtime backend").
//
// SwiftPM wants a C target to have a source file; this one has nothing else to say.
#include "onnxruntime_c_api.h"
