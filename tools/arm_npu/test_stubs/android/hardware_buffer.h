// Minimal host-only declarations for mock tests of npu_pipe.cc. No Android or
// vendor library is loaded by these tests; production builds use the NDK header.
#ifndef PIXELFORGE_TEST_ANDROID_HARDWARE_BUFFER_H_
#define PIXELFORGE_TEST_ANDROID_HARDWARE_BUFFER_H_
#include <stdint.h>
typedef struct AHardwareBuffer AHardwareBuffer;
typedef struct AHardwareBuffer_Desc {
  uint32_t width, height, layers, format;
  uint64_t usage;
  uint32_t stride, rfu0;
  uint64_t rfu1;
} AHardwareBuffer_Desc;
enum {
  AHARDWAREBUFFER_FORMAT_BLOB = 0x21,
  AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN = 3,
  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN = 3 << 4,
};
extern "C" {
int AHardwareBuffer_allocate(const AHardwareBuffer_Desc*, AHardwareBuffer**);
void AHardwareBuffer_release(AHardwareBuffer*);
int AHardwareBuffer_lock(AHardwareBuffer*, uint64_t, int32_t, const void*, void**);
int AHardwareBuffer_unlock(AHardwareBuffer*, int32_t*);
}
#endif
