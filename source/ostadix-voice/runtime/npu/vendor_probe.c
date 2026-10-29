#include <dlfcn.h>
#include <stdio.h>
int main(void) {
  void *lib = dlopen("libedgetpu_litert.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  puts("vendor_library_loaded=true");
  const char *symbols[] = {"thrInitialize", "thrGetVendorApiVersion",
                           "thrContextCreate", "thrInvocationContextInvokeOnce"};
  int rc = 0;
  for (unsigned i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
    int present = dlsym(lib, symbols[i]) != NULL;
    printf("%s=%s\n", symbols[i], present ? "present" : "missing");
    if (!present) rc = 2;
  }
  dlclose(lib);
  return rc;
}
