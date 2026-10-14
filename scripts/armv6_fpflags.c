// LD_PRELOAD shim for ADR-078 gate 3: which ggml ops raise which VFP exception flags.
// With LOOM_PROFILE set the engine computes one node per ggml_graph_compute call, so clearing FPSCR's
// cumulative flags before the real call and reading them after names the node that raised them.
// At one thread the compute runs on the calling thread, whose FPSCR this reads. IXC is ignored.
// Python loads the engine's libraries RTLD_LOCAL, so neither RTLD_NEXT nor this file's own references to
// ggml resolve through the global scope: both go through handles to the already-loaded libraries.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "ggml.h"
#include "ggml-cpu.h"

typedef enum ggml_status (*fn_t)(struct ggml_cgraph *, struct ggml_cplan *);
static fn_t real;
static int (*p_n_nodes)(struct ggml_cgraph *);
static struct ggml_tensor * (*p_node)(struct ggml_cgraph *, int);
static const char * (*p_op_desc)(const struct ggml_tensor *);
static const char * (*p_get_name)(const struct ggml_tensor *);

static void * handle_of(const char * part) {
    struct { const char * part; char path[512]; } q = { part, "" };
    int cb(struct dl_phdr_info * info, size_t size, void * data) {
        (void) size;
        if (strstr(info->dlpi_name, ((typeof(q) *) data)->part)) {
            snprintf(((typeof(q) *) data)->path, 512, "%s", info->dlpi_name); return 1;
        }
        return 0;
    }
    dl_iterate_phdr(cb, &q);
    return q.path[0] ? dlopen(q.path, RTLD_NOW | RTLD_NOLOAD) : NULL;
}

static void resolve(void) {
    void * cpu = handle_of("libggml-cpu"), * base = handle_of("libggml-base");
    real = (fn_t) dlsym(cpu, "ggml_graph_compute");
    p_n_nodes = dlsym(base, "ggml_graph_n_nodes");
    p_node = dlsym(base, "ggml_graph_node");
    p_op_desc = dlsym(base, "ggml_op_desc");
    p_get_name = dlsym(base, "ggml_get_name");
    if (!real || !p_n_nodes || !p_node || !p_op_desc || !p_get_name) {
        fprintf(stderr, "fpflags: could not resolve ggml (cpu %p base %p)\n", cpu, base);
        _exit(3);
    }
}
static struct { char op[40]; unsigned flags; long nodes; char example[120]; } tab[512];
static int ntab;
static long calls, flagged;

static unsigned get(void) { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static void put(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f) : "memory"); }

enum ggml_status ggml_graph_compute(struct ggml_cgraph * g, struct ggml_cplan * p) {
    if (!real) resolve();
    put(get() & ~0x9fu);
    enum ggml_status s = real(g, p);
    const unsigned f = get() & 0x8fu;  // IDC | UFC | OFC | DZC | IOC
    ++calls;
    if (f && p_n_nodes(g) > 0) {
        ++flagged;
        struct ggml_tensor * t = p_node(g, p_n_nodes(g) - 1);
        const char * op = p_op_desc(t);
        int i = 0;
        for (; i < ntab; ++i) if (tab[i].flags == f && !strcmp(tab[i].op, op)) break;
        if (i == ntab && ntab < 512) {
            ++ntab; snprintf(tab[i].op, sizeof tab[i].op, "%s", op); tab[i].flags = f;
            snprintf(tab[i].example, sizeof tab[i].example, "%s [%lld,%lld,%lld,%lld] n_nodes=%d", p_get_name(t),
                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                     p_n_nodes(g));
        }
        if (i < ntab) tab[i].nodes++;
    }
    return s;
}

__attribute__((destructor)) static void report(void) {
    fprintf(stderr, "fpflags: %ld computes, %ld raised a flag\n", calls, flagged);
    for (int i = 0; i < ntab; ++i)
        fprintf(stderr, "fpflags: %-14s %s%s%s%s%s %6ld  e.g. %s\n", tab[i].op, tab[i].flags & 1 ? "IOC " : "",
                tab[i].flags & 2 ? "DZC " : "", tab[i].flags & 4 ? "OFC " : "", tab[i].flags & 8 ? "UFC " : "",
                tab[i].flags & 0x80 ? "IDC " : "", tab[i].nodes, tab[i].example);
}
