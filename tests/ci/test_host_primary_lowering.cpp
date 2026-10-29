// P4.44: `backend_can_run` answers for the CPU fallback when the primary's tensors are host memory.
//
// A host-memory primary (ggml-vc4 on a Pi Zero, BLAS) declines almost every op, and the CPU beside it
// runs those on the same memory with no copy. Asking the primary alone gave such graphs the lowerings
// composed for a GPU -- VITS Q8_0_PC's folded convolutions as im2col + mul_mat, 5.7x slower on the
// board -- so the CPU's answer counts there, and only there: a primary with its own memory still gets
// the composition, because sending a node to the CPU costs it a copy each way.
//
// No such device exists on a CI runner, so this builds one: the CPU device with every op declined,
// over the CPU's buffer type (host) or a copy of it that says it is not host memory (a GPU, as far as
// the question goes). The node is never computed; only the answer is.
//
// P4.47: where such a device's weights go (weights_buffer_type): into its host buffer type when it has
// one -- ggml-vc4's VC4_HOST, RAM, where its own buffers are the GPU's small pool -- and into its own
// buffer type otherwise; checked as a truth table over the fake device, and through GgufModel::load,
// whose weights must land in the fake host buffer type.

#include "test_util.h"

#include "loom/loom.h"
#include "loom/ops/primitive_registry.h"

#include "cpu_backend.h"

#include "ggml-backend-impl.h"

#include <cstdio>
#include <string>

namespace {

ggml_backend_buffer_type g_device_buft;   // the CPU's buffer type, minus is_host
ggml_backend_buffer_type g_host_buft;     // the CPU's buffer type under another name: the "host buffer type"
bool g_host = true, g_offers_host = false;
enum ggml_backend_dev_type g_type = GGML_BACKEND_DEVICE_TYPE_ACCEL;

bool decline_everything(ggml_backend_dev_t, const ggml_tensor*) { return false; }
ggml_backend_buffer_type_t fake_buffer_type(ggml_backend_dev_t) {
    return g_host ? ggml_backend_cpu_buffer_type() : &g_device_buft;
}
ggml_backend_buffer_type_t fake_host_buffer_type(ggml_backend_dev_t) { return g_offers_host ? &g_host_buft : nullptr; }
enum ggml_backend_dev_type fake_type(ggml_backend_dev_t) { return g_type; }
const char* host_buft_name(ggml_backend_buffer_type_t) { return "FAKE_HOST"; }
void* no_proc(ggml_backend_reg_t, const char*) { return nullptr; }   // no extra buffer types

} // namespace

int main() {
    ggml_backend_ptr cpu(loom_test::cpu_backend());
    LOOM_CHECK(cpu != nullptr);

    g_device_buft = *ggml_backend_cpu_buffer_type();
    g_device_buft.iface.is_host = nullptr;
    ggml_backend_device fake_dev = *ggml_backend_get_device(cpu.get());
    fake_dev.iface.supports_op = decline_everything;
    fake_dev.iface.get_buffer_type = fake_buffer_type;
    ggml_backend fake = *cpu;   // a struct copy, never freed: only its device is consulted
    fake.device = &fake_dev;

    ggml_context_ptr ctx(ggml_init(ggml_init_params{8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true}));
    ggml_tensor* a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 4);
    ggml_tensor* node = ggml_pad_reflect_1d(ctx.get(), a, 3, 3);
    LOOM_CHECK(ggml_backend_supports_op(cpu.get(), node));
    LOOM_CHECK(!ggml_backend_supports_op(&fake, node));

    loom::SymbolEnv symbols;
    auto ask = [&](loom::Backends backends) {
        loom::PrimitiveContext pc{ctx.get(), symbols};
        pc.backends = backends;
        return loom::backend_can_run(pc, node);
    };

    g_host = true;
    LOOM_CHECK(ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(&fake)));
    const bool host_with_cpu = ask(loom::Backends(&fake, cpu.get()));
    const bool host_alone = ask(loom::Backends(&fake));
    g_host = false;
    LOOM_CHECK(!ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(&fake)));
    const bool device_with_cpu = ask(loom::Backends(&fake, cpu.get()));
    const bool cpu_alone = ask(loom::Backends(cpu.get()));

    LOOM_CHECK(host_with_cpu);     // VC4 / BLAS + CPU: the CPU's lowering
    LOOM_CHECK(!host_alone);       // nothing beside it to run the node
    LOOM_CHECK(!device_with_cpu);  // a GPU + CPU: the composition, as before
    LOOM_CHECK(cpu_alone);
    std::printf("host primary + CPU %d, host primary alone %d, device primary + CPU %d, CPU %d\n",
                host_with_cpu, host_alone, device_with_cpu, cpu_alone);

    // --- P4.47: weights_buffer_type --------------------------------------------------------------
    g_host_buft = *ggml_backend_cpu_buffer_type();
    g_host_buft.iface.get_name = host_buft_name;
    fake_dev.iface.get_host_buffer_type = fake_host_buffer_type;
    fake_dev.iface.get_type = fake_type;
    ggml_backend_reg fake_reg = *fake_dev.reg;
    fake_reg.iface.get_proc_address = no_proc;
    fake_dev.reg = &fake_reg;
    struct Row { bool host, offers; enum ggml_backend_dev_type type; bool want_host_buft; const char* what; };
    const Row rows[] = {
        {true, true, GGML_BACKEND_DEVICE_TYPE_ACCEL, true, "host-memory accelerator with a host buffer type (ggml-vc4)"},
        {true, false, GGML_BACKEND_DEVICE_TYPE_ACCEL, false, "host-memory accelerator without one (BLAS)"},
        {false, true, GGML_BACKEND_DEVICE_TYPE_GPU, false, "a GPU with pinned host buffers (CUDA, Vulkan)"},
        {true, true, GGML_BACKEND_DEVICE_TYPE_CPU, false, "the CPU"},
    };
    for (const Row& r : rows) {
        g_host = r.host;
        g_offers_host = r.offers;
        g_type = r.type;
        ggml_backend_buffer_type_t got = loom::weights_buffer_type(&fake_dev);
        const bool is_host_buft = got == &g_host_buft;
        std::printf("weights_buffer_type, %s: %s\n", r.what, ggml_backend_buft_name(got));
        LOOM_CHECK(is_host_buft == r.want_host_buft);
        if (!r.want_host_buft) LOOM_CHECK(got == fake_buffer_type(&fake_dev));
    }
    LOOM_CHECK(loom::weights_buffer_type(ggml_backend_get_device(cpu.get())) == ggml_backend_cpu_buffer_type());

    // ... and the loader puts a model's weights there
    g_host = true;
    g_offers_host = true;
    g_type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    {
        auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/minimal.gguf", loom::Backends(&fake, cpu.get()));
        ggml_tensor* w = model->weight("test.weight");
        LOOM_CHECK(w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) == &g_host_buft);
        std::printf("GgufModel::load beside the CPU: weights in %s\n", w->buffer ? ggml_backend_buffer_name(w->buffer) : "(none)");
    }
    g_offers_host = false;
    {
        auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/minimal.gguf", loom::Backends(&fake, cpu.get()));
        ggml_tensor* w = model->weight("test.weight");
        LOOM_CHECK(w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) == ggml_backend_cpu_buffer_type());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
