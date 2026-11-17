// LOOM_HEAL_LOG names every layout guess ("heal") a graph reaches -- the list the hub item "Who still
// depends on the layout-healing heuristics?" is built from. Its own executable because the sink is
// chosen once per process from the environment, so the variable must be set before the first op runs.
//
// What must hold, since a run over the whole zoo is only as good as this log:
//  * a heal that fires writes one line naming the op, the branch, the topology node and the operand
//    shapes BEFORE the heal;
//  * the same line is written once per process (a decode loop rebuilds its graph every token);
//  * operands that already combine reach no heal and write nothing -- the guards, seen from the log.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include "loom/loom.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using loom_test::GgmlScratch;

namespace {

const loom::PrimitiveFn& op(const std::string& name) {
    return loom::PrimitiveRegistry::instance().get(name);
}

std::string read_all(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main() {
    const char* tmpdir = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmpdir != nullptr && *tmpdir != '\0' ? tmpdir : "/tmp") + "/loom_heal_log_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const int fd = mkstemp(buf.data());
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed for %s\n", buf.data());
        return 1;
    }
    close(fd);
    const std::string log_path(buf.data());
    setenv("LOOM_HEAL_LOG", log_path.c_str(), 1);

    GgmlScratch s;
    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};

    // REPEAT [2,3] -> [3,2]: does not repeat, square-swapped -> the axis-0/1 heal. Built twice, as a
    // decode loop would, under a node label.
    const std::string rep_label = "blk.0.rep";
    pc.node_label = &rep_label;
    for (int i = 0; i < 2; ++i) {
        ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 3);
        ggml_tensor* out = op("REPEAT")(pc, {a}, {{"shape", {3, 2}}})[0];
        LOOM_CHECK(out->ne[0] == 3 && out->ne[1] == 2);
    }

    // Grouped-query attention's K/V repeat at n_tokens == head_dim: it ALREADY repeats, so no heal
    // (the 2026-11-16 guard) and no line.
    const std::string gqa_label = "blk.0.kv_rep";
    pc.node_label = &gqa_label;
    {
        ggml_tensor* kv = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 4, 4, 1, 2);
        op("REPEAT")(pc, {kv}, {{"shape", {4, 4, 2, 2}}});
    }

    // MUL_MAT [4,3,2] x [4,2,3]: will not multiply, axes 1/2 look swapped -> permute a. No label, as
    // in a hand-built context.
    pc.node_label = nullptr;
    {
        ggml_tensor* a = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 3, 2);
        ggml_tensor* b = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 2, 3);
        op("MUL_MAT")(pc, {a, b}, {});
    }

    // ADD of two operands that already broadcast: no line.
    {
        ggml_tensor* a = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 4, 4);
        ggml_tensor* b = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 4, 1);
        op("ADD")(pc, {a, b}, {});
    }

    const std::string got = read_all(log_path);
    const std::string want =
        "loom-heal\tREPEAT.swap01\tblk.0.rep\ta=[2,3,1,1]\ttarget=[3,2,1,1]\n"
        "loom-heal\tMUL_MAT.permute12(a)\t?\ta=[4,3,2,1]\tb=[4,2,3,1]\n";
    LOOM_CHECK(got == want);
    if (got != want) std::fprintf(stderr, "log was:\n%s\n", got.c_str());
    std::remove(log_path.c_str());
    LOOM_TEST_REPORT_AND_RETURN();
}
