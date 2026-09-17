// `loom.run_ode` / `loom.run_ode_and_retain`: integrating `dx/dt = f(x, t)` with the loop and the
// state on this side of the boundary (ADR-031).
//
// **The oracle is the closed form, not another implementation.** The estimator here is
// `f(x, t) = c*x + t` -- linear, and deliberately NOT autonomous, because midpoint and Heun agree
// exactly on an autonomous linear f and a test built on one cannot tell them apart. Each method's
// per-step recurrence is written out below in the same `double` arithmetic the binding uses, so a
// wrong Butcher weight, a wrong stage time or a stage evaluated from the wrong base is a mismatch
// rather than a plausible number.
//
// Euler additionally has to be BIT-identical to what `render_sampler`'s Lua loop produced, because
// that is what every flow-matching model in the zoo shipped with -- so `euler_marshalled` runs the old
// shape (state through Lua, update in Lua) against the new binding.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double kC = -0.5;   // matches the SCALE attr below
constexpr int kN = 4;         // state width

// f(x, t) = c*x + t, with `t` a 1-element tensor broadcast over the state.
const char* kEstimatorJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"z","dtype":"f32","shape":["n_elems"]},
             {"name":"t","dtype":"f32","shape":["1"]}],
  "outputs": ["v"],
  "nodes": [
    {"op": "SCALE", "inputs": ["z"], "outputs": ["cz"], "attrs": {"s": -0.5}},
    {"op": "ADD", "inputs": ["cz", "t"], "outputs": ["v"]}
  ]
})JSON";

const char* kScript = R"LUA(
    -- The bridge's Value variant is numbers and arrays, so the method arrives as an INDEX into this
    -- list rather than as a string. Nothing in the binding cares; the name is what it matches on.
    local METHODS = {'euler', 'midpoint', 'heun', 'rk4'}

    function ode(inputs)
        return loom.run_ode('estimator', {n_elems = inputs.n_elems[1], n_past = 0}, {}, {
            carried = 'z', time = 't', method = METHODS[inputs.method_id[1]],
            times = inputs.times, state = inputs.state,
        })
    end

    -- The shape this replaces: the state crosses twice per step and the update is a Lua loop.
    function euler_marshalled(inputs)
        local z = inputs.state
        local times = inputs.times
        for step = 1, #times - 1 do
            local h = times[step + 1] - times[step]
            local v = loom.run_subgraph('estimator', {n_elems = inputs.n_elems[1], n_past = 0},
                                         {z = z, t = { times[step] }})
            for i = 1, #z do z[i] = z[i] + v[i] * h end
        end
        return z
    end

    -- Retained: the result stays in the engine and a second module reads it by name.
    function ode_retained(inputs)
        loom.run_ode_and_retain('estimator', {n_elems = inputs.n_elems[1], n_past = 0}, {}, {
            carried = 'z', time = 't', method = METHODS[inputs.method_id[1]],
            times = inputs.times, state = inputs.state,
        })
        return loom.get_output('estimator', 1)
    end

    function bad_method(inputs)
        local ok, err = pcall(function()
            loom.run_ode('estimator', {n_elems = inputs.n_elems[1], n_past = 0}, {},
                          {carried = 'z', time = 't', method = 'rk9', times = inputs.times,
                           state = inputs.state})
        end)
        if ok then return {0} end
        if string.find(err, "unknown method 'rk9'") then return {1} end
        io.stderr:write("unexpected: " .. tostring(err) .. "\n")
        return {0}
    end
)LUA";

std::vector<double> f(const std::vector<double>& x, double t) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = kC * x[i] + t;
    return out;
}

// Each method's recurrence, written out. The binding reads f's output back as float (it is a tensor),
// so the oracle rounds every stage the same way -- otherwise the comparison would be measuring the
// tensor round trip rather than the integrator.
std::vector<double> as_f32(std::vector<double> v) {
    for (double& x : v) x = static_cast<double>(static_cast<float>(x));
    return v;
}

std::vector<double> integrate(const char* method, std::vector<double> x,
                               const std::vector<double>& times) {
    for (size_t s = 0; s + 1 < times.size(); ++s) {
        const double t = times[s], h = times[s + 1] - t;
        auto probe = [&](double scale, const std::vector<double>& k) {
            std::vector<double> p(x.size());
            for (size_t i = 0; i < x.size(); ++i) p[i] = x[i] + h * scale * k[i];
            return as_f32(p);
        };
        const std::vector<double> k1 = as_f32(f(as_f32(x), t));
        std::vector<double> delta(x.size(), 0.0);
        if (std::string(method) == "euler") {
            delta = k1;
        } else if (std::string(method) == "midpoint") {
            const std::vector<double> k2 = as_f32(f(probe(0.5, k1), t + 0.5 * h));
            delta = k2;
        } else if (std::string(method) == "heun") {
            const std::vector<double> k2 = as_f32(f(probe(1.0, k1), t + h));
            for (size_t i = 0; i < x.size(); ++i) delta[i] = 0.5 * k1[i] + 0.5 * k2[i];
        } else {  // rk4
            const std::vector<double> k2 = as_f32(f(probe(0.5, k1), t + 0.5 * h));
            const std::vector<double> k3 = as_f32(f(probe(0.5, k2), t + 0.5 * h));
            const std::vector<double> k4 = as_f32(f(probe(1.0, k3), t + h));
            for (size_t i = 0; i < x.size(); ++i) {
                delta[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += h * delta[i];
    }
    return x;
}

std::vector<double> as_array(const loom::LoomLuaBridge::Value& v) {
    return std::get<std::vector<double>>(v);
}

}  // namespace

int main() {
    ggml_backend_ptr backend(loom_test::test_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("estimator", *model, loom::GraphTopology::parse(kEstimatorJson));
    bridge.load_script(kScript);

    const std::vector<double> state = {0.25, -1.5, 2.0, 0.0};
    std::vector<double> times;
    for (int i = 0; i <= 8; ++i) times.push_back(i / 8.0);   // eight steps of 0.125

    // --- 1. Every method matches its own closed-form recurrence. ---
    const char* kMethodNames[] = {"euler", "midpoint", "heun", "rk4"};
    for (int method_id = 1; method_id <= 4; ++method_id) {
        const char* method = kMethodNames[method_id - 1];
        const std::vector<double> got = as_array(bridge.call("ode", {
            {"n_elems", std::vector<double>{static_cast<double>(kN)}},
            {"method_id", std::vector<double>{static_cast<double>(method_id)}},
            {"times", times},
            {"state", state},
        }));
        const std::vector<double> want = integrate(method, state, times);
        LOOM_CHECK(got.size() == want.size());
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
        std::fprintf(stderr, "  %-9s max |diff| vs closed form %.3e\n", method, worst);
        LOOM_CHECK(worst < 1e-6);
    }

    // --- 2. The methods are not the same function. A binding that silently ran Euler for all four
    // would pass nothing above only if the oracle were shared -- it is not, but this says it aloud. ---
    const auto run = [&](int method_id) {
        return as_array(bridge.call("ode", {
            {"n_elems", std::vector<double>{static_cast<double>(kN)}},
            {"method_id", std::vector<double>{static_cast<double>(method_id)}}, {"times", times}, {"state", state}}));
    };
    const std::vector<double> e = run(1), m = run(2), hh = run(3), r = run(4);
    const auto differs = [](const std::vector<double>& a, const std::vector<double>& b) {
        for (size_t i = 0; i < a.size(); ++i) if (std::abs(a[i] - b[i]) > 1e-9) return true;
        return false;
    };
    LOOM_CHECK(differs(e, m));
    LOOM_CHECK(differs(m, hh));   // equal for an AUTONOMOUS linear f; this one is not autonomous
    LOOM_CHECK(differs(hh, r));

    // --- 3. Euler is bit-identical to the Lua loop it replaces. This is the one that protects every
    // flow-matching model that already shipped. ---
    const std::vector<double> marshalled = as_array(bridge.call("euler_marshalled", {
        {"n_elems", std::vector<double>{static_cast<double>(kN)}},
        {"times", times}, {"state", state}}));
    LOOM_CHECK(marshalled.size() == e.size());
    for (size_t i = 0; i < e.size(); ++i) LOOM_CHECK(e[i] == marshalled[i]);
    std::fprintf(stderr, "  euler vs the Lua loop it replaces: bit-identical\n");

    // --- 4. The retaining form leaves the same state in the module's store. ---
    const std::vector<double> retained = as_array(bridge.call("ode_retained", {
        {"n_elems", std::vector<double>{static_cast<double>(kN)}},
        {"method_id", std::vector<double>{2}}, {"times", times}, {"state", state}}));
    LOOM_CHECK(retained.size() == m.size());
    for (size_t i = 0; i < m.size(); ++i) LOOM_CHECK(std::abs(retained[i] - m[i]) < 1e-6);

    // --- 5. An unknown method is named, not defaulted to Euler. ---
    LOOM_CHECK(as_array(bridge.call("bad_method", {
        {"n_elems", std::vector<double>{static_cast<double>(kN)}},
        {"times", times}, {"state", state}}))[0] == 1.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
