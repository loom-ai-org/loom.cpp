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

// The same f with a FIXED input added: `f(x, t, b) = c*x + t + b`. Guidance needs an input the two
// runs can differ in, and `b` is the smallest thing that is one -- with `b` standing for everything a
// real estimator conditions on (F5-TTS's reference mel and text embedding).
const char* kGuidedEstimatorJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"z","dtype":"f32","shape":["n_elems"]},
             {"name":"t","dtype":"f32","shape":["1"]},
             {"name":"b","dtype":"f32","shape":["n_elems"]}],
  "outputs": ["v"],
  "nodes": [
    {"op": "SCALE", "inputs": ["z"], "outputs": ["cz"], "attrs": {"s": -0.5}},
    {"op": "ADD", "inputs": ["cz", "t"], "outputs": ["ct"]},
    {"op": "ADD", "inputs": ["ct", "b"], "outputs": ["v"]}
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

    -- Classifier-free guidance: one module, one graph, two fixed-input tables.
    function ode_guided(inputs)
        return loom.run_ode('guided', {n_elems = inputs.n_elems[1], n_past = 0},
                             {b = inputs.b_cond}, {
            carried = 'z', time = 't', method = METHODS[inputs.method_id[1]],
            times = inputs.times, state = inputs.state,
            guidance = {scale = inputs.scale[1], inputs = {b = inputs.b_uncond}},
        })
    end

    -- The same call with NO guidance table, for the control arm: guidance at scale 0 must reproduce
    -- it exactly, and the conditional run is the one that survives.
    function ode_unguided(inputs)
        return loom.run_ode('guided', {n_elems = inputs.n_elems[1], n_past = 0},
                             {b = inputs.b_cond}, {
            carried = 'z', time = 't', method = METHODS[inputs.method_id[1]],
            times = inputs.times, state = inputs.state,
        })
    end

    function guidance_without_scale(inputs)
        local ok, err = pcall(function()
            loom.run_ode('guided', {n_elems = inputs.n_elems[1], n_past = 0}, {b = inputs.b_cond},
                          {carried = 'z', time = 't', times = inputs.times, state = inputs.state,
                           guidance = {inputs = {b = inputs.b_uncond}}})
        end)
        if ok then return {0} end
        if string.find(err, "guidance.scale is required") then return {1} end
        io.stderr:write("unexpected: " .. tostring(err) .. "\n")
        return {0}
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

// The guided recurrence, written out the same way. `f_b(x, t) = c*x + t + b`, and the velocity the
// integrator actually sees is `v_cond + scale * (v_cond - v_uncond)` -- which for this f collapses to
// `c*x + t + b_cond + scale * (b_cond - b_uncond)`, a different affine field. Every stage is rounded
// through f32 exactly where the binding reads a tensor back.
std::vector<double> integrate_guided(const char* method, std::vector<double> x,
                                      const std::vector<double>& times,
                                      const std::vector<double>& b_cond,
                                      const std::vector<double>& b_uncond, double scale) {
    auto fg = [&](const std::vector<double>& p, double t) {
        std::vector<double> c(p.size()), u(p.size()), out(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            c[i] = kC * p[i] + t + b_cond[i];
            u[i] = kC * p[i] + t + b_uncond[i];
        }
        c = as_f32(c);
        u = as_f32(u);
        for (size_t i = 0; i < p.size(); ++i) out[i] = c[i] + scale * (c[i] - u[i]);
        return out;
    };
    for (size_t s = 0; s + 1 < times.size(); ++s) {
        const double t = times[s], h = times[s + 1] - t;
        auto probe = [&](double sc, const std::vector<double>& k) {
            std::vector<double> p(x.size());
            for (size_t i = 0; i < x.size(); ++i) p[i] = x[i] + h * sc * k[i];
            return as_f32(p);
        };
        const std::vector<double> k1 = fg(as_f32(x), t);
        std::vector<double> delta(x.size(), 0.0);
        if (std::string(method) == "euler") {
            delta = k1;
        } else {  // midpoint, the second-order arm this test exercises
            delta = fg(probe(0.5, k1), t + 0.5 * h);
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
    bridge.register_module("guided", *model, loom::GraphTopology::parse(kGuidedEstimatorJson));
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

    // --- 6. Classifier-free guidance, inside the integrator. ---
    const std::vector<double> b_cond = {0.5, -0.25, 1.0, 0.125};
    const std::vector<double> b_uncond = {-0.5, 0.75, 0.0, 0.5};
    const auto guided = [&](int method_id, double scale) {
        return as_array(bridge.call("ode_guided", {
            {"n_elems", std::vector<double>{static_cast<double>(kN)}},
            {"method_id", std::vector<double>{static_cast<double>(method_id)}},
            {"times", times}, {"state", state},
            {"b_cond", b_cond}, {"b_uncond", b_uncond},
            {"scale", std::vector<double>{scale}}}));
    };
    for (int method_id : {1, 2}) {
        const char* method = kMethodNames[method_id - 1];
        const std::vector<double> got = guided(method_id, 2.0);
        const std::vector<double> want =
            integrate_guided(method, state, times, b_cond, b_uncond, 2.0);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
        std::fprintf(stderr, "  guided %-9s max |diff| vs closed form %.3e\n", method, worst);
        LOOM_CHECK(worst < 1e-6);
    }

    // The arm that says guidance is DOING something: a different scale is a different answer, and
    // scale 0 is the unguided call bit for bit (`v_c + 0*(v_c - v_u)` is `v_c`).
    const std::vector<double> g2 = guided(1, 2.0), g0 = guided(1, 0.0), g5 = guided(1, 0.5);
    LOOM_CHECK(differs(g2, g5));
    LOOM_CHECK(differs(g2, g0));
    const std::vector<double> plain = as_array(bridge.call("ode_unguided", {
        {"n_elems", std::vector<double>{static_cast<double>(kN)}},
        {"method_id", std::vector<double>{1}}, {"times", times}, {"state", state},
        {"b_cond", b_cond}}));
    LOOM_CHECK(plain.size() == g0.size());
    for (size_t i = 0; i < g0.size(); ++i) LOOM_CHECK(g0[i] == plain[i]);
    std::fprintf(stderr, "  guidance at scale 0 == the unguided call: bit-identical\n");

    // --- 7. A guidance table without a scale is named, not defaulted. ---
    LOOM_CHECK(as_array(bridge.call("guidance_without_scale", {
        {"n_elems", std::vector<double>{static_cast<double>(kN)}},
        {"times", times}, {"state", state},
        {"b_cond", b_cond}, {"b_uncond", b_uncond}}))[0] == 1.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
