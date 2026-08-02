#include "loom/core/graph_topology.h"
#include "loom/loom_errors.h"

namespace loom {
namespace {

using Json = nlohmann::json;

TopologyNode parse_node(const Json& j) {
    TopologyNode node;
    node.op = j.at("op").get<std::string>();
    node.name = j.value("name", std::string());
    node.inputs = j.value("inputs", std::vector<std::string>());
    node.outputs = j.value("outputs", std::vector<std::string>());
    node.attrs = j.value("attrs", Json::object());
    return node;
}

TopologyItem parse_item(const Json& j) {
    TopologyItem item;
    if (j.contains("repeat_for")) {
        item.is_repeat = true;
        item.repeat.count_symbol = j.at("repeat_for").get<std::string>();
        item.repeat.index_var = j.at("index_var").get<std::string>();
        for (const Json& child : j.at("nodes")) {
            item.repeat.nodes.push_back(parse_node(child));
        }
    } else {
        item.is_repeat = false;
        item.node = parse_node(j);
    }
    return item;
}

} // namespace

GraphTopology GraphTopology::parse(const std::string& json_text) {
    Json j;
    try {
        j = Json::parse(json_text);
    } catch (const Json::parse_error& e) {
        throw SchemaError(std::string("GraphTopology::parse: invalid JSON: ") + e.what());
    }

    try {
        GraphTopology topo;
        topo.version = j.at("version").get<int>();
        if (topo.version != 1) {
            throw SchemaError("GraphTopology::parse: unsupported schema version " + std::to_string(topo.version) +
                               " (only version 1 is supported)");
        }

        for (const Json& in : j.value("inputs", Json::array())) {
            TensorSpec spec;
            spec.name = in.at("name").get<std::string>();
            spec.dtype = in.at("dtype").get<std::string>();
            spec.shape = in.at("shape").get<std::vector<std::string>>();
            topo.inputs.push_back(std::move(spec));
        }

        // "outputs" (plural, JSON array) declares a multi-output topology (P2); "output" (singular
        // string) is both the original schema and the byte-identical serialization every single-output
        // topology still uses -- see graph_topology.h's own comment on the two fields.
        if (j.contains("outputs")) {
            topo.outputs = j.at("outputs").get<std::vector<std::string>>();
            if (topo.outputs.empty()) {
                throw SchemaError("GraphTopology::parse: 'outputs' array must not be empty");
            }
            topo.output = topo.outputs.front();
        } else {
            topo.output = j.at("output").get<std::string>();
            topo.outputs = {topo.output};
        }

        for (const Json& node_json : j.at("nodes")) {
            topo.items.push_back(parse_item(node_json));
        }

        return topo;
    } catch (const SchemaError&) {
        throw;
    } catch (const Json::exception& e) {
        throw SchemaError(std::string("GraphTopology::parse: malformed topology document: ") + e.what());
    }
}

bool GraphTopology::uses_kv_cache() const {
    // `kv_cache` defaults to TRUE when absent, matching op_attention's own default -- every milestone-1
    // LLM topology relies on that default and never sets the attr, so reading it as false here would
    // report exactly the models that need a cache as not needing one.
    const auto node_uses_cache = [](const TopologyNode& node) {
        return node.op == "ATTENTION" && node.attrs.value("kv_cache", true);
    };
    for (const TopologyItem& item : items) {
        if (item.is_repeat) {
            for (const TopologyNode& node : item.repeat.nodes) {
                if (node_uses_cache(node)) return true;
            }
        } else if (node_uses_cache(item.node)) {
            return true;
        }
    }
    return false;
}

} // namespace loom
