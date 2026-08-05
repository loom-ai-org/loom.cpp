#include "loom/core/tdt_decoder.h"

#include "loom/loom_errors.h"

#include <ggml-backend.h>

namespace loom {
namespace {

int32_t argmax(const float* data, size_t n) {
    int32_t best = 0;
    float best_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

} // namespace

TdtDecoder::TdtDecoder(GgufModel& model, std::vector<GraphTopology> lstm_h_topos,
                        std::vector<GraphTopology> lstm_c_topos, GraphTopology joint_topo, TdtDecoderConfig cfg,
                        ggml_backend_t backend, uint32_t pred_hidden)
    : model_(model),
      cfg_(std::move(cfg)),
      backend_(backend),
      pred_hidden_(pred_hidden),
      num_layers_(static_cast<uint32_t>(lstm_h_topos.size())),
      lstm_h_topos_(std::move(lstm_h_topos)),
      lstm_c_topos_(std::move(lstm_c_topos)),
      joint_topo_(std::move(joint_topo)) {
    if (cfg_.blank_id < 0) {
        throw Error("TdtDecoder: blank_id must be set (non-negative)");
    }
    // durations may be EMPTY: that's plain RNN-T mode (see TdtDecoderConfig's own comment), not an error.
    if (num_layers_ == 0 || lstm_c_topos_.size() != num_layers_) {
        throw Error("TdtDecoder: lstm_h_topos and lstm_c_topos must be the same non-zero size");
    }
    for (uint32_t i = 0; i < num_layers_; ++i) {
        lstm_h_builders_.push_back(std::make_unique<GraphBuilder>(lstm_h_topos_[i], model, backend, /*kv_cache=*/nullptr));
        lstm_c_builders_.push_back(std::make_unique<GraphBuilder>(lstm_c_topos_[i], model, backend, /*kv_cache=*/nullptr));
    }
    joint_builder_ = std::make_unique<GraphBuilder>(joint_topo_, model, backend, /*kv_cache=*/nullptr);
}

TdtDecoder::Result TdtDecoder::decode_greedy(const std::vector<std::vector<float>>& encoder_output) {
    Result result;
    std::vector<std::vector<float>> h(num_layers_, std::vector<float>(pred_hidden_, 0.0f));
    std::vector<std::vector<float>> c(num_layers_, std::vector<float>(pred_hidden_, 0.0f));
    int32_t last_label = cfg_.blank_id; // NeMo's own SOS sentinel for the very first step

    const auto n_frames = static_cast<uint32_t>(encoder_output.size());
    const auto n_durations = static_cast<uint32_t>(cfg_.durations.size());

    uint32_t time_idx = 0;
    while (time_idx < n_frames) {
        const std::vector<float>& frame = encoder_output[time_idx];

        uint32_t symbols_added = 0;
        bool advanced = false;
        while (symbols_added < cfg_.max_symbols_per_step) {
            // Run the LSTM stack: layer 0 embeds `last_label`; layer i>0 takes layer i-1's h_new as its
            // own "layer_input" -- no fresh embedding lookup partway up the stack.
            std::vector<std::vector<float>> h_new(num_layers_);
            std::vector<std::vector<float>> c_new(num_layers_);
            std::vector<float> layer_input; // only populated/used for layers > 0

            for (uint32_t layer = 0; layer < num_layers_; ++layer) {
                const GraphBuilder::BuildResult& hr = lstm_h_builders_[layer]->build({{"n_tokens", /*n_tokens=*/0}, {"n_past", /*n_past=*/0}});
                if (layer == 0) {
                    ggml_backend_tensor_set(hr.input_tensors.at("last_label"), &last_label, 0, sizeof(int32_t));
                } else {
                    ggml_backend_tensor_set(hr.input_tensors.at("layer_input"), layer_input.data(), 0,
                                             layer_input.size() * sizeof(float));
                }
                ggml_backend_tensor_set(hr.input_tensors.at("h_prev"), h[layer].data(), 0, h[layer].size() * sizeof(float));
                ggml_backend_tensor_set(hr.input_tensors.at("c_prev"), c[layer].data(), 0, c[layer].size() * sizeof(float));
                ggml_backend_graph_compute(backend_, hr.graph);
                h_new[layer].resize(pred_hidden_);
                ggml_backend_tensor_get(hr.output, h_new[layer].data(), 0, h_new[layer].size() * sizeof(float));

                const GraphBuilder::BuildResult& cr = lstm_c_builders_[layer]->build({{"n_tokens", /*n_tokens=*/0}, {"n_past", /*n_past=*/0}});
                if (layer == 0) {
                    ggml_backend_tensor_set(cr.input_tensors.at("last_label"), &last_label, 0, sizeof(int32_t));
                } else {
                    ggml_backend_tensor_set(cr.input_tensors.at("layer_input"), layer_input.data(), 0,
                                             layer_input.size() * sizeof(float));
                }
                ggml_backend_tensor_set(cr.input_tensors.at("h_prev"), h[layer].data(), 0, h[layer].size() * sizeof(float));
                ggml_backend_tensor_set(cr.input_tensors.at("c_prev"), c[layer].data(), 0, c[layer].size() * sizeof(float));
                ggml_backend_graph_compute(backend_, cr.graph);
                c_new[layer].resize(pred_hidden_);
                ggml_backend_tensor_get(cr.output, c_new[layer].data(), 0, c_new[layer].size() * sizeof(float));

                layer_input = h_new[layer]; // feeds the next layer, if any
            }
            const std::vector<float>& top_h = h_new[num_layers_ - 1];

            const GraphBuilder::BuildResult& j_res = joint_builder_->build({{"n_tokens", /*n_tokens=*/0}, {"n_past", /*n_past=*/0}});
            ggml_backend_tensor_set(j_res.input_tensors.at("encoder_frame"), frame.data(), 0, frame.size() * sizeof(float));
            ggml_backend_tensor_set(j_res.input_tensors.at("decoder_out"), top_h.data(), 0, top_h.size() * sizeof(float));
            ggml_backend_graph_compute(backend_, j_res.graph);

            const auto n_combined = static_cast<uint32_t>(j_res.output->ne[0]);
            if (n_combined <= n_durations) {
                throw Error("TdtDecoder: joint output width (" + std::to_string(n_combined) +
                            ") must exceed the number of duration classes (" + std::to_string(n_durations) + ")");
            }
            const uint32_t n_token_classes = n_combined - n_durations;
            std::vector<float> combined(n_combined);
            ggml_backend_tensor_get(j_res.output, combined.data(), 0, combined.size() * sizeof(float));

            const int32_t k = argmax(combined.data(), n_token_classes);
            // Plain RNN-T (n_durations==0): no duration head at all -- every blank advances exactly one
            // frame (standard RNN-T greedy decoding), never a predicted duration. Guarded explicitly
            // rather than falling through to the TDT duration-argmax path below, which would read past
            // `combined`'s own end (n_durations==0) and index an empty `cfg_.durations`.
            uint32_t skip = 1;
            if (n_durations > 0) {
                const uint32_t d_idx = static_cast<uint32_t>(argmax(combined.data() + n_token_classes, n_durations));
                skip = cfg_.durations[d_idx];
            }

            if (k != cfg_.blank_id) {
                result.tokens.push_back(k);
                result.frame_indices.push_back(time_idx);
                h = h_new;
                c = c_new;
                last_label = k;
                if (n_durations == 0) skip = 0; // plain RNN-T: stay on this frame after a non-blank emission
            } else if (skip == 0) {
                skip = 1; // blank is forced to advance at least one frame -- otherwise decoding could spin forever
            }
            ++symbols_added;
            time_idx += skip;
            if (skip > 0) {
                advanced = true;
                break;
            }
        }
        if (!advanced) {
            // Defensive termination bound, not part of the real TDT algorithm itself (which relies on
            // blank-forcing alone): guards against a pathological model that keeps emitting non-blank
            // tokens with duration 0 forever, which would otherwise spin on the same frame indefinitely.
            // Same fallback as tools/fixture_gen/tdt_step_common.py's reference implementation (hit this
            // for real against that fixture's own randomly generated weights before adding it here too).
            ++time_idx;
        }
    }
    return result;
}

} // namespace loom
