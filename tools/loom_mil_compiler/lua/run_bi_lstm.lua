-- Host-steps one BiLSTM instance over a T x input_dim sequence, driving the per-timestep cell
-- topologies (`<ns>_fwd`/`_bwd`) the exporter's `recurrent.py` generates. ggml has no LSTM op; the
-- recurrence is genuinely host-side.
--
-- Each cell topology declares BOTH of the step's outputs, so one call per timestep per direction
-- returns `h_new, c_new` together. It used to be two calls against two topologies that shared an
-- identical node list and differed only in which output they declared -- which meant computing the
-- gate stack, the four gate VIEWs and the six elementwise ops twice for every timestep of every
-- BiLSTM. See `recurrent.py::_lstm_cell_topology`.
local function run_bi_lstm(namespace_, seq, hidden_dim)
    local T = #seq
    local out = {}
    for t = 1, T do out[t] = {} end

    local h_fwd, c_fwd = {}, {}
    for i = 1, hidden_dim do h_fwd[i], c_fwd[i] = 0.0, 0.0 end
    for t = 1, T do
        local h_new, c_new = loom.run_subgraph(namespace_ .. "_fwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_fwd, c_prev = c_fwd})
        h_fwd, c_fwd = h_new, c_new
        for i = 1, hidden_dim do out[t][i] = h_new[i] end
    end

    local h_bwd, c_bwd = {}, {}
    for i = 1, hidden_dim do h_bwd[i], c_bwd[i] = 0.0, 0.0 end
    for i = 0, T - 1 do
        local t = T - i
        local h_new, c_new = loom.run_subgraph(namespace_ .. "_bwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_bwd, c_prev = c_bwd})
        h_bwd, c_bwd = h_new, c_new
        for j = 1, hidden_dim do out[t][hidden_dim + j] = h_new[j] end
    end
    return out
end
