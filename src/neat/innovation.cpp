#include "innovation.hpp"

namespace neat {

int InnovationTracker::connection(int in, int out) {
    auto key = std::make_pair(in, out);
    auto it = conn_this_gen_.find(key);
    if (it != conn_this_gen_.end()) return it->second;
    int innov = next_innov_++;
    conn_this_gen_[key] = innov;
    return innov;
}

InnovationTracker::Split InnovationTracker::split(int split_innov, int in, int out) {
    auto it = split_this_gen_.find(split_innov);
    if (it != split_this_gen_.end()) return it->second;
    Split s{next_node_id_++, 0, 0};
    s.innov_in  = connection(in, s.node_id);
    s.innov_out = connection(s.node_id, out);
    split_this_gen_[split_innov] = s;
    return s;
}

int InnovationTracker::duplicate(int node_id) {
    auto it = dup_this_gen_.find(node_id);
    if (it != dup_this_gen_.end()) return it->second;
    int id = next_node_id_++;
    dup_this_gen_[node_id] = id;
    return id;
}

}  // namespace neat
