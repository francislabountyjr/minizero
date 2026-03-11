#include "configuration.h"
#include "data_loader.h"
#include "environment.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace minizero;

std::shared_ptr<Environment> kEnvInstance;

Environment& getEnvInstance()
{
    if (!kEnvInstance) { kEnvInstance = std::make_shared<Environment>(); }
    return *kEnvInstance;
}

PYBIND11_MODULE(minizero_py, m)
{
    m.def("load_config_file", [](std::string file_name) {
        minizero::env::setUpEnv();
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromFile(file_name);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("load_config_string", [](std::string conf_str) {
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromString(conf_str);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("use_gumbel", []() { return config::actor_use_gumbel; });
    m.def("get_zero_replay_buffer", []() { return config::zero_replay_buffer; });
    m.def("use_per", []() { return config::learner_use_per; });
    m.def("get_training_step", []() { return config::learner_training_step; });
    m.def("get_training_display_step", []() { return config::learner_training_display_step; });
    m.def("get_batch_size", []() { return config::learner_batch_size; });
    m.def("get_muzero_unrolling_step", []() { return config::learner_muzero_unrolling_step; });
    m.def("get_n_step_return", []() { return config::learner_n_step_return; });
    m.def("get_optimizer", []() { return config::learner_optimizer; });
    m.def("get_learning_rate", []() { return config::learner_learning_rate; });
    m.def("get_momentum", []() { return config::learner_momentum; });
    m.def("get_weight_decay", []() { return config::learner_weight_decay; });
    m.def("get_value_loss_scale", []() { return config::learner_value_loss_scale; });
    m.def("get_game_name", []() { return getEnvInstance().name(); });
    m.def("get_nn_num_input_channels", []() { return getEnvInstance().getNumInputChannels(); });
    m.def("get_nn_input_channel_height", []() { return getEnvInstance().getInputChannelHeight(); });
    m.def("get_nn_input_channel_width", []() { return getEnvInstance().getInputChannelWidth(); });
    m.def("get_nn_num_hidden_channels", []() { return config::nn_num_hidden_channels; });
    m.def("get_nn_hidden_channel_height", []() { return getEnvInstance().getHiddenChannelHeight(); });
    m.def("get_nn_hidden_channel_width", []() { return getEnvInstance().getHiddenChannelWidth(); });
    m.def("get_nn_num_action_feature_channels", []() { return getEnvInstance().getNumActionFeatureChannels(); });
    m.def("get_nn_num_blocks", []() { return config::nn_num_blocks; });
    m.def("get_nn_action_size", []() { return getEnvInstance().getPolicySize(); });
    m.def("get_nn_num_value_hidden_channels", []() { return config::nn_num_value_hidden_channels; });
    m.def("get_nn_discrete_value_size", []() { return kEnvInstance->getDiscreteValueSize(); });
    m.def("get_nn_type_name", []() { return config::nn_type_name; });

    py::class_<learner::DataLoader>(m, "DataLoader")
        .def(py::init<std::string>())
        .def("initialize", &learner::DataLoader::initialize)
        .def("load_data_from_file", &learner::DataLoader::loadDataFromFile, py::call_guard<py::gil_scoped_release>())
        .def(
            "update_priority", [](learner::DataLoader& data_loader, py::array_t<int>& sampled_index, py::array_t<float>& batch_values) {
                data_loader.updatePriority(static_cast<int*>(sampled_index.request().ptr), static_cast<float*>(batch_values.request().ptr));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](learner::DataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& action_features, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& reward, py::array_t<float>& loss_scale, py::array_t<int>& sampled_index) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->action_features_ = static_cast<float*>(action_features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.getSharedData()->getDataPtr()->reward_ = static_cast<float*>(reward.request().ptr);
                data_loader.getSharedData()->getDataPtr()->loss_scale_ = static_cast<float*>(loss_scale.request().ptr);
                data_loader.getSharedData()->getDataPtr()->sampled_index_ = static_cast<int*>(sampled_index.request().ptr);
                data_loader.sampleData();
            },
            py::call_guard<py::gil_scoped_release>());

#if GUNGI
    auto piece_symbol = [](minizero::env::gungi::GungiPieceType type) {
        using minizero::env::gungi::GungiPieceType;
        switch (type) {
            case GungiPieceType::kMarshal: return std::string{u8"\u5e25"};
            case GungiPieceType::kGeneral: return std::string{u8"\u5927"};
            case GungiPieceType::kLieutenantGeneral: return std::string{u8"\u4e2d"};
            case GungiPieceType::kMajorGeneral: return std::string{u8"\u5c0f"};
            case GungiPieceType::kWarrior: return std::string{u8"\u4f8d"};
            case GungiPieceType::kLancer: return std::string{u8"\u69cd"};
            case GungiPieceType::kRider: return std::string{u8"\u99ac"};
            case GungiPieceType::kSpy: return std::string{u8"\u5fcd"};
            case GungiPieceType::kFortress: return std::string{u8"\u7826"};
            case GungiPieceType::kSoldier: return std::string{u8"\u5175"};
            case GungiPieceType::kCannon: return std::string{u8"\u7832"};
            case GungiPieceType::kArcher: return std::string{u8"\u5f13"};
            case GungiPieceType::kMusketeer: return std::string{u8"\u7b52"};
            case GungiPieceType::kTactician: return std::string{u8"\u8b00"};
            default: return std::string{};
        }
    };

    auto piece_to_dict = [piece_symbol](const minizero::env::gungi::GungiPiece& piece) {
        py::dict result;
        result["type"] = piece_symbol(piece.type);
        result["color"] = std::string(1, piece.player == minizero::env::Player::kPlayer1 ? 'b' : 'w');
        result["square"] = std::to_string(piece.rank) + "-" + std::to_string(piece.file);
        result["tier"] = piece.tier;
        return result;
    };

    auto move_to_dict = [piece_symbol, piece_to_dict](const minizero::env::gungi::GungiMove& move) {
        py::dict result;
        result["san"] = move.san;
        result["piece"] = piece_symbol(move.piece_type);
        result["color"] = std::string(1, move.player == minizero::env::Player::kPlayer1 ? 'b' : 'w');
        result["from"] = (move.from_tier == 0 ? "" : std::to_string(move.from_rank) + "-" + std::to_string(move.from_file) + "-" + std::to_string(move.from_tier));
        result["to"] = std::to_string(move.to_rank) + "-" + std::to_string(move.to_file) + "-" + std::to_string(move.to_tier);
        switch (move.type) {
            case minizero::env::gungi::GungiMoveType::kRoute: result["type"] = "route"; break;
            case minizero::env::gungi::GungiMoveType::kCapture: result["type"] = "capture"; break;
            case minizero::env::gungi::GungiMoveType::kTsuke: result["type"] = "tsuke"; break;
            case minizero::env::gungi::GungiMoveType::kBetray: result["type"] = "betray"; break;
            case minizero::env::gungi::GungiMoveType::kArata: result["type"] = "arata"; break;
        }
        if (move.draft_finished) {
            result["draftFinished"] = py::bool_(true);
        } else {
            result["draftFinished"] = py::none();
        }
        py::list captured;
        for (const auto& piece : move.captured) { captured.append(piece_to_dict(piece)); }
        result["captured"] = captured;
        result["before"] = move.before_fen;
        result["after"] = move.after_fen;
        return result;
    };

    py::class_<minizero::env::gungi::GungiEnv>(m, "GungiEnv")
        .def(py::init<>())
        .def(py::init([](const std::string& fen) {
            auto env = std::make_unique<minizero::env::gungi::GungiEnv>();
            env->loadFen(fen);
            return env;
        }))
        .def("reset", &minizero::env::gungi::GungiEnv::reset)
        .def("load", &minizero::env::gungi::GungiEnv::load)
        .def("load_fen", &minizero::env::gungi::GungiEnv::loadFen)
        .def("fen", &minizero::env::gungi::GungiEnv::getFen)
        .def("turn", &minizero::env::gungi::GungiEnv::getTurnString)
        .def("move_number", &minizero::env::gungi::GungiEnv::getMoveNumber)
        .def("drafting", &minizero::env::gungi::GungiEnv::getDraftingRights)
        .def("in_draft", &minizero::env::gungi::GungiEnv::inDraft)
        .def("moves", &minizero::env::gungi::GungiEnv::getMoveStrings)
        .def("moves_verbose", [move_to_dict](const minizero::env::gungi::GungiEnv& env) {
            py::list moves;
            for (const auto& move : env.getMovesVerbose()) { moves.append(move_to_dict(move)); }
            return moves;
        })
        .def("legal_moves", &minizero::env::gungi::GungiEnv::getLegalMoveStrings)
        .def("legal_moves_verbose", [move_to_dict](const minizero::env::gungi::GungiEnv& env) {
            py::list moves;
            for (const auto& move : env.getLegalMovesVerbose()) { moves.append(move_to_dict(move)); }
            return moves;
        })
        .def("history", &minizero::env::gungi::GungiEnv::getMoveHistorySan)
        .def("history_verbose", [move_to_dict](const minizero::env::gungi::GungiEnv& env) {
            py::list moves;
            for (const auto& move : env.getMoveHistoryVerbose()) { moves.append(move_to_dict(move)); }
            return moves;
        })
        .def("move", &minizero::env::gungi::GungiEnv::move)
        .def("act_san", &minizero::env::gungi::GungiEnv::actSan)
        .def("undo", &minizero::env::gungi::GungiEnv::undo)
        .def("in_check", [](const minizero::env::gungi::GungiEnv& env) { return env.inCheck(); })
        .def("is_checkmate", &minizero::env::gungi::GungiEnv::isCheckmate)
        .def("is_stalemate", &minizero::env::gungi::GungiEnv::isStalemate)
        .def("is_insufficient_material", &minizero::env::gungi::GungiEnv::isInsufficientMaterial)
        .def("is_fourfold_repetition", &minizero::env::gungi::GungiEnv::isFourfoldRepetition)
        .def("is_draw", &minizero::env::gungi::GungiEnv::isDraw)
        .def("is_game_over", &minizero::env::gungi::GungiEnv::isTerminal)
        .def("eval_score", &minizero::env::gungi::GungiEnv::getEvalScore)
        .def("features", [](const minizero::env::gungi::GungiEnv& env) { return env.getFeatures(); });
#endif
}
