#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace minizero::env::gungi {

const std::string kGungiName = "gungi";
const int kGungiBoardSize = 9;
const int kGungiNumPlayer = 2;
const int kGungiNumPieceTypes = 14;
const int kGungiMaxTier = 3;
const int kGungiBoardDeltaPlaneCount = 76;
const int kGungiBoardVariantCount = 5;
const int kGungiBoardPlaneCount = kGungiBoardDeltaPlaneCount * kGungiBoardVariantCount;
const int kGungiDropVariantCount = 8;
const int kGungiDropPlaneCount = kGungiNumPieceTypes * kGungiDropVariantCount;
const int kGungiPolicyPlaneCount = kGungiBoardPlaneCount + kGungiDropPlaneCount;
const int kGungiPolicySize = kGungiBoardSize * kGungiBoardSize * kGungiPolicyPlaneCount;

enum class GungiPieceType : int {
    kMarshal = 0,
    kGeneral,
    kLieutenantGeneral,
    kMajorGeneral,
    kWarrior,
    kLancer,
    kRider,
    kSpy,
    kFortress,
    kSoldier,
    kCannon,
    kArcher,
    kMusketeer,
    kTactician,
    kSize
};

enum class GungiSetupMode : int {
    kIntro = 0,
    kBeginner = 1,
    kIntermediate = 2,
    kAdvanced = 3
};

enum class GungiMoveType : int {
    kRoute = 0,
    kCapture,
    kTsuke,
    kBetray,
    kArata
};

struct GungiPiece {
    GungiPieceType type = GungiPieceType::kMarshal;
    Player player = Player::kPlayerNone;
    int rank = 0;
    int file = 0;
    int tier = 0;
};

struct GungiHandPiece {
    GungiPieceType type = GungiPieceType::kMarshal;
    Player player = Player::kPlayerNone;
    int count = 0;
};

struct GungiMove {
    GungiMoveType type = GungiMoveType::kRoute;
    GungiPieceType piece_type = GungiPieceType::kMarshal;
    Player player = Player::kPlayerNone;
    int from_rank = 0;
    int from_file = 0;
    int from_tier = 0;
    int to_rank = 0;
    int to_file = 0;
    int to_tier = 0;
    bool draft_finished = false;
    std::vector<GungiPiece> captured;
    std::string san;
    std::string before_fen;
    std::string after_fen;
};

void initialize();

std::string getIntroPosition();
std::string getBeginnerPosition();
std::string getIntermediatePosition();
std::string getAdvancedPosition();

class GungiAction : public BaseAction {
public:
    GungiAction() : BaseAction() {}
    GungiAction(int action_id, Player player) : BaseAction(action_id, player) {}
    explicit GungiAction(const std::vector<std::string>& action_string_args);

    Player nextPlayer() const override { return getNextPlayer(getPlayer(), kGungiNumPlayer); }
    std::string toConsoleString() const override;

    inline int getPlane() const { return action_id_ / (kGungiBoardSize * kGungiBoardSize); }
    inline int getCell() const { return action_id_ % (kGungiBoardSize * kGungiBoardSize); }
};

class GungiEnv : public BaseBoardEnv<GungiAction> {
public:
    struct GungiState {
        std::array<std::array<std::vector<GungiPiece>, kGungiBoardSize>, kGungiBoardSize> board{};
        std::vector<GungiHandPiece> hand;
        Player turn = Player::kPlayer2;
        GungiSetupMode mode = GungiSetupMode::kIntro;
        std::array<bool, static_cast<int>(Player::kPlayerSize)> drafting{{false, false, false}};
        int move_number = 1;
    };

    GungiEnv();

    void reset() override;
    bool act(const GungiAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<GungiAction> getLegalActions() const override;
    bool isLegalAction(const GungiAction& action) const override;
    bool isTerminal() const override;
    float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const GungiAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return 116; }
    inline int getNumActionFeatureChannels() const override { return 0; }
    inline int getPolicySize() const override { return kGungiPolicySize; }
    std::string toString() const override;
    inline std::string name() const override { return kGungiName; }
    inline int getNumPlayer() const override { return kGungiNumPlayer; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }

    bool loadFen(const std::string& fen);
    bool load(const std::string& fen);
    bool actSan(const std::string& san);
    bool move(const std::string& san);
    bool undo();

    std::string getFen() const;
    std::string getTurnString() const;
    int getMoveNumber() const;
    bool inDraft() const;
    bool inCheck(Player player = Player::kPlayerNone) const;
    bool isCheckmate() const;
    bool isStalemate() const;
    bool isInsufficientMaterial() const;
    bool isDraw() const;
    bool isFourfoldRepetition() const;
    std::unordered_map<std::string, bool> getDraftingRights() const;
    std::vector<std::string> getMoveStrings() const;
    std::vector<GungiMove> getMovesVerbose() const;
    std::vector<std::string> getLegalMoveStrings() const;
    std::vector<GungiMove> getLegalMovesVerbose() const;
    std::vector<std::string> getMoveHistorySan() const;
    std::vector<GungiMove> getMoveHistoryVerbose() const;

private:
    void invalidateLegalCache();
    void rebuildLegalCache() const;
    bool applyMove(const GungiMove& move, int action_id);

    bool loadFenInternal(const std::string& fen, bool reset_history);
    void setInitialFen(const std::string& fen);

    std::string initial_fen_;
    GungiState state_;
    std::vector<GungiState> state_history_;
    std::vector<GungiMove> move_history_;
    std::vector<std::string> position_history_;

    mutable bool legal_cache_dirty_ = true;
    mutable std::vector<GungiAction> legal_actions_cache_;
    mutable std::vector<GungiMove> legal_moves_cache_;
    mutable std::vector<int> legal_lookup_;
};

class GungiEnvLoader : public BaseBoardEnvLoader<GungiAction, GungiEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::vector<float> getValue(const int pos) const override { return {getReturn()}; }
    inline std::string name() const override { return kGungiName; }
    inline int getPolicySize() const override { return kGungiPolicySize; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
};

} // namespace minizero::env::gungi
