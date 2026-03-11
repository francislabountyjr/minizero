#include "gungi.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace minizero::env::gungi {

namespace {

using Board = std::array<std::array<std::vector<GungiPiece>, kGungiBoardSize>, kGungiBoardSize>;
using GungiState = GungiEnv::GungiState;

constexpr int kBoardSpatialSize = kGungiBoardSize * kGungiBoardSize;
constexpr int kBoardNormalPlaneOffset = 0;
constexpr int kBoardCapturePlaneOffset = kGungiBoardDeltaPlaneCount;
constexpr int kBoardBetrayMask1PlaneOffset = 2 * kGungiBoardDeltaPlaneCount;
constexpr int kBoardBetrayMask2PlaneOffset = 3 * kGungiBoardDeltaPlaneCount;
constexpr int kBoardBetrayMask3PlaneOffset = 4 * kGungiBoardDeltaPlaneCount;
constexpr int kDropPlaneOffset = kGungiBoardPlaneCount;
constexpr int kHandFeatureOffset = 84;
constexpr int kTurnFeatureOffset = 112;
constexpr int kDraftFeatureOffset = 114;

struct Probe {
    int pval = 0;
    int carry = 1;
    bool infinite = false;
};

struct GeneratedMove {
    GungiMove move;
    GungiState after_state;
};

const std::array<std::string, kGungiNumPieceTypes> kPieceSymbols = {
    std::string{u8"\u5e25"},
    std::string{u8"\u5927"},
    std::string{u8"\u4e2d"},
    std::string{u8"\u5c0f"},
    std::string{u8"\u4f8d"},
    std::string{u8"\u69cd"},
    std::string{u8"\u99ac"},
    std::string{u8"\u5fcd"},
    std::string{u8"\u7826"},
    std::string{u8"\u5175"},
    std::string{u8"\u7832"},
    std::string{u8"\u5f13"},
    std::string{u8"\u7b52"},
    std::string{u8"\u8b00"},
};

const std::array<char, kGungiNumPieceTypes> kFenCodes = {'m', 'g', 'i', 'j', 'w', 'n', 'r', 's', 'f', 'd', 'c', 'a', 'k', 't'};
const std::array<int, kGungiNumPieceTypes> kInitialHandCounts = {1, 1, 1, 2, 2, 3, 2, 2, 2, 4, 1, 2, 1, 1};
const std::array<std::pair<int, int>, 8> kDirections = {{{-1, 1}, {-1, 0}, {-1, -1}, {0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}}};
const std::array<std::array<Probe, 8>, kGungiNumPieceTypes> kPieceProbes = {{
    {{{1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}}},
    {{{1, 1, false}, {1, 1, true}, {1, 1, false}, {1, 1, true}, {1, 1, true}, {1, 1, false}, {1, 1, true}, {1, 1, false}}},
    {{{1, 1, true}, {1, 1, false}, {1, 1, true}, {1, 1, false}, {1, 1, false}, {1, 1, true}, {1, 1, false}, {1, 1, true}}},
    {{{1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{1, 1, false}, {1, 1, false}, {1, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{1, 1, false}, {1, 2, false}, {1, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{0, 1, false}, {1, 2, false}, {0, 1, false}, {1, 1, false}, {1, 1, false}, {0, 1, false}, {1, 2, false}, {0, 1, false}}},
    {{{1, 2, false}, {0, 1, false}, {1, 2, false}, {0, 1, false}, {0, 1, false}, {1, 2, false}, {0, 1, false}, {1, 2, false}}},
    {{{0, 1, false}, {1, 1, false}, {0, 1, false}, {1, 1, false}, {1, 1, false}, {1, 1, false}, {0, 1, false}, {1, 1, false}}},
    {{{0, 1, false}, {1, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{0, 1, false}, {3, 1, false}, {0, 1, false}, {1, 1, false}, {1, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{2, 1, false}, {2, 1, false}, {2, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
    {{{0, 1, false}, {2, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}, {1, 1, false}}},
    {{{1, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}, {0, 1, false}, {0, 1, false}, {1, 1, false}, {0, 1, false}}},
}};

bool g_initialized = false;
std::vector<std::pair<int, int>> g_policy_deltas;
std::unordered_map<int, int> g_delta_to_index;

const std::string& getTsukeMarker()
{
    static const std::string marker{u8"\u4ed8"};
    return marker;
}

const std::string& getTakeMarker()
{
    static const std::string marker{u8"\u53d6"};
    return marker;
}

const std::string& getBetrayMarker()
{
    static const std::string marker{u8"\u8fd4"};
    return marker;
}

const std::string& getArataMarker()
{
    static const std::string marker{u8"\u65b0"};
    return marker;
}

const std::string& getDraftEndMarker()
{
    static const std::string marker{u8"\u7d42"};
    return marker;
}

inline bool isBlack(Player player) { return player == Player::kPlayer1; }
inline bool isWhite(Player player) { return player == Player::kPlayer2; }
inline char playerToFenChar(Player player) { return isBlack(player) ? 'b' : 'w'; }
inline std::string playerToFenString(Player player) { return std::string(1, playerToFenChar(player)); }

int deltaKey(int dr, int df)
{
    return (dr + 8) * 32 + (df + 8);
}

int pieceIndex(GungiPieceType piece_type)
{
    return static_cast<int>(piece_type);
}

char pieceTypeToFenCode(GungiPieceType piece_type)
{
    return kFenCodes[pieceIndex(piece_type)];
}

std::string pieceTypeToSymbol(GungiPieceType piece_type)
{
    return kPieceSymbols[pieceIndex(piece_type)];
}

GungiPieceType fenCodeToPieceType(char code)
{
    char lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(code)));
    for (int i = 0; i < kGungiNumPieceTypes; ++i) {
        if (kFenCodes[i] == lowered) { return static_cast<GungiPieceType>(i); }
    }
    throw std::runtime_error("invalid piece code");
}

bool onBoard(int rank, int file)
{
    return rank >= 1 && rank <= kGungiBoardSize && file >= 1 && file <= kGungiBoardSize;
}

int squareToCell(int rank, int file)
{
    return (rank - 1) * kGungiBoardSize + (file - 1);
}

std::pair<int, int> cellToSquare(int cell)
{
    return {cell / kGungiBoardSize + 1, cell % kGungiBoardSize + 1};
}

std::string squareToString(int rank, int file)
{
    return std::to_string(rank) + "-" + std::to_string(file);
}

std::string squareTierToString(int rank, int file, int tier)
{
    return squareToString(rank, file) + "-" + std::to_string(tier);
}

std::vector<GungiPiece>& getTower(Board& board, int rank, int file)
{
    return board[rank - 1][kGungiBoardSize - file];
}

const std::vector<GungiPiece>& getTower(const Board& board, int rank, int file)
{
    return board[rank - 1][kGungiBoardSize - file];
}

const GungiPiece* getTopPiece(const Board& board, int rank, int file)
{
    if (!onBoard(rank, file)) { return nullptr; }
    const auto& tower = getTower(board, rank, file);
    return tower.empty() ? nullptr : &tower.back();
}

bool samePiece(const GungiPiece& lhs, const GungiPiece& rhs)
{
    return lhs.type == rhs.type && lhs.player == rhs.player && lhs.rank == rhs.rank && lhs.file == rhs.file && lhs.tier == rhs.tier;
}

void putPiece(GungiState& state, const GungiPiece& piece)
{
    auto& tower = getTower(state.board, piece.rank, piece.file);
    tower.push_back(piece);
}

void removeTopPiece(GungiState& state, int rank, int file)
{
    auto& tower = getTower(state.board, rank, file);
    if (!tower.empty()) { tower.pop_back(); }
}

void removePieces(GungiState& state, int rank, int file, const std::vector<GungiPiece>& pieces)
{
    auto& tower = getTower(state.board, rank, file);
    tower.erase(std::remove_if(tower.begin(), tower.end(), [&](const GungiPiece& piece) {
                   return std::any_of(pieces.begin(), pieces.end(), [&](const GungiPiece& target) { return samePiece(piece, target); });
               }),
               tower.end());
}

void convertPieces(GungiState& state, int rank, int file, const std::vector<GungiPiece>& pieces)
{
    auto& tower = getTower(state.board, rank, file);
    for (auto& piece : tower) {
        if (std::any_of(pieces.begin(), pieces.end(), [&](const GungiPiece& target) { return samePiece(piece, target); })) {
            piece.player = getNextPlayer(piece.player, kGungiNumPlayer);
        }
    }
}

void updateHand(const std::vector<GungiPiece>& pieces, std::vector<GungiHandPiece>& hand, bool opposite_color = false)
{
    for (const auto& piece : pieces) {
        Player target_player = piece.player;
        if (opposite_color) { target_player = getNextPlayer(target_player, kGungiNumPlayer); }

        for (auto it = hand.begin(); it != hand.end(); ++it) {
            if (it->type == piece.type && it->player == target_player) {
                --it->count;
                if (it->count == 0) { hand.erase(it); }
                break;
            }
        }
    }
}

std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> parts;
    std::string token;
    std::istringstream iss(str);
    while (std::getline(iss, token, delimiter)) { parts.push_back(token); }
    return parts;
}

std::string normalizeSanInput(const std::string& san)
{
    if (san.empty()) { return san; }
    char suffix = san.back();
    if (suffix == '#' || suffix == '=') { return san.substr(0, san.size() - 1); }
    return san;
}

bool validateFen(const std::string& fen, std::string& error)
{
    auto parts = split(fen, ' ');
    if (parts.size() != 6) {
        error = "expected 6 fields";
        return false;
    }

    const auto& placement = parts[0];
    const auto& handpieces = parts[1];
    const auto& turn = parts[2];
    const auto& mode = parts[3];
    const auto& drafting = parts[4];
    const auto& move_number = parts[5];

    auto ranks = split(placement, '/');
    if (ranks.size() != kGungiBoardSize) {
        error = "invalid rank count";
        return false;
    }

    for (const auto& rank : ranks) {
        auto file_parts = split(rank, '|');
        int file_count = 0;
        for (const auto& file_part : file_parts) {
            if (file_part.find(':') != std::string::npos) {
                auto tower_pieces = split(file_part, ':');
                if (tower_pieces.empty() || tower_pieces.size() > kGungiMaxTier) {
                    error = "invalid tower";
                    return false;
                }
                ++file_count;
                for (const auto& tower_piece : tower_pieces) {
                    if (tower_piece.size() != 1 || std::string("mgijwnrsfdcaktMGIJWNRSFDCAKT").find(tower_piece[0]) == std::string::npos) {
                        error = "invalid tower piece";
                        return false;
                    }
                }
            } else {
                for (char c : file_part) {
                    if (std::isdigit(static_cast<unsigned char>(c))) {
                        file_count += c - '0';
                    } else if (std::string("mgijwnrsfdcaktMGIJWNRSFDCAKT").find(c) != std::string::npos) {
                        ++file_count;
                    } else {
                        error = "invalid board piece";
                        return false;
                    }
                }
            }
        }
        if (file_count != kGungiBoardSize) {
            error = "invalid file count";
            return false;
        }
    }

    auto hands = split(handpieces, '/');
    if (hands.size() != 2) {
        error = "invalid hand count";
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        const auto& hand = hands[i];
        if (hand == "-") { continue; }
        if (hand.size() % 2 != 0) {
            error = "invalid hand encoding";
            return false;
        }
        const std::string allowed = (i == 0 ? "MGIJWNRSFDCAKT" : "mgijwnrsfdcakt");
        for (size_t j = 0; j < hand.size(); j += 2) {
            if (allowed.find(hand[j]) == std::string::npos || !std::isdigit(static_cast<unsigned char>(hand[j + 1]))) {
                error = "invalid hand piece";
                return false;
            }
        }
    }

    if (turn != "b" && turn != "w") {
        error = "invalid turn";
        return false;
    }
    if (mode.size() != 1 || std::string("0123").find(mode[0]) == std::string::npos) {
        error = "invalid mode";
        return false;
    }
    if (drafting != "-" && drafting != "w" && drafting != "b" && drafting != "wb") {
        error = "invalid drafting";
        return false;
    }
    try {
        std::stoi(move_number);
    } catch (...) {
        error = "invalid move number";
        return false;
    }
    return true;
}

std::optional<GungiSetupMode> configuredRuleset()
{
    const auto& ruleset = minizero::config::env_gungi_ruleset;
    if (ruleset == "any") { return std::nullopt; }
    if (ruleset == "intro") { return GungiSetupMode::kIntro; }
    if (ruleset == "beginner") { return GungiSetupMode::kBeginner; }
    if (ruleset == "intermediate") { return GungiSetupMode::kIntermediate; }
    if (ruleset == "advanced") { return GungiSetupMode::kAdvanced; }
    return GungiSetupMode::kAdvanced;
}

bool rulesetMatchesConfiguration(GungiSetupMode mode)
{
    const auto expected = configuredRuleset();
    return !expected.has_value() || *expected == mode;
}

bool parseFen(const std::string& fen, GungiState& state, std::string& error)
{
    if (!validateFen(fen, error)) { return false; }

    for (auto& rank : state.board) {
        for (auto& tower : rank) { tower.clear(); }
    }
    state.hand.clear();

    auto parts = split(fen, ' ');
    const auto& placement = parts[0];
    const auto& handpieces = parts[1];

    auto ranks = split(placement, '/');
    for (int rank_index = 0; rank_index < kGungiBoardSize; ++rank_index) {
        auto file_parts = split(ranks[rank_index], '|');
        int cell_index = 0;
        for (const auto& file_part : file_parts) {
            if (file_part.find(':') != std::string::npos) {
                auto tower_pieces = split(file_part, ':');
                std::vector<GungiPiece> tower;
                int file = kGungiBoardSize - cell_index;
                for (int tier = 0; tier < static_cast<int>(tower_pieces.size()); ++tier) {
                    char code = tower_pieces[tier][0];
                    tower.push_back(GungiPiece{
                        fenCodeToPieceType(code),
                        std::islower(static_cast<unsigned char>(code)) ? Player::kPlayer1 : Player::kPlayer2,
                        rank_index + 1,
                        file,
                        tier + 1,
                    });
                }
                state.board[rank_index][cell_index] = tower;
                ++cell_index;
            } else {
                for (char c : file_part) {
                    if (std::isdigit(static_cast<unsigned char>(c))) {
                        cell_index += c - '0';
                    } else {
                        int file = kGungiBoardSize - cell_index;
                        state.board[rank_index][cell_index] = {GungiPiece{
                            fenCodeToPieceType(c),
                            std::islower(static_cast<unsigned char>(c)) ? Player::kPlayer1 : Player::kPlayer2,
                            rank_index + 1,
                            file,
                            1,
                        }};
                        ++cell_index;
                    }
                }
            }
        }
    }

    auto hands = split(handpieces, '/');
    for (int hand_index = 0; hand_index < 2; ++hand_index) {
        const auto& hand = hands[hand_index];
        if (hand == "-") { continue; }
        for (size_t i = 0; i < hand.size(); i += 2) {
            state.hand.push_back(GungiHandPiece{
                fenCodeToPieceType(hand[i]),
                std::islower(static_cast<unsigned char>(hand[i])) ? Player::kPlayer1 : Player::kPlayer2,
                hand[i + 1] - '0',
            });
        }
    }

    state.turn = (parts[2] == "b" ? Player::kPlayer1 : Player::kPlayer2);
    state.mode = static_cast<GungiSetupMode>(std::stoi(parts[3]));
    state.drafting.fill(false);
    state.drafting[static_cast<int>(Player::kPlayer1)] = (parts[4].find('b') != std::string::npos);
    state.drafting[static_cast<int>(Player::kPlayer2)] = (parts[4].find('w') != std::string::npos);
    state.move_number = std::stoi(parts[5]);
    return true;
}

std::string encodeFen(const GungiState& state)
{
    std::ostringstream placement;
    for (int rank_index = 0; rank_index < kGungiBoardSize; ++rank_index) {
        int empty_count = 0;
        for (int cell_index = 0; cell_index < kGungiBoardSize; ++cell_index) {
            const auto& tower = state.board[rank_index][cell_index];
            if (tower.empty()) {
                ++empty_count;
                continue;
            }

            if (empty_count) {
                placement << empty_count;
                empty_count = 0;
            }

            if (tower.size() > 1) {
                placement << "|";
                for (size_t i = 0; i < tower.size(); ++i) {
                    char code = pieceTypeToFenCode(tower[i].type);
                    placement << (isBlack(tower[i].player) ? code : static_cast<char>(std::toupper(static_cast<unsigned char>(code))));
                    if (i + 1 < tower.size()) { placement << ":"; }
                }
                placement << "|";
            } else {
                char code = pieceTypeToFenCode(tower[0].type);
                placement << (isBlack(tower[0].player) ? code : static_cast<char>(std::toupper(static_cast<unsigned char>(code))));
            }
        }
        if (empty_count) { placement << empty_count; }
        if (rank_index + 1 < kGungiBoardSize) { placement << "/"; }
    }

    auto appendHand = [](std::ostringstream& oss, const std::vector<GungiHandPiece>& hand, Player player) {
        bool wrote_any = false;
        for (const auto& hand_piece : hand) {
            if (hand_piece.player != player) { continue; }
            char code = pieceTypeToFenCode(hand_piece.type);
            oss << (isBlack(player) ? code : static_cast<char>(std::toupper(static_cast<unsigned char>(code)))) << hand_piece.count;
            wrote_any = true;
        }
        if (!wrote_any) { oss << "-"; }
    };

    std::ostringstream hand_stream;
    appendHand(hand_stream, state.hand, Player::kPlayer2);
    hand_stream << "/";
    appendHand(hand_stream, state.hand, Player::kPlayer1);

    std::string drafting;
    if (state.drafting[static_cast<int>(Player::kPlayer2)]) { drafting += "w"; }
    if (state.drafting[static_cast<int>(Player::kPlayer1)]) { drafting += "b"; }
    if (drafting.empty()) { drafting = "-"; }

    return placement.str() + " " + hand_stream.str() + " " + playerToFenString(state.turn) + " " + std::to_string(static_cast<int>(state.mode)) + " " + drafting + " " + std::to_string(state.move_number);
}

std::string boardPlacementOnly(const GungiState& state)
{
    std::string fen = encodeFen(state);
    return fen.substr(0, fen.find(' '));
}

std::vector<GungiPiece> getAllPieces(const GungiState& state)
{
    std::vector<GungiPiece> pieces;
    for (const auto& rank : state.board) {
        for (const auto& tower : rank) {
            for (const auto& piece : tower) { pieces.push_back(piece); }
        }
    }
    return pieces;
}

bool boardGameOver(const GungiState& state)
{
    int marshal_count = 0;
    for (const auto& piece : getAllPieces(state)) {
        if (piece.type == GungiPieceType::kMarshal) { ++marshal_count; }
    }
    return marshal_count != 2;
}

Player remainingMarshalWinner(const GungiState& state)
{
    int black_marshals = 0;
    int white_marshals = 0;
    for (const auto& piece : getAllPieces(state)) {
        if (piece.type != GungiPieceType::kMarshal) { continue; }
        if (isBlack(piece.player)) {
            ++black_marshals;
        } else {
            ++white_marshals;
        }
    }

    if (black_marshals == 1 && white_marshals == 0) { return Player::kPlayer1; }
    if (white_marshals == 1 && black_marshals == 0) { return Player::kPlayer2; }
    return Player::kPlayerNone;
}

std::vector<std::pair<int, int>> getAvailableSquares(const GungiState& state, std::pair<int, int> direction, std::pair<int, int> start, std::pair<int, int> origin, int max_length)
{
    int y = start.first;
    int x = start.second;
    int py = origin.first;
    int px = origin.second;
    int dy = direction.first;
    int dx = direction.second;
    const GungiPiece* origin_piece = getTopPiece(state.board, py, px);
    if (!origin_piece) { return {}; }

    if (origin_piece->type == GungiPieceType::kArcher && std::abs(dy) == 1 && std::abs(dx) == 1) {
        const GungiPiece* wing = getTopPiece(state.board, py + dy, px + dx);
        if (wing && wing->tier > origin_piece->tier) { return {}; }
    }

    std::vector<std::pair<int, int>> available_squares;

    int reverse_y = y;
    int reverse_x = x;
    while (reverse_x != px || reverse_y != py) {
        reverse_x -= dx;
        reverse_y -= dy;

        const GungiPiece* piece = getTopPiece(state.board, reverse_y, reverse_x);
        if (piece && piece->tier > origin_piece->tier) { return {}; }

        int side = isBlack(origin_piece->player) ? -1 : 1;
        const GungiPiece* below = getTopPiece(state.board, reverse_y + side, reverse_x);
        if (below && below->rank == py && below->file == px) { break; }
    }

    int forward_y = y;
    int forward_x = x;
    int step = 0;
    while (step < max_length) {
        if (!onBoard(forward_y, forward_x)) { break; }

        const GungiPiece* piece = getTopPiece(state.board, forward_y, forward_x);
        if (piece && piece->tier > origin_piece->tier) { break; }

        available_squares.emplace_back(forward_y, forward_x);

        if (piece) {
            bool leap_piece = (origin_piece->type == GungiPieceType::kCannon || origin_piece->type == GungiPieceType::kMusketeer || origin_piece->type == GungiPieceType::kArcher);
            if (!leap_piece) { break; }

            bool moving_forward = isWhite(origin_piece->player) ? (dy < 0) : (dy > 0);
            if (!moving_forward) { break; }
        }

        forward_x += dx;
        forward_y += dy;
        ++step;
    }

    return available_squares;
}

std::vector<std::vector<GungiPiece>> generateCombinations(const std::vector<GungiPiece>& items)
{
    std::vector<std::vector<GungiPiece>> result;
    std::vector<GungiPiece> current;
    std::function<void(size_t)> helper = [&](size_t start) {
        if (!current.empty()) { result.push_back(current); }
        for (size_t index = start; index < items.size(); ++index) {
            current.push_back(items[index]);
            helper(index + 1);
            current.pop_back();
        }
    };
    helper(0);
    return result;
}

std::vector<std::vector<GungiPiece>> getBetrayalCombos(const std::vector<GungiPiece>& tower, const std::vector<GungiHandPiece>& hand, Player player)
{
    std::vector<GungiPiece> enemies;
    for (const auto& piece : tower) {
        if (piece.player != player) { enemies.push_back(piece); }
    }
    if (enemies.empty()) { return {}; }

    std::unordered_map<int, int> enemy_count_map;
    for (const auto& enemy : enemies) { ++enemy_count_map[pieceIndex(enemy.type)]; }

    std::vector<GungiPiece> betrayal_options;
    for (const auto& [type_index, count] : enemy_count_map) {
        bool can_pay = std::any_of(hand.begin(), hand.end(), [&](const GungiHandPiece& hand_piece) {
            return hand_piece.player == player && pieceIndex(hand_piece.type) == type_index && hand_piece.count >= count;
        });
        if (!can_pay) { continue; }
        for (const auto& enemy : enemies) {
            if (pieceIndex(enemy.type) == type_index) { betrayal_options.push_back(enemy); }
        }
    }
    return generateCombinations(betrayal_options);
}

std::pair<GungiState, std::string> makeMove(const GungiState& state, const GungiMove& move)
{
    GungiState next_state = state;
    GungiPiece placed_piece{move.piece_type, move.player, move.to_rank, move.to_file, move.to_tier};

    switch (move.type) {
        case GungiMoveType::kRoute:
        case GungiMoveType::kTsuke:
            if (move.from_tier != 0) { removeTopPiece(next_state, move.from_rank, move.from_file); }
            break;
        case GungiMoveType::kCapture:
            if (move.from_tier != 0) { removeTopPiece(next_state, move.from_rank, move.from_file); }
            removePieces(next_state, move.to_rank, move.to_file, move.captured);
            break;
        case GungiMoveType::kBetray:
            if (move.from_tier != 0) { removeTopPiece(next_state, move.from_rank, move.from_file); }
            convertPieces(next_state, move.to_rank, move.to_file, move.captured);
            updateHand(move.captured, next_state.hand, true);
            break;
        case GungiMoveType::kArata:
            updateHand({placed_piece}, next_state.hand, false);
            if (!move.captured.empty()) {
                convertPieces(next_state, move.to_rank, move.to_file, move.captured);
                updateHand(move.captured, next_state.hand, true);
            }
            if (move.draft_finished) {
                next_state.drafting[static_cast<int>(move.player)] = false;
                if (isBlack(move.player)) { next_state.drafting[static_cast<int>(Player::kPlayer2)] = false; }
            }
            break;
    }

    putPiece(next_state, placed_piece);

    if (state.turn == Player::kPlayer1 && (next_state.drafting[static_cast<int>(Player::kPlayer2)] || !next_state.drafting[static_cast<int>(Player::kPlayer1)])) {
        ++next_state.move_number;
    }
    if (state.turn == Player::kPlayer2 && !next_state.drafting[static_cast<int>(Player::kPlayer1)] && !next_state.drafting[static_cast<int>(Player::kPlayer2)] && move.draft_finished) {
        ++next_state.move_number;
    }

    if (next_state.drafting[static_cast<int>(Player::kPlayer2)] == next_state.drafting[static_cast<int>(Player::kPlayer1)]) {
        if (!move.draft_finished || state.turn != Player::kPlayer2) { next_state.turn = getNextPlayer(state.turn, kGungiNumPlayer); }
    } else {
        if (!next_state.drafting[static_cast<int>(Player::kPlayer1)] && state.turn == Player::kPlayer1) {
            next_state.turn = Player::kPlayer2;
        } else if (!next_state.drafting[static_cast<int>(Player::kPlayer2)] && state.turn == Player::kPlayer2) {
            next_state.turn = Player::kPlayer1;
        }
    }

    bool marshal_captured = std::any_of(move.captured.begin(), move.captured.end(), [](const GungiPiece& piece) { return piece.type == GungiPieceType::kMarshal; });
    return {next_state, marshal_captured ? "#" : ""};
}

GungiMove buildMove(const GungiState& state,
                    const GungiPiece& piece,
                    int to_rank,
                    int to_file,
                    int to_tier,
                    GungiMoveType move_type,
                    const std::vector<GungiPiece>& captured = {},
                    bool draft_finished = false)
{
    GungiMove move;
    move.type = move_type;
    move.piece_type = piece.type;
    move.player = piece.player;
    move.from_rank = (piece.tier == 0 ? 0 : piece.rank);
    move.from_file = (piece.tier == 0 ? 0 : piece.file);
    move.from_tier = piece.tier;
    move.to_rank = to_rank;
    move.to_file = to_file;
    move.to_tier = to_tier;
    move.draft_finished = draft_finished;
    move.captured = captured;
    move.before_fen = encodeFen(state);

    std::string from_string = (piece.tier == 0 ? "" : "(" + squareTierToString(piece.rank, piece.file, piece.tier) + ")");
    std::string arata = (move_type == GungiMoveType::kArata ? getArataMarker() : "");
    std::string capture = (move_type == GungiMoveType::kCapture ? getTakeMarker() : "");
    std::string tsuke = ((move_type == GungiMoveType::kTsuke) || (to_tier != 1 && to_tier - piece.tier > 0) ? getTsukeMarker() : "");
    std::string betray;
    if (move_type == GungiMoveType::kBetray || (move_type == GungiMoveType::kArata && !captured.empty())) {
        betray = getBetrayMarker();
        for (const auto& captured_piece : captured) { betray += pieceTypeToSymbol(captured_piece.type); }
    }
    std::string draft_done = (draft_finished ? getDraftEndMarker() : "");
    move.san = arata + pieceTypeToSymbol(piece.type) + from_string + capture + "(" + squareTierToString(to_rank, to_file, to_tier) + ")" + (!betray.empty() ? betray : tsuke) + draft_done;

    auto [after_state, game_over_san] = makeMove(state, move);
    move.after_fen = encodeFen(after_state);
    move.san += game_over_san;
    return move;
}

std::vector<GeneratedMove> generateMovesForSquare(const GungiState& state, int rank, int file)
{
    const GungiPiece* piece_ptr = getTopPiece(state.board, rank, file);
    if (!piece_ptr || state.turn != piece_ptr->player) { return {}; }
    GungiPiece piece = *piece_ptr;

    const auto& probes = kPieceProbes[pieceIndex(piece.type)];
    std::vector<std::pair<int, int>> targets;

    for (size_t i = 0; i < probes.size(); ++i) {
        const auto& probe = probes[i];
        if (probe.pval < 1 && !probe.infinite) { continue; }
        int dy = kDirections[i].first;
        int dx = kDirections[i].second;
        if (isBlack(piece.player)) {
            dy *= -1;
            dx *= -1;
        }

        int start_rank = (probe.infinite ? rank + dy : rank + probe.pval * dy);
        int start_file = file + dx;
        int max_length = (probe.infinite ? 8 : piece.tier + probe.carry - 1);

        auto available = getAvailableSquares(state, {dy, dx}, {start_rank, start_file}, {rank, file}, max_length);
        targets.insert(targets.end(), available.begin(), available.end());
    }

    std::vector<GeneratedMove> generated_moves;
    int max_tier = (state.mode == GungiSetupMode::kAdvanced ? 3 : 2);
    bool marshal_can_stack = (state.mode == GungiSetupMode::kAdvanced || state.mode == GungiSetupMode::kIntermediate);

    for (const auto& [to_rank, to_file] : targets) {
        const GungiPiece* top_piece = getTopPiece(state.board, to_rank, to_file);
        const auto& tower = getTower(state.board, to_rank, to_file);

        if (!top_piece || tower.empty()) {
            auto move = buildMove(state, piece, to_rank, to_file, 1, GungiMoveType::kRoute);
            generated_moves.push_back({move, makeMove(state, move).first});
            continue;
        }

        if (top_piece->tier < max_tier && top_piece->type != GungiPieceType::kMarshal) {
            if (piece.type != GungiPieceType::kMarshal || marshal_can_stack) {
                auto move = buildMove(state, piece, to_rank, to_file, top_piece->tier + 1, GungiMoveType::kTsuke);
                generated_moves.push_back({move, makeMove(state, move).first});
            }

            if (piece.type == GungiPieceType::kTactician && std::any_of(tower.begin(), tower.end(), [&](const GungiPiece& tower_piece) { return tower_piece.player != piece.player; })) {
                for (const auto& combo : getBetrayalCombos(tower, state.hand, piece.player)) {
                    auto move = buildMove(state, piece, to_rank, to_file, top_piece->tier + 1, GungiMoveType::kBetray, combo);
                    generated_moves.push_back({move, makeMove(state, move).first});
                }
            }
        }

        if (top_piece->player != piece.player) {
            int new_tier = static_cast<int>(std::count_if(tower.begin(), tower.end(), [&](const GungiPiece& tower_piece) { return tower_piece.player == piece.player; })) + 1;
            std::vector<GungiPiece> captured;
            for (const auto& tower_piece : tower) {
                if (tower_piece.player != piece.player) { captured.push_back(tower_piece); }
            }
            auto move = buildMove(state, piece, to_rank, to_file, new_tier, GungiMoveType::kCapture, captured);
            generated_moves.push_back({move, makeMove(state, move).first});
        }
    }

    return generated_moves;
}

std::vector<GeneratedMove> generateArata(const GungiState& state, const GungiHandPiece& hand_piece)
{
    if (state.turn != hand_piece.player) { return {}; }

    bool is_marshal_placed = !std::any_of(state.hand.begin(), state.hand.end(), [&](const GungiHandPiece& piece) {
        return piece.type == GungiPieceType::kMarshal && piece.player == hand_piece.player;
    });
    if (!is_marshal_placed && hand_piece.type != GungiPieceType::kMarshal) { return {}; }

    bool is_draft = state.drafting[static_cast<int>(Player::kPlayer1)] || state.drafting[static_cast<int>(Player::kPlayer2)];
    int max_tier = (state.mode == GungiSetupMode::kAdvanced ? 3 : 2);
    std::vector<int> ranks;
    std::vector<int> maybe;

    if (is_draft) {
        ranks = (isWhite(hand_piece.player) ? std::vector<int>{7, 8, 9} : std::vector<int>{1, 2, 3});
    } else {
        int start = (isBlack(hand_piece.player) ? 1 : 9);
        int end = (isBlack(hand_piece.player) ? 9 : 1);
        int step = (isBlack(hand_piece.player) ? 1 : -1);
        for (int current_rank = start; (isBlack(hand_piece.player) ? current_rank <= end : current_rank >= end); current_rank += step) {
            bool has_own_top = false;
            for (int file = 1; file <= kGungiBoardSize; ++file) {
                const GungiPiece* top = getTopPiece(state.board, current_rank, file);
                if (top && top->player == hand_piece.player) {
                    has_own_top = true;
                    break;
                }
            }
            if (has_own_top) {
                ranks.insert(ranks.end(), maybe.begin(), maybe.end());
                ranks.push_back(current_rank);
                maybe.clear();
            } else {
                maybe.push_back(current_rank);
            }
        }
    }

    std::vector<GeneratedMove> generated_moves;
    int player_hand_count = std::accumulate(state.hand.begin(), state.hand.end(), 0, [&](int sum, const GungiHandPiece& candidate) {
        return sum + (candidate.player == hand_piece.player ? candidate.count : 0);
    });
    bool is_last_piece = (player_hand_count == 1);

    for (int rank : ranks) {
        for (int file = 1; file <= kGungiBoardSize; ++file) {
            const GungiPiece* top_piece = getTopPiece(state.board, rank, file);
            const auto& tower = getTower(state.board, rank, file);

            if (top_piece && (top_piece->player != hand_piece.player || top_piece->tier >= max_tier || top_piece->type == GungiPieceType::kMarshal)) { continue; }

            GungiPiece arata_piece{hand_piece.type, hand_piece.player, rank, file, 0};
            int tier = (top_piece ? top_piece->tier : 0) + 1;
            std::vector<std::vector<GungiPiece>> betrayal_combos;
            if (hand_piece.type == GungiPieceType::kTactician && !tower.empty()) { betrayal_combos = getBetrayalCombos(tower, state.hand, hand_piece.player); }

            bool drafting_enabled = state.drafting[static_cast<int>(hand_piece.player)];
            auto push_move = [&](const std::vector<GungiPiece>& captured, bool draft_finished) {
                auto move = buildMove(state, arata_piece, rank, file, tier, GungiMoveType::kArata, captured, draft_finished);
                generated_moves.push_back({move, makeMove(state, move).first});
            };

            if (drafting_enabled) {
                if (!is_last_piece) {
                    push_move({}, false);
                    for (const auto& combo : betrayal_combos) { push_move(combo, false); }
                }
                push_move({}, true);
                for (const auto& combo : betrayal_combos) { push_move(combo, true); }
            } else {
                push_move({}, false);
                for (const auto& combo : betrayal_combos) { push_move(combo, false); }
            }
        }
    }

    return generated_moves;
}

std::vector<std::pair<int, int>> getAttackedSquares(const GungiState& state, int rank, int file)
{
    const GungiPiece* piece_ptr = getTopPiece(state.board, rank, file);
    if (!piece_ptr) { return {}; }
    GungiPiece piece = *piece_ptr;

    std::vector<std::pair<int, int>> attacked_squares;
    const auto& probes = kPieceProbes[pieceIndex(piece.type)];
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto& probe = probes[i];
        if (probe.pval < 1 && !probe.infinite) { continue; }
        int dy = kDirections[i].first;
        int dx = kDirections[i].second;
        if (isBlack(piece.player)) {
            dy *= -1;
            dx *= -1;
        }
        int start_rank = (probe.infinite ? rank + dy : rank + probe.pval * dy);
        int start_file = file + dx;
        int max_length = (probe.infinite ? 8 : piece.tier + probe.carry - 1);
        auto squares = getAvailableSquares(state, {dy, dx}, {start_rank, start_file}, {rank, file}, max_length);
        attacked_squares.insert(attacked_squares.end(), squares.begin(), squares.end());
    }
    return attacked_squares;
}

bool isSquareAttacked(const GungiState& state, int rank, int file, Player by_player)
{
    for (int source_rank = 1; source_rank <= kGungiBoardSize; ++source_rank) {
        for (int source_file = kGungiBoardSize; source_file >= 1; --source_file) {
            const GungiPiece* piece = getTopPiece(state.board, source_rank, source_file);
            if (!piece || piece->player != by_player) { continue; }
            auto attacked = getAttackedSquares(state, source_rank, source_file);
            if (std::any_of(attacked.begin(), attacked.end(), [&](const std::pair<int, int>& square) { return square.first == rank && square.second == file; })) {
                return true;
            }
        }
    }
    return false;
}

bool playerInCheck(const GungiState& state, Player player)
{
    Player check_player = (player == Player::kPlayerNone ? state.turn : player);
    Player opposite = getNextPlayer(check_player, kGungiNumPlayer);

    for (int rank = 1; rank <= kGungiBoardSize; ++rank) {
        for (int file = kGungiBoardSize; file >= 1; --file) {
            const GungiPiece* piece = getTopPiece(state.board, rank, file);
            if (piece && piece->type == GungiPieceType::kMarshal && piece->player == check_player) {
                return isSquareAttacked(state, rank, file, opposite);
            }
        }
    }
    return false;
}

int getBetrayMask(const GungiState& state, const GungiMove& move)
{
    if (move.type != GungiMoveType::kBetray && !(move.type == GungiMoveType::kArata && !move.captured.empty())) { return 0; }
    const auto& tower = getTower(state.board, move.to_rank, move.to_file);
    int mask = 0;
    for (size_t i = 0; i < tower.size(); ++i) {
        if (std::any_of(move.captured.begin(), move.captured.end(), [&](const GungiPiece& captured) { return samePiece(captured, tower[i]); })) {
            mask |= (1 << static_cast<int>(i));
        }
    }
    return mask;
}

int encodeActionId(const GungiState& state, const GungiMove& move)
{
    if (move.type == GungiMoveType::kArata) {
        int mask = getBetrayMask(state, move);
        int variant = (mask == 0 ? 0 : mask * 2) + (move.draft_finished ? 1 : 0);
        int plane = kDropPlaneOffset + pieceIndex(move.piece_type) * kGungiDropVariantCount + variant;
        return plane * kBoardSpatialSize + squareToCell(move.to_rank, move.to_file);
    }

    int delta_rank = move.to_rank - move.from_rank;
    int delta_file = move.to_file - move.from_file;
    auto it = g_delta_to_index.find(deltaKey(delta_rank, delta_file));
    if (it == g_delta_to_index.end()) { return -1; }

    int plane_offset = kBoardNormalPlaneOffset;
    if (move.type == GungiMoveType::kCapture) {
        plane_offset = kBoardCapturePlaneOffset;
    } else if (move.type == GungiMoveType::kBetray) {
        int mask = getBetrayMask(state, move);
        if (mask == 1) {
            plane_offset = kBoardBetrayMask1PlaneOffset;
        } else if (mask == 2) {
            plane_offset = kBoardBetrayMask2PlaneOffset;
        } else {
            plane_offset = kBoardBetrayMask3PlaneOffset;
        }
    }
    return (plane_offset + it->second) * kBoardSpatialSize + squareToCell(move.from_rank, move.from_file);
}

std::string decodeActionString(int action_id)
{
    if (action_id < 0 || action_id >= kGungiPolicySize) { return "invalid"; }

    int plane = action_id / kBoardSpatialSize;
    int cell = action_id % kBoardSpatialSize;
    auto [rank, file] = cellToSquare(cell);

    if (plane < kGungiBoardPlaneCount) {
        int variant = plane / kGungiBoardDeltaPlaneCount;
        auto [dr, df] = g_policy_deltas[plane % kGungiBoardDeltaPlaneCount];
        int to_rank = rank + dr;
        int to_file = file + df;
        std::string suffix = "n";
        if (variant == 1) {
            suffix = "x";
        } else if (variant == 2) {
            suffix = "b1";
        } else if (variant == 3) {
            suffix = "b2";
        } else if (variant == 4) {
            suffix = "b3";
        }
        return squareToString(rank, file) + "->" + squareToString(to_rank, to_file) + ":" + suffix;
    }

    int drop_plane = plane - kDropPlaneOffset;
    int piece_type_index = drop_plane / kGungiDropVariantCount;
    int variant = drop_plane % kGungiDropVariantCount;
    int mask = variant / 2;
    bool draft_finished = (variant % 2 == 1);
    std::string suffix = (mask == 0 ? "n" : "b" + std::to_string(mask));
    if (draft_finished) { suffix += "+end"; }
    return "@" + squareToString(rank, file) + ":" + std::string(1, kFenCodes[piece_type_index]) + ":" + suffix;
}

bool isCurrentPositionFourfold(const std::vector<std::string>& position_history)
{
    std::unordered_map<std::string, int> frequency;
    int best = 0;
    for (const auto& placement : position_history) {
        best = std::max(best, ++frequency[placement]);
    }
    return best == 4;
}

bool isInsufficientMaterialInternal(const GungiState& state)
{
    if (state.drafting[static_cast<int>(Player::kPlayer1)] || state.drafting[static_cast<int>(Player::kPlayer2)]) { return false; }

    if (std::any_of(state.hand.begin(), state.hand.end(), [](const GungiHandPiece& hand_piece) { return hand_piece.type != GungiPieceType::kMarshal; })) { return false; }

    auto pieces = getAllPieces(state);
    if (std::any_of(pieces.begin(), pieces.end(), [](const GungiPiece& piece) { return piece.type != GungiPieceType::kMarshal; })) { return false; }

    std::vector<GungiPiece> marshals;
    for (const auto& piece : pieces) {
        if (piece.type == GungiPieceType::kMarshal) { marshals.push_back(piece); }
    }
    if (marshals.size() != 2) { return false; }

    int rank_diff = std::abs(marshals[0].rank - marshals[1].rank);
    int file_diff = std::abs(marshals[0].file - marshals[1].file);
    bool is_adjacent = rank_diff <= 1 && file_diff <= 1;
    return !is_adjacent;
}

std::vector<GeneratedMove> generatePseudoMoves(const GungiState& state)
{
    std::vector<GeneratedMove> moves;
    bool drafting = state.drafting[static_cast<int>(Player::kPlayer1)] || state.drafting[static_cast<int>(Player::kPlayer2)];
    if (drafting) {
        for (const auto& hand_piece : state.hand) {
            auto hand_moves = generateArata(state, hand_piece);
            moves.insert(moves.end(), hand_moves.begin(), hand_moves.end());
        }
        return moves;
    }

    for (int rank = 1; rank <= kGungiBoardSize; ++rank) {
        for (int file = kGungiBoardSize; file >= 1; --file) {
            auto board_moves = generateMovesForSquare(state, rank, file);
            moves.insert(moves.end(), board_moves.begin(), board_moves.end());
        }
    }
    for (const auto& hand_piece : state.hand) {
        auto hand_moves = generateArata(state, hand_piece);
        moves.insert(moves.end(), hand_moves.begin(), hand_moves.end());
    }
    return moves;
}

bool hasEscapingMove(const GungiState& state)
{
    for (const auto& generated_move : generatePseudoMoves(state)) {
        if (!playerInCheck(generated_move.after_state, generated_move.move.player)) { return true; }
    }
    return false;
}

} // namespace

void initialize()
{
    if (g_initialized) { return; }
    g_initialized = true;

    std::set<std::pair<int, int>> deltas;
    for (int type_index = 0; type_index < kGungiNumPieceTypes; ++type_index) {
        for (int tier = 1; tier <= kGungiMaxTier; ++tier) {
            for (size_t probe_index = 0; probe_index < kDirections.size(); ++probe_index) {
                const auto& probe = kPieceProbes[type_index][probe_index];
                if (probe.pval < 1 && !probe.infinite) { continue; }

                int dy = kDirections[probe_index].first;
                int dx = kDirections[probe_index].second;
                int y = (probe.infinite ? dy : probe.pval * dy);
                int x = dx;
                int max_length = (probe.infinite ? 8 : tier + probe.carry - 1);
                for (int step = 0; step < max_length; ++step) {
                    if (std::abs(y) <= 8 && std::abs(x) <= 8) {
                        deltas.insert({y, x});
                        deltas.insert({-y, -x});
                    }
                    y += dy;
                    x += dx;
                }
            }
        }
    }

    g_policy_deltas.assign(deltas.begin(), deltas.end());
    if (static_cast<int>(g_policy_deltas.size()) != kGungiBoardDeltaPlaneCount) {
        throw std::runtime_error("unexpected Gungi policy delta count");
    }
    for (int i = 0; i < static_cast<int>(g_policy_deltas.size()); ++i) {
        g_delta_to_index[deltaKey(g_policy_deltas[i].first, g_policy_deltas[i].second)] = i;
    }
}

std::string getIntroPosition()
{
    return "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1 w 0 - 1";
}

std::string getBeginnerPosition()
{
    return "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1";
}

std::string getIntermediatePosition()
{
    return "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1";
}

std::string getAdvancedPosition()
{
    return "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1";
}

GungiAction::GungiAction(const std::vector<std::string>& action_string_args)
    : BaseAction(-1, action_string_args.empty() ? Player::kPlayerNone : charToPlayer(action_string_args[0][0]))
{
    if (action_string_args.size() >= 2 && !action_string_args.back().empty() && std::all_of(action_string_args.back().begin(), action_string_args.back().end(), ::isdigit)) {
        action_id_ = std::stoi(action_string_args.back());
    }
}

std::string GungiAction::toConsoleString() const
{
    return decodeActionString(action_id_);
}

GungiEnv::GungiEnv()
    : BaseBoardEnv<GungiAction>(kGungiBoardSize),
      initial_fen_(getAdvancedPosition())
{
    initialize();
    reset();
}

void GungiEnv::invalidateLegalCache()
{
    legal_cache_dirty_ = true;
}

bool GungiEnv::loadFenInternal(const std::string& fen, bool reset_history)
{
    GungiState next_state;
    std::string error;
    if (!parseFen(fen, next_state, error)) { return false; }
    if (!rulesetMatchesConfiguration(next_state.mode)) { return false; }

    state_ = next_state;
    turn_ = state_.turn;
    invalidateLegalCache();

    if (reset_history) {
        actions_.clear();
        observations_.clear();
        move_history_.clear();
        state_history_.clear();
        state_history_.push_back(state_);
        position_history_.clear();
        position_history_.push_back(boardPlacementOnly(state_));
    }
    return true;
}

void GungiEnv::setInitialFen(const std::string& fen)
{
    initial_fen_ = fen;
}

void GungiEnv::reset()
{
    loadFenInternal(initial_fen_, true);
}

bool GungiEnv::loadFen(const std::string& fen)
{
    setInitialFen(fen);
    return loadFenInternal(fen, true);
}

bool GungiEnv::load(const std::string& fen)
{
    return loadFenInternal(fen, true);
}

bool GungiEnv::actSan(const std::string& san)
{
    std::string normalized_san = normalizeSanInput(san);
    rebuildLegalCache();
    for (size_t i = 0; i < legal_moves_cache_.size(); ++i) {
        if (legal_moves_cache_[i].san == san || legal_moves_cache_[i].san == normalized_san) { return act(legal_actions_cache_[i]); }
    }
    return false;
}

bool GungiEnv::move(const std::string& san)
{
    std::string normalized_san = normalizeSanInput(san);
    for (const auto& generated_move : generatePseudoMoves(state_)) {
        if (generated_move.move.san == san || generated_move.move.san == normalized_san) {
            int action_id = encodeActionId(state_, generated_move.move);
            return applyMove(generated_move.move, action_id);
        }
    }
    return false;
}

bool GungiEnv::undo()
{
    if (state_history_.size() <= 1) { return false; }

    state_history_.pop_back();
    state_ = state_history_.back();
    turn_ = state_.turn;
    if (!actions_.empty()) { actions_.pop_back(); }
    if (!move_history_.empty()) { move_history_.pop_back(); }
    if (!position_history_.empty()) { position_history_.pop_back(); }
    invalidateLegalCache();
    return true;
}

bool GungiEnv::act(const std::vector<std::string>& action_string_args)
{
    if (action_string_args.size() < 2) { return false; }
    std::string action_string = action_string_args.back();
    std::string normalized_action_string = normalizeSanInput(action_string);
    rebuildLegalCache();

    for (size_t i = 0; i < legal_moves_cache_.size(); ++i) {
        if (legal_moves_cache_[i].san == action_string || legal_moves_cache_[i].san == normalized_action_string || legal_actions_cache_[i].toConsoleString() == action_string) { return act(legal_actions_cache_[i]); }
    }

    if (!action_string.empty() && std::all_of(action_string.begin(), action_string.end(), ::isdigit)) {
        return act(GungiAction(std::stoi(action_string), charToPlayer(action_string_args[0][0])));
    }
    return false;
}

void GungiEnv::rebuildLegalCache() const
{
    if (!legal_cache_dirty_) { return; }

    legal_actions_cache_.clear();
    legal_moves_cache_.clear();
    legal_lookup_.assign(kGungiPolicySize, -1);

    if (!inDraft() && (boardGameOver(state_) || isCurrentPositionFourfold(position_history_) || isInsufficientMaterialInternal(state_))) {
        legal_cache_dirty_ = false;
        return;
    }

    auto pseudo_moves = generatePseudoMoves(state_);
    for (const auto& generated_move : pseudo_moves) {
        int action_id = encodeActionId(state_, generated_move.move);
        if (action_id < 0 || action_id >= kGungiPolicySize) { continue; }
        if (legal_lookup_[action_id] != -1) { continue; }

        int index = static_cast<int>(legal_moves_cache_.size());
        legal_moves_cache_.push_back(generated_move.move);
        legal_actions_cache_.emplace_back(action_id, state_.turn);
        legal_lookup_[action_id] = index;
    }

    legal_cache_dirty_ = false;
}

bool GungiEnv::applyMove(const GungiMove& input_move, int action_id)
{
    GungiMove move = input_move;
    GungiState next_state = makeMove(state_, move).first;

    state_ = next_state;
    turn_ = state_.turn;
    actions_.emplace_back(action_id, move.player);

    bool has_hash_suffix = !move.san.empty() && move.san.back() == '#';
    bool has_equal_suffix = !move.san.empty() && move.san.back() == '=';
    state_history_.push_back(state_);
    position_history_.push_back(boardPlacementOnly(state_));
    if (!has_hash_suffix && !has_equal_suffix) {
        if (isCheckmate()) {
            move.san += "#";
        } else if (isStalemate() || isCurrentPositionFourfold(position_history_) || isInsufficientMaterial()) {
            move.san += "=";
        }
    }
    move_history_.push_back(move);
    invalidateLegalCache();
    return true;
}

bool GungiEnv::act(const GungiAction& action)
{
    rebuildLegalCache();
    if (action.getPlayer() != Player::kPlayerNone && action.getPlayer() != state_.turn) { return false; }
    if (action.getActionID() < 0 || action.getActionID() >= kGungiPolicySize) { return false; }

    int index = legal_lookup_[action.getActionID()];
    if (index < 0) { return false; }

    return applyMove(legal_moves_cache_[index], action.getActionID());
}

std::vector<GungiAction> GungiEnv::getLegalActions() const
{
    rebuildLegalCache();
    return legal_actions_cache_;
}

bool GungiEnv::isLegalAction(const GungiAction& action) const
{
    rebuildLegalCache();
    if (action.getPlayer() != Player::kPlayerNone && action.getPlayer() != state_.turn) { return false; }
    return action.getActionID() >= 0 && action.getActionID() < kGungiPolicySize && legal_lookup_[action.getActionID()] >= 0;
}

bool GungiEnv::isTerminal() const
{
    if (inDraft()) { return false; }
    if (boardGameOver(state_)) { return true; }
    if (isCurrentPositionFourfold(position_history_)) { return true; }
    if (isInsufficientMaterialInternal(state_)) { return true; }
    return isCheckmate() || isStalemate();
}

float GungiEnv::getEvalScore(bool is_resign) const
{
    Player winner = Player::kPlayerNone;
    if (is_resign) {
        winner = getNextPlayer(state_.turn, kGungiNumPlayer);
    } else if (boardGameOver(state_)) {
        winner = remainingMarshalWinner(state_);
    } else if (isCheckmate()) {
        winner = getNextPlayer(state_.turn, kGungiNumPlayer);
    }

    if (winner == Player::kPlayer1) { return 1.0f; }
    if (winner == Player::kPlayer2) { return -1.0f; }
    return 0.0f;
}

std::vector<float> GungiEnv::getFeatures(utils::Rotation rotation) const
{
    (void)rotation;
    std::vector<float> features(getNumInputChannels() * kBoardSpatialSize, 0.0f);

    for (int rank = 1; rank <= kGungiBoardSize; ++rank) {
        for (int file = 1; file <= kGungiBoardSize; ++file) {
            int cell = squareToCell(rank, file);
            const auto& tower = getTower(state_.board, rank, file);
            for (const auto& piece : tower) {
                int color_offset = (isBlack(piece.player) ? 0 : 42);
                int channel = color_offset + pieceIndex(piece.type) * 3 + (piece.tier - 1);
                features[channel * kBoardSpatialSize + cell] = 1.0f;
            }
        }
    }

    for (const auto& hand_piece : state_.hand) {
        int channel = kHandFeatureOffset + (isBlack(hand_piece.player) ? 0 : 14) + pieceIndex(hand_piece.type);
        float value = static_cast<float>(hand_piece.count) / static_cast<float>(kInitialHandCounts[pieceIndex(hand_piece.type)]);
        for (int cell = 0; cell < kBoardSpatialSize; ++cell) { features[channel * kBoardSpatialSize + cell] = value; }
    }

    for (int cell = 0; cell < kBoardSpatialSize; ++cell) {
        features[(kTurnFeatureOffset + 0) * kBoardSpatialSize + cell] = static_cast<float>(state_.turn == Player::kPlayer1);
        features[(kTurnFeatureOffset + 1) * kBoardSpatialSize + cell] = static_cast<float>(state_.turn == Player::kPlayer2);
        features[(kDraftFeatureOffset + 0) * kBoardSpatialSize + cell] = static_cast<float>(state_.drafting[static_cast<int>(Player::kPlayer1)]);
        features[(kDraftFeatureOffset + 1) * kBoardSpatialSize + cell] = static_cast<float>(state_.drafting[static_cast<int>(Player::kPlayer2)]);
    }

    return features;
}

std::vector<float> GungiEnv::getActionFeatures(const GungiAction& action, utils::Rotation rotation) const
{
    (void)action;
    (void)rotation;
    return {};
}

std::string GungiEnv::toString() const
{
    std::ostringstream oss;
    oss << "   ";
    for (int file = kGungiBoardSize; file >= 1; --file) { oss << " " << file << " "; }
    oss << "\n";
    for (int rank = 1; rank <= kGungiBoardSize; ++rank) {
        oss << " " << rank << " ";
        for (int file = kGungiBoardSize; file >= 1; --file) {
            const GungiPiece* top = getTopPiece(state_.board, rank, file);
            if (!top) {
                oss << " . ";
            } else {
                char code = pieceTypeToFenCode(top->type);
                oss << " " << (isBlack(top->player) ? code : static_cast<char>(std::toupper(static_cast<unsigned char>(code)))) << top->tier;
            }
        }
        oss << " " << rank << "\n";
    }
    oss << "   ";
    for (int file = kGungiBoardSize; file >= 1; --file) { oss << " " << file << " "; }
    return oss.str();
}

std::string GungiEnv::getFen() const
{
    return encodeFen(state_);
}

std::string GungiEnv::getTurnString() const
{
    return playerToFenString(state_.turn);
}

int GungiEnv::getMoveNumber() const
{
    return state_.move_number;
}

bool GungiEnv::inDraft() const
{
    return state_.drafting[static_cast<int>(Player::kPlayer1)] || state_.drafting[static_cast<int>(Player::kPlayer2)];
}

bool GungiEnv::inCheck(Player player) const
{
    return playerInCheck(state_, player);
}

bool GungiEnv::isCheckmate() const
{
    if (inDraft()) { return false; }
    if (boardGameOver(state_)) { return false; }
    return playerInCheck(state_, state_.turn) && !hasEscapingMove(state_);
}

bool GungiEnv::isStalemate() const
{
    if (inDraft()) { return false; }
    if (boardGameOver(state_)) { return false; }
    return !playerInCheck(state_, state_.turn) && !hasEscapingMove(state_);
}

bool GungiEnv::isInsufficientMaterial() const
{
    return isInsufficientMaterialInternal(state_);
}

bool GungiEnv::isDraw() const
{
    return isStalemate() || isCurrentPositionFourfold(position_history_) || isInsufficientMaterial();
}

bool GungiEnv::isFourfoldRepetition() const
{
    return isCurrentPositionFourfold(position_history_);
}

std::unordered_map<std::string, bool> GungiEnv::getDraftingRights() const
{
    return {
        {"b", state_.drafting[static_cast<int>(Player::kPlayer1)]},
        {"w", state_.drafting[static_cast<int>(Player::kPlayer2)]},
    };
}

std::vector<std::string> GungiEnv::getMoveStrings() const
{
    if (isTerminal()) { return {}; }
    std::vector<std::string> moves;
    auto generated_moves = generatePseudoMoves(state_);
    moves.reserve(generated_moves.size());
    for (const auto& generated_move : generated_moves) { moves.push_back(generated_move.move.san); }
    return moves;
}

std::vector<GungiMove> GungiEnv::getMovesVerbose() const
{
    if (isTerminal()) { return {}; }
    std::vector<GungiMove> moves;
    auto generated_moves = generatePseudoMoves(state_);
    moves.reserve(generated_moves.size());
    for (const auto& generated_move : generated_moves) { moves.push_back(generated_move.move); }
    return moves;
}

std::vector<std::string> GungiEnv::getLegalMoveStrings() const
{
    rebuildLegalCache();
    std::vector<std::string> moves;
    moves.reserve(legal_moves_cache_.size());
    for (const auto& move : legal_moves_cache_) { moves.push_back(move.san); }
    return moves;
}

std::vector<GungiMove> GungiEnv::getLegalMovesVerbose() const
{
    rebuildLegalCache();
    return legal_moves_cache_;
}

std::vector<std::string> GungiEnv::getMoveHistorySan() const
{
    std::vector<std::string> history;
    history.reserve(move_history_.size());
    for (const auto& move : move_history_) { history.push_back(move.san); }
    return history;
}

std::vector<GungiMove> GungiEnv::getMoveHistoryVerbose() const
{
    return move_history_;
}

std::vector<float> GungiEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation) const
{
    (void)pos;
    (void)rotation;
    return {};
}

} // namespace minizero::env::gungi
