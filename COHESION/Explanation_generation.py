def calc_level(score, max_score):
    if score >= 0.8 * max_score:
        return "very high"
    elif score >= 0.6 * max_score:
        return "high"
    elif score >= 0.4 * max_score:
        return "moderate"
    elif score >= 0.2 * max_score:
        return "low"
    else:
        return "very low"

def calc_friendship_level(score, max_score):
    if score > 0.8 * max_score:
        return "very friendly"
    elif score > 0.6 * max_score:
        return "friendly"
    elif score > 0.4 * max_score:
        return "neutral"
    elif score > 0.2 * max_score:
        return "unfriendly"
    else:
        return "very unfriendly"


def transition(level1, level2):
    if "low" in level1 and "low" in level2:
        return "and"
    elif "high" in level1 and "high" in level2:
        return "and"
    else:
        return "but"

def generate_explanation(measure_weights, normalized_scores, S):
    maxS = sum(measure_weights)
    basic_info = f"The community shows {calc_level(S, maxS)} cohesion, reflecting its general orientation toward developing and maintaining social relationships.\n"
    
    # Individual level explanation
    EI_explanation = f"users in this community generally experience {calc_level(normalized_scores[0], 1)} enjoyment from interacting with other members"
    SIT_explanation = f"their relationships are generally {calc_friendship_level(normalized_scores[1], 1)}"
    CED_explanation = f"Compared to the enjoyment they received outside the community, they experience {calc_level(normalized_scores[2], 1)} enjoyment with community members"
    individual_level = f"--> From the individual level, {EI_explanation}, {SIT_explanation}. {CED_explanation}."

    # Group level explanation
    GIP_level, GID_level = calc_level(normalized_scores[3], 1), calc_level(normalized_scores[4], 1)
    connector = transition(GIP_level, GID_level)
    GIP_explanation = f"compared to post information alone, group members demonstrate {GIP_level} interaction preference"
    GID_explanation = f"their interaction frequency is {GID_level}"
    group_level = f"--> From the group level, {GIP_explanation}, {connector} {GID_explanation}."

    explanation = basic_info + individual_level + "\n" + group_level
    return explanation