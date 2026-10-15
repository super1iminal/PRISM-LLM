"""Instance-specific conditions, read with the rule parser (src/rule_specificity.py)."""
from rule_specificity import blame_follow, kinds, list_counts, parse

POSITIONS, CONSTANTS = {"x", "y"}, {"N", "G2R", "G2C", "G3R", "G3C"}


def test_kinds_of_instance_specific_conditions():
    def k(condition):
        return kinds(parse(condition), POSITIONS, CONSTANTS)
    assert k("g2 & !g3 & y = G3C & x > G3R") == {"position"}
    assert k("y = G2C - 1 & !wall") == {"position", "offset"}
    assert k("G2R = G3R + 1 & toward") == {"constants", "offset"}
    assert k("x = 0 & !wall") == set() and k("!wall & closer") == set()


def test_conditions_count_once_per_list():
    counts = list_counts(["!wall & x = G2R", "!wall&x=G2R", "!wall & x = G2R", "x = 0", "true"], POSITIONS, CONSTANTS)
    assert counts == {"conditions": 3, "positions": 2, "specific": 1, "position": 1, "offset": 0, "constants": 0}


def test_blamed_rules_rewritten():
    prompt = ("## Your current rules\n\n```\n"
              '1. {"condition": "!wall & closer", "action": "up"}\n'
              '2. {"condition": "!wall", "action": "down"}\n'
              '3. {"condition": "true", "action": "left"}\n```\n\n'
              "## Where probability is lost\n\nShare ...:\n- 80%: rule 2 `!wall (action down)`. Most in: ...\n\n## Task\n")
    assert blame_follow(prompt, ["!wall & closer", "!wall & toward", "true"]) == (1, 1, 0, 2)
    assert blame_follow("## Your current rules\n\n```\n```\n\n## Task\n", []) is None
