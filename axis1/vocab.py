"""Donor vocabularies for L3 (misleading names).

Each domain borrows the names of a semantically different domain from the benchmark set,
so the LLM sees a plausible, coherent, but wrong story (Blocksworld's `on` becomes, say,
Transport's `in`). Names are matched by arity where possible. If the donor domain runs out
of names, the extra words below are used, and only then a numeric suffix.
"""

DONOR = {
    "blocksworld": "transport",
    "transport": "blocksworld",
    "childsnack": "rovers",
    "rovers": "childsnack",
    "floortile": "miconic",
    "miconic": "floortile",
    "sokoban": "spanner",
    "spanner": "sokoban",
}

# Extra words in the donor's theme, used after the donor domain's own names run out.
EXTRA = {
    "transport": {
        "predicate": ["loaded", "parked", "fuelled", "delivered", "connected", "empty-truck", "destination"],
        "action": ["load-truck", "unload-truck", "refuel", "park", "deliver"],
        "type": ["truck", "depot", "cargo", "road-segment"],
    },
    "blocksworld": {
        "predicate": ["on", "on-table", "clear", "holding", "hand-empty", "above", "stacked"],
        "action": ["stack", "unstack", "pick-up", "put-down", "restack"],
        "type": ["block", "table", "tower", "hand"],
    },
    "rovers": {
        "predicate": ["calibrated", "have-image", "visible", "can-traverse", "equipped", "full", "communicated"],
        "action": ["navigate", "calibrate", "take-image", "sample-rock", "communicate", "drop-sample"],
        "type": ["rover", "waypoint", "camera", "lander", "objective", "store"],
    },
    "childsnack": {
        "predicate": ["served", "waiting", "ontray", "allergic", "no-gluten", "at-kitchen", "notexist"],
        "action": ["make-sandwich", "put-on-tray", "serve", "move-tray"],
        "type": ["child", "sandwich", "tray", "place", "bread", "filling"],
    },
    "miconic": {
        "predicate": ["boarded", "served", "lift-at", "origin", "destin", "above", "waiting"],
        "action": ["board", "depart", "up", "down", "open-door"],
        "type": ["passenger", "floor", "elevator", "button"],
    },
    "floortile": {
        "predicate": ["painted", "robot-at", "robot-has", "clear", "up", "down", "right", "left"],
        "action": ["paint-up", "paint-down", "change-color", "move-up", "move-down", "move-left", "move-right"],
        "type": ["robot", "tile", "color", "brush"],
    },
    "spanner": {
        "predicate": ["carrying", "usable", "tightened", "loose", "link", "at-man", "broken"],
        "action": ["walk", "pickup-spanner", "tighten-nut", "drop-spanner"],
        "type": ["man", "nut", "spanner", "location", "gate"],
    },
    "sokoban": {
        "predicate": ["at-robot", "box-at", "clear", "goal", "adjacent", "at-goal", "pushable"],
        "action": ["move", "push-to-goal", "push-to-nongoal", "pull"],
        "type": ["location", "box", "direction", "wall"],
    },
}

# Name used for untyped objects and variables at L3 (e.g. Blocksworld blocks become packages).
DEFAULT_TYPE = {
    "transport": "package",
    "blocksworld": "block",
    "rovers": "waypoint",
    "childsnack": "sandwich",
    "miconic": "passenger",
    "floortile": "tile",
    "spanner": "spanner",
    "sokoban": "box",
}
