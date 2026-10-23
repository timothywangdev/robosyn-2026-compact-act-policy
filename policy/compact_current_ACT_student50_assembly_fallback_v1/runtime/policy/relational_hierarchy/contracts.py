"""Versioned current-observation goal contract, shared by annotation and inference.

Validation establishes syntax, bounds and encoder compatibility, never whether
Qwen's named object, box, arm assignment or desired effect is visually correct.
No demonstration futures or task-ID features enter this encoder.
"""

import hashlib
import json
import math
import re

VERSION = "relational-goal-v2"
COORDINATE_SCALES = {"normalized_1": 1.0, "normalized_1000": 1000.0}
WIRE_COORDINATE_SYSTEM = "normalized_1000"
SLOTS = ("left", "right", "object_destination")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ROLES = ("unknown", "idle", "reach", "grasp", "hold", "move", "place", "release",
         "insert", "align", "transfer", "stabilize", "open", "close", "pour", "press")
RELATIONS = ("unknown", "near", "above", "inside", "on", "aligned_with", "touching",
             "toward", "away_from", "left_of", "right_of", "handover_to", "separated")
GRIPPERS = ("unknown", "open", "closed", "hold")
NAME_DIM = 32
ENTITY_DIM = NAME_DIM + len(CAMERAS) + 4 + 1
FEATURE_DIM = len(ROLES) + len(RELATIONS) + len(GRIPPERS) + 2 * ENTITY_DIM
MAX_RESPONSE_BYTES = 16384


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{name} requires exactly {sorted(expected)}")


def _choice(value, choices, name):
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")


def _entity(value, scale):
    _keys(value, ("name", "camera", "bbox"), "entity")
    if all(value[k] is None for k in ("name", "camera", "bbox")):
        return {"name": None, "camera": None, "bbox": None}
    name, camera, bbox = value["name"], value["camera"], value["bbox"]
    if type(name) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9 _-]{0,63}", name) or name.strip() != name:
        raise ValueError("Entity name must be 1..64 lowercase ASCII letters/digits/spaces/_/-")
    _choice(camera, CAMERAS, "entity.camera")
    if type(bbox) is not list or len(bbox) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox):
        raise ValueError("Entity bbox must contain four finite numeric coordinates")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= scale and 0 <= y0 < y1 <= scale):
        raise ValueError(f"Entity bbox requires increasing coordinates in declared0..{scale:g} units")
    return {"name": name, "camera": camera, "bbox": [float(x) / scale for x in bbox]}


def validate_contract(payload):
    _keys(payload, ("version", "coordinate_system", *SLOTS), "contract")
    if payload["version"] != VERSION:
        raise ValueError(f"Expected contract version {VERSION}")
    _choice(payload["coordinate_system"], COORDINATE_SCALES, "coordinate_system")
    scale = COORDINATE_SCALES[payload["coordinate_system"]]
    # Canonical storage/features are always0..1; subsequent validation is idempotent.
    result = {"version": VERSION, "coordinate_system": "normalized_1"}
    for name in SLOTS:
        slot = payload[name]
        _keys(slot, ("role", "source", "destination", "relation", "gripper"), name)
        _choice(slot["role"], ROLES, "role")
        _choice(slot["relation"], RELATIONS, "relation")
        _choice(slot["gripper"], GRIPPERS, "gripper")
        result[name] = {"role": slot["role"], "source": _entity(slot["source"], scale),
                        "destination": _entity(slot["destination"], scale),
                        "relation": slot["relation"], "gripper": slot["gripper"]}
    return result


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def parse_contract(text):
    if type(text) is not str or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Response must be a bounded JSON string")
    text = text.strip()
    # Accept only one exact outer JSON fence, not prose, nested or multiple blocks.
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
        if "```" in text:
            raise ValueError("Nested or multiple JSON fences are forbidden")
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value}")
    return validate_contract(json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=invalid_constant))


def unknown_contract():
    # New dictionaries on each call; caller edits cannot alias roles or examples.
    def entity():
        return {"name": None, "camera": None, "bbox": None}
    return {"version": VERSION, "coordinate_system": "normalized_1", **{name: {"role": "unknown", "source": entity(),
            "destination": entity(), "relation": "unknown", "gripper": "unknown"} for name in SLOTS}}


def _one_hot(value, choices):
    return [float(value == x) for x in choices]


def _entity_features(entity):
    if entity["name"] is None:
        return [0.0] * ENTITY_DIM
    # Fixed identity code, NOT a semantic embedding or grounding certificate.
    canonical_name = " ".join(entity["name"].split())
    digest = hashlib.sha256(canonical_name.encode("ascii")).digest()
    name_features = [(x - 127.5) / (127.5 * math.sqrt(NAME_DIM)) for x in digest]
    return name_features + _one_hot(entity["camera"], CAMERAS) + entity["bbox"] + [1.0]


def encode_contract(contract):
    contract = validate_contract(contract)
    features, valid = [], []
    for name in SLOTS:
        slot = contract[name]
        features.append(_one_hot(slot["role"], ROLES) + _one_hot(slot["relation"], RELATIONS)
                        + _one_hot(slot["gripper"], GRIPPERS) + _entity_features(slot["source"])
                        + _entity_features(slot["destination"]))
        grounded_source = slot["source"]["name"] is not None
        if name == "object_destination":
            valid.append(grounded_source and slot["destination"]["name"] is not None and slot["relation"] != "unknown")
        else:
            valid.append(grounded_source and slot["role"] != "unknown")
    return features, valid


def encode_contract_batch(contracts):
    if not contracts:
        raise ValueError("At least one contract is required")
    rows = [encode_contract(contract) for contract in contracts]
    return [x[0] for x in rows], [x[1] for x in rows]


def build_prompt(task, qpos):
    if type(task) is not str or not task.strip() or len(task) > 1000 or any(ord(c) < 32 for c in task):
        raise ValueError("Task must be a nonempty single line of at most1000 characters")
    if type(qpos) is not list or len(qpos) != 14 or any(type(x) not in (float, int) or not math.isfinite(x) for x in qpos):
        raise ValueError("qpos must be14 finite numeric joint/gripper values")
    return (
        "Use only these three CURRENT RGB views, measured qpos and task. No future images or expert actions are provided. "
        "Choose the NEXT COMMANDED ACTION for each arm and the desired object/destination relation. "
        "Use camera labels to identify arms; screen-left is not robot-left. An inactive arm may idle. "
        "Ground each named entity in its clearest camera. A robot finger or empty gripper is NOT the task object. "
        "Do not infer an object, grasp or contact from the task text or a closed gripper. "
        "Use consistent brief object names from the task only when visually supported. "
        "If identity is unclear or overexposed, leave that entity unknown; do not fabricate a box. "
        "Return ONE MINIFIED JSON object on one line, no explanation or markdown. "
        f"Top-level keys: version=\"{VERSION}\", coordinate_system=\"{WIRE_COORDINATE_SYSTEM}\", "
        "left, right, object_destination. Those three values are objects with exactly role,source,destination,relation,gripper. "
        "source and destination MUST be objects, NEVER bare null. Each entity has exactly name,camera,bbox. "
        'The ONLY unknown entity is {"name":null,"camera":null,"bbox":null}; do not use name="unknown". '
        "Known entity: lowercase short name, one allowed camera, tight bbox [x0,y0,x1,y1]. "
        "Coordinates use a 0..1000 grid for that camera: left/top=0, right/bottom=1000; increasing bounds, NOT pixels. "
        "Use a distinct destination/reference entity where relevant; otherwise use the unknown entity. "
        "role is a NEXT action verb; relation is its desired effect, not necessarily the current relation. "
        "gripper MUST be unknown,open,closed,hold: closed means desired closure; open means desired opening/release; "
        "hold means preserve opening. NEVER use close or release as gripper values (they are role verbs). "
        + f"Cameras={json.dumps(CAMERAS,separators=(',',':'))}; roles={json.dumps(ROLES,separators=(',',':'))}; "
        + f"relations={json.dumps(RELATIONS,separators=(',',':'))}. "
        "qpos: first7 values left arm, last7 right arm. Indices6/13 are normalized measured gripper openings: "
        "0=near-closed,1=near-open, small overshoot allowed; NOT meters, commands or proof of contact. "
        + f"Task={json.dumps(task)}; qpos={json.dumps(qpos,separators=(',',':'))}"
    )
