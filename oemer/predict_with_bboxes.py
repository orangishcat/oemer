from typing import Any, Dict, List, Optional

import math
import numpy as np

from oemer import layers
from oemer.bbox import BBox
from oemer.build_system import (
    Barline,
    Clef,
    ClefType,
    F_CLEF_POS_TO_PITCH,
    FLAT_KEY_ORDER,
    G_CLEF_POS_TO_PITCH,
    SHARP_KEY_ORDER,
    Sfn,
    SfnType,
    Rest,
    gen_measure,
    get_chroma_pitch,
    get_duration,
    get_total_track_nums,
    get_voices,
    sort_symbols,
)
from oemer.dewarp import estimate_coords, dewarp
from oemer.ete import generate_pred, register_note_id
from oemer.logger import get_logger
from oemer.note_group_extraction import extract as group_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.staffline_extraction import extract as staff_extract
from oemer.symbol_extraction import extract as symbol_extract


logger = get_logger(__name__)

INITIAL_BBOX: BBox = (100000, 100000, 0, 0)
TRACKS_PER_GROUP = 2


def note_pos_to_midi(pos: int, acc: Optional[SfnType], clef_type: ClefType) -> int:
    """Compute MIDI number from staff position + accidental and clef."""
    if clef_type == ClefType.G_CLEF:
        order, base_oct, pitch_off = G_CLEF_POS_TO_PITCH, 5, 1
    else:
        order, base_oct, pitch_off = F_CLEF_POS_TO_PITCH, 3, 3

    step = order[pos % 7] if pos >= 0 else order[pos % -7]
    if pos - pitch_off >= 0:
        octave = math.floor((pos + pitch_off) / 7) + base_oct
    else:
        octave = -math.ceil((pos + pitch_off) / -7) + base_oct

    semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + 12 * octave
    if acc == SfnType.SHARP:
        semis += 1
    elif acc == SfnType.FLAT:
        semis -= 1
    return semis


def expand_bbox(b1: BBox, b2: BBox) -> BBox:
    return (
        int(min(b1[0], b2[0])),
        int(min(b1[1], b2[1])),
        int(max(b1[2], b2[2])),
        int(max(b1[3], b2[3])),
    )


def get_voices_info() -> List[Dict[str, Any]]:
    """Get information about all voices including bbox, clef, and position."""

    voice_info: Dict[int, Dict[str, Any]] = {}

    def slot(group: int, track: int) -> Dict[str, Any]:
        idx = group * TRACKS_PER_GROUP + track
        if idx not in voice_info:
            voice_info[idx] = {
                "clef": "UNKNOWN",
                "track": track,
                "group": group,
                "bbox": INITIAL_BBOX,
            }
        return voice_info[idx]

    for clef in layers.get_layer("clefs"):
        entry = slot(clef.group, clef.track)
        entry["clef"] = "TREBLE" if clef.label == ClefType.G_CLEF else "BASS"
        entry["bbox"] = expand_bbox(entry["bbox"], tuple(map(int, clef.bbox)))

    for barline in layers.get_layer("barlines"):
        bbox = tuple(map(int, barline.bbox))
        half_y = int((bbox[3] - bbox[1]) * 0.6)
        boxes = (
            (bbox[0], bbox[1], bbox[2], bbox[3] - half_y),
            (bbox[0], bbox[1] + half_y, bbox[2], bbox[3]),
        )
        for track, track_bbox in enumerate(boxes):
            entry = slot(barline.group, track)
            entry["bbox"] = expand_bbox(entry["bbox"], track_bbox)

    for note in layers.get_layer("notes"):
        entry = slot(note.group, note.track)
        entry["bbox"] = expand_bbox(entry["bbox"], tuple(map(int, note.bbox)))

    return [voice_info[key] for key in sorted(voice_info.keys())]


def get_line_info(voices_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    line_map: Dict[int, Dict[str, Any]] = {}
    for voice in voices_info:
        group = voice["group"]
        entry = line_map.setdefault(group, {"clefs": [], "group": group, "bbox": INITIAL_BBOX})
        entry["clefs"].append(voice["clef"])
        entry["bbox"] = expand_bbox(entry["bbox"], voice["bbox"])
    return [line_map[key] for key in sorted(line_map.keys())]


def note_events() -> List[Dict[str, Any]]:
    notes_layer = layers.get_layer("notes")
    voices = get_voices()
    group_container = sort_symbols(voices)

    measures: List[Any] = []
    num = 1
    for grp, instances in group_container.items():
        buffer: List[Any] = []
        at_beginning, dbl = True, False
        for inst in instances:
            if isinstance(inst, Barline):
                if buffer:
                    measures.append(gen_measure(buffer, grp, num, at_beginning, dbl))
                    num += 1
                    buffer, at_beginning, dbl = [], False, False
                else:
                    dbl = True
                continue
            buffer.append(inst)
        if buffer:
            measures.append(gen_measure(buffer, grp, num, at_beginning, dbl))

    if not measures:
        return []

    total_tracks = get_total_track_nums()
    current_clefs: List[Clef] = []
    for track_idx in range(total_tracks):
        clef = Clef()
        clef.track = track_idx
        clef.label = ClefType.G_CLEF
        current_clefs.append(clef)

    accidental_state: Dict[str, Optional[SfnType]] = {letter: None for letter in "ABCDEFG"}
    first_key = measures[0].get_key().value
    if first_key > 0:
        for letter in SHARP_KEY_ORDER[:first_key]:
            accidental_state[letter] = SfnType.SHARP
    elif first_key < 0:
        for letter in FLAT_KEY_ORDER[:abs(first_key)]:
            accidental_state[letter] = SfnType.FLAT

    events: List[Dict[str, Any]] = []
    measure_offset = 0

    for measure in measures:
        local_time = [0] * total_tracks

        for sym in measure.symbols:
            if isinstance(sym, Clef):
                current_clefs[sym.track] = sym
                continue
            if isinstance(sym, Sfn):
                continue

            track = sym.track
            duration = get_duration(sym)
            start_local = local_time[track]

            if isinstance(sym, Rest):
                local_time[track] += duration
                continue

            clef_type = current_clefs[track].label
            for nid in sym.note_ids:
                note = notes_layer[nid]
                letter = get_chroma_pitch(note.staff_line_pos, clef_type)
                if note.sfn is not None:
                    accidental_state[letter] = note.sfn
                    acc = note.sfn
                else:
                    acc = accidental_state[letter]

                bbox = getattr(note, "bbox", None)
                events.append(
                    {
                        "pitch": note_pos_to_midi(int(note.staff_line_pos), acc, clef_type),
                        "duration": duration,
                        "track": track,
                        "bbox": list(map(int, bbox)) if bbox is not None else None,
                        "start_time": measure_offset + start_local,
                    }
                )

            local_time[track] += duration

        measure_offset += max(local_time, default=0)

    return events


def predict_bboxes(img_path: str, deskew: bool):
    staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(img_path)

    preds = {
        "staff": staff,
        "symbols": symbols,
        "stems_rests": stems_rests,
        "notehead": notehead,
        "clefs_keys": clefs_keys,
    }

    if deskew:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(preds["staff"])
        for key, layer in preds.items():
            preds[key] = dewarp(layer, coords_x, coords_y)

    layers._layers.clear()
    layers._access_count.clear()

    layers.register_layer("stems_rests_pred", preds["stems_rests"])
    layers.register_layer("clefs_keys_pred", preds["clefs_keys"])
    layers.register_layer("notehead_pred", preds["notehead"])
    merged_symbols = np.clip(
        preds["symbols"] + preds["clefs_keys"] + preds["stems_rests"],
        0,
        1,
    )
    layers.register_layer("symbols_pred", merged_symbols)
    layers.register_layer("staff_pred", preds["staff"])

    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", np.asarray(staffs))
    layers.register_layer("zones", zones)

    logger.info("Extracting noteheads")
    notes = note_extract()
    layers.register_layer("notes", np.asarray(notes))
    layers.register_layer("note_id", np.full(merged_symbols.shape, -1, dtype=np.int64))
    register_note_id()

    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer("note_groups", np.asarray(groups))
    layers.register_layer("group_map", group_map)

    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer("barlines", np.asarray(barlines))
    layers.register_layer("clefs", np.asarray(clefs))
    layers.register_layer("sfns", np.asarray(sfns))
    layers.register_layer("rests", np.asarray(rests))

    logger.info("Extracting rhythm types")
    rhythm_extract()

    voice_info = get_voices_info()
    return {
        "size": [preds["staff"].shape[1], preds["staff"].shape[0]],
        "notes": note_events(),
        "voices": voice_info,
        "lines": get_line_info(voice_info),
    }
