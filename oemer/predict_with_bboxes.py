from collections import defaultdict

from oemer.bbox import BBox
from oemer.build_system import *
from oemer.dewarp import estimate_coords, dewarp
from oemer.ete import generate_pred
from oemer.ete import register_note_id
from oemer.note_group_extraction import extract as group_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.staffline_extraction import extract as staff_extract
from oemer.symbol_extraction import extract as symbol_extract


def note_pos_to_midi(pos: int, acc: Optional[SfnType], clef_type: ClefType) -> int:
    """Compute MIDI number from staff position + accidental and clef."""
    # choose mapping
    if clef_type == ClefType.G_CLEF:
        order, base_oct, pitch_off = G_CLEF_POS_TO_PITCH, 5, 1
    else:
        order, base_oct, pitch_off = F_CLEF_POS_TO_PITCH, 3, 3

    step = order[pos % 7] if pos >= 0 else order[pos % -7]
    # exact floor/ceil octave math
    if pos - pitch_off >= 0:
        octave = math.floor((pos + pitch_off) / 7) + base_oct
    else:
        octave = -math.ceil((pos + pitch_off) / -7) + base_oct

    semis = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[step] + 12 * octave
    if acc == SfnType.SHARP:
        semis += 1
    elif acc == SfnType.FLAT:
        semis -= 1
    return semis


def expand_bbox(b1: BBox, b2: BBox) -> BBox:
    return min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])


def get_voices_info() -> List[Dict[str, Any]]:
    """Get information about all voices including bbox, clef, and position."""

    voice_info = defaultdict(lambda: {
        "clef": "unknown",
        "track": -1,
        "group": -1,
        "bbox": (100000, 100000, 0, 0)
    })

    for clef in layers.get_layer("clefs"):
        d = voice_info[clef.group * 2 + clef.track]
        d["clef"] = "treble" if clef.label.name == "G_CLEF" else "bass"
        d["track"] = clef.track
        d["group"] = clef.group
        d["bbox"] = expand_bbox(d["bbox"], clef.bbox)

    for barline in layers.get_layer("barlines"):
        for track in range(2):
            d = voice_info[barline.group * 2 + track]
            bbox = barline.bbox.copy()
            half_y = (bbox[3] - bbox[1]) * 0.6
            if track == 0:
                bbox[3] -= half_y
            else:
                bbox[1] += half_y
            d["bbox"] = expand_bbox(d["bbox"], bbox)

    for note in layers.get_layer("notes"):
        d = voice_info[note.group * 2 + note.track]
        d["bbox"] = expand_bbox(d["bbox"], note.bbox)

    return sorted(voice_info.values(), key=lambda values: values["group"] * 2 + values["track"])


def get_line_info(voices_info):
    line = defaultdict(lambda: {
        "clefs": [],
        "group": -1,
        "bbox": (100000, 100000, 0, 0)
    })
    for voice in voices_info:
        d = line[voice["group"]]
        d["clefs"].append(voice["clef"])
        d["group"] = voice["group"]
        d["bbox"] = expand_bbox(d["bbox"], voice["bbox"])
    return sorted(line.values(), key=lambda values: values["group"])


def note_events() -> List[Dict[str, Any]]:
    notes_layer = layers.get_layer('notes')
    voices = get_voices()
    group_container = sort_symbols(voices)

    # Build measures (same as MusicXMLBuilder.gen_measures)
    measures = []
    num = 1
    for grp, instances in group_container.items():
        buffer, at_beginning, dbl = [], True, False
        for inst in instances:
            if isinstance(inst, Barline):
                if not buffer:
                    dbl = True
                else:
                    measures.append(gen_measure(buffer, grp, num, at_beginning, dbl))
                    num += 1
                    buffer, at_beginning, dbl = [], False, False
                continue
            buffer.append(inst)
        if buffer:
            measures.append(gen_measure(buffer, grp, num, at_beginning, dbl))

    total_tracks = get_total_track_nums()

    # Continuous clef and accidental state across the entire page
    current_clefs = []
    for t in range(total_tracks):
        c = Clef();
        c.track = t;
        c.label = ClefType.G_CLEF
        current_clefs.append(c)

    # Initialize accidentals once from the first measure’s key
    accidental_state: dict[str, SfnType | None] = {L: None for L in "ABCDEFG"}
    first_key = measures[0].get_key().value
    if first_key > 0:
        for L in SHARP_KEY_ORDER[:first_key]:
            accidental_state[L] = SfnType.SHARP
    elif first_key < 0:
        for L in FLAT_KEY_ORDER[:abs(first_key)]:
            accidental_state[L] = SfnType.FLAT

    events: List[Dict[str, Any]] = []
    measure_offset = 0  # cumulative time from all previous measures

    for m in measures:
        # per-measure local time trackers
        local_time = [0] * total_tracks
        measure_events: List[Dict[str, Any]] = []

        for sym in m.symbols:
            # 1) clef changes take effect immediately
            if isinstance(sym, Clef):
                current_clefs[sym.track] = sym
                continue
            # 2) skip accidental symbols (we’re not resetting per measure)
            if isinstance(sym, Sfn):
                continue

            tr = sym.track
            dur = get_duration(sym)
            start_local = local_time[tr]

            if isinstance(sym, Rest):
                local_time[tr] += dur
                continue

            # 3) for notes (Voice), collect events with local start
            clef_type = current_clefs[tr].label
            for nid in sym.note_ids:
                n = notes_layer[nid]
                letter = get_chroma_pitch(n.staff_line_pos, clef_type)

                # explicit accidental? update; else inherit
                if n.sfn is not None:
                    accidental_state[letter] = n.sfn
                    acc = n.sfn
                else:
                    acc = accidental_state[letter]

                midi = note_pos_to_midi(int(n.staff_line_pos), acc, clef_type)
                measure_events.append({
                    "pitch": midi,
                    "duration": dur,
                    "track": tr,
                    "bbox": getattr(n, "bbox", None),
                    "start_local": start_local
                })

            # advance local time for this track
            local_time[tr] += dur

        # 4) compute how long this measure lasted (max across tracks)
        measure_length = max(local_time)

        # 5) finalize each event’s absolute start_time
        for ev in measure_events:
            ev["start_time"] = measure_offset + ev.pop("start_local")
            events.append(ev)

        # 6) bump the offset for the next measure
        measure_offset += measure_length

    return events


def predict_bboxes(img_path: str, deskew: bool):
    staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(img_path)

    if deskew:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)

    layers._layers = {}
    layers._access_count = {}
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols > 1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)

    # ---- Extract staff lines and group informations ---- #
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.

    # ---- Extract noteheads ---- #
    logger.info("Extracting noteheads")
    notes = note_extract()

    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))

    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64) - 1)
    register_note_id()

    # ---- Extract groups of note ---- #
    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    # ---- Extract symbols ---- #
    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    # ---- Parse rhythm ---- #
    logger.info("Extracting rhythm types")
    rhythm_extract()

    return {
        "size": [staff.shape[1], staff.shape[0]],
        "note_events": note_events(),
        "voice_info": (voice_info := get_voices_info()),
        "line_info": get_line_info(voice_info)
    }
