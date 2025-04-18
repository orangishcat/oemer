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
    """Compute MIDI number from staff position + accidental + clef."""
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


def note_events() -> List[Dict[str, Any]]:
    notes_layer = layers.get_layer('notes')
    voices = get_voices()
    group_container = sort_symbols(voices)

    # build measures…
    measures, num = [], 1
    for grp, insts in group_container.items():
        buf, at_beg, dbl = [], True, False
        for inst in insts:
            if isinstance(inst, Barline):
                if not buf:
                    dbl = True
                else:
                    m = gen_measure(buf, grp, num, at_beg, dbl)
                    measures.append(m)
                    num += 1
                    buf, at_beg, dbl = [], False, False
                continue
            buf.append(inst)
        if buf:
            measures.append(gen_measure(buf, grp, num, at_beg, dbl))
    total_tracks = get_total_track_nums()
    track_time = [0] * total_tracks

    # default clefs
    current_clefs = []
    for t in range(total_tracks):
        c = Clef()
        c.track = t
        c.label = ClefType.G_CLEF
        current_clefs.append(c)

    events = []
    for m in measures:
        # init accidental state from key
        ks = m.get_key().value
        acc_state = {L: None for L in "ABCDEFG"}
        if ks > 0:
            for L in SHARP_KEY_ORDER[:ks]:
                acc_state[L] = SfnType.SHARP
        elif ks < 0:
            for L in FLAT_KEY_ORDER[:abs(ks)]:
                acc_state[L] = SfnType.FLAT

        for sym in m.symbols:
            # handle clef change first
            if isinstance(sym, Clef):
                current_clefs[sym.track] = sym
                continue
            if isinstance(sym, Sfn):
                continue

            tr, dur = sym.track, get_duration(sym)
            start = track_time[tr]

            if isinstance(sym, Rest):
                track_time[tr] += dur
                continue

            # Voice → note(s)
            clef = current_clefs[tr].label
            for nid in sym.note_ids:
                n = notes_layer[nid]
                # determine letter
                letter = get_chroma_pitch(n.staff_line_pos, clef)
                # pick explicit or inherited
                if n.sfn is not None:
                    acc_state[letter] = n.sfn
                    acc = n.sfn
                else:
                    acc = acc_state[letter]
                midi = note_pos_to_midi(int(n.staff_line_pos), acc, clef)
                events.append({
                    "pitch": midi,
                    "duration": dur,
                    "start_time": start,
                    "track": tr,
                    "bbox": getattr(n, "bbox", None)
                })

            track_time[tr] += dur

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

    return {"size": [staff.shape[1], staff.shape[0]], "notes_info": note_events()}
