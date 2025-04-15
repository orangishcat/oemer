from oemer.build_system import *
from oemer.dewarp import estimate_coords, dewarp
from oemer.ete import generate_pred
from oemer.ete import register_note_id
from oemer.note_group_extraction import extract as group_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.staffline_extraction import extract as staff_extract
from oemer.symbol_extraction import extract as symbol_extract


def note_to_midi(note, clef_type):
    """
    Convert a note (with a staff_line_pos and optional accidental) to a MIDI number.
    For G-clef, we use the order ['D', 'E', 'F', 'G', 'A', 'B', 'C'] with an octave base of 4;
    for F-clef, we use ['F', 'G', 'A', 'B', 'C', 'D', 'E'] with an octave base of 2.
    """
    if clef_type == ClefType.G_CLEF:
        pitch_order = ['D', 'E', 'F', 'G', 'A', 'B', 'C']
        base_octave = 5
        pitch_offset = 1
    else:  # Assume F-clef.
        pitch_order = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
        base_octave = 3
        pitch_offset = 3

    pos = int(note.staff_line_pos)
    # Get the note letter (step) from the staff position.
    step = pitch_order[pos % 7] if pos >= 0 else pitch_order[pos % -7]

    # Compute the octave.
    if pos - pitch_offset >= 0:
        octave = (pos + pitch_offset) // 7 + base_octave
    else:
        octave = -((-pos - pitch_offset) // 7) + base_octave

    # Map the step to a semitone offset relative to C.
    step_to_semitone = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    midi_num = step_to_semitone[step] + (octave * 12)

    # Adjust for accidentals.
    if note.sfn == SfnType.SHARP:
        midi_num += 1
    elif note.sfn == SfnType.FLAT:
        midi_num -= 1

    return midi_num


def note_events() -> list[dict]:
    """
    Process the notes layer and return a list of note events.
    Each event is a dictionary with keys:
      - "pitch": MIDI number (where middle C = 60)
      - "duration": Duration (in internal division units)
      - "start_time": Cumulative time offset on its track
      - "track": Track index (starting at 0)
      - "bbox": Bounding box of the note (if available)

    In each measure:
      - The key signature initializes an accidental state for each letter.
      - If a note explicitly shows an accidental (including natural), that accidental is stored.
      - Otherwise, a note without an explicit accidental will inherit the accidental
        (if any) from an earlier note in the same measure with the same letter.
    """
    # Get layers and voices.
    notes_layer = layers.get_layer('notes')
    voices = get_voices()
    group_container = sort_symbols(voices)

    # Generate measures (using a similar buffering strategy as MusicXMLBuilder.gen_measures).
    measures = []
    num = 1  # Measure numbering starts at 1.
    for grp, insts in group_container.items():
        buffer = []
        at_beginning = True
        double_barline = False
        for inst in insts:
            if isinstance(inst, Barline):
                if len(buffer) == 0:
                    # Double barline encountered.
                    double_barline = True
                else:
                    m = gen_measure(buffer, grp, num, at_beginning, double_barline)
                    measures.append(m)
                    num += 1
                    buffer = []
                    at_beginning = False
                    double_barline = False
                continue
            buffer.append(inst)
        if buffer:
            m = gen_measure(buffer, grp, num, at_beginning, double_barline)
            measures.append(m)

    # Initialize per-track cumulative time.
    total_tracks = get_total_track_nums()
    track_time = [0] * total_tracks

    # Initialize current clefs for each track (default to G-clef if not updated).
    current_clefs = [None] * total_tracks
    for t in range(total_tracks):
        default_clef = Clef()
        default_clef.track = t
        default_clef.label = ClefType.G_CLEF
        current_clefs[t] = default_clef

    events = []  # List to collect note events.

    # Process each measure sequentially.
    for m in measures:
        # Retrieve the measure's key and initialize accidental state.
        measure_key = m.get_key()
        accidental_state = {letter: None for letter in "ABCDEFG"}
        if measure_key.value > 0:
            for letter in SHARP_KEY_ORDER[:measure_key.value]:
                accidental_state[letter] = SfnType.SHARP
        elif measure_key.value < 0:
            for letter in FLAT_KEY_ORDER[:abs(measure_key.value)]:
                accidental_state[letter] = SfnType.FLAT

        # Update current clefs from the measure.
        for clef in m.clefs:
            current_clefs[clef.track] = clef

        # Process each symbol in the measure.
        for sym in m.symbols:
            # Skip symbols that don't affect note timing.
            if isinstance(sym, (Clef, Sfn)):
                continue

            track = sym.track
            dur = get_duration(sym)
            start_time = track_time[track]

            if isinstance(sym, Rest):
                # For rests, simply update the cumulative time.
                track_time[track] += dur
            elif isinstance(sym, Voice):
                # For a note (or chord), process each note in the Voice.
                for nid in sym.note_ids:
                    note_obj = notes_layer[nid]
                    # Determine the note's chroma (letter) from its staff position using the current clef.
                    chroma = get_chroma_pitch(note_obj.staff_line_pos, current_clefs[track].label)
                    # If the note shows an explicit accidental, update the accidental state.
                    if note_obj.sfn is not None:
                        accidental_state[chroma] = note_obj.sfn
                    else:
                        # Otherwise, inherit the accidental (if any) from a previous note in the measure.
                        note_obj.sfn = accidental_state[chroma]

                    # Compute the MIDI pitch (with middle C = 60) using the updated accidental.
                    pitch = note_to_midi(note_obj, current_clefs[track].label)
                    event = {
                        "pitch": pitch,
                        "duration": dur,
                        "start_time": start_time,
                        "track": track,
                        "bbox": note_obj.bbox if hasattr(note_obj, "bbox") else None
                    }
                    events.append(event)
                # After processing the note/chord, update the track's cumulative time.
                track_time[track] += dur

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
