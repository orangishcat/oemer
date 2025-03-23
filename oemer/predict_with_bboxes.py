from oemer.build_system import *
from oemer.dewarp import estimate_coords, dewarp
from oemer.ete import generate_pred
from oemer.ete import register_note_id
from oemer.note_group_extraction import extract as group_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.staffline_extraction import extract as staff_extract
from oemer.symbol_extraction import extract as symbol_extract
from music21 import converter

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

    builder = MusicXMLBuilder()
    builder.build()
    xml = builder.to_musicxml()

    score = converter.parse(xml.decode())
    notes_info = []
    note_bboxes = [list(n.bbox) for n in notes if not n.invalid]
    note_index = 0
    for note in score.flatten().notes:
        for pitch in note.pitches:
            notes_info.append({"pitch": pitch.midi, "start_time": float(note.offset), "duration": note.duration.quarterLength, "bbox": note_bboxes[note_index]})
            note_index += 1

    return {"size": [staff.shape[1], staff.shape[0]], "notes_info": notes_info}