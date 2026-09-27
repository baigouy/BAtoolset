"""
Cell and bond CSV exports of a movie (pyTA 'Export cell data' / 'Export bond data', and the command line pipeline):
the per image tables of all the images are combined in a master database, joined, and saved with the calibrated
columns (see units.py).
"""
from batoolset.ta.database.sql import createMasterDB
from batoolset.ta.database.units import calibrated_query


def cells_export_query(master_db):
    """SQL of the cell export: cells_2D joined with the 3D measures, tracks, clones and image properties."""
    sql_command = 'SELECT * FROM cells_2D'
    available_tables = master_db.get_tables(force_lower_case=True)
    if 'cells_3d' in available_tables:
        sql_command += ' NATURAL JOIN cells_3D'
    if 'cell_tracks' in available_tables:
        sql_command += ' NATURAL JOIN cell_tracks'
    if 'cell_clones' in available_tables:
        sql_command += ' NATURAL LEFT JOIN cell_clones'
    if 'properties' in available_tables:
        sql_command += ' NATURAL JOIN properties'
    return calibrated_query(master_db, sql_command)


def bonds_export_query(master_db):
    """SQL of the bond export: bonds_2D joined with the 3D measures, bond tracks and image properties."""
    sql_command = 'SELECT * FROM bonds_2D'
    available_tables = master_db.get_tables(force_lower_case=True)
    if 'bonds_3d' in available_tables:
        sql_command += ' NATURAL JOIN bonds_3D'
    if 'bond_tracks' in available_tables:
        sql_command += ' NATURAL JOIN bond_tracks'
    if 'properties' in available_tables:
        sql_command += ' NATURAL JOIN properties'
    return calibrated_query(master_db, sql_command)


def export_cells_csv(lst, output_file):
    """Saves the cell data of all the images of lst to output_file (CSV)."""
    master_db = createMasterDB(lst, force_track_cells_db_update=True)
    try:
        master_db.save_query_to_csv_file(cells_export_query(master_db), output_file)
    finally:
        master_db.close()


def export_bonds_csv(lst, output_file):
    """Saves the bond data of all the images of lst to output_file (CSV)."""
    master_db = createMasterDB(lst)
    try:
        master_db.save_query_to_csv_file(bonds_export_query(master_db), output_file)
    finally:
        master_db.close()
