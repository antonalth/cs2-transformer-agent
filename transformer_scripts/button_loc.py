#!/usr/bin/env python3
'''
- Parse demo,
- extract all player inputs/tick; table name PLAYERINPUT
- round start - round stop;
- player recording timeframes (round#, starttick, endtick, isdeath?)
    - separate tables: playerinput, info for player pov recording (round#, starttick ,endtick, isdeath, (video_path, audio_path), (team)), separate table auxiliary information (money, roundcount)

'''

from __future__ import annotations
import argparse, sys, sqlite3, json
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser

# ── helper: define weapon categories for switch inference ─────────────────────
# Maps common in-game weapon names (as seen in `active_weapon_name` from demoparser)
# to a specific SWITCH key.
# This allows us to infer a weapon switch "button press" when the active weapon changes.
WEAPON_CATEGORIES = {
    # SWITCH_1: Primary Weapons (Rifles, SMGs, Shotguns, LMGs)
    "AK-47": "SWITCH_1", "M4A4": "SWITCH_1", "M4A1-S": "SWITCH_1",
    "Galil AR": "SWITCH_1", "FAMAS": "SWITCH_1", "AUG": "SWITCH_1",
    "SG 553": "SWITCH_1", "AWP": "SWITCH_1", "SSG 08": "SWITCH_1",
    "G3SG1": "SWITCH_1", "SCAR-20": "SWITCH_1", "MP9": "SWITCH_1",
    "MAC-10": "SWITCH_1", "MP7": "SWITCH_1", "MP5-SD": "SWITCH_1",
    "UMP-45": "SWITCH_1", "P90": "SWITCH_1", "PP-Bizon": "SWITCH_1",
    "Nova": "SWITCH_1", "XM1014": "SWITCH_1", "MAG-7": "SWITCH_1",
    "Sawed-Off": "SWITCH_1", "M249": "SWITCH_1", "Negev": "SWITCH_1",

    # SWITCH_2: Secondary Weapons (Pistols)
    "Glock-18": "SWITCH_2", "USP-S": "SWITCH_2", "P250": "SWITCH_2",
    "P2000": "SWITCH_2", "Dual Berettas": "SWITCH_2", "Five-SeveN": "SWITCH_2",
    "Tec-9": "SWITCH_2", "CZ75-Auto": "SWITCH_2", "Desert Eagle": "SWITCH_2",
    "R8 Revolver": "SWITCH_2",

    # SWITCH_3: Melee Weapons (Knives often just show "Knife" but can sometimes be more specific)
    "knife":"SWITCH_3","knife_ct": "SWITCH_3", "knife_t": "SWITCH_3", "Bayonet": "SWITCH_3", "Flip Knife": "SWITCH_3",
    "Gut Knife": "SWITCH_3", "Karambit": "SWITCH_3", "M9 Bayonet": "SWITCH_3",
    "Huntsman Knife": "SWITCH_3", "Falchion Knife": "SWITCH_3", "Bowie Knife": "SWITCH_3",
    "Butterfly Knife": "SWITCH_3", "Shadow Daggers": "SWITCH_3", "Ursus Knife": "SWITCH_3",
    "Navaja Knife": "SWITCH_3", "Stiletto Knife": "SWITCH_3", "Talon Knife": "SWITCH_3",
    "Classic Knife": "SWITCH_3", "Paracord Knife": "SWITCH_3", "Survival Knife": "SWITCH_3",
    "Nomad Knife": "SWITCH_3", "Skeleton Knife": "SWITCH_3",

    # SWITCH_4: Grenades
    "High Explosive Grenade": "SWITCH_4", "Flashbang": "SWITCH_4", "Smoke Grenade": "SWITCH_4",
    "Molotov": "SWITCH_4", "Incendiary Grenade": "SWITCH_4", "Decoy Grenade": "SWITCH_4",

    # SWITCH_5: Other (C4, Defuse Kit, Zeus)
    "C4 Explosive": "SWITCH_5", "Defuse Kit": "SWITCH_5", "Zeus x27": "SWITCH_3",
}

def get_weapon_switch_type(weapon_name: str | None) -> str | None:
    """Returns the SWITCH_X category for a given weapon name."""
    if not weapon_name:
        return None
    return WEAPON_CATEGORIES.get(weapon_name, f"SWITCH_UNDEFINED_{weapon_name}")


# ── helper: decode "buttons" bit-field ────────────────────────────────────────
KEY_MAPPING = {
    "IN_ATTACK":    1 << 0,
    "IN_JUMP":      1 << 1,
    "IN_DUCK":      1 << 2,
    "IN_FORWARD":   1 << 3,
    "IN_BACK":      1 << 4,
    "IN_USE":       1 << 5,
    "IN_CANCEL":    1 << 6,
    "IN_TURNLEFT":  1 << 7,
    "IN_TURNRIGHT": 1 << 8,
    "IN_MOVELEFT":  1 << 9,
    "IN_MOVERIGHT": 1 << 10,
    "IN_ATTACK2":   1 << 11,
    "IN_RELOAD":    1 << 13,
    "IN_ALT1":      1 << 14,
    "IN_ALT2":      1 << 15,
    "IN_SPEED":     1 << 16,
    "IN_WALK":      1 << 17,
    "IN_ZOOM":      1 << 18,
    "IN_WEAPON1":   1 << 19,
    "IN_WEAPON2":   1 << 20,
    "IN_BULLRUSH":  1 << 21,
    "IN_GRENADE1":  1 << 22,
    "IN_GRENADE2":  1 << 23,
    "IN_ATTACK3":   1 << 24,
    "UNKNOWN_25":   1 << 25,
    "UNKNOWN_26":   1 << 26,
    "UNKNOWN_27":   1 << 27,
    "UNKNOWN_28":   1 << 28,
    "UNKNOWN_29":   1 << 29,
    "UNKNOWN_30":   1 << 30,
    "UNKNOWN_31":   1 << 31,
    "IN_SCORE":     1 << 33,
    "IN_INSPECT":   1 << 35,
}

def extract_buttons(bits) -> list[str]:
    """
    Given a bit-field (possibly passed as a JS decimal string),
    return the list of all key names currently held.
    """
    bits_int = int(bits)
    return [
        name
        for name, mask in KEY_MAPPING.items()
        if bits_int & mask
    ]

# ── helpers: SQLite export ────────────────────────────────────────────────────
def _sanitize_inventory(inv):
    """Convert *inv* to a JSON string SQLite can store (lists/tuples → JSON)."""
    if isinstance(inv, (list, tuple, set)):
        try:
            return json.dumps(list(inv))
        except TypeError:
            return ",".join(map(str, inv))
    return str(inv)

# Modified _export_sqlite function
def _export_sqlite(db_path: Path, tick_df: pd.DataFrame) -> None:
    """Write combined *tick_df* info into an SQLite DB at *db_path*."""
    df = tick_df.copy()

    # stringify inventory lists
    df["inventory"] = df["inventory"].apply(_sanitize_inventory)

    # final shape - The `keyboard_input` column is now pre-computed in main()
    df.rename(columns={"name": "playername"}, inplace=True)
    df = df[["tick", "steamid", "playername", "keyboard_input", "inventory", "X", "Y", "Z", "active_weapon_name"]]

    # write to DB
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inputs (
            tick INTEGER NOT NULL,
            steamid INTEGER NOT NULL,
            playername TEXT,
            keyboard_input TEXT,
            inventory TEXT,
            x REAL,
            y REAL,
            z REAL,
            active_weapon TEXT,
            PRIMARY KEY (tick, steamid)
        )
        """
    )
    # Updated column list for INSERT OR REPLACE INTO - now 9 columns
    cur.executemany(
        "INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?,?,?)",
        df.to_records(index=False).tolist(),
    )
    con.commit()
    con.close()

# ── CLI setup ────────────────────────────────────────────────────────────────
def cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny CS-2 demo inspector / exporter")
    p.add_argument("demofile", type=Path, help="Path to .dem file")
    p.add_argument("--sqlout", type=Path, metavar="FILE.db",
                   help="Write results to SQLite DB instead of opening the CLI")
    return p

# ── main logic ───────────────────────────────────────────────────────────────
def main() -> None:
    args = cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo – one-off parse, please wait …]")
    dp = DemoParser(args.demofile.as_posix())

    # 1) per-tick table - Updated wanted_props
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick", "steamid", "userid", "name",
                "buttons", "inventory",
                "X", "Y", "Z",
                "active_weapon_name", "is_defusing",
            ]
        ).pipe(pd.DataFrame)
    )

    # 2) gap-fill so every (tick,steamid) exists
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid", group_keys=False)
        .apply(lambda g: g.reindex(range(g.index.min(), g.index.max() + 1)).ffill())
        .reset_index()
    )

    # 3) Infer weapon switches from active weapon changes
    print("[inferring weapon switches...]")
    # Get the previous tick's weapon for each player
    tick_df['prev_weapon'] = tick_df.groupby('steamid')['active_weapon_name'].shift(1)

    # Determine if the weapon changed from the previous tick
    # Also ensure that both current and previous weapons are not None/NaN to avoid false positives
    weapon_changed_mask = (tick_df['active_weapon_name'] != tick_df['prev_weapon']) & \
                          (tick_df['active_weapon_name'].notna()) & \
                          (tick_df['prev_weapon'].notna())

    # Get the SWITCH_X category for the *new* weapon
    tick_df['switch_type'] = tick_df['active_weapon_name'].apply(get_weapon_switch_type)

    # Create a new column that only contains a value on the tick a switch occurred
    tick_df['inferred_switch'] = ''
    tick_df.loc[weapon_changed_mask, 'inferred_switch'] = tick_df.loc[weapon_changed_mask, 'switch_type']
    tick_df['inferred_switch'].fillna('', inplace=True)


    # 4) Create final keyboard_input by combining real and inferred inputs
    # First, get the list of real keys from the bitfield
    tick_df['real_keys_list'] = tick_df['buttons'].apply(lambda b: extract_buttons(int(b)))

    # Then, combine the list of real keys with the inferred switch string
    def combine_inputs(row):
        keys = row['real_keys_list']
        switch = row['inferred_switch']
        if switch: # If an inferred switch exists for this tick...
            # Ensure we don't add duplicate switch types if somehow both IN_WEAPON1 and a switch were inferred
            # Although IN_WEAPON1/2 are often specific binds, the inferred switch is more generic.
            if switch not in keys:
                keys.append(switch) # ...add it to the list of inputs.
        return ",".join(keys)

    tick_df['keyboard_input'] = tick_df.apply(combine_inputs, axis=1)

    # Clean up temporary columns
    tick_df.drop(columns=['prev_weapon', 'switch_type', 'inferred_switch', 'real_keys_list', 'buttons'], inplace=True)


    # ── SQLite export path ────────────────────────────────────────────────────
    if args.sqlout:
        _export_sqlite(args.sqlout, tick_df)
        print(f"✓ wrote {args.sqlout} – done")
        return

    # 5) tiny REPL - Now uses the final `keyboard_input` column
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit / exit   leave\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in {"quit", "exit"}:
            break
        if not cmd.startswith("seek"):
            print("Unknown command. Use 'seek <tick>' or 'quit'.")
            continue

        parts = cmd.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit():
            print("Usage: seek <tick_number>")
            continue
        tgt_tick = int(parts[1])

        rows = tick_df.query("tick == @tgt_tick")
        if rows.empty:
            print(f"No data at tick {tgt_tick}.")
            continue

        players = rows[["steamid", "name"]].drop_duplicates().reset_index(drop=True)
        for i, r in players.iterrows():
            print(f"{i+1}. {r['name']}  (sid {r['steamid']})")

        sel = input("Player # > ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(players)):
            print("Invalid choice.")
            continue
        sid = players.iloc[int(sel)-1]["steamid"]

        r = rows.loc[rows["steamid"] == sid].iloc[-1]
        # We no longer call extract_buttons here; we use the pre-computed column
        print(
            f"\nTick {tgt_tick} – {r['name']}:\n"
            f"  Inputs      : {r['keyboard_input'] or '(none)'}\n"
            f"  Position    : x={r['X']:.2f}, y={r['Y']:.2f}, z={r['Z']:.2f}\n"
            f"  Inventory   : {_sanitize_inventory(r['inventory'])}\n"
            f"  Active Weapon: {r['active_weapon_name']}"
        )
        print()

if __name__ == "__main__":
    main()