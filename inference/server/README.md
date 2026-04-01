# CS2 Server Mod Stack

This directory captures the dedicated-server setup we need for the sim harness when we want deterministic scenario resets instead of a mostly vanilla server.

## What We Need

We need three layers on the `server` user's CS2 install:

1. A bootstrapped Steam install for the `server` user.
2. Metamod:Source loaded through `game/csgo/gameinfo.gi`.
3. CounterStrikeSharp loaded through Metamod, plus a small custom plugin that can:
   - reset the round on demand
   - force players onto T or CT
   - respawn them
   - teleport them to indexed spawn points
   - remove weapons and reissue a deterministic loadout

The plugin scaffold in this directory is named `Cs2SimHarness`.

## Current Host Facts

- The CS2 dedicated server launch path works as `server`:
  - `/home/server/steam-lib/steamapps/common/Counter-Strike Global Offensive/game/cs2.sh -dedicated ...`
- The first failure mode was missing `~/.steam/sdk64/steamclient.so`.
- Running `steam -version` once as `server` created the needed Steam bootstrap files.
- The host currently does **not** have the .NET SDK installed:
  - `dotnet --info` returns `command not found`

That means:

- Runtime install of CounterStrikeSharp is fine on the server once we use the `with-runtime` package.
- Building the custom plugin on this host is not yet possible until .NET 8 SDK is installed, or until we build elsewhere and deploy the DLLs here.

## Files

- `bootstrap_server_user.sh`
  - one-time Steam bootstrap for the `server` user
- `install_mod_stack.sh`
  - installs Metamod and CounterStrikeSharp from local archives, patches `gameinfo.gi`, and clears the executable-stack flag on `counterstrikesharp.so`
- `deploy_plugin.sh`
  - deploys a built `Cs2SimHarness` plugin plus config templates
- `start_server.sh`
  - starts the dedicated server in a tmux session on the host
  - defaults to `-insecure +sv_lan 1` on the command line instead of putting `sv_lan` in `server.cfg`
- `setup_steamrt.sh`
  - installs Steam Runtime 3 for hosts where CounterStrikeSharp fails to load directly
- `runtime/cfg/server.cfg`
  - stable server-wide overrides
- `runtime/cfg/gamemode_competitive_server.cfg`
  - competitive-mode overrides for deterministic local testing
- `runtime/plugins/Cs2SimHarness/cs2-sim-harness.example.json`
  - example scenario config for the plugin
- `plugins/Cs2SimHarness/`
  - CounterStrikeSharp plugin source scaffold

## Install Plan

1. Bootstrap Steam for the `server` user:

```bash
./inference/server/bootstrap_server_user.sh
```

2. Download the current Linux packages:
   - Metamod:Source 2.x dev build
   - CounterStrikeSharp `with-runtime`

3. Install them into the live server tree:

```bash
./inference/server/install_mod_stack.sh \
  --metamod-archive /path/to/mmsource-latest-linux.tar.gz \
  --cssharp-archive /path/to/counterstrikesharp-with-runtime-linux.zip
```

4. Build the plugin on a machine with .NET 8 SDK, then deploy it:

```bash
dotnet build inference/server/plugins/Cs2SimHarness/Cs2SimHarness.csproj

./inference/server/deploy_plugin.sh \
  --build-dir inference/server/plugins/Cs2SimHarness/bin/Debug/net8.0
```

5. Start the server:

```bash
./inference/server/start_server.sh
```

If CounterStrikeSharp fails with:

```text
cannot enable executable stack as shared object requires: Invalid argument
```

install Steam Runtime 3 and rerun the server start script:

```bash
./inference/server/setup_steamrt.sh
./inference/server/start_server.sh
```

There are currently three launcher modes in `start_server.sh`:

- default
  - plain `./cs2.sh -dedicated ...`
- `SERVER_LAUNCH_MODE=steamrt-wrapper`
  - `steamrt/run -- ./cs2.sh ...`
- `SERVER_LAUNCH_MODE=steamrt-direct`
  - `steamrt/run -- game/bin/linuxsteamrt64/cs2 --graphics-provider "" -- ...`

The default mode is the intended path. After clearing the executable-stack flag on `counterstrikesharp.so`, CounterStrikeSharp itself loads under the normal launcher on this host.

6. Verify in the server console:

```text
meta list
css_sim_list
css_sim_dump_spawns
```

## How Reset Control Should Work

The plugin is deliberately shaped for harness control from tmux or server console:

- `css_sim_reload_config`
- `css_sim_list`
- `css_sim_dump_spawns`
- `css_sim_apply <scenario>`
- `css_sim_reset <scenario>`

Recommended harness flow:

1. Ensure the five clients are connected.
2. Send `css_sim_reset <scenario>` into the server tmux pane.
3. Plugin forces the matched human clients onto the configured controlled team.
4. Plugin kicks and recreates the opposing bot team to the configured bot count.
5. Plugin issues `mp_warmup_end` and then `mp_restartgame 1`.
6. On `player_spawn`, plugin reapplies the configured loadout and teleports each player or bot to the chosen indexed spawn.

This is better than trying to do all of it with raw console cfgs because:

- spawn placement must be per-player
- loadouts may differ per player
- we want one command that re-establishes a known state

## Scenario Config Shape

The example JSON supports:

- one controlled human team per scenario
- one opposing bot team per scenario
- exact bot count
- per-human and per-bot loadouts/spawns

Human assignments support matching by any of:

- `steamid64`
- `slot`
- `name`

If a human assignment does not specify any of those selectors, it falls back to the next unmatched connected human ordered by slot. That gives us a migration path:

- use `slot` or `name` while prototyping
- move to `steamid64` once the five client accounts are fixed and known

Human and bot assignments can define:

- `team`
- `spawn_index`
- `primary`
- `secondary`
- `armor`
- `defuser`
- `grenades`
- `health`

Each scenario can also define:

- `controlledTeam`
- `botTeam`
- `botCount`
- `botDifficulty`
- `spectateUnassignedHumans`
- `restartDelaySeconds`

## Remaining Gaps

- We have not installed the mod stack yet.
- We have not built the C# plugin yet because `.NET 8 SDK` is missing on this host.
- We have not yet wired the Python sim harness to call `css_sim_reset` automatically.

The next practical step is to install Metamod + CounterStrikeSharp into the `server` user's CS2 tree, then either:

- install `.NET 8 SDK` on the host and build here, or
- build the plugin elsewhere and use `deploy_plugin.sh`.
