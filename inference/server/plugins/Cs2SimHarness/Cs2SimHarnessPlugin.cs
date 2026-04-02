using System.Globalization;
using System.Text.Json;
using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes;
using CounterStrikeSharp.API.Modules.Commands;
using CounterStrikeSharp.API.Modules.Utils;
using Microsoft.Extensions.Logging;

namespace Cs2SimHarness;

[MinimumApiVersion(80)]
public sealed class Cs2SimHarnessPlugin : BasePlugin
{
    private const string ConfigFileName = "cs2-sim-harness.json";
    private const string StateFileName = "cs2-sim-state.json";

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    private HarnessConfig _config = new();
    private string? _activeScenarioName;
    private string _currentMapName = "";
    private FreezeMode _freezeMode = FreezeMode.None;
    private readonly Dictionary<int, FrozenPlayerState> _frozenPlayers = new();

    public override string ModuleName => "CS2 Sim Harness";
    public override string ModuleVersion => "0.2.0";
    public override string ModuleAuthor => "OpenAI";
    public override string ModuleDescription => "Deterministic scenario reset/loadout control for the sim harness.";

    public override void Load(bool hotReload)
    {
        LoadConfigFromDisk();

        AddCommand("css_sim_reload_config", "Reload the sim harness scenario config.", OnReloadConfigCommand);
        AddCommand("css_sim_list", "List available sim harness scenarios.", OnListScenariosCommand);
        AddCommand("css_sim_dump_spawns", "Log indexed T and CT spawn points for the current map.", OnDumpSpawnsCommand);
        AddCommand("css_sim_apply", "Apply a scenario immediately to connected players.", OnApplyScenarioCommand);
        AddCommand("css_sim_reset", "Force teams, rebuild bots, and restart the round for a scenario.", OnResetScenarioCommand);
        AddCommand("css_sim_freeze", "Freeze all, human, or bot players in place.", OnFreezeCommand);
        AddCommand("css_sim_unfreeze", "Unfreeze all current players and bots.", OnUnfreezeCommand);
        AddCommand("css_sim_write_state", "Write current player/map state to disk for the harness.", OnWriteStateCommand);
        AddCommand("css_sim_set_view", "Set the single alive human player's view angles: <pitch> <yaw>.", OnSetViewCommand);

        RegisterEventHandler<EventPlayerSpawn>(OnPlayerSpawn);
        RegisterListener<Listeners.OnMapStart>(OnMapStart);
        RegisterListener<Listeners.OnTick>(OnTick);
        RegisterListener<Listeners.OnPlayerTakeDamagePre>(OnPlayerTakeDamagePre);

        Logger.LogInformation("Cs2SimHarness loaded from {ModuleDirectory}", ModuleDirectory);
    }

    public override void Unload(bool hotReload)
    {
        _freezeMode = FreezeMode.None;
        RestoreAllFrozenPlayers();
    }

    private string ConfigPath => Path.Combine(ModuleDirectory, ConfigFileName);
    private string StatePath => Path.Combine(ModuleDirectory, StateFileName);

    private void LoadConfigFromDisk()
    {
        if (!File.Exists(ConfigPath))
        {
            Logger.LogWarning("Scenario config not found at {Path}", ConfigPath);
            _config = new HarnessConfig();
            return;
        }

        var json = File.ReadAllText(ConfigPath);
        _config = JsonSerializer.Deserialize<HarnessConfig>(json, _jsonOptions) ?? new HarnessConfig();
        Logger.LogInformation("Loaded {ScenarioCount} scenarios from {Path}", _config.Scenarios.Count, ConfigPath);
    }

    private void OnReloadConfigCommand(CCSPlayerController? player, CommandInfo command)
    {
        LoadConfigFromDisk();
        command.ReplyToCommand($"Reloaded {_config.Scenarios.Count} scenarios from {ConfigPath}");
    }

    private void OnListScenariosCommand(CCSPlayerController? player, CommandInfo command)
    {
        if (_config.Scenarios.Count == 0)
        {
            command.ReplyToCommand("No scenarios configured.");
            return;
        }

        foreach (var entry in _config.Scenarios.OrderBy(kv => kv.Key, StringComparer.OrdinalIgnoreCase))
        {
            var scenario = entry.Value;
            command.ReplyToCommand(
                $"{entry.Key} map={scenario.Map} humans={scenario.Assignments.Count} botCount={scenario.BotCount} controlledTeam={scenario.ControlledTeam}");
        }
    }

    private void OnDumpSpawnsCommand(CCSPlayerController? player, CommandInfo command)
    {
        LogSpawnPoints("ct", GetCtSpawns());
        LogSpawnPoints("t", GetTSpawns());
        command.ReplyToCommand("Spawn points dumped to server log.");
    }

    private void OnApplyScenarioCommand(CCSPlayerController? player, CommandInfo command)
    {
        var scenario = ResolveScenario(command);
        if (scenario is null)
        {
            return;
        }

        _activeScenarioName = scenario.Value.Name;
        ApplyScenarioNow(scenario.Value.Name, scenario.Value.Config);
        command.ReplyToCommand($"Applied scenario {scenario.Value.Name}");
    }

    private void OnResetScenarioCommand(CCSPlayerController? player, CommandInfo command)
    {
        var scenario = ResolveScenario(command);
        if (scenario is null)
        {
            return;
        }

        var (scenarioName, config) = scenario.Value;
        if (!string.IsNullOrWhiteSpace(_currentMapName) &&
            !string.Equals(config.Map, _currentMapName, StringComparison.OrdinalIgnoreCase))
        {
            command.ReplyToCommand($"Scenario map is {config.Map}, but current map is {_currentMapName}. Change map first.");
            return;
        }

        _activeScenarioName = scenarioName;
        PrepareScenarioReset(scenarioName, config);
        command.ReplyToCommand($"Reset queued for scenario {scenarioName}");
    }

    private void OnFreezeCommand(CCSPlayerController? player, CommandInfo command)
    {
        var requestedMode = ParseFreezeMode(command.ArgCount >= 2 ? command.ArgByIndex(1) : null);
        if (requestedMode == FreezeMode.None)
        {
            command.ReplyToCommand("Usage: css_sim_freeze [all|bots|humans]");
            return;
        }

        if (_freezeMode != FreezeMode.None)
        {
            RestoreAllFrozenPlayers();
        }

        _freezeMode = requestedMode;
        FreezeTrackedPlayers();
        WriteServerStateSnapshot();
        command.ReplyToCommand($"Sim freeze active for {_frozenPlayers.Count} tracked players in mode {GetFreezeModeLabel(_freezeMode)}.");
    }

    private void OnUnfreezeCommand(CCSPlayerController? player, CommandInfo command)
    {
        if (_freezeMode == FreezeMode.None && _frozenPlayers.Count == 0)
        {
            command.ReplyToCommand("Sim freeze is not active.");
            return;
        }

        _freezeMode = FreezeMode.None;
        var restored = RestoreAllFrozenPlayers();
        WriteServerStateSnapshot();
        command.ReplyToCommand($"Sim freeze disabled. Restored {restored} players.");
    }

    private void OnWriteStateCommand(CCSPlayerController? player, CommandInfo command)
    {
        var path = WriteServerStateSnapshot();
        command.ReplyToCommand($"Wrote sim state to {path}");
    }

    private void OnSetViewCommand(CCSPlayerController? player, CommandInfo command)
    {
        if (command.ArgCount < 3)
        {
            command.ReplyToCommand("Usage: css_sim_set_view <pitch> <yaw>");
            return;
        }

        if (!float.TryParse(command.ArgByIndex(1), NumberStyles.Float, CultureInfo.InvariantCulture, out var pitch) ||
            !float.TryParse(command.ArgByIndex(2), NumberStyles.Float, CultureInfo.InvariantCulture, out var yaw))
        {
            command.ReplyToCommand("Pitch and yaw must be floats.");
            return;
        }

        var human = ResolveSingleAliveHuman(command);
        if (human is null || human.PlayerPawn.Value is not { IsValid: true } pawn)
        {
            return;
        }

        var origin = pawn.AbsOrigin ?? new Vector(0.0f, 0.0f, 0.0f);
        pawn.Teleport(origin, new QAngle(pitch, yaw, 0.0f), new Vector(0.0f, 0.0f, 0.0f));

        Server.NextFrame(() =>
        {
            WriteServerStateSnapshot();
            Logger.LogInformation("Set calibration view for {PlayerName} to pitch={Pitch:F3} yaw={Yaw:F3}", human.PlayerName, pitch, yaw);
        });

        command.ReplyToCommand($"Set view for {human.PlayerName} to pitch={pitch.ToString("F3", CultureInfo.InvariantCulture)} yaw={yaw.ToString("F3", CultureInfo.InvariantCulture)}");
    }

    private (string Name, ScenarioConfig Config)? ResolveScenario(CommandInfo command)
    {
        string? requested = null;

        if (command.ArgCount >= 2)
        {
            requested = command.ArgByIndex(1);
        }

        requested ??= _config.DefaultScenario;

        if (string.IsNullOrWhiteSpace(requested))
        {
            command.ReplyToCommand("No scenario specified and no defaultScenario set.");
            return null;
        }

        if (!_config.Scenarios.TryGetValue(requested, out var scenario))
        {
            command.ReplyToCommand($"Unknown scenario: {requested}");
            return null;
        }

        return (requested, scenario);
    }

    private void PrepareScenarioReset(string scenarioName, ScenarioConfig scenario)
    {
        ConfigureHumanTeam(scenario);
        ForceAllHumansToControlledTeam(scenario);
        var matchedHumans = ResolveHumanAssignments(scenario);
        SpectateUnmatchedHumans(matchedHumans, scenario);
        RebuildBotTeam(scenario);

        var restartDelay = Math.Max(0.0f, scenario.RestartDelaySeconds);
        AddTimer(restartDelay, () =>
        {
            Server.ExecuteCommand("mp_warmup_end");
            Server.ExecuteCommand("mp_restartgame 1");
        });

        Logger.LogInformation(
            "Queued reset for scenario {ScenarioName}: humans={HumanCount} bots={BotCount} controlledTeam={ControlledTeam} botTeam={BotTeam}",
            scenarioName,
            matchedHumans.Count,
            scenario.BotCount,
            GetControlledTeam(scenario),
            GetBotTeam(scenario));
    }

    private void ApplyScenarioNow(string scenarioName, ScenarioConfig scenario)
    {
        foreach (var matchedHuman in ResolveHumanAssignments(scenario))
        {
            ApplyAssignment(matchedHuman.Player, matchedHuman.Assignment, scenario, isBotAssignment: false);
        }

        foreach (var matchedBot in ResolveBotAssignments(scenario))
        {
            ApplyAssignment(matchedBot.Player, matchedBot.Assignment, scenario, isBotAssignment: true);
        }

        Logger.LogInformation("Applied scenario {ScenarioName}", scenarioName);
    }

    private HookResult OnPlayerSpawn(EventPlayerSpawn @event, GameEventInfo info)
    {
        if (string.IsNullOrWhiteSpace(_activeScenarioName))
        {
            return HookResult.Continue;
        }

        if (!_config.Scenarios.TryGetValue(_activeScenarioName, out var scenario))
        {
            Logger.LogWarning("Active scenario {ScenarioName} was not found in config.", _activeScenarioName);
            _activeScenarioName = null;
            return HookResult.Continue;
        }

        var player = @event.Userid;
        if (player is null || !player.IsValid)
        {
            return HookResult.Continue;
        }

        if (player.IsBot)
        {
            var matchedBot = ResolveBotAssignments(scenario)
                .FirstOrDefault(entry => entry.Player.Slot == player.Slot);
            if (matchedBot is null)
            {
                return HookResult.Continue;
            }

            Server.NextFrame(() => ApplyAssignment(player, matchedBot.Assignment, scenario, isBotAssignment: true));
            return HookResult.Continue;
        }

        var matchedHuman = ResolveHumanAssignments(scenario)
            .FirstOrDefault(entry => entry.Player.Slot == player.Slot);
        if (matchedHuman is null)
        {
            return HookResult.Continue;
        }

        Server.NextFrame(() => ApplyAssignment(player, matchedHuman.Assignment, scenario, isBotAssignment: false));
        return HookResult.Continue;
    }

    private void OnTick()
    {
        if (_freezeMode == FreezeMode.None)
        {
            return;
        }

        FreezeTrackedPlayers();
    }

    private HookResult OnPlayerTakeDamagePre(CCSPlayerPawn player, CTakeDamageInfo info)
    {
        if (_freezeMode == FreezeMode.None)
        {
            return HookResult.Continue;
        }

        if (IsFrozenPawn(player))
        {
            return HookResult.Handled;
        }

        if (info.Attacker.Value is CCSPlayerPawn attackerPawn && IsFrozenPawn(attackerPawn))
        {
            return HookResult.Handled;
        }

        return HookResult.Continue;
    }

    private void OnMapStart(string mapName)
    {
        _currentMapName = mapName;
        _freezeMode = FreezeMode.None;
        _frozenPlayers.Clear();
        Logger.LogInformation("Current map set to {MapName}", mapName);
        WriteServerStateSnapshot();
    }

    private void FreezeTrackedPlayers()
    {
        var activeSlots = new HashSet<int>();
        foreach (var player in GetPlayersForFreezeMode(_freezeMode))
        {
            activeSlots.Add(player.Slot);
            FreezePlayer(player);
        }

        foreach (var staleSlot in _frozenPlayers.Keys.Where(slot => !activeSlots.Contains(slot)).ToList())
        {
            _frozenPlayers.Remove(staleSlot);
        }
    }

    private void FreezePlayer(CCSPlayerController player)
    {
        if (player.PlayerPawn.Value is not { IsValid: true } pawn)
        {
            return;
        }

        if (!_frozenPlayers.TryGetValue(player.Slot, out var state) || state.PawnIndex != pawn.Index)
        {
            state = CaptureFrozenPlayerState(player, pawn);
            _frozenPlayers[player.Slot] = state;
        }

        var movement = pawn.MovementServices?.As<CCSPlayer_MovementServices>();
        if (movement is not null)
        {
            movement.ForwardMove = 0.0f;
            movement.LeftMove = 0.0f;
            movement.UpMove = 0.0f;
            movement.Maxspeed = 0.0f;
            movement.Impulse = 0;
            movement.ButtonDoublePressed = 0;
            ClearButtonStates(movement.Buttons);
            ClearSpan(movement.ButtonPressedCmdNumber);
            ClearSpan(movement.ForceSubtickMoveWhen);
        }

        pawn.MoveType = MoveType_t.MOVETYPE_NONE;
        pawn.Flags |= (uint)Flags_t.FL_FROZEN;
        pawn.Teleport(state.Origin, state.Angles, new Vector(0.0f, 0.0f, 0.0f));
    }

    private int RestoreAllFrozenPlayers()
    {
        var restored = 0;
        foreach (var entry in _frozenPlayers.Values)
        {
            if (RestoreFrozenPlayer(entry))
            {
                restored++;
            }
        }

        _frozenPlayers.Clear();
        return restored;
    }

    private static FrozenPlayerState CaptureFrozenPlayerState(CCSPlayerController player, CCSPlayerPawn pawn)
    {
        var origin = pawn.AbsOrigin ?? new Vector(0.0f, 0.0f, 0.0f);
        var angles = pawn.AbsRotation ?? new QAngle(0.0f, 0.0f, 0.0f);
        var movement = pawn.MovementServices?.As<CCSPlayer_MovementServices>();
        var maxspeed = movement?.Maxspeed ?? 250.0f;
        return new FrozenPlayerState(
            player.Slot,
            pawn.Index,
            new Vector(origin.X, origin.Y, origin.Z),
            new QAngle(angles.X, angles.Y, angles.Z),
            pawn.MoveType,
            pawn.Flags,
            maxspeed);
    }

    private static bool RestoreFrozenPlayer(FrozenPlayerState state)
    {
        var player = Utilities.GetPlayerFromSlot(state.PlayerSlot);
        if (player is null || !CanFreezePlayer(player) || player.PlayerPawn.Value is not { IsValid: true } pawn)
        {
            return false;
        }

        var movement = pawn.MovementServices?.As<CCSPlayer_MovementServices>();
        if (movement is not null)
        {
            movement.Maxspeed = state.Maxspeed;
            movement.ForwardMove = 0.0f;
            movement.LeftMove = 0.0f;
            movement.UpMove = 0.0f;
            movement.Impulse = 0;
            movement.ButtonDoublePressed = 0;
            ClearButtonStates(movement.Buttons);
        }

        pawn.MoveType = state.MoveType;
        pawn.Flags = state.Flags;
        pawn.Teleport(pawn.AbsOrigin, pawn.AbsRotation, new Vector(0.0f, 0.0f, 0.0f));
        return true;
    }

    private static bool CanFreezePlayer(CCSPlayerController player)
    {
        return player.IsValid &&
               player.Connected == PlayerConnectedState.PlayerConnected &&
               player.Team is CsTeam.CounterTerrorist or CsTeam.Terrorist &&
               player.PlayerPawn.Value is { IsValid: true } pawn &&
               pawn.Health > 0;
    }

    private static bool IsFrozenPawn(CCSPlayerPawn pawn)
    {
        if (pawn.Controller.Value is not CCSPlayerController controller || !controller.IsValid)
        {
            return false;
        }

        return CanFreezePlayer(controller) && (pawn.Flags & (uint)Flags_t.FL_FROZEN) != 0;
    }

    private static IEnumerable<CCSPlayerController> GetPlayersForFreezeMode(FreezeMode mode)
    {
        return mode switch
        {
            FreezeMode.All => Utilities.GetPlayers().Where(CanFreezePlayer),
            FreezeMode.Bots => Utilities.GetPlayers().Where(player => player.IsBot && CanFreezePlayer(player)),
            FreezeMode.Humans => Utilities.GetPlayers().Where(player => !player.IsBot && CanFreezePlayer(player)),
            _ => Enumerable.Empty<CCSPlayerController>(),
        };
    }

    private static FreezeMode ParseFreezeMode(string? rawMode)
    {
        return rawMode?.Trim().ToLowerInvariant() switch
        {
            null or "" or "all" => FreezeMode.All,
            "bot" or "bots" => FreezeMode.Bots,
            "human" or "humans" => FreezeMode.Humans,
            _ => FreezeMode.None,
        };
    }

    private static string GetFreezeModeLabel(FreezeMode mode)
    {
        return mode switch
        {
            FreezeMode.All => "all",
            FreezeMode.Bots => "bots",
            FreezeMode.Humans => "humans",
            _ => "none",
        };
    }

    private string WriteServerStateSnapshot()
    {
        var snapshot = BuildServerStateSnapshot();
        var json = JsonSerializer.Serialize(snapshot, _jsonOptions);
        File.WriteAllText(StatePath, json);
        return StatePath;
    }

    private ServerStateSnapshot BuildServerStateSnapshot()
    {
        var snapshot = new ServerStateSnapshot
        {
            Map = _currentMapName,
            ActiveScenario = _activeScenarioName ?? "",
            FreezeMode = GetFreezeModeLabel(_freezeMode),
            GeneratedAtUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
        };

        foreach (var player in Utilities.GetPlayers().Where(player => player.IsValid).OrderBy(player => player.Slot))
        {
            var connected = player.Connected == PlayerConnectedState.PlayerConnected;
            var alive = CanFreezePlayer(player);
            var pawn = player.PlayerPawn.Value;
            var origin = pawn?.AbsOrigin ?? new Vector(0.0f, 0.0f, 0.0f);
            var angles = GetPlayerViewAngles(player, pawn);
            snapshot.Players.Add(new PlayerStateSnapshot
            {
                Slot = player.Slot,
                Name = player.PlayerName,
                IsBot = player.IsBot,
                Connected = connected,
                Alive = alive,
                Frozen = pawn is not null && pawn.IsValid && IsFrozenPawn(pawn),
                Team = GetTeamCommandName(player.Team),
                Pitch = angles.X,
                Yaw = angles.Y,
                Roll = angles.Z,
                OriginX = origin.X,
                OriginY = origin.Y,
                OriginZ = origin.Z,
            });
        }

        return snapshot;
    }

    private static QAngle GetPlayerViewAngles(CCSPlayerController player, CCSPlayerPawn? pawn)
    {
        if (pawn is not null && pawn.IsValid && pawn.EyeAngles is not null)
        {
            return pawn.EyeAngles;
        }

        if (pawn is not null && pawn.IsValid && pawn.AbsRotation is not null)
        {
            return pawn.AbsRotation;
        }

        return new QAngle(0.0f, 0.0f, 0.0f);
    }

    private CCSPlayerController? ResolveSingleAliveHuman(CommandInfo command)
    {
        var humans = GetHumanPlayers()
            .Where(CanFreezePlayer)
            .ToList();

        if (humans.Count != 1)
        {
            command.ReplyToCommand($"Expected exactly one alive human player, found {humans.Count}.");
            return null;
        }

        return humans[0];
    }

    private static void ClearButtonStates(CInButtonState buttonState)
    {
        for (var i = 0; i < buttonState.ButtonStates.Length; i++)
        {
            buttonState.ButtonStates[i] = 0;
        }
    }

    private static void ClearSpan(Span<uint> values)
    {
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = 0;
        }
    }

    private static void ClearSpan(Span<float> values)
    {
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = 0.0f;
        }
    }

    private void ConfigureHumanTeam(ScenarioConfig scenario)
    {
        var controlledTeam = GetControlledTeam(scenario);
        if (controlledTeam == CsTeam.None)
        {
            return;
        }

        Server.ExecuteCommand("mp_autoteambalance 0");
        Server.ExecuteCommand("mp_limitteams 0");
        Server.ExecuteCommand($"mp_humanteam {GetTeamCommandName(controlledTeam)}");
    }

    private void ForceAllHumansToControlledTeam(ScenarioConfig scenario)
    {
        var desiredTeam = GetControlledTeam(scenario);
        if (desiredTeam == CsTeam.None)
        {
            return;
        }

        foreach (var player in GetHumanPlayers())
        {
            if (player.Team == desiredTeam)
            {
                continue;
            }

            player.SwitchTeam(desiredTeam);
            Logger.LogInformation("Switched human {PlayerName} to {Team}", player.PlayerName, desiredTeam);
        }
    }

    private void SpectateUnmatchedHumans(
        IReadOnlyCollection<MatchedHumanAssignment> matchedHumans,
        ScenarioConfig scenario)
    {
        if (!scenario.SpectateUnassignedHumans)
        {
            return;
        }

        var assignedSlots = matchedHumans.Select(entry => entry.Player.Slot).ToHashSet();
        foreach (var player in GetHumanPlayers().Where(player => !assignedSlots.Contains(player.Slot)))
        {
            if (player.Team == CsTeam.Spectator)
            {
                continue;
            }

            player.SwitchTeam(CsTeam.Spectator);
            Logger.LogInformation("Moved unmatched human {PlayerName} to spectator", player.PlayerName);
        }
    }

    private void RebuildBotTeam(ScenarioConfig scenario)
    {
        var botTeam = GetBotTeam(scenario);
        var botJoinTeam = botTeam == CsTeam.CounterTerrorist ? "ct" : "t";

        Server.ExecuteCommand("mp_autoteambalance 0");
        Server.ExecuteCommand("mp_limitteams 0");
        Server.ExecuteCommand("bot_quota_mode normal");
        Server.ExecuteCommand("bot_join_after_player 0");
        Server.ExecuteCommand($"bot_join_team {botJoinTeam}");
        Server.ExecuteCommand("bot_quota 0");
        Server.ExecuteCommand("bot_kick");

        if (scenario.BotDifficulty.HasValue)
        {
            var botDifficulty = Math.Clamp(scenario.BotDifficulty.Value, 0, 3);
            Server.ExecuteCommand($"bot_difficulty {botDifficulty}");
        }

        if (scenario.BotCount > 0)
        {
            Server.ExecuteCommand($"bot_quota {scenario.BotCount}");
        }
    }

    private void ApplyAssignment(
        CCSPlayerController player,
        LoadoutAssignment assignment,
        ScenarioConfig scenario,
        bool isBotAssignment)
    {
        if (!player.IsValid || player.PlayerPawn.Value is null || !player.PlayerPawn.Value.IsValid)
        {
            Logger.LogWarning("Player {PlayerName} does not have a valid pawn yet.", player.PlayerName);
            return;
        }

        var desiredTeam = ResolveAssignmentTeam(assignment, scenario, isBotAssignment);
        if (desiredTeam != CsTeam.None && player.Team != desiredTeam)
        {
            player.SwitchTeam(desiredTeam);
        }

        var pawn = player.PlayerPawn.Value;
        player.RemoveWeapons();
        player.GiveNamedItem("weapon_knife");

        if (!string.IsNullOrWhiteSpace(assignment.Secondary))
        {
            player.GiveNamedItem(assignment.Secondary);
        }

        if (!string.IsNullOrWhiteSpace(assignment.Primary))
        {
            player.GiveNamedItem(assignment.Primary);
        }

        switch (assignment.Armor.ToLowerInvariant())
        {
            case "kevlar":
                player.GiveNamedItem("item_kevlar");
                break;
            case "helmet":
            case "assaultsuit":
                player.GiveNamedItem("item_assaultsuit");
                break;
        }

        if (assignment.Defuser)
        {
            player.GiveNamedItem("item_defuser");
        }

        foreach (var grenade in assignment.Grenades)
        {
            if (!string.IsNullOrWhiteSpace(grenade))
            {
                player.GiveNamedItem(grenade);
            }
        }

        pawn.Health = assignment.Health;
        pawn.ArmorValue = assignment.Armor.Equals("none", StringComparison.OrdinalIgnoreCase) ? 0 : 100;
        Utilities.SetStateChanged(pawn, "CBaseEntity", "m_iHealth");
        Utilities.SetStateChanged(pawn, "CCSPlayerPawn", "m_ArmorValue");

        TeleportToConfiguredSpawn(player, assignment, desiredTeam);
        Logger.LogInformation("Applied assignment to {PlayerName} team={Team} spawnIndex={SpawnIndex}", player.PlayerName, desiredTeam, assignment.SpawnIndex);
    }

    private void TeleportToConfiguredSpawn(CCSPlayerController player, LoadoutAssignment assignment, CsTeam desiredTeam)
    {
        if (player.PlayerPawn.Value is null || !player.PlayerPawn.Value.IsValid)
        {
            return;
        }

        var spawns = desiredTeam == CsTeam.CounterTerrorist ? GetCtSpawns() : GetTSpawns();
        if (assignment.SpawnIndex < 0 || assignment.SpawnIndex >= spawns.Count)
        {
            Logger.LogWarning(
                "Spawn index {SpawnIndex} is out of range for {Team} on map {MapName}. Available={Count}",
                assignment.SpawnIndex,
                desiredTeam,
                _currentMapName,
                spawns.Count);
            return;
        }

        var spawn = spawns[assignment.SpawnIndex];
        player.PlayerPawn.Value.Teleport(spawn.Position, spawn.Angles, null);
    }

    private List<MatchedHumanAssignment> ResolveHumanAssignments(ScenarioConfig scenario)
    {
        var humans = GetHumanPlayers();
        var matchedSlots = new HashSet<int>();
        var results = new List<MatchedHumanAssignment>();

        foreach (var assignment in scenario.Assignments.Where(assignment => assignment.HasSelector))
        {
            var player = humans.FirstOrDefault(player => MatchesPlayer(player, assignment));
            if (player is null)
            {
                Logger.LogWarning("No connected human matches assignment {Assignment}", assignment.DisplayName);
                continue;
            }

            if (!matchedSlots.Add(player.Slot))
            {
                Logger.LogWarning("Multiple assignments resolved to the same player {PlayerName}", player.PlayerName);
                continue;
            }

            results.Add(new MatchedHumanAssignment(assignment, player));
        }

        var unmatchedHumans = humans.Where(player => !matchedSlots.Contains(player.Slot)).ToList();
        foreach (var pair in scenario.Assignments
                     .Where(assignment => !assignment.HasSelector)
                     .Zip(unmatchedHumans, (assignment, player) => new MatchedHumanAssignment(assignment, player)))
        {
            matchedSlots.Add(pair.Player.Slot);
            results.Add(pair);
        }

        var unmatchedAssignmentCount = scenario.Assignments.Count - results.Count;
        if (unmatchedAssignmentCount > 0)
        {
            Logger.LogWarning("{Count} human assignments could not be matched on map {MapName}", unmatchedAssignmentCount, _currentMapName);
        }

        return results;
    }

    private List<MatchedBotAssignment> ResolveBotAssignments(ScenarioConfig scenario)
    {
        var botTeam = GetBotTeam(scenario);
        var bots = GetBotPlayers()
            .Where(player => player.Team == botTeam)
            .OrderBy(player => player.Slot)
            .ToList();

        return scenario.BotAssignments
            .Zip(bots, (assignment, player) => new MatchedBotAssignment(assignment, player))
            .ToList();
    }

    private static bool MatchesPlayer(CCSPlayerController player, PlayerAssignment assignment)
    {
        if (!player.IsValid || player.IsBot)
        {
            return false;
        }

        if (assignment.SteamId64.GetValueOrDefault() != 0)
        {
            var authorizedSteamId = player.AuthorizedSteamID;
            var expectedSteamId64 = assignment.SteamId64.GetValueOrDefault();
            if (authorizedSteamId is null || authorizedSteamId.SteamId64 != expectedSteamId64)
            {
                return false;
            }
        }

        if (assignment.Slot.HasValue && player.Slot != assignment.Slot.Value)
        {
            return false;
        }

        if (!string.IsNullOrWhiteSpace(assignment.Name) &&
            !string.Equals(player.PlayerName, assignment.Name, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return assignment.HasSelector;
    }

    private static List<CCSPlayerController> GetHumanPlayers()
    {
        return Utilities.GetPlayers()
            .Where(player => player.IsValid && !player.IsBot)
            .OrderBy(player => player.Slot)
            .ToList();
    }

    private static List<CCSPlayerController> GetBotPlayers()
    {
        return Utilities.GetPlayers()
            .Where(player => player.IsValid && player.IsBot)
            .OrderBy(player => player.Slot)
            .ToList();
    }

    private static CsTeam GetControlledTeam(ScenarioConfig scenario)
    {
        var parsed = ParseTeam(scenario.ControlledTeam);
        return parsed == CsTeam.None ? CsTeam.CounterTerrorist : parsed;
    }

    private static CsTeam GetBotTeam(ScenarioConfig scenario)
    {
        if (!string.IsNullOrWhiteSpace(scenario.BotTeam))
        {
            var parsed = ParseTeam(scenario.BotTeam);
            if (parsed != CsTeam.None)
            {
                return parsed;
            }
        }

        return GetControlledTeam(scenario) == CsTeam.CounterTerrorist
            ? CsTeam.Terrorist
            : CsTeam.CounterTerrorist;
    }

    private static CsTeam ResolveAssignmentTeam(LoadoutAssignment assignment, ScenarioConfig scenario, bool isBotAssignment)
    {
        if (!string.IsNullOrWhiteSpace(assignment.Team))
        {
            var parsed = ParseTeam(assignment.Team);
            if (parsed != CsTeam.None)
            {
                return parsed;
            }
        }

        return isBotAssignment ? GetBotTeam(scenario) : GetControlledTeam(scenario);
    }

    private static string GetTeamCommandName(CsTeam team)
    {
        return team switch
        {
            CsTeam.CounterTerrorist => "ct",
            CsTeam.Terrorist => "t",
            _ => "any",
        };
    }

    private static CsTeam ParseTeam(string? rawTeam)
    {
        return rawTeam?.Trim().ToLowerInvariant() switch
        {
            "spec" or "spectator" => CsTeam.Spectator,
            "ct" or "counterterrorist" => CsTeam.CounterTerrorist,
            "t" or "terrorist" => CsTeam.Terrorist,
            _ => CsTeam.None,
        };
    }

    private void LogSpawnPoints(string label, List<SpawnPoint> spawns)
    {
        for (var i = 0; i < spawns.Count; i++)
        {
            var spawn = spawns[i];
            Logger.LogInformation(
                "{Team} spawn[{Index}] pos=({X:F1},{Y:F1},{Z:F1}) ang=({Pitch:F1},{Yaw:F1},{Roll:F1})",
                label,
                i,
                spawn.Position.X,
                spawn.Position.Y,
                spawn.Position.Z,
                spawn.Angles.X,
                spawn.Angles.Y,
                spawn.Angles.Z);
        }
    }

    private static List<SpawnPoint> GetCtSpawns()
    {
        return Utilities.FindAllEntitiesByDesignerName<CBaseEntity>("info_player_counterterrorist")
            .Select(ToSpawnPoint)
            .Where(spawn => spawn is not null)
            .Cast<SpawnPoint>()
            .ToList();
    }

    private static List<SpawnPoint> GetTSpawns()
    {
        return Utilities.FindAllEntitiesByDesignerName<CBaseEntity>("info_player_terrorist")
            .Select(ToSpawnPoint)
            .Where(spawn => spawn is not null)
            .Cast<SpawnPoint>()
            .ToList();
    }

    private static SpawnPoint? ToSpawnPoint(CBaseEntity entity)
    {
        if (entity.AbsOrigin is null || entity.AbsRotation is null)
        {
            return null;
        }

        return new SpawnPoint(entity.AbsOrigin, entity.AbsRotation);
    }

    private sealed record SpawnPoint(Vector Position, QAngle Angles);
    private sealed record MatchedHumanAssignment(PlayerAssignment Assignment, CCSPlayerController Player);
    private sealed record MatchedBotAssignment(BotAssignment Assignment, CCSPlayerController Player);
    private sealed record FrozenPlayerState(
        int PlayerSlot,
        uint PawnIndex,
        Vector Origin,
        QAngle Angles,
        MoveType_t MoveType,
        uint Flags,
        float Maxspeed);

    private enum FreezeMode
    {
        None,
        All,
        Bots,
        Humans,
    }
}
