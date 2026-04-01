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

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    private HarnessConfig _config = new();
    private string? _activeScenarioName;
    private string _currentMapName = "";

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

        RegisterEventHandler<EventPlayerSpawn>(OnPlayerSpawn);
        RegisterListener<Listeners.OnMapStart>(OnMapStart);

        Logger.LogInformation("Cs2SimHarness loaded from {ModuleDirectory}", ModuleDirectory);
    }

    private string ConfigPath => Path.Combine(ModuleDirectory, ConfigFileName);

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

    private void OnMapStart(string mapName)
    {
        _currentMapName = mapName;
        Logger.LogInformation("Current map set to {MapName}", mapName);
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
}
