using System.Text.Json.Serialization;

namespace Cs2SimHarness;

public sealed class HarnessConfig
{
    public string? DefaultScenario { get; set; }

    public Dictionary<string, ScenarioConfig> Scenarios { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}

public sealed class ScenarioConfig
{
    public string Map { get; set; } = "";

    public string ControlledTeam { get; set; } = "ct";

    public string? BotTeam { get; set; }

    public int BotCount { get; set; } = 5;

    public int? BotDifficulty { get; set; }

    public bool SpectateUnassignedHumans { get; set; } = true;

    public float RestartDelaySeconds { get; set; } = 0.25f;

    public List<PlayerAssignment> Assignments { get; set; } = [];

    public List<BotAssignment> BotAssignments { get; set; } = [];
}

public abstract class LoadoutAssignment
{
    public string? Team { get; set; }

    public int SpawnIndex { get; set; }

    public string? Primary { get; set; }

    public string? Secondary { get; set; }

    public string Armor { get; set; } = "helmet";

    public bool Defuser { get; set; }

    public List<string> Grenades { get; set; } = [];

    public int Health { get; set; } = 100;
}

public sealed class PlayerAssignment : LoadoutAssignment
{
    public string? Label { get; set; }

    public ulong? SteamId64 { get; set; }

    public int? Slot { get; set; }

    public string? Name { get; set; }

    [JsonIgnore]
    public string DisplayName => Label ?? Name ?? SteamId64?.ToString() ?? Slot?.ToString() ?? "unknown";

    [JsonIgnore]
    public bool HasSelector => SteamId64.GetValueOrDefault() != 0 || Slot.HasValue || !string.IsNullOrWhiteSpace(Name);
}

public sealed class BotAssignment : LoadoutAssignment
{
    public string? Label { get; set; }

    [JsonIgnore]
    public string DisplayName => Label ?? $"bot_spawn_{SpawnIndex}";
}

public sealed class ServerStateSnapshot
{
    public string Map { get; set; } = "";

    public string ActiveScenario { get; set; } = "";

    public string FreezeMode { get; set; } = "none";

    public long GeneratedAtUnixMs { get; set; }

    public List<PlayerStateSnapshot> Players { get; set; } = [];
}

public sealed class PlayerStateSnapshot
{
    public int Slot { get; set; }

    public string Name { get; set; } = "";

    public bool IsBot { get; set; }

    public bool Connected { get; set; }

    public bool Alive { get; set; }

    public bool Frozen { get; set; }

    public string Team { get; set; } = "none";

    public float Pitch { get; set; }

    public float Yaw { get; set; }

    public float Roll { get; set; }

    public float OriginX { get; set; }

    public float OriginY { get; set; }

    public float OriginZ { get; set; }
}
