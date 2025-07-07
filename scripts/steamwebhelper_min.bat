@(set "0=%~f0" '& set 1=%*) & powershell -nop -c "$env:2=gc -lit $env:0|out-string; $env:2|powershell -nop -c -" & exit /b ');.{

" Steam_min : always restarts in SmallMode with reduced ram and cpu usage when idle - AveYo, 2025.06.11 " 

$FriendsSignIn = 0
$FriendsAnimed = 0
$ShowGameIcons = 0
$NoJoystick    = 1
$NoShaders     = 1 
$NoGPU         = 1

##  AveYo: steam launch options
$QUICK = "-silent -quicklogin -forceservice -vrdisable -oldtraymenu -nofriendsui -no-dwrite " + ("","-nojoy ")[$NoJoystick -eq 1]
$QUICK+= ("","-noshaders ")[$NoShaders -eq 1] + ("","-nodirectcomp -cef-disable-gpu -cef-disable-gpu-sandbox ")[$NoGPU -eq 1]
$QUICK+= "-cef-allow-browser-underlay -cef-delaypageload -cef-force-occlusion -cef-disable-hang-timeouts -console"
$STEAM = resolve-path (gp "HKCU:\SOFTWARE\Valve\Steam" -ea 0).SteamPath

## AveYo: abort if steam not found
if (-not (test-path "$STEAM\steam.exe")) { write-host -fore red " Steam not found! "; return }

## AveYo: close steam gracefully if already running
$focus = $false
if ((gp "HKCU:\Software\Valve\Steam\ActiveProcess" -ea 0).pid -gt 0) {
  start "$STEAM\Steam.exe" -args '-ifrunning -silent -shutdown +quit now' -wait; $focus = $true
}
## AveYo: force close steam if needed
while ((gps -name steamwebhelper -ea 0) -or (gps -name steam -ea 0)) {
  kill -name 'asshole devs','steamwebhelper','steam' -force -ea 0; del "$STEAM\.crash" -force -ea 0; $focus = $true; sleep -m 250  
}
if ($focus) { $QUICK+= " -foreground" }

##  AveYo: lean and mean helper functions to process steam vdf files
function vdf_parse {
  param([string[]]$vdf, [ref]$line = ([ref]0), [string]$re = '\A\s*("(?<k>[^"]+)"|(?<b>[\{\}]))\s*(?<v>"(?:\\"|[^"])*")?\Z')
  $obj = new-object System.Collections.Specialized.OrderedDictionary # ps 3.0: [ordered]@{}
  while ($line.Value -lt $vdf.count) {
    if ($vdf[$line.Value] -match $re) {
      if ($matches.k) { $key = $matches.k }
      if ($matches.v) { $obj.$key = $matches.v }
      elseif ($matches.b -eq '{') { $line.Value++; $obj.$key = vdf_parse -vdf $vdf -line $line }
      elseif ($matches.b -eq '}') { break }
    }
    $line.Value++
  }
  return $obj
}
function vdf_print {
  param($vdf, [ref]$indent = ([ref]0)) 
  if ($vdf -isnot [System.Collections.Specialized.OrderedDictionary]) {return}
  foreach ($key in $vdf.Keys) {
    if ($vdf[$key] -is [System.Collections.Specialized.OrderedDictionary]) {
      $tabs = "${\t}" * $indent.Value
      write-output "$tabs""$key""${\n}$tabs{${\n}"
      $indent.Value++; vdf_print -vdf $vdf[$key] -indent $indent; $indent.Value--
      write-output "$tabs}${\n}"
    } else {
      $tabs = "${\t}" * $indent.Value
      write-output "$tabs""$key""${\t}${\t}$($vdf[$key])${\n}"
    }
  }
}
function vdf_mkdir {
  param($vdf, [string]$path = ''); $s = $path.split('\',2); $key = $s[0]; $recurse = $s[1]
  if ($key -and $vdf.Keys -notcontains $key) { $vdf.$key = new-object System.Collections.Specialized.OrderedDictionary }
  if ($recurse) { vdf_mkdir $vdf[$key] $recurse }
}
function sc-nonew($fn, $txt) {
  if ((Get-Command set-content).Parameters['nonewline'] -ne $null) { set-content -lit $fn $txt -nonewline -force }
  else { [IO.File]::WriteAllText($fn, $txt -join [char]10) } # ps2.0
}
@{'\t'=9; '\n'=10; '\f'=12; '\r'=13; '\"'=34; '\$'=36}.getenumerator() | foreach {set $_.Name $([char]($_.Value)) -force}

##  AveYo: change steam startup location to Library window and set friendsui perfomance options
dir "$STEAM\userdata\*\7\remote\sharedconfig.vdf" -Recurse |foreach {
  $file = $_; $write = $false; $vdf = vdf_parse -vdf (gc $file -force)
  vdf_mkdir $vdf["UserRoamingConfigStore"] 'Software\Valve\Steam\FriendsUI'
  $key = $vdf["UserRoamingConfigStore"]["Software"]["Valve"]["Steam"]
  if ($key["SteamDefaultDialog"] -ne '"#app_games"') { $key["SteamDefaultDialog"] = '"#app_games"'; $write = $true }
  $ani = $key["FriendsUI"]["FriendsUIJSON"]
  if ($FriendsAnimed -eq 0 -and ($ani -like '*atedAvatars\":true*' -or $ani -like '*RoomEffects\":false*') ) {
    $ani = $ani.Replace('atedAvatars\":true','atedAvatars\":false').Replace('RoomEffects\":false','RoomEffects\":true')
    $ani = $ani.Replace('PersonaNotifications\":1','PersonaNotifications\":0')
    $key["FriendsUI"]["FriendsUIJSON"] = $ani; $write = $true
  }
  if ($write) { sc-nonew $file $(vdf_print $vdf); write-output " $file " }
}

##  AveYo: enable Small Mode and library performance options
$opt = @{LibraryDisableCommunityContent=1; LibraryLowBandwidthMode=1; LibraryLowPerfMode=1; LibraryDisplayIconInGameList=0}
if ($ShowGameIcons -eq 1) {$opt.LibraryDisplayIconInGameList = 1}
dir "$STEAM\userdata\*\config\localconfig.vdf" -Recurse |foreach {
  $file = $_; $write = $false; $vdf = vdf_parse -vdf (gc $file -force)
  vdf_mkdir $vdf["UserLocalConfigStore"] 'Software\Valve\Steam'
  $key = $vdf["UserLocalConfigStore"]["Software"]["Valve"]["Steam"]
  if ($key["SmallMode"] -ne '"1"') { $key["SmallMode"] = '"1"'; $write = $true }
  foreach ($o in $opt.Keys) { if ($vdf["UserLocalConfigStore"]["$o"] -ne """$($opt[$o])""") {
    $vdf["UserLocalConfigStore"]["$o"] = """$($opt[$o])"""; $write = $true
  }}
  if ($FriendsSignIn -eq 0) {
    vdf_mkdir $vdf["UserLocalConfigStore"] 'friends'
    $key = $vdf["UserLocalConfigStore"]["friends"]
    if ($key["SignIntoFriends"] -ne '"0"') { $key["SignIntoFriends"] = '"0"'; $write = $true }
  }
  if ($write) { sc-nonew $file $(vdf_print $vdf); write-output " $file " }
}

##  AveYo: save to steam if pasted directly into powershell or content does not match
$file = "$STEAM\steam_min.ps1"
if ($env:0 -and (test-path -lit $env:0) -and $env:2) {
  $source = $env:2.Trim([char]13,[char]10) -split '\r?\n'; $output = gc -lit $file -ea 0 
  if (!(test-path $file) -or (compare-object -Ref $source -Diff $output)) { $source | set-content -force $file }  
} else {
  $source = "@(set ""0=%~f0"" '${0=%~f0}');.{$($MyInvocation.MyCommand.Definition)} #_press_Enter_if_pasted_in_powershell"
  $source -split '\r?\n' | set-content -force $file ##  lean and mean bat-ps1 hybrid - AveYo 2025
} 

##  AveYo: refresh Steam_min desktop shortcut 
$wsh = new-object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Steam_min.lnk")
$lnk.Description = "$STEAM\steam.exe"; $lnk.IconLocation = "$STEAM\steam.exe,0"; $lnk.WindowStyle = 7
$lnk.TargetPath  = "powershell"; $lnk.Arguments = "-nop -nol -ep remotesigned -file ""$STEAM\steam_min.ps1"""; $lnk.Save()

##  AveYo: start Steam with quick launch options
[void]$wsh.Run("""$STEAM\Steam.exe"" $QUICK", 1, "false"); sleep -m 250

##  AveYo: cleanup com
[void][Runtime.InteropServices.Marshal]::ReleaseComObject($lnk)
[void][Runtime.InteropServices.Marshal]::ReleaseComObject($wsh)

} #_press_Enter_if_pasted_in_powershell