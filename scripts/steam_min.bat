@(set "0=%~f0" '& set 1=%*) & powershell -nop -c "type -lit $env:0 | out-string | powershell -nop -c -" & exit /b ');.{

" Steam_min : always restarts in SmallMode with reduced ram and cpu usage when idle - AveYo, 2025.06.28 "

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

## AveYo: abort if steam not found
$STEAM = resolve-path (gp "HKCU:\SOFTWARE\Valve\Steam" -ea 0).SteamPath
if (-not (test-path "$STEAM\steam.exe") -or -not (test-path "$STEAM\steamapps\libraryfolders.vdf")) {
  write-host " Steam not found! " -fore Black -back Yellow; sleep 7; return
}

## AveYo: close steam gracefully if already running
$focus = $false
if ((gp "HKCU:\Software\Valve\Steam\ActiveProcess" -ea 0).pid -gt 0 -and (gps -name steamwebhelper -ea 0)) {
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
      if ($matches.v) { $obj[$key] = $matches.v }
      elseif ($matches.b -eq '{') { $line.Value++; $obj[$key] = vdf_parse -vdf $vdf -line $line }
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
  if ($key -and $vdf.Keys -notcontains $key) { $vdf[$key] = new-object System.Collections.Specialized.OrderedDictionary }
  if ($recurse) { vdf_mkdir $vdf[$key] $recurse }
}
function sc-nonew($fn, $txt) {
  if ((Get-Command set-content).Parameters['nonewline']) { set-content -lit $fn $txt -nonewline -force }
  else { [IO.File]::WriteAllText($fn, $txt -join [char]10) } # ps2.0
}
@{'\t'=9; '\n'=10; '\f'=12; '\r'=13; '\"'=34; '\$'=36}.getenumerator() | foreach {set $_.Name $([char]($_.Value)) -force}

##  AveYo: change steam startup location to Library window and set friendsui perfomance options
dir "$STEAM\userdata\*\7\remote\sharedconfig.vdf" -Recurse |foreach {
  $file = $_; $write = $false; $vdf = vdf_parse -vdf (gc $file -force)
  if ($vdf.count -eq 0) { $vdf = vdf_parse @('"UserRoamingConfigStore"','{','}') }
  vdf_mkdir $vdf[0] 'Software\Valve\Steam\FriendsUI'
  $key = $vdf[0]["Software"]["Valve"]["Steam"]
  if ($key["SteamDefaultDialog"] -ne '"#app_games"') { $key["SteamDefaultDialog"] = '"#app_games"'; $write = $true }
  $ui = $key["FriendsUI"]["FriendsUIJSON"]; if ($ui -notlike '*{*') { $ui = '' }
  if ($FriendsSignIn -eq 0 -and ($ui -like '*bSignIntoFriends\":true*' -or $ui -like '*PersonaNotifications\":1*') ) {
	$ui = $ui.Replace('bSignIntoFriends\":true','bSignIntoFriends\":false')
    $ui = $ui.Replace('PersonaNotifications\":1','PersonaNotifications\":0'); $write = $true
  }
  if ($FriendsAnimed -eq 0 -and ($ui -like '*bAnimatedAvatars\":true*' -or $ui -like '*bDisableRoomEffects\":false*') ) {
    $ui = $ui.Replace('bAnimatedAvatars\":true','bAnimatedAvatars\":false')
    $ui = $ui.Replace('bDisableRoomEffects\":false','bDisableRoomEffects\":true'); $write = $true
  }
  $key["FriendsUI"]["FriendsUIJSON"] = $ui; if ($write) { sc-nonew $file $(vdf_print $vdf); write-output " $file " }
}

##  AveYo: enable Small Mode and library performance options
$opt = @{LibraryDisableCommunityContent=1; LibraryLowBandwidthMode=1; LibraryLowPerfMode=1; LibraryDisplayIconInGameList=0}
if ($ShowGameIcons -eq 1) {$opt.LibraryDisplayIconInGameList = 1}
dir "$STEAM\userdata\*\config\localconfig.vdf" -Recurse |foreach {
  $file = $_; $write = $false; $vdf = vdf_parse -vdf (gc $file -force)
  if ($vdf.count -eq 0) { $vdf = vdf_parse @('"UserLocalConfigStore"','{','}') }
  vdf_mkdir $vdf[0] 'Software\Valve\Steam'
  $key = $vdf[0]["Software"]["Valve"]["Steam"]
  if ($key["SmallMode"] -ne '"1"') { $key["SmallMode"] = '"1"'; $write = $true }
  foreach ($o in $opt.Keys) { if ($vdf[0]["$o"] -ne """$($opt[$o])""") {
    $vdf[0]["$o"] = """$($opt[$o])"""; $write = $true
  }}
  if ($FriendsSignIn -eq 0) {
    vdf_mkdir $vdf[0] 'friends'
    $key = $vdf[0]["friends"]
    if ($key["SignIntoFriends"] -ne '"0"') { $key["SignIntoFriends"] = '"0"'; $write = $true }
  }
  if ($write) { sc-nonew $file $(vdf_print $vdf); write-output " $file " }
}

##  AveYo: save to steam if pasted directly into powershell or content does not match
$file = "$STEAM\steam_min.ps1"; $file_lines = if (test-path -lit $file) {(gc -lit $file) -ne ''} else {'file'}
$env0 = if ($env:0 -and (test-path -lit $env:0)) {gc -lit $env:0} else {'env0'} ; $env0_lines = $env0 -ne ''
$text = "@(set ""0=%~f0"" '${0=%~f0}');.{$($MyInvocation.MyCommand.Definition)} #_press_Enter_if_pasted_in_powershell"
$text = $text -split '\r?\n'; $text_lines = $text -ne ''
if (diff $text_lines $env0_lines) { if (diff $file_lines $text_lines) { $text | set-content -force $file} }
else { if (diff $file_lines $env0_lines) {$env0 | set-content -force $file} }

##  AveYo: refresh Steam_min desktop shortcut
$wsh = new-object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Steam_min.lnk")
$lnk.Description = "$STEAM\steam.exe"; $lnk.IconLocation = "$STEAM\steam.exe,0"; $lnk.WindowStyle = 7
$lnk.TargetPath  = "powershell"; $lnk.Arguments = "-nop -nol -ep remotesigned -file ""$STEAM\steam_min.ps1"""
$lnk.Save(); $lnk = $null

##  AveYo: start Steam with quick launch options
[void]$wsh.Run("""$STEAM\Steam.exe"" $QUICK", 1, "false"); $wsh = $null

} #_press_Enter_if_pasted_in_powershell