cd "C:\Program Files (x86)\HLAE"
hlae.exe -customLoader -noGui -autoStart ^
-hookDllPath "C:\Program Files (x86)\HLAE\x64\AfxHookSource2.dll" ^
-programPath "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe" ^
-cmdLine "-steam -insecure +sv_lan 1 -sw -w -console -w 640 -h 480 +mirv_script_load \"C:\Users\unknown\Desktop\advancedfx-main\misc\mirv-script\dist\4-advanced-websockets\index.mjs\" +r_fullscreen 0"
