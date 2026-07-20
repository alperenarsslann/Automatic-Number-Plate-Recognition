; Inno Setup script for ANPR — builds ANPR-Setup-x64.exe
; Prerequisites:
;   1) Build the pipeline:  cd cmake/anpr && cmake --preset x64-release && cmake --build out/build/x64-release
;   2) Fetch models:        cd cmake/anpr/models && python prepare_models.py --vehicle
;   3) Stage files:         cd installer && ./package.ps1
;   4) Compile installer:   iscc anpr.iss   (or open in the Inno Setup IDE)

#define AppName "ANPR"
#define AppVersion "1.0.0"
#define AppPublisher "Alperen Arslan"
#define AppExeName "AnprStudio.exe"
#define MyAppSetupIcon "..\images\Assets\ANPR_icon_5.ico"

#ifexist MyAppSetupIcon
	#define HasSetupIcon
#endif

[Setup]
AppId={{7E1C3B94-2A6D-4B1E-9F3C-ANPR00000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=ANPR-Setup-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile=..\LICENSE
UninstallDisplayIcon={app}\studio\{#AppExeName}
#ifdef HasSetupIcon
SetupIconFile={#MyAppSetupIcon}
#endif

#ifdef HasSetupIcon
  #pragma message "Setup icon FOUND"
#else
  #pragma message "Setup icon NOT FOUND — check path"
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Everything staged by package.ps1
Source: "staging\pipeline\*"; DestDir: "{app}\pipeline"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "staging\studio\*";   DestDir: "{app}\studio";   Flags: recursesubdirs createallsubdirs ignoreversion
Source: "staging\LICENSE";               DestDir: "{app}"; Flags: ignoreversion
Source: "staging\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "staging\README.md";             DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

#ifdef HasSetupIcon
Source: "{#MyAppSetupIcon}"; DestDir: "{app}"; DestName: "app.ico"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\ANPR Studio";        Filename: "{app}\studio\{#AppExeName}"
#ifdef HasSetupIcon
Name: "{group}\ANPR (pipeline)";    Filename: "{app}\pipeline\anpr.exe"; Parameters: "--config=""{app}\pipeline\config\anpr.json"""; IconFilename: "{app}\app.ico"
Name: "{group}\Edit config";        Filename: "notepad.exe"; Parameters: "{app}\pipeline\config\anpr.json"; IconFilename: "{app}\app.ico"
#else
Name: "{group}\ANPR (pipeline)";    Filename: "{app}\pipeline\anpr.exe"; Parameters: "--config=""{app}\pipeline\config\anpr.json"""
Name: "{group}\Edit config";        Filename: "notepad.exe"; Parameters: "{app}\pipeline\config\anpr.json"
#endif
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ANPR Studio";  Filename: "{app}\studio\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\studio\{#AppExeName}"; Description: "{cm:LaunchProgram,ANPR Studio}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove runtime-generated detection snapshots on uninstall.
Type: filesandordirs; Name: "{app}\pipeline\detections"
