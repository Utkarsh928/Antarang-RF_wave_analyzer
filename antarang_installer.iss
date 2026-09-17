; =====================================================================
; Antarang Signal Analyzer - Inno Setup Script
; Generates: Antarang_Setup_v1.0.exe ("Next -> Next -> Install" Wizard)
; =====================================================================

#define MyAppName "Antarang Signal Analyzer"
#define MyAppVersion "1.0"
#define MyAppPublisher "Antarang Project Team"
#define MyAppURL "https://github.com/Utkarsh928/antarang-installer"
#define MyAppExeName "Antarang.exe"

[Setup]
; Unique AppId (do not change across updates)
AppId={{E58482E2-8182-4523-B101-38478A52FA89}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\Antarang
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist_installer
OutputBaseFilename=Antarang_Setup_v1.0
SetupIconFile=assets\branding\tarang.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full Installation (Antarang + Local AI Overview)"
Name: "standard"; Description: "Standard Installation (Antarang Signal Analyzer only - Fast & Lightweight)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Antarang Core Application (Signal Analysis, Demodulation, FEC, Protocol Detection)"; Types: full standard custom; Flags: fixed
Name: "ai"; Description: "AI Overview Engine (Bundled Ollama + Local LLaMA 3.1 Model ~5 GB)"; Types: full

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc_iq"; Description: "Associate .iq signal files with Antarang"; GroupDescription: "File Associations:"
Name: "assoc_wav"; Description: "Associate .wav RF recordings with Antarang"; GroupDescription: "File Associations:"; Flags: unchecked

[Files]
; Core Antarang Application (from PyInstaller dist\Antarang)
Source: "dist\Antarang\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core

; Optional AI Overview Bundle (placed inside {app}\ollama if checked)
; If packaging with offline Ollama bundle:
Source: "ollama_bundle\*"; DestDir: "{app}\ollama"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist; Components: ai

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\branding\tarang.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\branding\tarang.ico"; Tasks: desktopicon

[Registry]
; .iq File Association
Root: HKA; Subkey: "Software\Classes\.iq"; ValueType: string; ValueName: ""; ValueData: "AntarangSignalFile"; Flags: uninsdeletevalue; Tasks: assoc_iq
Root: HKA; Subkey: "Software\Classes\AntarangSignalFile"; ValueType: string; ValueName: ""; ValueData: "IQ Signal Data File"; Flags: uninsdeletekey; Tasks: assoc_iq
Root: HKA; Subkey: "Software\Classes\AntarangSignalFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\branding\tarang.ico,0"; Tasks: assoc_iq
Root: HKA; Subkey: "Software\Classes\AntarangSignalFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assoc_iq

; .wav File Association (optional)
Root: HKA; Subkey: "Software\Classes\.wav\OpenWithProgids"; ValueType: string; ValueName: "AntarangAudioFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_wav
Root: HKA; Subkey: "Software\Classes\AntarangAudioFile"; ValueType: string; ValueName: ""; ValueData: "WAV RF Recording"; Flags: uninsdeletekey; Tasks: assoc_wav
Root: HKA; Subkey: "Software\Classes\AntarangAudioFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\branding\tarang.ico,0"; Tasks: assoc_wav
Root: HKA; Subkey: "Software\Classes\AntarangAudioFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assoc_wav

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
