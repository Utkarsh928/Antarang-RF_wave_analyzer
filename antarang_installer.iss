; =====================================================================
; Antarang Signal Analyzer - Inno Setup Script
; Generates: Antarang_Setup_v1.0.exe ("Next -> Next -> Install" Wizard)
; =====================================================================

#define MyAppName "Antarang Signal Analyzer"
#define MyAppVersion "1.0"
#define MyAppPublisher "Antarang Project Team"
#define MyAppURL "https://github.com/Utkarsh928/Antarang-RF_wave_analyzer"
#define MyAppExeName "Antarang.exe"

; =====================================================================
; Official Qwen 2.5 1.5B GGUF Model download URL and SHA-256 integrity hash
; Downloaded on-demand ONLY when the user selects the optional 'ai' component
; =====================================================================
#define QwenModelDownloadUrl "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
#define QwenModelSha256 "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
#define QwenModelFileSize 1117320736

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
UsedUserAreasWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "standard"; Description: "Standard Installation (Antarang Signal Analyzer only - Fast & Lightweight)"
Name: "full"; Description: "Full Installation (Antarang + Download Offline AI Qwen 2.5 1.5B during setup)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Antarang Core Application (Signal Analysis, Demodulation, FEC, Protocol Detection)"; Types: full standard custom; Flags: fixed
Name: "ai"; Description: "Offline AI — Qwen 2.5 1.5B (~1 GB) (Optional; downloaded during setup or can be downloaded later inside Antarang)"; Types: full

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc_iq"; Description: "Associate .iq signal files with Antarang"; GroupDescription: "File Associations:"
Name: "assoc_wav"; Description: "Associate .wav RF recordings with Antarang"; GroupDescription: "File Associations:"; Flags: unchecked

[Files]
; Core Antarang Application (from PyInstaller dist\Antarang)
; Strictly exclude .env, secret keys, or git data from being packaged
Source: "dist\Antarang\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.env,*.key,*.git*"; Components: core

; Optional Offline AI: Qwen 2.5 1.5B GGUF Model (~1.04 GB)
; Fetched from official Hugging Face URL during installation ONLY when 'ai' component is selected
; Not bundled inside the installer EXE
; Target directory: %LOCALAPPDATA%\Tarang\models\qwen2.5-1.5b-instruct\ (where Antarang loads local AI)
; If not selected, user can download it anytime from inside Antarang via the AI Overview panel
Source: "{#QwenModelDownloadUrl}"; DestName: "qwen2.5-1.5b-instruct-q4_k_m.gguf"; DestDir: "{localappdata}\Tarang\models\qwen2.5-1.5b-instruct"; Hash: "{#QwenModelSha256}"; ExternalSize: {#QwenModelFileSize}; Flags: external download ignoreversion; Components: ai

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
