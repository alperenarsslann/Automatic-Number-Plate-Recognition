using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Win32;

namespace AnprStudio;

/// <summary>One row of the detected-plates table.</summary>
public sealed record PlateRow(string Time, string Camera, string Text, string DetConf,
                              string OcrConf, string Box, string Seq);

/// <summary>
/// Test & configuration companion for the ANPR multi-camera pipeline.
/// Edits the pipeline's JSON config (only the fields shown; everything else
/// in the file is preserved), launches/stops anpr.exe, parses its log stream
/// into a live plate table, and talks to the pipeline's REST API for live
/// per-camera status and ALPR toggling.
/// </summary>
public partial class MainWindow : Window
{
    // Pipeline log: PLATE KH05ZZK  cam=cam8 det=0.87 ocr=0.97 box=[x,y wxh] seq=204
    private static readonly Regex PlateRegex = new(
        @"PLATE\s+(?<text>\S+)\s+cam=(?<cam>\S+)\s+det=(?<det>[\d.]+)\s+ocr=(?<ocr>[\d.]+)\s+box=\[(?<box>[^\]]*)\]\s+seq=(?<seq>\d+)",
        RegexOptions.Compiled);

    // Pipeline log: stats: cams=8 fps_total=115.3 motion_active=7 alpr_submitted=154
    //               alpr_runs=3 alpr_mean_ms=1717.6 queue=32 reports=4
    private static readonly Regex StatsRegex = new(
        @"stats: cams=(?<cams>\d+)\s+fps_total=(?<fps>[\d.]+)\s+motion_active=(?<motion>\d+).*alpr_mean_ms=(?<ms>[\d.]+)\s+queue=(?<queue>\d+)\s+reports=(?<reports>\d+)",
        RegexOptions.Compiled);

    private readonly ObservableCollection<PlateRow> _plates = new();
    private JsonNode? _config;
    private Process? _process;
    private int _selectedCamera = -1;
    private bool _loadingCameraDetail;

    // Log lines arrive on threadpool threads at a high rate; they are queued
    // here and flushed to the UI in batches by a timer. Touching the UI per
    // line (Dispatcher.Invoke + AppendText) freezes the window under load.
    private readonly ConcurrentQueue<string> _pendingLines = new();
    private readonly DispatcherTimer _logFlushTimer;

    // Live status via the pipeline's REST API.
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly DispatcherTimer _apiPollTimer;
    private readonly Dictionary<string, bool> _liveAlpr = new();

    // Embedded camera wall: per-camera Image controls fed by the pipeline's
    // /api/cameras/<id>/snapshot endpoint. The UniformGrid recomputes a
    // near-square layout automatically as cameras are added/removed, and the
    // tiles stretch with the (resizable) window.
    private readonly Dictionary<string, Image> _wallImages = new();
    private readonly DispatcherTimer _wallTimer;
    private bool _wallBusy;

    public MainWindow()
    {
        InitializeComponent();
        PlatesGrid.ItemsSource = _plates;

        _logFlushTimer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromMilliseconds(250),
        };
        _logFlushTimer.Tick += (_, _) => FlushPendingLines();
        _logFlushTimer.Start();

        _apiPollTimer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromSeconds(2),
        };
        _apiPollTimer.Tick += async (_, _) => await PollApiAsync();
        _apiPollTimer.Start();

        _wallTimer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromMilliseconds(200),
        };
        _wallTimer.Tick += async (_, _) => await RefreshWallAsync();
        _wallTimer.Start();

        GuessDefaultPaths();
        TryLoadConfig(silent: true);
    }

    // ------------------------------------------------------------------- wall

    /// <summary>Rebuild the wall tiles from the config's enabled cameras.</summary>
    private void RebuildWall()
    {
        WallGrid.Children.Clear();
        _wallImages.Clear();
        if (_config == null) return;

        foreach (var camNode in Cameras())
        {
            if (camNode!["enabled"]?.GetValue<bool>() == false) continue;
            var id = camNode["id"]?.GetValue<string>() ?? "?";

            var image = new Image { Stretch = Stretch.Uniform };
            var placeholder = new TextBlock
            {
                Text = id,
                Foreground = Brushes.DimGray,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
            };
            var tile = new Border
            {
                Margin = new Thickness(1),
                Background = new SolidColorBrush(Color.FromRgb(24, 24, 24)),
                Child = new Grid { Children = { placeholder, image } },
                Cursor = Cursors.Hand,
                ToolTip = $"{id} — tıkla: ALPR aç/kapat",
            };
            tile.MouseLeftButtonDown += async (_, _) => await ToggleCameraAsync(id);

            WallGrid.Children.Add(tile);
            _wallImages[id] = image;
        }

        // Snapshot polling budget scales with tile count (~40 req/s total).
        _wallTimer.Interval = TimeSpan.FromMilliseconds(Math.Max(200, _wallImages.Count * 25));
    }

    /// <summary>Fetch fresh snapshots for every tile (parallel, best-effort).</summary>
    private async Task RefreshWallAsync()
    {
        if (_wallBusy || _process == null || _wallImages.Count == 0) return;
        _wallBusy = true;
        try
        {
            var port = ApiPort();
            var tasks = _wallImages.Keys.ToDictionary(
                id => id,
                id => Http.GetByteArrayAsync($"http://127.0.0.1:{port}/api/cameras/{id}/snapshot"));
            foreach (var (id, task) in tasks)
            {
                try
                {
                    var bytes = await task;
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = new MemoryStream(bytes);
                    bitmap.EndInit();
                    bitmap.Freeze();
                    _wallImages[id].Source = bitmap;
                }
                catch
                {
                    // Camera not producing frames yet (503) or API briefly away — keep the old picture.
                }
            }
        }
        finally
        {
            _wallBusy = false;
        }
    }

    // ------------------------------------------------------------------ paths

    /// <summary>Locate anpr.exe / config by walking up from the app directory.</summary>
    private void GuessDefaultPaths()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var probe = dir; probe != null; probe = probe.Parent)
        {
            var projectRoot = Path.Combine(probe.FullName, "cmake", "anpr");
            if (!Directory.Exists(projectRoot)) continue;

            ConfigPathBox.Text = Path.Combine(projectRoot, "config", "anpr.json");
            var release = Path.Combine(projectRoot, "out", "build", "x64-release", "control", "anpr.exe");
            var debug = Path.Combine(projectRoot, "out", "build", "x64-debug", "control", "anpr.exe");
            ExePathBox.Text = File.Exists(release) ? release : debug;
            return;
        }
    }

    private void BrowseExe_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "anpr.exe|anpr.exe|Executables|*.exe" };
        if (dlg.ShowDialog() == true) ExePathBox.Text = dlg.FileName;
    }

    private void BrowseConfig_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "JSON config|*.json" };
        if (dlg.ShowDialog() == true)
        {
            ConfigPathBox.Text = dlg.FileName;
            TryLoadConfig(silent: false);
        }
    }

    private void BrowseVideo_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Videos|*.mp4;*.avi;*.mkv;*.mov|All files|*.*" };
        if (dlg.ShowDialog() == true) CamVideoBox.Text = dlg.FileName;
    }

    // ----------------------------------------------------------------- config

    private void ReloadConfig_Click(object sender, RoutedEventArgs e) => TryLoadConfig(silent: false);

    private void TryLoadConfig(bool silent)
    {
        try
        {
            _config = JsonNode.Parse(File.ReadAllText(ConfigPathBox.Text));
            EnsureCamerasNode();
            ConfigToUi();
            RebuildWall();
            ConfigPanel.IsEnabled = true;
            StatusText.Text = "Config loaded";
        }
        catch (Exception ex)
        {
            _config = null;
            ConfigPanel.IsEnabled = false;
            if (!silent)
                MessageBox.Show(this, ex.Message, "Cannot load config", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void SaveConfig_Click(object sender, RoutedEventArgs e)
    {
        if (_config == null) return;
        try
        {
            FlushCameraDetail();
            UiToConfig();
            File.WriteAllText(ConfigPathBox.Text,
                _config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            RefreshCameraList(keepSelection: true);
            RebuildWall();
            StatusText.Text = "Config saved";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Cannot save config", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>Get-or-create a nested object node, e.g. Node("processing").</summary>
    private JsonNode Node(params string[] path)
    {
        JsonNode node = _config!;
        foreach (var key in path)
        {
            node = node[key] ??= new JsonObject();
        }
        return node;
    }

    private JsonArray Cameras() => (_config!["cameras"] ??= new JsonArray()).AsArray();

    /// <summary>Mirror of the pipeline's legacy handling: a config that still
    /// uses the old single "capture" block gets a synthesized camera entry.</summary>
    private void EnsureCamerasNode()
    {
        if (_config!["cameras"] is JsonArray arr && arr.Count > 0) return;
        var capture = _config["capture"];
        var cam = new JsonObject
        {
            ["id"] = "cam1",
            ["name"] = "Camera 1",
            ["enabled"] = true,
            ["alpr_enabled"] = true,
            ["source"] = capture?["source"]?.GetValue<string>() ?? "simulation",
        };
        if (capture?["simulation"] is JsonNode sim) cam["simulation"] = sim.DeepClone();
        if (capture?["camera"] is JsonNode rtsp) cam["camera"] = rtsp.DeepClone();
        _config["cameras"] = new JsonArray(cam);
    }

    private JsonNode? ActiveProfileNode()
    {
        var name = (ProfileCombo.SelectedItem as string) ?? Node("processing")["active_model_profile"]?.GetValue<string>();
        return name == null ? null : Node("processing", "model_profiles")[name];
    }

    private void ConfigToUi()
    {
        RefreshCameraList(keepSelection: false);

        var proc = Node("processing");
        ProfileCombo.Items.Clear();
        if (proc["model_profiles"] is JsonObject profiles)
        {
            foreach (var (name, _) in profiles) ProfileCombo.Items.Add(name);
        }
        ProfileCombo.SelectedItem = proc["active_model_profile"]?.GetValue<string>();
        DedupBox.Text = NumText(proc["dedup_window_seconds"], "10");
        EveryNBox.Text = NumText(proc["process_every_n_frames"], "1");
        ThreadsBox.Text = NumText(proc["num_threads"], "0");
        MaxFrameWidthBox.Text = NumText(proc["max_frame_width"], "0");
        WorkerCountBox.Text = NumText(proc["worker_count"], "2");

        var consol = proc["consolidation"];
        ConsolEnabledCheck.IsChecked = consol?["enabled"]?.GetValue<bool>() ?? true;
        ConsolFinalizeBox.Text = NumText(consol?["finalize_after_seconds"], "1.5");
        ConsolMinSightBox.Text = NumText(consol?["min_sightings"], "2");
        ConsolEditBox.Text = NumText(consol?["max_edit_distance"], "2");
        ConsolMaxTrackBox.Text = NumText(consol?["max_track_seconds"], "10");

        ProfileToUi();

        var display = Node("display");
        DisplayCheck.IsChecked = display["enabled"]?.GetValue<bool>() ?? false;
        MaxWidthBox.Text = NumText(display["max_width"], "1280");

        var api = Node("api");
        ApiEnabledCheck.IsChecked = api["enabled"]?.GetValue<bool>() ?? true;
        ApiPortBox.Text = NumText(api["port"], "8088");
        var keys = api["api_keys"] as JsonArray;
        ApiKeyBox.Text = (keys != null && keys.Count > 0) ? keys[0]!.GetValue<string>() : "";
        ApplyApiKey();

        var detOut = Node("detection_output");
        DetOutEnabledCheck.IsChecked = detOut["enabled"]?.GetValue<bool>() ?? false;
        DetOutDirBox.Text = detOut["directory"]?.GetValue<string>() ?? "detections";
        DetOutTimestampCheck.IsChecked = detOut["draw_timestamp"]?.GetValue<bool>() ?? true;

        var net = Node("network");
        NetEnabledCheck.IsChecked = net["enabled"]?.GetValue<bool>() ?? false;
        NetHostBox.Text = net["server_host"]?.GetValue<string>() ?? "127.0.0.1";
        NetPortBox.Text = NumText(net["server_port"], "9000");

        SelectComboText(LogLevelCombo, Node("logging")["level"]?.GetValue<string>() ?? "info");
    }

    private void ProfileToUi()
    {
        if (ActiveProfileNode() is not JsonNode profile) return;
        SelectComboText(BackendCombo, profile["dnn_backend"]?.GetValue<string>() ?? "cpu");
        VehicleCheck.IsChecked = profile["vehicle_detection"]?["enabled"]?.GetValue<bool>() ?? false;
        VehicleConfBox.Text = NumText(profile["vehicle_detection"]?["confidence_threshold"], "0.3");
        DetConfBox.Text = NumText(profile["detection"]?["confidence_threshold"], "0.5");
        NmsBox.Text = NumText(profile["detection"]?["nms_threshold"], "0.45");
        OcrConfBox.Text = NumText(profile["ocr"]?["confidence_threshold"], "0.6");
    }

    private void ProfileCombo_SelectionChanged(object sender, SelectionChangedEventArgs e) => ProfileToUi();

    private void UiToConfig()
    {
        var proc = Node("processing");
        if (ProfileCombo.SelectedItem is string profileName)
        {
            proc["active_model_profile"] = profileName;
            if (ActiveProfileNode() is JsonNode profile)
            {
                profile["dnn_backend"] = ComboText(BackendCombo);
                (profile["vehicle_detection"] ??= new JsonObject())["enabled"] =
                    VehicleCheck.IsChecked == true;
                profile["vehicle_detection"]!["confidence_threshold"] =
                    ParseDouble(VehicleConfBox.Text, "Vehicle confidence");
                (profile["detection"] ??= new JsonObject())["confidence_threshold"] =
                    ParseDouble(DetConfBox.Text, "Det. conf.");
                profile["detection"]!["nms_threshold"] = ParseDouble(NmsBox.Text, "NMS");
                (profile["ocr"] ??= new JsonObject())["confidence_threshold"] =
                    ParseDouble(OcrConfBox.Text, "OCR conf.");
            }
        }
        proc["dedup_window_seconds"] = ParseDouble(DedupBox.Text, "Dedup (s)");
        proc["process_every_n_frames"] = ParseInt(EveryNBox.Text, "ALPR her N kare");
        proc["num_threads"] = ParseInt(ThreadsBox.Text, "CPU threads");
        proc["max_frame_width"] = ParseInt(MaxFrameWidthBox.Text, "Max frame width");
        proc["worker_count"] = ParseInt(WorkerCountBox.Text, "ALPR workers");

        var consol = (JsonObject)(proc["consolidation"] ??= new JsonObject());
        consol["enabled"] = ConsolEnabledCheck.IsChecked == true;
        consol["finalize_after_seconds"] = ParseDouble(ConsolFinalizeBox.Text, "Finalize after");
        consol["min_sightings"] = ParseInt(ConsolMinSightBox.Text, "Min sightings");
        consol["max_edit_distance"] = ParseInt(ConsolEditBox.Text, "Max edit distance");
        consol["max_track_seconds"] = ParseDouble(ConsolMaxTrackBox.Text, "Max track");

        var display = Node("display");
        display["enabled"] = DisplayCheck.IsChecked == true;
        display["max_width"] = ParseInt(MaxWidthBox.Text, "Wall width");

        var api = Node("api");
        api["enabled"] = ApiEnabledCheck.IsChecked == true;
        api["port"] = ParseInt(ApiPortBox.Text, "API port");
        var apiKey = ApiKeyBox.Text.Trim();
        api["api_keys"] = string.IsNullOrEmpty(apiKey)
            ? new JsonArray()
            : new JsonArray(JsonValue.Create(apiKey)!);
        ApplyApiKey();

        var detOut = Node("detection_output");
        detOut["enabled"] = DetOutEnabledCheck.IsChecked == true;
        detOut["directory"] = DetOutDirBox.Text;
        detOut["draw_timestamp"] = DetOutTimestampCheck.IsChecked == true;

        var net = Node("network");
        net["enabled"] = NetEnabledCheck.IsChecked == true;
        net["server_host"] = NetHostBox.Text;
        net["server_port"] = ParseInt(NetPortBox.Text, "Server port");

        Node("logging")["level"] = ComboText(LogLevelCombo);
    }

    // ---------------------------------------------------------------- cameras

    private void RefreshCameraList(bool keepSelection)
    {
        var previous = keepSelection ? _selectedCamera : 0;
        _selectedCamera = -1; // Suppress flush during rebuild.
        CamerasList.Items.Clear();
        foreach (var camNode in Cameras())
        {
            CamerasList.Items.Add(CameraLabel(camNode!));
        }
        if (CamerasList.Items.Count > 0)
        {
            CamerasList.SelectedIndex = Math.Clamp(previous, 0, CamerasList.Items.Count - 1);
        }
    }

    private string CameraLabel(JsonNode cam)
    {
        var id = cam["id"]?.GetValue<string>() ?? "?";
        var name = cam["name"]?.GetValue<string>() ?? id;
        var enabled = cam["enabled"]?.GetValue<bool>() ?? true;
        var label = $"{id} — {name}";
        if (!enabled) label += "  (disabled)";
        if (_liveAlpr.TryGetValue(id, out var live)) label += live ? "  [ALPR ✓]" : "  [ALPR ✗]";
        return label;
    }

    private void CamerasList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        FlushCameraDetail();
        _selectedCamera = CamerasList.SelectedIndex;
        LoadCameraDetail();
    }

    private void LoadCameraDetail()
    {
        if (_selectedCamera < 0 || _selectedCamera >= Cameras().Count) return;
        _loadingCameraDetail = true;
        var cam = Cameras()[_selectedCamera]!;
        CamIdBox.Text = cam["id"]?.GetValue<string>() ?? "";
        CamNameBox.Text = cam["name"]?.GetValue<string>() ?? "";
        CamEnabledCheck.IsChecked = cam["enabled"]?.GetValue<bool>() ?? true;
        CamAlprCheck.IsChecked = cam["alpr_enabled"]?.GetValue<bool>() ?? true;
        SelectComboText(CamSourceCombo, cam["source"]?.GetValue<string>() ?? "simulation");
        CamVideoBox.Text = cam["simulation"]?["video_path"]?.GetValue<string>() ?? "";
        CamRtspBox.Text = cam["camera"]?["rtsp_url"]?.GetValue<string>() ?? "";
        CamMotionCheck.IsChecked = cam["motion"]?["enabled"]?.GetValue<bool>() ?? true;
        CamMotionCooldownBox.Text = NumText(cam["motion"]?["cooldown_seconds"], "2");
        _loadingCameraDetail = false;
    }

    /// <summary>Write the detail fields back into the currently selected camera node.</summary>
    private void FlushCameraDetail()
    {
        if (_loadingCameraDetail || _selectedCamera < 0 || _selectedCamera >= Cameras().Count) return;
        var cam = Cameras()[_selectedCamera]!;
        if (!string.IsNullOrWhiteSpace(CamIdBox.Text)) cam["id"] = CamIdBox.Text.Trim();
        cam["name"] = CamNameBox.Text;
        cam["enabled"] = CamEnabledCheck.IsChecked == true;
        cam["alpr_enabled"] = CamAlprCheck.IsChecked == true;
        cam["source"] = ComboText(CamSourceCombo);
        (cam["simulation"] ??= new JsonObject())["video_path"] = CamVideoBox.Text;
        (cam["camera"] ??= new JsonObject())["rtsp_url"] = CamRtspBox.Text;
        (cam["motion"] ??= new JsonObject())["enabled"] = CamMotionCheck.IsChecked == true;
        cam["motion"]!["cooldown_seconds"] = ParseDouble(CamMotionCooldownBox.Text, "Motion cooldown");
    }

    private void AddCamera_Click(object sender, RoutedEventArgs e)
    {
        if (_config == null) return;
        FlushCameraDetail();
        var cams = Cameras();
        if (cams.Count >= 64)
        {
            MessageBox.Show(this, "En fazla 64 kamera destekleniyor.", "Limit",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        cams.Add(new JsonObject
        {
            ["id"] = $"cam{cams.Count + 1}",
            ["name"] = $"Camera {cams.Count + 1}",
            ["enabled"] = true,
            ["alpr_enabled"] = true,
            ["source"] = "simulation",
            ["simulation"] = new JsonObject { ["video_path"] = "", ["realtime"] = true, ["loop"] = true },
            ["motion"] = new JsonObject { ["enabled"] = true, ["cooldown_seconds"] = 2.0 },
        });
        RefreshCameraList(keepSelection: false);
        CamerasList.SelectedIndex = CamerasList.Items.Count - 1;
        RebuildWall();
    }

    private void RemoveCamera_Click(object sender, RoutedEventArgs e)
    {
        if (_config == null || _selectedCamera < 0 || Cameras().Count <= 1) return;
        var index = _selectedCamera;
        _selectedCamera = -1; // The node is going away; don't flush into it.
        Cameras().RemoveAt(index);
        RefreshCameraList(keepSelection: false);
        RebuildWall();
    }

    // Pipeline prints one "DISCOVERED ip=... desc=... serial=..." line per device.
    private static readonly Regex DiscoverRegex = new(
        @"DISCOVERED ip=(?<ip>\S+).*?desc=(?<desc>.*?)(?:\s+serial=(?<serial>\S+))?(?:\s+mac=|\s*$)",
        RegexOptions.Compiled);

    /// <summary>Run `anpr.exe --discover`, list found cameras, add the chosen ones.</summary>
    private async void Discover_Click(object sender, RoutedEventArgs e)
    {
        if (_config == null) return;
        if (!File.Exists(ExePathBox.Text))
        {
            MessageBox.Show(this, "anpr.exe not found:\n" + ExePathBox.Text, "Cannot discover",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        DiscoverButton.IsEnabled = false;
        StatusText.Text = "Scanning the LAN for cameras...";
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = ExePathBox.Text,
                Arguments = "--discover",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            using var proc = Process.Start(psi)!;
            var stdout = await proc.StandardOutput.ReadToEndAsync();
            await proc.WaitForExitAsync();

            var found = new List<(string Ip, string Desc, string Serial)>();
            foreach (Match m in DiscoverRegex.Matches(stdout))
            {
                found.Add((m.Groups["ip"].Value, m.Groups["desc"].Value.Trim(),
                           m.Groups["serial"].Value));
            }

            if (found.Count == 0)
            {
                MessageBox.Show(this,
                    "No cameras found.\n\nCheck that the devices are powered and on the same " +
                    "subnet. SADP replies may be blocked by the firewall; the TCP fallback scan " +
                    "needs ports 8000/554 reachable.",
                    "Discovery", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var existing = Cameras().Select(c => c!["camera"]?["rtsp_url"]?.GetValue<string>() ?? "")
                                    .ToList();
            var added = 0;
            foreach (var (ip, desc, _) in found)
            {
                if (existing.Any(u => u.Contains(ip))) continue; // Already configured.
                var url = $"rtsp://admin:<password>@{ip}:554/Streaming/Channels/101";
                Cameras().Add(new JsonObject
                {
                    ["id"] = $"cam{Cameras().Count + 1}",
                    ["name"] = string.IsNullOrEmpty(desc) ? ip : desc,
                    ["enabled"] = true,
                    ["alpr_enabled"] = true,
                    ["source"] = "camera",
                    ["camera"] = new JsonObject
                    {
                        ["rtsp_url"] = url,
                        ["substream_url"] = "",
                        ["transport"] = "tcp",
                        ["reconnect_interval_seconds"] = 5.0,
                    },
                    ["motion"] = new JsonObject { ["enabled"] = true, ["cooldown_seconds"] = 2.0 },
                });
                added++;
            }

            RefreshCameraList(keepSelection: false);
            RebuildWall();
            MessageBox.Show(this,
                $"Found {found.Count} camera(s), added {added} new.\n\n" +
                "Set each camera's RTSP password (the URL has a <password> placeholder), " +
                "then Save Config.",
                "Discovery", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Discovery error", MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        finally
        {
            DiscoverButton.IsEnabled = true;
            StatusText.Text = _process == null ? "Stopped" : StatusText.Text;
        }
    }

    // -------------------------------------------------------------------- api

    private int ApiPort() => int.TryParse(ApiPortBox.Text, out var p) ? p : 8088;

    /// <summary>Attach (or clear) the X-API-Key header used for all live API
    /// calls, matching what the running pipeline expects.</summary>
    private void ApplyApiKey()
    {
        Http.DefaultRequestHeaders.Remove("X-API-Key");
        var key = ApiKeyBox.Text.Trim();
        if (!string.IsNullOrEmpty(key)) Http.DefaultRequestHeaders.Add("X-API-Key", key);
    }

    private async Task PollApiAsync()
    {
        if (_process == null || _config == null) return;
        try
        {
            var json = await Http.GetStringAsync($"http://127.0.0.1:{ApiPort()}/api/cameras");
            var cams = JsonNode.Parse(json)!.AsArray();
            _liveAlpr.Clear();
            var online = 0;
            foreach (var cam in cams)
            {
                var id = cam!["id"]!.GetValue<string>();
                _liveAlpr[id] = cam["alpr_enabled"]!.GetValue<bool>();
                if (cam["online"]!.GetValue<bool>()) online++;
            }
            StatusText.Text = $"Running (pid {_process.Id}) — {online}/{cams.Count} camera(s) online";
            LiveToggleButton.IsEnabled = true;

            // Refresh list labels in place to avoid disturbing the selection.
            for (var i = 0; i < CamerasList.Items.Count && i < Cameras().Count; ++i)
            {
                var label = CameraLabel(Cameras()[i]!);
                if (!label.Equals(CamerasList.Items[i])) CamerasList.Items[i] = label;
            }
        }
        catch
        {
            LiveToggleButton.IsEnabled = false; // API not reachable (yet).
        }
    }

    private async void LiveToggleAlpr_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedCamera < 0 || _selectedCamera >= Cameras().Count) return;
        var id = Cameras()[_selectedCamera]!["id"]?.GetValue<string>();
        if (id != null) await ToggleCameraAsync(id);
    }

    /// <summary>Flip a camera's ALPR state on the RUNNING pipeline via the API.
    /// Shared by the detail button and the wall tile click.</summary>
    private async Task ToggleCameraAsync(string id)
    {
        if (_process == null) return;
        var target = !(_liveAlpr.TryGetValue(id, out var current) && current);
        try
        {
            var body = new StringContent($"{{\"enabled\": {(target ? "true" : "false")}}}",
                Encoding.UTF8, "application/json");
            var response = await Http.PostAsync(
                $"http://127.0.0.1:{ApiPort()}/api/cameras/{id}/alpr", body);
            response.EnsureSuccessStatusCode();
            _liveAlpr[id] = target;
            await PollApiAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "API error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    // ---------------------------------------------------------------- process

    private void Start_Click(object sender, RoutedEventArgs e)
    {
        if (_process != null) return;
        ApplyApiKey(); // Live calls must carry the current key even if unsaved.
        if (!File.Exists(ExePathBox.Text))
        {
            MessageBox.Show(this, "anpr.exe not found:\n" + ExePathBox.Text, "Cannot start",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        // Run from the pipeline project root (= parent of the config folder)
        // so relative model/charset paths in the config resolve.
        var workingDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(ConfigPathBox.Text)))
                         ?? Environment.CurrentDirectory;

        var psi = new ProcessStartInfo
        {
            FileName = ExePathBox.Text,
            Arguments = $"--config=\"{ConfigPathBox.Text}\"",
            WorkingDirectory = workingDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,   // the pipeline logs to stderr
            RedirectStandardOutput = true,
        };

        try
        {
            _process = Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Cannot start", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }
        if (_process == null) return;

        _plates.Clear();
        LogBox.Clear();
        while (_pendingLines.TryDequeue(out _)) { } // Discard leftovers from a previous run.
        _process.EnableRaisingEvents = true;
        _process.Exited += (_, _) => Dispatcher.BeginInvoke(OnProcessExited);
        _process.ErrorDataReceived += (_, args) => { if (args.Data != null) _pendingLines.Enqueue(args.Data); };
        _process.OutputDataReceived += (_, args) => { if (args.Data != null) _pendingLines.Enqueue(args.Data); };
        _process.BeginErrorReadLine();
        _process.BeginOutputReadLine();

        StartButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        StatusText.Text = "Running (pid " + _process.Id + ")";
    }

    private void Stop_Click(object sender, RoutedEventArgs e) => StopProcess();

    private void StopProcess()
    {
        if (_process == null) return;
        try
        {
            if (!_process.HasExited) _process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Already exiting; nothing to do.
        }
    }

    private void OnProcessExited()
    {
        _process?.Dispose();
        _process = null;
        FlushPendingLines(); // Show the tail of the log immediately.
        StartButton.IsEnabled = true;
        StopButton.IsEnabled = false;
        LiveToggleButton.IsEnabled = false;
        _liveAlpr.Clear();
        StatusText.Text = "Stopped";
    }

    /// <summary>Timer tick: move queued log lines to the UI in one batch.</summary>
    private void FlushPendingLines()
    {
        if (_pendingLines.IsEmpty) return;

        var batch = new StringBuilder();
        while (_pendingLines.TryDequeue(out var line))
        {
            batch.AppendLine(line);
            ParseLine(line);
        }

        LogBox.AppendText(batch.ToString());
        if (LogBox.Text.Length > 300_000)
        {
            // Rare O(n) trim; never per-line.
            LogBox.Text = LogBox.Text[^150_000..];
        }
        LogBox.ScrollToEnd();
    }

    private void ParseLine(string line)
    {
        var plate = PlateRegex.Match(line);
        if (plate.Success)
        {
            _plates.Insert(0, new PlateRow(
                DateTime.Now.ToString("HH:mm:ss"),
                plate.Groups["cam"].Value,
                plate.Groups["text"].Value,
                plate.Groups["det"].Value,
                plate.Groups["ocr"].Value,
                plate.Groups["box"].Value,
                plate.Groups["seq"].Value));
            if (_plates.Count > 500) _plates.RemoveAt(_plates.Count - 1);
            return;
        }

        var stats = StatsRegex.Match(line);
        if (stats.Success)
        {
            StatsText.Text =
                $"cams: {stats.Groups["cams"].Value}   fps: {stats.Groups["fps"].Value}   " +
                $"motion: {stats.Groups["motion"].Value}   alpr: {stats.Groups["ms"].Value} ms   " +
                $"queue: {stats.Groups["queue"].Value}   reports: {stats.Groups["reports"].Value}";
        }
    }

    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e) => StopProcess();

    // ---------------------------------------------------------------- helpers

    private static string NumText(JsonNode? node, string fallback) =>
        node == null ? fallback : Convert.ToString(node.AsValue().GetValue<object>(), CultureInfo.InvariantCulture) ?? fallback;

    private static string ComboText(ComboBox combo) =>
        (combo.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "";

    private static void SelectComboText(ComboBox combo, string text)
    {
        foreach (var item in combo.Items)
        {
            if ((item as ComboBoxItem)?.Content?.ToString() == text)
            {
                combo.SelectedItem = item;
                return;
            }
        }
    }

    private static double ParseDouble(string text, string field)
    {
        if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            throw new FormatException($"{field}: \"{text}\" is not a number (use '.' as decimal separator)");
        return value;
    }

    private static int ParseInt(string text, string field)
    {
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
            throw new FormatException($"{field}: \"{text}\" is not an integer");
        return value;
    }
}
