using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using Microsoft.Win32;

namespace AnprStudio;

/// <summary>One row of the detected-plates table.</summary>
public sealed record PlateRow(string Time, string Text, string DetConf, string OcrConf, string Box, string Seq);

/// <summary>
/// Test & configuration companion for the ANPR pipeline. Edits the pipeline's
/// JSON config (only the fields shown; everything else in the file is
/// preserved), launches/stops anpr.exe and turns its log stream into a live
/// plate table + statistics.
/// </summary>
public partial class MainWindow : Window
{
    private static readonly Regex PlateRegex = new(
        @"PLATE\s+(?<text>\S+)\s+det=(?<det>[\d.]+)\s+ocr=(?<ocr>[\d.]+)\s+box=\[(?<box>[^\]]*)\]\s+seq=(?<seq>\d+)",
        RegexOptions.Compiled);

    private static readonly Regex StatsRegex = new(
        @"stats: seq=(?<seq>\d+)\s+fps=(?<fps>[\d.]+)\s+alpr_ms=(?<ms>[\d.]+).*reports=(?<reports>\d+)",
        RegexOptions.Compiled);

    private readonly ObservableCollection<PlateRow> _plates = new();
    private JsonNode? _config;
    private Process? _process;

    // Log lines arrive on threadpool threads at a high rate; they are queued
    // here and flushed to the UI in batches by a timer. Touching the UI per
    // line (Dispatcher.Invoke + AppendText) freezes the window under load.
    private readonly ConcurrentQueue<string> _pendingLines = new();
    private readonly DispatcherTimer _logFlushTimer;

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
        GuessDefaultPaths();
        TryLoadConfig(silent: true);
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
        if (dlg.ShowDialog() == true) VideoPathBox.Text = dlg.FileName;
    }

    // ----------------------------------------------------------------- config

    private void ReloadConfig_Click(object sender, RoutedEventArgs e) => TryLoadConfig(silent: false);

    private void TryLoadConfig(bool silent)
    {
        try
        {
            _config = JsonNode.Parse(File.ReadAllText(ConfigPathBox.Text));
            ConfigToUi();
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
            UiToConfig();
            File.WriteAllText(ConfigPathBox.Text,
                _config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            StatusText.Text = "Config saved";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Cannot save config", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>Get-or-create a nested object node, e.g. Node("capture", "simulation").</summary>
    private JsonNode Node(params string[] path)
    {
        JsonNode node = _config!;
        foreach (var key in path)
        {
            node = node[key] ??= new JsonObject();
        }
        return node;
    }

    private JsonNode? ActiveProfileNode()
    {
        var name = (ProfileCombo.SelectedItem as string) ?? Node("processing")["active_model_profile"]?.GetValue<string>();
        return name == null ? null : Node("processing", "model_profiles")[name];
    }

    private void ConfigToUi()
    {
        var sim = Node("capture", "simulation");
        SelectComboText(SourceCombo, Node("capture")["source"]?.GetValue<string>() ?? "simulation");
        VideoPathBox.Text = sim["video_path"]?.GetValue<string>() ?? "";
        RealtimeCheck.IsChecked = sim["realtime"]?.GetValue<bool>() ?? true;
        LoopCheck.IsChecked = sim["loop"]?.GetValue<bool>() ?? true;
        TargetFpsBox.Text = NumText(sim["target_fps"], "0");
        FrameSkipBox.Text = NumText(sim["frame_skip"], "0");
        StartTimeBox.Text = NumText(sim["start_time_seconds"], "0");

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
        ProfileToUi();

        var display = Node("display");
        DisplayCheck.IsChecked = display["enabled"]?.GetValue<bool>() ?? false;
        MaxWidthBox.Text = NumText(display["max_width"], "1280");

        var net = Node("network");
        NetEnabledCheck.IsChecked = net["enabled"]?.GetValue<bool>() ?? true;
        HostBox.Text = net["server_host"]?.GetValue<string>() ?? "127.0.0.1";
        PortBox.Text = NumText(net["server_port"], "9000");

        SelectComboText(LogLevelCombo, Node("logging")["level"]?.GetValue<string>() ?? "info");
    }

    private void ProfileToUi()
    {
        if (ActiveProfileNode() is not JsonNode profile) return;
        DetConfBox.Text = NumText(profile["detection"]?["confidence_threshold"], "0.5");
        NmsBox.Text = NumText(profile["detection"]?["nms_threshold"], "0.45");
        OcrConfBox.Text = NumText(profile["ocr"]?["confidence_threshold"], "0.6");
    }

    private void ProfileCombo_SelectionChanged(object sender, SelectionChangedEventArgs e) => ProfileToUi();

    private void UiToConfig()
    {
        var sim = Node("capture", "simulation");
        Node("capture")["source"] = ComboText(SourceCombo);
        sim["video_path"] = VideoPathBox.Text;
        sim["realtime"] = RealtimeCheck.IsChecked == true;
        sim["loop"] = LoopCheck.IsChecked == true;
        sim["target_fps"] = ParseDouble(TargetFpsBox.Text, "Target FPS");
        sim["frame_skip"] = ParseInt(FrameSkipBox.Text, "Frame skip");
        sim["start_time_seconds"] = ParseDouble(StartTimeBox.Text, "Start (s)");

        var proc = Node("processing");
        if (ProfileCombo.SelectedItem is string profileName)
        {
            proc["active_model_profile"] = profileName;
            if (ActiveProfileNode() is JsonNode profile)
            {
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

        var display = Node("display");
        display["enabled"] = DisplayCheck.IsChecked == true;
        display["max_width"] = ParseInt(MaxWidthBox.Text, "Max window width");

        var net = Node("network");
        net["enabled"] = NetEnabledCheck.IsChecked == true;
        net["server_host"] = HostBox.Text;
        net["server_port"] = ParseInt(PortBox.Text, "Port");

        Node("logging")["level"] = ComboText(LogLevelCombo);
    }

    // ---------------------------------------------------------------- process

    private void Start_Click(object sender, RoutedEventArgs e)
    {
        if (_process != null) return;
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
                $"fps: {stats.Groups["fps"].Value}   alpr: {stats.Groups["ms"].Value} ms   " +
                $"reports: {stats.Groups["reports"].Value}   frame: {stats.Groups["seq"].Value}";
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
