using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;
using System.Web.Script.Serialization;
using System.Security.Cryptography;

namespace WushuBridgeLauncher
{
    internal static class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                if (args.Length == 3 && args[0] == "--self-test")
                    return Launcher.SelfTest(args[1], args[2]);
                if (args.Length == 3 && args[0] == "--self-test-explicit-plan")
                    return Launcher.SelfTestExplicitPlan(args[1], args[2]);
                Application.Run(new Launcher(AppDomain.CurrentDomain.BaseDirectory));
                return 0;
            }
            catch (Exception ex)
            {
                if (args.Length > 0 && args[0].StartsWith("--self-test", StringComparison.Ordinal))
                {
                    Console.Error.WriteLine(ex.ToString());
                    if (args.Length == 3 && args[0] == "--self-test") { Directory.CreateDirectory(args[2]); File.WriteAllText(Path.Combine(args[2], "self-test-error.txt"), ex.ToString()); }
                    return 1;
                }
                MessageBox.Show(ex.Message, "WushuBridge 启动器", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }

    internal sealed class Launcher : Form
    {
        readonly string root;
        readonly Dictionary<string, TextBox> paths = new Dictionary<string, TextBox>();
        readonly List<Control> editable = new List<Control>();
        readonly List<Control> alignedFields = new List<Control>();
        readonly Dictionary<string, Control[]> pathControls = new Dictionary<string, Control[]>();
        readonly Dictionary<string, Label> pathHints = new Dictionary<string, Label>();
        readonly ToolTip help = new ToolTip { AutoPopDelay = 20000 };
        bool restoringSettings;
        TableLayoutPanel fieldPanel, mainLayout;
        Button resumeButton, checkButton;
        readonly Label footer = new Label();
        const string ExplicitMode = "explicit_sentence_target_v1";
        string activeMode;
        readonly RichTextBox log = new RichTextBox();
        readonly Label status = new Label();
        readonly ComboBox trainingMode = new ComboBox();
        readonly NumericUpDown bridgeEpochs = new NumericUpDown { Minimum = 1, Maximum = 100000, Value = 150, Width = 80 };
        readonly NumericUpDown judgeEpochs = new NumericUpDown { Minimum = 1, Maximum = 100000, Value = 200, Width = 80 };
        readonly Label epochHint = new Label();
        FlowLayoutPanel epochControls;
        Label epochLabel;
        readonly CheckBox allRows = new CheckBox();
        readonly Label modeSummary = new Label();
        readonly Button stop = new Button();
        readonly object processLock = new object();
        Process activeProcess;
        volatile bool cancelled;
        bool busy;
        bool updating;
        bool installingUpdate;
        bool updateAvailable;
        readonly Button updateButton = new Button();
        readonly CheckBox autoUpdate = new CheckBox();
        readonly Label updateStatus = new Label();
        StreamWriter logFile;
        string activeRun;
        const string EnvironmentProbe = "import sys,torch,numpy,safetensors; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); assert v >= (2,3), 'PyTorch >= 2.3 required'; print('Python: '+sys.version.split()[0]); print('PyTorch: '+torch.__version__); print('CUDA: '+str(torch.cuda.is_available())); print('GPU: '+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))";
        // A fresh CPU subprocess verifies reuse before any encoding or training.
        const string PreparedProbe = @"import json,sys
from pathlib import Path
from tools.prepare_aligned_dataset import tokenizer_provenance, sha256, METHOD, TRANSPORT_METHOD
p,d,j,e,s,c=map(lambda v:Path(v).resolve(),sys.argv[1:7]); expected=sys.argv[7]; expected_policy=sys.argv[8]
if expected not in (METHOD,TRANSPORT_METHOD): raise ValueError('Unknown preparation method')
r=json.loads((p/'preparation_receipt.json').read_text(encoding='utf-8-sig'))
if r.get('status')!='completed' or r.get('method')!=expected: raise ValueError('Invalid preparation receipt for selected mode')
if r.get('selection_policy')!=expected_policy: raise ValueError('Prepared selection policy changed')
sources={'pairs':j,'dataset':d,'metadata':d.with_suffix('.json'),'encoding_receipt':Path(str(d)+'.receipt.json'),'encoder':e,'split':s}
for name,path in sources.items():
 if sha256(path)!=r['source_sha256'].get(name): raise ValueError('Prepared source changed: '+name)
required={'pairs.npz','pairs.json','pairs.jsonl','pairs.npz.receipt.json','split_manifest.json'}
if set(r['artifacts'])!=required: raise ValueError('Unexpected prepared artifact manifest')
for name in required:
 path=p/name
 if path.stat().st_size!=r['artifacts'][name]['bytes'] or sha256(path)!=r['artifacts'][name]['sha256']: raise ValueError('Prepared artifact changed: '+name)
import tools.prepare_aligned_dataset as prep
if sha256(prep.__file__)!=r['script_sha256']: raise ValueError('Preparation code changed; keep the original environment')
identity=json.loads(json.loads(d.with_suffix('.json').read_text(encoding='utf-8-sig'))['encoder'])
if tokenizer_provenance(c,identity)!=r['tokenizer_provenance']: raise ValueError('Tokenizer provenance changed')
if expected==TRANSPORT_METHOD:
 policy=r.get('transport_policy',{})
 if policy.get('token_alignment_source_sha256')!=sha256(Path(prep.__file__).resolve().parents[1]/'wushu_bridge/token_alignment.py'): raise ValueError('Transport code changed')
print('Prepared source/artifact/tokenizer receipt integrity: OK; resume also requires identical run signature')";

        internal Launcher(string directory, bool loadSavedSettings = true)
        {
            root = Path.GetFullPath(directory);
            Text = "WushuBridge 训练器 · By T8star";
            Font = new Font("Microsoft YaHei UI", 9F);
            BackColor = Color.FromArgb(246, 248, 250);
            ClientSize = new Size(1080, 1030);
            MinimumSize = new Size(900, 1010);
            StartPosition = FormStartPosition.CenterScreen;
            AutoScaleMode = AutoScaleMode.Dpi;

            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24), ColumnCount = 1, RowCount = 9 };
            mainLayout = layout;
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 32));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 462));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 47));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 32));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 32));
            Controls.Add(layout);
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
            layout.Controls.Add(new Label { Text = "WushuBridge 训练器 · By T8star", Font = new Font(Font.FontFamily, 22, FontStyle.Bold), AutoSize = true, ForeColor = Color.FromArgb(23, 54, 66) }, 0, 0);
            modeSummary.AutoSize = true; modeSummary.ForeColor = Color.DimGray;
            layout.Controls.Add(modeSummary, 0, 1);

            var fields = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 4, RowCount = 14 };
            fieldPanel = fields;
            fields.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 154));
            fields.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            fields.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 86));
            fields.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 260));
            layout.Controls.Add(fields, 0, 2);
            AddPath(fields, 0, "Python", "Python 环境", false, "Python|python.exe|所有文件|*.*");
            AddPath(fields, 1, "ComfyRoot", "ComfyUI 目录", true, null);
            AddPath(fields, 2, "Encoder", "H3 文本编码器", false, "模型权重|*.safetensors;*.gguf;*.pt;*.pth|所有文件|*.*");
            AddPath(fields, 3, "Pairs", "源训练对 JSONL", false, "JSONL|*.jsonl|所有文件|*.*");
            AddPath(fields, 4, "Dataset", "源编码数据 NPZ", false, "数据集|*.npz", true);
            AddPath(fields, 5, "RunDirectory", "训练输出目录", true, null);
            AddPath(fields, 6, "Config", "训练参数 JSON", false, "配置|*.json");
            AddPath(fields, 7, "SplitManifest", "源偏好分组划分", false, "分组清单 JSON|*.json");
            AddPath(fields, 8, "PreparedDirectory", "实验准备数据目录", true, null);
            AddPath(fields, 9, "VideoFeedbackRoot", "视频反馈证据目录", true, null);
            AddPath(fields, 10, "TargetManifest", "显式 train 目标清单", false, "目标清单 JSON|*.json");
            AddPath(fields, 11, "ExplicitFitConfig", "拟合参数（可选）", false, "参数 JSON|*.json");
            fields.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            fields.Controls.Add(new Label { Text = "训练方式", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, 12);
            trainingMode.DropDownStyle = ComboBoxStyle.DropDownList;
            trainingMode.Items.AddRange(new object[] { "数值评分数据：语义桥 + JEV（原流程）", "无评分偏好对：仅语义桥（必须选择分组划分）", "实验：等长逐 token 对齐 + identity（仅语义桥）", "实验：变长 token 单调映射 + identity（仅语义桥）", "实验：显式句段目标拟合（CPU，仅语义桥）" });
            trainingMode.Dock = DockStyle.Fill; trainingMode.Margin = new Padding(0, 7, 8, 5);
            fields.Controls.Add(trainingMode, 1, 12); editable.Add(trainingMode);
            allRows.Text = "全变体"; allRows.Dock = DockStyle.Fill; allRows.Margin = new Padding(0, 7, 0, 5);
            fields.Controls.Add(allRows, 2, 12); editable.Add(allRows);
            new ToolTip().SetToolTip(allRows, "实验模式：保留同组所有已审核训练对；仍按 group_id 划分，不视作独立场景。默认每组一条。");
            trainingMode.SelectedIndex = 1;
            trainingMode.SelectedIndexChanged += delegate { UpdateTrainingMode(!restoringSettings); };
            fields.Controls.Add(new Label { Text = "新手优先选无评分偏好对；实验模式需专门准备", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, ForeColor = Color.DimGray }, 3, 12);
            fields.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
            epochLabel = new Label { Text = "训练轮次（最大）", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };
            fields.Controls.Add(epochLabel, 0, 13);
            epochControls = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false, Padding = new Padding(0, 6, 0, 0) };
            epochControls.Controls.Add(new Label { Text = "语义桥", AutoSize = true, Margin = new Padding(0, 4, 5, 0) });
            epochControls.Controls.Add(bridgeEpochs);
            epochControls.Controls.Add(new Label { Text = "JEV", AutoSize = true, Margin = new Padding(14, 4, 5, 0) });
            epochControls.Controls.Add(judgeEpochs);
            fields.Controls.Add(epochControls, 1, 13); fields.SetColumnSpan(epochControls, 2);
            epochHint.Text = "验证集无改善可提前停止；续训沿用原轮次";
            epochHint.Dock = DockStyle.Fill; epochHint.ForeColor = Color.DimGray; epochHint.TextAlign = ContentAlignment.MiddleLeft;
            fields.Controls.Add(epochHint, 3, 13);
            editable.Add(bridgeEpochs); editable.Add(judgeEpochs);
            help.SetToolTip(bridgeEpochs, "最多训练多少轮。新任务可调整；早停仍由 patience 控制。不会修改原 JSON。");
            help.SetToolTip(judgeEpochs, "仅数值评分模式训练 JEV；偏好模式不使用。");
            paths["Config"].TextChanged += delegate { RefreshEpochDefaults(); };

            var actions = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            layout.Controls.Add(actions, 0, 3);
            checkButton = AddAction(actions, "检查环境", async delegate { await Start(false, true); }, true, false);
            AddAction(actions, "开始训练", async delegate { await Start(false, false); }, true, true);
            resumeButton = AddAction(actions, "断点续训", async delegate { await Start(true, false); }, true, false);
            stop.Text = "停止任务"; stop.Size = new Size(105, 34); stop.Enabled = false;
            stop.Click += delegate { RequestStop(); };
            actions.Controls.Add(stop);
            AddAction(actions, "新输出目录", delegate { paths["RunDirectory"].Text = NewRunPath(); if (trainingMode.SelectedIndex != 4) paths["Dataset"].Text = NewDatasetPath(trainingMode.SelectedIndex != 0); if (trainingMode.SelectedIndex == 2 || trainingMode.SelectedIndex == 3) paths["PreparedDirectory"].Text = NewPreparedPath(trainingMode.SelectedIndex == 3); }, true, false);
            AddAction(actions, "查看输出", delegate { OpenFolder(activeRun ?? paths["RunDirectory"].Text); }, false, false);
            AddAction(actions, "使用说明", delegate { OpenFile(File.Exists(Path.Combine(root, "如何训练.md")) ? Path.Combine(root, "如何训练.md") : Path.Combine(root, "docs", File.Exists(Path.Combine(root, "python", "python.exe")) ? "便携整合包.md" : "可复用训练器.md")); }, false, false);
            AddAction(actions, "模型用法", delegate {
                string guide = Path.Combine(paths["RunDirectory"].Text, "模型使用说明.md");
                if (File.Exists(guide)) OpenFile(guide);
                else MessageBox.Show(this, "训练完成后，此处会打开该模型专属的 alpha 与配套参数说明。先选对应的训练输出目录。", "模型使用说明");
            }, false, false);

            status.Text = "就绪 · 请选择空闲时段开始训练";
            status.Dock = DockStyle.Fill; status.TextAlign = ContentAlignment.MiddleLeft;
            status.ForeColor = Color.FromArgb(0, 108, 91);
            layout.Controls.Add(status, 0, 4);
            log.Dock = DockStyle.Fill; log.ReadOnly = true; log.BackColor = Color.FromArgb(23, 32, 42);
            log.ForeColor = Color.FromArgb(221, 232, 240); log.BorderStyle = BorderStyle.None;
            log.Font = new Font("Consolas", 10); log.WordWrap = false;
            layout.Controls.Add(log, 0, 5);
            footer.Dock = DockStyle.Fill; footer.TextAlign = ContentAlignment.MiddleLeft; footer.ForeColor = Color.DimGray;
            layout.Controls.Add(footer, 0, 6);
            var social = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            layout.Controls.Add(social, 0, 7);
            foreach (var link in new[] {
                new[] { "B站", "https://space.bilibili.com/385085361" },
                new[] { "YouTube", "https://www.youtube.com/@T8star-Aix/" },
                new[] { "API", "https://api.seedance.nz/sign-up?aff=5f4w" },
                new[] { "免费画廊", "https://www.openzhenzhen.com" },
                new[] { "在线AI应用", "https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121" },
                new[] { "ComfyUI整合包", "https://pan.quark.cn/s/264edb7e36bd" },
                new[] { "Hugging Face", "https://huggingface.co/t8star" } })
            {
                string url = link[1];
                var item = new LinkLabel { Text = link[0], AutoSize = true, Margin = new Padding(0, 7, 18, 0) };
                help.SetToolTip(item, url);
                item.LinkClicked += delegate { try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); } catch (Exception ex) { MessageBox.Show(this, ex.Message); } };
                social.Controls.Add(item);
            }
            var updateBar = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            layout.Controls.Add(updateBar, 0, 8);
            updateButton.Text = "检查更新"; updateButton.Size = new Size(105, 30);
            updateButton.Click += async delegate { await UpdateApplication(updateAvailable); };
            updateBar.Controls.Add(updateButton);
            autoUpdate.Text = "启动检查更新"; autoUpdate.Checked = true; autoUpdate.AutoSize = true; autoUpdate.Margin = new Padding(10, 7, 12, 0);
            updateBar.Controls.Add(autoUpdate);
            updateStatus.Text = "当前 " + ApplicationVersion() + " · 更新保留 Python、模型、数据和设置";
            updateStatus.AutoSize = true; updateStatus.Margin = new Padding(0, 7, 0, 0);
            updateBar.Controls.Add(updateStatus);
            LoadSettings(loadSavedSettings);
            UpdateTrainingMode(false);
            Append("选择“检查环境”确认依赖；“开始训练”会依次校验数据、编码、训练和评估。");
            Append("已有 NPZ 会核对训练对和编码器 SHA256。已有训练目录请使用“断点续训”。");
            Append("偏好模式只训练桥，并输出两种 magnitude 的验证集诊断；不生成 JEV，也不自动选择 alpha。");
            Append("逐 token 实验使用原始输入与真实 tokenizer 准备独立数据；变长模式将目标映射到原有 token 数。固定 alpha / per_token 辅助诊断不代表视频质量。");
            Append("只有使用完整声画双 seed 复核的 video_feedback_v1 训练对时才填写私有证据目录；程序会在检查、编码、准备和训练阶段重新核验。普通作者偏好对留空。");
            Append("实验 epoch 选择以训练 run 的 correction + identity 验证损失为准；辅助诊断不选择 alpha，也不表示实验通过。");
            Append("优先使用包内 Python/ComfyUI；不会自动安装依赖或卸载其他任务的模型。");
            Append("首次已填写包内偏好格式示例和配套分组；示例未验证视频效果。正式自训请更换自己的 JSONL 与分组，并选择真实 H3 文本编码器。其余默认路径通常无需修改。");
            FormClosing += OnClosing;
            if (loadSavedSettings) Shown += async delegate {
                string error = Path.Combine(root, "local", "updater", "last_error.json");
                if (File.Exists(error)) { Append("上次更新提示：" + File.ReadAllText(error, Encoding.UTF8)); }
                if (autoUpdate.Checked) await UpdateApplication(false);
            };
        }

        string ApplicationVersion()
        {
            try { return Convert.ToString(ReadMap(Path.Combine(root, "app_version.json"))["version"]); }
            catch { return "0.0.0"; }
        }

        Dictionary<string, object> UpdateCommand(string python, string[] args)
        {
            using (var p = new Process())
            {
                p.StartInfo = ProcessInfo(python, args, root);
                p.Start();
                var stdout = p.StandardOutput.ReadToEndAsync();
                var stderr = p.StandardError.ReadToEndAsync();
                if (!p.WaitForExit(180000)) { p.Kill(); throw new Exception("更新请求超时，可稍后重试。"); }
                Task.WaitAll(stdout, stderr);
                var result = ObjectMap(Json().DeserializeObject(stdout.Result));
                if (p.ExitCode != 0) throw new Exception(result.ContainsKey("error") ? Convert.ToString(result["error"]) : stderr.Result);
                return result;
            }
        }

        async Task UpdateApplication(bool install)
        {
            if (busy || updating) return;
            string python = paths["Python"].Text.Trim();
            string bundled = Path.Combine(root, "python", "python.exe");
            if (File.Exists(bundled)) python = bundled;
            if (!File.Exists(python)) { updateStatus.Text = "先选择 Python 3.10+，再检查更新"; return; }
            updating = true;
            installingUpdate = install;
            if (install) foreach (var control in editable) control.Enabled = false;
            updateButton.Enabled = false;
            updateStatus.Text = install ? "下载并校验更新包…" : "正在检查 GitHub Release…";
            bool closingForUpdate = false;
            try
            {
                string command = install ? "stage" : "check";
                var result = await Task.Run(() => UpdateCommand(python, new[] { "-B", "-X", "utf8", Path.Combine(root, "tools", "updater.py"), command, "--root", root }));
                if (install)
                {
                    SaveSettings(Snapshot());
                    var info = ProcessInfo(python, new[] { "-B", "-X", "utf8", Convert.ToString(result["helper"]), "apply", "--root", root,
                        "--staging", Convert.ToString(result["staging"]), "--wait-pid", Process.GetCurrentProcess().Id.ToString(), "--restart" }, root);
                    info.RedirectStandardOutput = false; info.RedirectStandardError = false;
                    info.StandardOutputEncoding = null; info.StandardErrorEncoding = null;
                    Process.Start(info);
                    closingForUpdate = true;
                    updating = false;
                    Close();
                }
                else
                {
                    updateAvailable = Convert.ToBoolean(result["available"]);
                    updateButton.Text = updateAvailable ? "安装更新" : "检查更新";
                    updateStatus.Text = Convert.ToString(result["message"]) + (result.ContainsKey("latest") ? " · " + Convert.ToString(result["latest"]) : "");
                    if (updateAvailable) Append("发现新版 " + result["latest"] + "，空闲时点“安装更新”。安装会关闭并重启本启动器。");
                }
            }
            catch (Exception ex) { updateStatus.Text = "更新未完成，可稍后重试"; Append("更新：" + ex.Message); }
            finally
            {
                if (!closingForUpdate && !IsDisposed) {
                    updating = false; installingUpdate = false; updateButton.Enabled = !busy;
                    if (install) {
                        foreach (var control in editable) control.Enabled = true;
                        UpdateTrainingMode(false);
                    }
                }
            }
        }

        void AddPath(TableLayoutPanel panel, int row, string key, string title, bool folder, string filter, bool output = false)
        {
            panel.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            var label = new Label { Text = title, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };
            panel.Controls.Add(label, 0, row);
            var box = new TextBox { Dock = DockStyle.Fill, Margin = new Padding(0, 7, 8, 5) };
            paths.Add(key, box); editable.Add(box); panel.Controls.Add(box, 1, row);
            var browse = new Button { Text = "浏览…", Dock = DockStyle.Fill, Margin = new Padding(0, 5, 0, 6) };
            editable.Add(browse); panel.Controls.Add(browse, 2, row);
            var hint = new Label { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, ForeColor = Color.DimGray, Font = new Font(Font.FontFamily, 8.5F), Margin = new Padding(8, 0, 0, 0) };
            panel.Controls.Add(hint, 3, row); pathHints[key] = hint;
            pathControls[key] = new Control[] { label, box, browse, hint };
            box.TextChanged += delegate { RefreshPathHints(); };
            if (key == "PreparedDirectory") alignedFields.AddRange(new Control[] { label, box, browse, hint });
            browse.Click += delegate
            {
                if (folder)
                {
                    using (var dialog = new FolderBrowserDialog { Description = key == "PreparedDirectory" ? "选择已有准备回执的目录以复用；否则会在所选目录下使用新的 aligned 子目录。" : title, SelectedPath = box.Text })
                        if (dialog.ShowDialog(this) == DialogResult.OK)
                            box.Text = key == "RunDirectory" && trainingMode.SelectedIndex == 4
                                ? Path.Combine(dialog.SelectedPath, "explicit_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff"))
                                : key == "PreparedDirectory" && !File.Exists(Path.Combine(dialog.SelectedPath, "preparation_receipt.json"))
                                ? Path.Combine(dialog.SelectedPath, "aligned_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff")) : dialog.SelectedPath;
                }
                else
                {
                    using (FileDialog dialog = output ? (FileDialog)new SaveFileDialog { OverwritePrompt = false } : new OpenFileDialog())
                    {
                        dialog.Filter = filter; dialog.FileName = box.Text;
                        if (dialog.ShowDialog(this) == DialogResult.OK) box.Text = dialog.FileName;
                    }
                }
            };
        }

        Button AddAction(Control panel, string title, EventHandler action, bool disableWhileBusy, bool primary)
        {
            var button = new Button { Text = title, Size = new Size(105, 34), FlatStyle = FlatStyle.Flat, Margin = new Padding(0, 3, 9, 3) };
            if (primary) { button.BackColor = Color.FromArgb(0, 108, 91); button.ForeColor = Color.White; }
            button.Click += action; panel.Controls.Add(button);
            if (disableWhileBusy) editable.Add(button);
            return button;
        }

        string NewRunPath()
        {
            return Path.Combine(root, "runs", "wushu_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff"));
        }

        string NewPreparedPath(bool transport = false)
        {
            return Path.Combine(root, "local", (transport ? "transport_" : "aligned_") + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff"));
        }

        string NewDatasetPath(bool preference = true)
        {
            return Path.Combine(root, "local", (preference ? "preference_" : "wushu_") + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff") + ".npz");
        }

        string PreferencePairs()
        {
            string bundled = Path.Combine(root, "datasets", "preference_example.jsonl");
            string fallback = Path.Combine(root, "examples", "preference_starter", "pairs.format-example.jsonl");
            return File.Exists(bundled) ? bundled : File.Exists(fallback) ? fallback : "";
        }

        string PreferenceSplit()
        {
            string bundled = Path.Combine(root, "datasets", "preference_example_split.json");
            string fallback = Path.Combine(root, "examples", "preference_starter", "split.format-example.json");
            return File.Exists(bundled) ? bundled : File.Exists(fallback) ? fallback : "";
        }

        string AuthorPairs()
        {
            string bundled = Path.Combine(root, "datasets", "wushu_pairs_v1_pairs.jsonl");
            return File.Exists(bundled) ? bundled : Path.Combine(root, "models", "wushu_bridge", "datasets", "wushu_pairs_v1_pairs.jsonl");
        }

        void RefreshPathHints()
        {
            foreach (var item in pathHints)
            {
                string key = item.Key, value = paths[key].Text;
                string note = key == "Python" ? "自动：包内 Python，通常不用修改"
                    : key == "ComfyRoot" ? "自动：包内 H3 编码源码，通常不用修改"
                    : key == "Encoder" ? (File.Exists(value) ? "已找到编码器；请核对与实际推理版本一致" : "必选：另备 H3 文本编码器，与推理版本一致")
                    : key == "Pairs" ? (value == PreferencePairs() && value != "" ? "8对格式示例；正式训练请换自己的 JSONL" : value == AuthorPairs() ? "内置作者676对；数值模式同时训练 JEV" : "自备：审核过的 bad/good 训练对 JSONL")
                    : key == "SplitManifest" ? (trainingMode.SelectedIndex == 0 ? "数值模式无需填写，可留空" : value == PreferenceSplit() && value != "" ? "示例配套分组；自训时与 JSONL 一起更换" : "必选：与上方 JSONL 匹配的分组划分")
                    : key == "Dataset" ? "自动：首次由程序生成；换数据点新输出目录"
                    : key == "RunDirectory" ? "自动：新运行目录；续训时选原目录"
                    : key == "Config" ? "自动：随训练方式切换；初次保留默认"
                    : key == "PreparedDirectory" ? "仅 token 实验使用；普通训练保持默认"
                    : key == "VideoFeedbackRoot" ? "普通自编文本留空；仅真实反馈证据填写"
                    : key == "TargetManifest" ? "必选：实验专用 train 目标清单，见高级文档"
                    : "可选：实验拟合参数；初次可留空";
                item.Value.Text = note;
                item.Value.ForeColor = note.StartsWith("必选") || note.Contains("格式示例") || note.Contains("示例配套") ? Color.FromArgb(160, 88, 0) : Color.DimGray;
                help.SetToolTip(item.Value, note); help.SetToolTip(paths[key], note + "\n" + value);
            }
        }

        void LoadSettings(bool loadSavedSettings)
        {
            paths["Pairs"].Text = PreferencePairs();
            paths["SplitManifest"].Text = PreferenceSplit();
            paths["Dataset"].Text = NewDatasetPath();
            paths["RunDirectory"].Text = NewRunPath();
            paths["Config"].Text = Path.Combine(root, "configs", "preference_v1.json");
            paths["PreparedDirectory"].Text = NewPreparedPath();
            if (File.Exists(Path.Combine(root, "python", "python.exe")))
                paths["Python"].Text = Path.Combine(root, "python", "python.exe");
            if (Directory.Exists(Path.Combine(root, "ComfyUI", "comfy")))
                paths["ComfyRoot"].Text = Path.Combine(root, "ComfyUI");
            var encoders = new[] { Path.Combine(root, "models", "text_encoders"), Path.Combine(root, "ComfyUI", "models", "text_encoders") }
                .Where(Directory.Exists).SelectMany(p => Directory.GetFiles(p)).Where(p => new[] { ".safetensors", ".gguf", ".pt", ".pth" }.Contains(Path.GetExtension(p).ToLowerInvariant())).ToArray();
            if (encoders.Length == 1) paths["Encoder"].Text = encoders[0];
            string settings = Path.Combine(root, "local", "launcher.settings.xml");
            if (loadSavedSettings && File.Exists(settings))
            {
                try
                {
                    restoringSettings = true;
                    var doc = XDocument.Load(settings);
                    foreach (var item in paths)
                    {
                        var value = doc.Root.Element(item.Key);
                        if (value != null)
                            item.Value.Text = value.Value.StartsWith("relative:", StringComparison.Ordinal)
                                ? Path.GetFullPath(Path.Combine(root, value.Value.Substring(9))) : value.Value;
                    }
                    var mode = doc.Root.Element("TrainingMode");
                    if (mode != null) trainingMode.SelectedIndex = mode.Value == ExplicitMode ? 4 : mode.Value == "numeric_scores" ? 0 : mode.Value == "aligned_token_identity_v1" ? 2 : mode.Value == "monotone_token_identity_v1" ? 3 : 1;
                    var policy = doc.Root.Element("SelectionPolicy");
                    if (policy != null) allRows.Checked = policy.Value == "all_rows";
                    var updates = doc.Root.Element("AutoCheckUpdates");
                    if (updates != null) autoUpdate.Checked = updates.Value != "false";
                    decimal saved;
                    if (Decimal.TryParse((string)doc.Root.Element("BridgeEpochs"), out saved) && saved >= 1 && saved <= bridgeEpochs.Maximum) bridgeEpochs.Value = saved;
                    if (Decimal.TryParse((string)doc.Root.Element("JudgeEpochs"), out saved) && saved >= 1 && saved <= judgeEpochs.Maximum) judgeEpochs.Value = saved;
                }
                catch (Exception ex) { Append("无法读取上次设置：" + ex.Message); }
                finally { restoringSettings = false; }
            }
            RefreshPathHints();
        }

        Dictionary<string, string> Snapshot()
        {
            var snapshot = paths.ToDictionary(p => p.Key, p => p.Value.Text.Trim());
            snapshot["TrainingMode"] = trainingMode.SelectedIndex == 4 ? ExplicitMode : trainingMode.SelectedIndex == 3 ? "monotone_token_identity_v1" : trainingMode.SelectedIndex == 2 ? "aligned_token_identity_v1" : trainingMode.SelectedIndex == 1 ? "preference_only" : "numeric_scores";
            snapshot["SelectionPolicy"] = allRows.Checked ? "all_rows" : "one_per_group";
            snapshot["AutoCheckUpdates"] = autoUpdate.Checked ? "true" : "false";
            snapshot["BridgeEpochs"] = bridgeEpochs.Value.ToString(CultureInfo.InvariantCulture);
            snapshot["JudgeEpochs"] = judgeEpochs.Value.ToString(CultureInfo.InvariantCulture);
            return snapshot;
        }

        void UpdateTrainingMode(bool changeDefaults)
        {
            bool explicitFit = trainingMode.SelectedIndex == 4;
            allRows.Enabled = (trainingMode.SelectedIndex == 2 || trainingMode.SelectedIndex == 3) && !busy;
            mainLayout.RowStyles[2].Height = explicitFit ? 210 : 500;
            epochControls.Visible = epochHint.Visible = epochLabel.Visible = !explicitFit;
            fieldPanel.RowStyles[13].Height = explicitFit ? 0 : 38;
            bridgeEpochs.Enabled = !busy && !explicitFit;
            judgeEpochs.Enabled = !busy && trainingMode.SelectedIndex == 0;
            foreach (var entry in pathControls)
            {
                bool explicitField = entry.Key == "TargetManifest" || entry.Key == "ExplicitFitConfig";
                bool common = entry.Key == "Python" || entry.Key == "RunDirectory";
                bool visible = common || (explicitFit == explicitField);
                foreach (var control in entry.Value) { control.Visible = visible; control.Enabled = visible && !busy; }
                fieldPanel.RowStyles[fieldPanel.GetRow(entry.Value[0])].Height = visible ? 42 : 0;
            }
            if (resumeButton != null) resumeButton.Enabled = !busy && !explicitFit;
            if (checkButton != null) checkButton.Text = explicitFit ? "检查目标" : "检查环境";
            footer.Text = explicitFit ? "CPU 拟合仅保存最终权重；不支持断点续训。完成不代表视频质量或泛化通过。" : "缓存与日志保存在 local/；模型保存在训练输出目录。停止后从上一完整 epoch 恢复。";
            if (explicitFit)
            {
                foreach (var control in alignedFields) control.Enabled = false;
                modeSummary.Text = "已构建 train 目标清单 → CPU 校验 → 固定步数拟合与重载验证（不编码，不验证视频质量）";
                return;
            }
            bool aligned = trainingMode.SelectedIndex == 2;
            bool transport = trainingMode.SelectedIndex == 3;
            bool experimental = aligned || transport;
            bool preference = trainingMode.SelectedIndex != 0;
            foreach (var control in alignedFields) control.Enabled = experimental && !busy;
            modeSummary.Text = transport ? "实验：原始 H3 数据 → CPU 变长单调映射 → 仅语义桥 → 固定强度辅助诊断（不是视频质量）" : aligned ? "实验：原始 H3 数据 → CPU 等长 token 对齐 → 仅语义桥 → 固定强度辅助诊断（不是视频质量）" : preference ? "无评分偏好对 → 仅语义桥 → CPU 权重检查 → validation 诊断（不是视频质量）"
                : "真实 H3 编码 → 语义桥 + JEV 训练 → 权重验证与评估（原流程）";
            RefreshPathHints();
            if (!changeDefaults) return;
            if (new[] { "author_v1.json", "preference_v1.json", "experimental_aligned_token_v1.json", "experimental_monotone_token_v1.json" }.Any(name => String.Equals(paths["Config"].Text, Path.Combine(root, "configs", name), StringComparison.OrdinalIgnoreCase)))
                paths["Config"].Text = Path.Combine(root, "configs", transport ? "experimental_monotone_token_v1.json" : aligned ? "experimental_aligned_token_v1.json" : preference ? "preference_v1.json" : "author_v1.json");
            string preparedPath = paths["PreparedDirectory"].Text;
            string preparedName = Path.GetFileName(preparedPath.TrimEnd(Path.DirectorySeparatorChar));
            bool autoPrepared = Path.GetDirectoryName(preparedPath) != null && String.Equals(Path.GetFullPath(Path.GetDirectoryName(preparedPath)), Path.Combine(root, "local"), StringComparison.OrdinalIgnoreCase)
                && (preparedName.StartsWith("aligned_", StringComparison.OrdinalIgnoreCase) || preparedName.StartsWith("transport_", StringComparison.OrdinalIgnoreCase));
            if (experimental && autoPrepared && ((transport && preparedName.StartsWith("aligned_", StringComparison.OrdinalIgnoreCase)) || (aligned && preparedName.StartsWith("transport_", StringComparison.OrdinalIgnoreCase))))
                paths["PreparedDirectory"].Text = NewPreparedPath(transport);
            string dataset = paths["Dataset"].Text;
            if (String.Equals(Path.GetDirectoryName(dataset), Path.Combine(root, "local"), StringComparison.OrdinalIgnoreCase)
                && (Path.GetFileName(dataset).StartsWith("preference_") || Path.GetFileName(dataset).StartsWith("wushu_")))
                paths["Dataset"].Text = NewDatasetPath(preference);
            string legacyPairs = Path.Combine(root, "models", "wushu_bridge", "datasets", "wushu_pairs_v1_pairs.jsonl");
            bool examplePairs = paths["Pairs"].Text == PreferencePairs();
            if (preference && (paths["Pairs"].Text == AuthorPairs() || paths["Pairs"].Text == legacyPairs || String.IsNullOrWhiteSpace(paths["Pairs"].Text)))
            {
                paths["Pairs"].Text = PreferencePairs();
                paths["SplitManifest"].Text = PreferenceSplit();
            }
            if (!preference && (examplePairs || String.IsNullOrWhiteSpace(paths["Pairs"].Text)))
            {
                paths["Pairs"].Text = AuthorPairs();
                if (paths["SplitManifest"].Text == PreferenceSplit()) paths["SplitManifest"].Text = "";
            }
            RefreshPathHints();
        }

        void RefreshEpochDefaults()
        {
            try {
                if (!File.Exists(paths["Config"].Text)) return;
                var values = ObjectMap(Json().DeserializeObject(File.ReadAllText(paths["Config"].Text)));
                if (values.ContainsKey("bridge_epochs")) bridgeEpochs.Value = Math.Max(bridgeEpochs.Minimum, Math.Min(bridgeEpochs.Maximum, Convert.ToDecimal(values["bridge_epochs"])));
                if (values.ContainsKey("judge_epochs")) judgeEpochs.Value = Math.Max(judgeEpochs.Minimum, Math.Min(judgeEpochs.Maximum, Convert.ToDecimal(values["judge_epochs"])));
            } catch (Exception) { /* The normal preflight reports invalid JSON. */ }
        }

        internal static int[] EpochOptions(Dictionary<string, string> settings, bool resume)
        {
            string run = Path.Combine(settings["RunDirectory"], "run.json");
            Dictionary<string, object> config;
            if (resume && File.Exists(run)) config = ObjectMap(ObjectMap(Json().DeserializeObject(File.ReadAllText(run)))["config"]);
            else config = ObjectMap(Json().DeserializeObject(File.ReadAllText(settings["Config"])));
            var values = new int[2];
            for (int i = 0; i < 2; i++) {
                string key = i == 0 ? "BridgeEpochs" : "JudgeEpochs", jsonKey = i == 0 ? "bridge_epochs" : "judge_epochs";
                string raw = (!resume || !File.Exists(run)) && settings.ContainsKey(key) ? settings[key]
                    : config.ContainsKey(jsonKey) ? Convert.ToString(config[jsonKey], CultureInfo.InvariantCulture) : i == 0 ? "150" : "200";
                if (!Int32.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out values[i]) || values[i] < 1 || values[i] > 100000)
                    throw new Exception("训练轮次必须是 1–100000 的整数。");
            }
            return values;
        }

        void SaveSettings(Dictionary<string, string> settings)
        {
            string dir = Path.Combine(root, "local"); Directory.CreateDirectory(dir);
            string prefix = root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            var doc = new XDocument(new XElement("Launcher", settings.Select(p => new XElement(p.Key,
                p.Value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ? "relative:" + p.Value.Substring(prefix.Length) : p.Value))));
            string target = Path.Combine(dir, "launcher.settings.xml");
            string temp = target + ".tmp";
            doc.Save(temp);
            if (File.Exists(target)) File.Replace(temp, target, null); else File.Move(temp, target);
        }

        internal static string Quote(string value)
        {
            // Windows CRT argv quoting: preserve spaces, quotes and trailing backslashes.
            var b = new StringBuilder("\""); int slashes = 0;
            foreach (char c in value)
            {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') b.Append('\\', slashes * 2 + 1);
                else b.Append('\\', slashes);
                b.Append(c); slashes = 0;
            }
            b.Append('\\', slashes * 2); b.Append('"'); return b.ToString();
        }

        static JavaScriptSerializer Json()
        {
            return new JavaScriptSerializer { MaxJsonLength = 32 * 1024 * 1024, RecursionLimit = 128 };
        }

        static Dictionary<string, object> ObjectMap(object value)
        {
            var map = value as Dictionary<string, object>;
            if (map == null) throw new Exception("需要 JSON object。");
            return map;
        }

        static Dictionary<string, object> ReadMap(string path)
        {
            return ObjectMap(Json().DeserializeObject(File.ReadAllText(path, Encoding.UTF8)));
        }

        static string FileSha(string path)
        {
            using (var input = File.OpenRead(path))
            using (var sha = SHA256.Create())
                return String.Concat(sha.ComputeHash(input).Select(b => b.ToString("x2", CultureInfo.InvariantCulture)));
        }

        internal static Dictionary<string, object> ExplicitConfig(string path)
        {
            var values = new Dictionary<string, object> {
                {"steps",300}, {"seed",1234}, {"lr",0.0003}, {"weight_decay",0.01}, {"batch_size",4},
                {"hidden",256}, {"layers",2}, {"heads",4}, {"max_tokens",4096}, {"alpha",1.0},
                {"y_identity_weight",1.0}, {"x_identity_weight",0.0}, {"x_identity_mask",null}, {"threads",2}
            };
            if (!String.IsNullOrWhiteSpace(path))
                foreach (var item in ReadMap(path))
                {
                    if (!values.ContainsKey(item.Key)) throw new Exception("未知拟合参数：" + item.Key);
                    values[item.Key] = item.Value;
                }
            var integers = new[] { "steps", "seed", "batch_size", "hidden", "layers", "heads", "max_tokens", "threads" };
            foreach (var item in values)
            {
                if (item.Key == "x_identity_mask")
                {
                    if (item.Value != null && (!(item.Value is string) || String.IsNullOrWhiteSpace((string)item.Value)))
                        throw new Exception("x_identity_mask 必须是非空名称或 null。");
                    continue;
                }
                if (!(item.Value is int || item.Value is long || item.Value is decimal || item.Value is double))
                    throw new Exception("拟合参数必须是数值：" + item.Key);
                double number = Convert.ToDouble(item.Value, CultureInfo.InvariantCulture);
                if (Double.IsNaN(number) || Double.IsInfinity(number) || number < 0)
                    throw new Exception("拟合参数必须是有限非负值：" + item.Key);
                if (integers.Contains(item.Key) && (!(item.Value is int || item.Value is long) || number > Int32.MaxValue || number < (item.Key == "seed" ? 0 : 1)))
                    throw new Exception("拟合参数必须是有效整数：" + item.Key);
            }
            int hidden = Convert.ToInt32(values["hidden"]), heads = Convert.ToInt32(values["heads"]);
            if (hidden % 2 != 0 || hidden % heads != 0 || Convert.ToDouble(values["lr"]) <= 0 ||
                Convert.ToDouble(values["alpha"]) <= 0 || Convert.ToDouble(values["alpha"]) > 1 ||
                (Convert.ToDouble(values["x_identity_weight"]) > 0 && values["x_identity_mask"] == null))
                throw new Exception("拟合参数不满足 hidden/head、alpha/lr 或 identity mask 约束。");
            return values;
        }

        internal static List<string[]> ExplicitPlan(Dictionary<string, string> s, bool resume)
        {
            if (resume) throw new Exception("显式目标拟合不支持断点续训；请选择新的输出目录。");
            string config;
            if (!s.TryGetValue("ExplicitFitConfig", out config)) config = "";
            string configSha = String.IsNullOrWhiteSpace(config) ? "" : FileSha(config);
            var values = ExplicitConfig(config);
            if (configSha != "" && FileSha(config) != configSha) throw new Exception("读取拟合参数时文件发生变化，请重新检查。");
            string manifest = Path.GetFullPath(s["TargetManifest"]);
            string digest = FileSha(manifest);
            s["ExplicitManifestSha"] = digest;
            s["ExplicitConfigSha"] = configSha;
            s["ExplicitConfigSnapshot"] = Json().Serialize(values);
            var args = new List<string> { "fit_explicit_targets.py", "--manifest", manifest, "--manifest-sha", digest };
            foreach (var item in values)
                if (item.Value != null)
                    args.AddRange(new[] { "--" + item.Key.Replace('_', '-'), Convert.ToString(item.Value, CultureInfo.InvariantCulture) });
            return new List<string[]> {
                args.Concat(new[] { "--dry-run" }).ToArray(),
                args.Concat(new[] { "--output", Path.GetFullPath(s["RunDirectory"]), "--execute" }).ToArray()
            };
        }

        static bool Within(string path, string directory)
        {
            string p = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string d = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return p.Equals(d, StringComparison.OrdinalIgnoreCase) || p.StartsWith(d + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
        }

        void ValidateExplicit(Dictionary<string, string> s, bool resume)
        {
            if (resume) throw new Exception("显式目标拟合不支持断点续训。");
            foreach (string path in new[] { s["Python"], s["TargetManifest"], Path.Combine(root, "tools", "fit_explicit_targets.py") })
                if (!File.Exists(path)) throw new Exception("找不到文件：" + path);
            if (String.IsNullOrWhiteSpace(s["RunDirectory"]) || !Path.IsPathRooted(s["RunDirectory"]))
                throw new Exception("训练输出必须是新的完整目录路径。");
            string output = Path.GetFullPath(s["RunDirectory"]);
            if (Directory.Exists(output) || File.Exists(output)) throw new Exception("显式拟合输出路径已存在，即使空目录也不可复用；请选择新输出目录。");
            if (Within(output, Path.Combine(root, "models")) || Within(output, Path.Combine(root, "ComfyUI", "models")))
                throw new Exception("请使用独立 runs 目录，保留模型目录。");
            for (var parent = Directory.GetParent(output); parent != null; parent = parent.Parent)
                if (File.Exists(Path.Combine(parent.FullName, "run.json")) ||
                    (File.Exists(Path.Combine(parent.FullName, "PLAN.json")) && File.Exists(Path.Combine(parent.FullName, "RESULT.json"))))
                    throw new Exception("请使用独立输出目录，不要放在已有训练 run 内。");
            string comfy;
            if (s.TryGetValue("ComfyRoot", out comfy) && !String.IsNullOrWhiteSpace(comfy) && Within(output, Path.Combine(comfy, "models")))
                throw new Exception("输出不可位于 ComfyUI 模型目录。");
            var manifest = ReadMap(s["TargetManifest"]);
            if (!manifest.ContainsKey("kind") || Convert.ToString(manifest["kind"]) != "explicit_train_target_manifest_v1" ||
                !manifest.ContainsKey("evidence_class") || Convert.ToString(manifest["evidence_class"]) != "train_fit_probe")
                throw new Exception("图形训练需 train_fit_probe 显式目标清单；合成 fixture 仅用于独立 CPU 测试。");
            string config = s.ContainsKey("ExplicitFitConfig") ? s["ExplicitFitConfig"] : "";
            ExplicitConfig(config);
            foreach (string input in new[] { s["TargetManifest"], config }.Where(v => !String.IsNullOrWhiteSpace(v)))
                if (Within(input, output)) throw new Exception("输出目录不可包含输入文件。");
        }

        static void VerifyDigestMap(Dictionary<string, object> bindings)
        {
            foreach (var item in bindings)
                if (FileSha(item.Key) != Convert.ToString(item.Value)) throw new Exception("校验后输入或源码已变化：" + item.Key);
        }

        static void MatchExplicitConfig(Dictionary<string, object> expected, Dictionary<string, object> actual)
        {
            if (expected.Count != actual.Count || expected.Keys.Any(k => !actual.ContainsKey(k))) throw new Exception("拟合配置字段不同。");
            foreach (var item in expected)
            {
                object value = actual[item.Key];
                bool same = item.Value == null ? value == null : item.Key == "x_identity_mask" ? Equals(item.Value, value) :
                    value != null && !(value is bool) && Convert.ToDouble(item.Value, CultureInfo.InvariantCulture) == Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (!same) throw new Exception("实际拟合配置不同：" + item.Key);
            }
        }

        internal static void ValidateExplicitResult(string output, int steps)
        {
            var result = ReadMap(Path.Combine(output, "RESULT.json"));
            if (Convert.ToString(result["status"]) != "completed_train_fit_only" || Convert.ToInt32(result["steps"]) != steps ||
                !Equals(result["video_quality_evidence"], false) || !Equals(result["generalization_evidence"], false) || !Equals(result["val_test_loaded"], false))
                throw new Exception("拟合完成回执不符合预期。");
            var weight = ObjectMap(result["weights"]);
            if (!Path.GetFullPath(Convert.ToString(weight["path"])).Equals(Path.Combine(Path.GetFullPath(output), "bridge.safetensors"), StringComparison.OrdinalIgnoreCase) ||
                FileSha(Convert.ToString(weight["path"])) != Convert.ToString(weight["sha256"]))
                throw new Exception("最终权重路径或 SHA 不匹配。");
            var checks = result["reload_checks"] as object[];
            if (checks == null || checks.Length == 0) throw new Exception("缺少重载验证。");
            foreach (var check in checks)
            {
                double error = Convert.ToDouble(ObjectMap(check)["max_abs_error"], CultureInfo.InvariantCulture);
                if (Double.IsNaN(error) || Double.IsInfinity(error) || error < 0 || error > 1e-6) throw new Exception("重载验证失败。");
            }
        }

        void RunExplicit(Dictionary<string, string> s, List<string[]> plan, bool checkOnly, string logPath)
        {
            string script = Path.Combine(root, "tools", "fit_explicit_targets.py");
            string scriptSha = FileSha(script);
            var dry = ObjectMap(Json().DeserializeObject(RunExplicitCommand(s["Python"], new[] { "-u", script }.Concat(plan[0].Skip(1)).ToArray(), "CPU 校验显式目标")));
            if (Convert.ToString(dry["status"]) != "dry_run_validated_no_model_or_optimizer_created" || Convert.ToString(dry["device"]) != "cpu" ||
                Convert.ToString(dry["evidence_class"]) != "train_fit_probe" || Convert.ToString(ObjectMap(dry["manifest"])["sha256"]) != s["ExplicitManifestSha"] ||
                !Equals(dry["val_test_loaded"], false) || !Equals(dry["aggregate_npz_loaded"], false))
                throw new Exception("实际 dry-run 回执不符合显式 train-only CPU 模式。");
            var config = ObjectMap(Json().DeserializeObject(s["ExplicitConfigSnapshot"]));
            MatchExplicitConfig(config, ObjectMap(dry["config"]));
            var inputs = ObjectMap(dry["input_bindings"]);
            foreach (string path in inputs.Keys)
                if (Within(path, s["RunDirectory"])) throw new Exception("输出目录包含绑定输入：" + path);
            File.WriteAllText(logPath + ".dry-run.json", Json().Serialize(dry), new UTF8Encoding(false));
            Append("CPU 目标校验：" + dry["pair_count"] + " 对 / " + dry["work_count"] + " 作品；不是视频质量验收。");
            if (checkOnly) return;
            if (cancelled) throw new OperationCanceledException();
            ValidateExplicit(s, false);
            if (FileSha(script) != scriptSha || FileSha(s["TargetManifest"]) != s["ExplicitManifestSha"] ||
                (s["ExplicitConfigSha"] != "" && FileSha(s["ExplicitFitConfig"]) != s["ExplicitConfigSha"]))
                throw new Exception("校验后清单、配置或工具已变化，请重新检查。");
            VerifyDigestMap(inputs); VerifyDigestMap(ObjectMap(dry["code"]));
            RunExplicitCommand(s["Python"], new[] { "-u", script }.Concat(plan[1].Skip(1)).ToArray(),
                "CPU 拟合（" + config["steps"] + " steps）；逐步进度见输出 OPTIMIZATION.jsonl");
            if (cancelled) throw new OperationCanceledException();
            ValidateExplicitResult(s["RunDirectory"], Convert.ToInt32(config["steps"]));
        }

        string RunExplicitCommand(string python, string[] args, string label)
        {
            if (cancelled) throw new OperationCanceledException();
            Ui(delegate { status.Text = label; });
            Append(label); Append("命令：" + Quote(python) + " " + String.Join(" ", args.Select(Quote)));
            var text = new StringBuilder();
            bool tooLong = false;
            using (var p = new Process())
            {
                p.StartInfo = ProcessInfo(python, args, root);
                p.StartInfo.EnvironmentVariables["CUDA_VISIBLE_DEVICES"] = "";
                p.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (e.Data == null) return;
                    lock (text) { if (text.Length + e.Data.Length < 32 * 1024 * 1024) text.AppendLine(e.Data); else tooLong = true; }
                    Append(e.Data);
                };
                p.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) Append(e.Data); };
                lock (processLock) { if (cancelled) throw new OperationCanceledException(); p.Start(); activeProcess = p; }
                try
                {
                    p.BeginOutputReadLine(); p.BeginErrorReadLine();
                    var elapsed = Stopwatch.StartNew();
                    while (!p.WaitForExit(500))
                    {
                        string progress = label + " · 已运行 " + elapsed.Elapsed.ToString(@"hh\:mm\:ss");
                        Ui(delegate { if (!cancelled) status.Text = progress; });
                    }
                    p.WaitForExit(); // Drain async stdout/stderr before accepting JSON.
                    if (cancelled) throw new OperationCanceledException();
                    if (p.ExitCode != 0) throw new Exception(label + " 失败，退出码 " + p.ExitCode);
                    if (tooLong) throw new Exception("工具 JSON 输出超过 32 MiB，未接受为成功回执。");
                    return text.ToString();
                }
                finally { lock (processLock) { activeProcess = null; } }
            }
        }

        internal static Dictionary<string, object> CheckGuiObjective(string configPath, string mode)
        {
            bool aligned = mode == "aligned_token_identity_v1";
            bool transport = mode == "monotone_token_identity_v1";
            bool experimental = aligned || transport;
            if (!File.Exists(configPath))
            {
                if (experimental) throw new Exception("逐 token 实验需要有效的实验参数 JSON。");
                return null; // Validate reports missing paths.
            }
            var config = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(File.ReadAllText(configPath, Encoding.UTF8));
            object objective;
            string configured = config != null && config.TryGetValue("semantic_objective", out objective) ? Convert.ToString(objective) : "";
            string expected = aligned ? "aligned_token_identity_v1" : transport ? "monotone_token_identity_v1" : "";
            if ((experimental && configured != expected) || (!experimental && (configured == "aligned_token_identity_v1" || configured == "monotone_token_identity_v1")))
                throw new Exception("逐 token 实验参数必须搭配对应的等长或变长实验训练方式；当前模式与 semantic_objective 不一致。未开始编码或训练。");
            if (experimental)
            {
                foreach (string key in new[] { "alpha_min", "alpha_max", "magnitude_mode", "residual_skip", "anchor_weight", "max_seq_tokens" })
                    if (!config.ContainsKey(key)) throw new Exception("实验参数缺少：" + key);
                double alpha = Convert.ToDouble(config["alpha_min"], CultureInfo.InvariantCulture);
                if (!(alpha > 0 && alpha <= 1) || alpha != Convert.ToDouble(config["alpha_max"], CultureInfo.InvariantCulture) ||
                    Convert.ToString(config["magnitude_mode"]) != "per_token" || !(config["residual_skip"] is bool) || !(bool)config["residual_skip"] ||
                    Convert.ToDouble(config["anchor_weight"], CultureInfo.InvariantCulture) != 0 || !(config["max_seq_tokens"] is int) || (int)config["max_seq_tokens"] < 1)
                    throw new Exception("实验参数要求固定 alpha、per_token、residual_skip=true、anchor_weight=0 和正整数 max_seq_tokens。");
            }
            return config;
        }

        internal static List<string[]> Plan(Dictionary<string, string> s, bool resume)
        {
            if (s["TrainingMode"] == ExplicitMode) return ExplicitPlan(s, resume);
            bool aligned = s.ContainsKey("TrainingMode") && s["TrainingMode"] == "aligned_token_identity_v1";
            bool transport = s.ContainsKey("TrainingMode") && s["TrainingMode"] == "monotone_token_identity_v1";
            bool experimental = aligned || transport;
            var config = CheckGuiObjective(s["Config"], s["TrainingMode"]);
            bool preference = experimental || (s.ContainsKey("TrainingMode") && s["TrainingMode"] == "preference_only");
            string selectionPolicy;
            if (!s.TryGetValue("SelectionPolicy", out selectionPolicy)) selectionPolicy = "one_per_group";
            if (experimental && selectionPolicy != "one_per_group" && selectionPolicy != "all_rows")
                throw new Exception("实验准备选择策略无效。");
            string evidenceRoot;
            bool videoFeedback = s.TryGetValue("VideoFeedbackRoot", out evidenceRoot) && !String.IsNullOrWhiteSpace(evidenceRoot);
            var result = new List<string[]>();
            var inspect = new List<string>(preference ? new[] { "inspect", s["Pairs"], "--require-supervision", "preference_only", "--split-manifest", s["SplitManifest"] } : new[] { "inspect", s["Pairs"] });
            if (videoFeedback) inspect.AddRange(new[] { "--video-feedback-root", evidenceRoot });
            result.Add(inspect.ToArray());
            if (File.Exists(s["Dataset"]))
                result.Add(new[] { "check-dataset", s["Dataset"], "--pairs", s["Pairs"], "--encoder", s["Encoder"] });
            else
            {
                var encode = new List<string> { "encode", s["Pairs"], "--comfy-root", s["ComfyRoot"], "--encoder", s["Encoder"], "--output", s["Dataset"] };
                if (videoFeedback) encode.AddRange(new[] { "--video-feedback-root", evidenceRoot });
                result.Add(encode.ToArray());
            }
            string dataset = s["Dataset"], split = s["SplitManifest"];
            if (experimental)
            {
                string prepared = s["PreparedDirectory"];
                if (!Directory.Exists(prepared))
                {
                    if (resume) throw new Exception("续训必须复用原实验对齐数据目录，不可重新准备。");
                    var prepare = new List<string> { "prepare-aligned", dataset, "--pairs", s["Pairs"], "--comfy-root", s["ComfyRoot"], "--encoder", s["Encoder"], "--split-manifest", split, "--output-dir", prepared, "--max-tokens", Convert.ToString(config["max_seq_tokens"], CultureInfo.InvariantCulture) };
                    if (transport) prepare.AddRange(new[] { "--method", "token_id_monotone_transport_v1" });
                    if (selectionPolicy == "all_rows") prepare.AddRange(new[] { "--selection-policy", "all_rows" });
                    if (videoFeedback) prepare.AddRange(new[] { "--video-feedback-root", evidenceRoot });
                    result.Add(prepare.ToArray());
                }
                dataset = Path.Combine(prepared, "pairs.npz"); split = Path.Combine(prepared, "split_manifest.json");
                result.Add(new[] { "check-dataset", dataset, "--pairs", Path.Combine(prepared, "pairs.jsonl"), "--encoder", s["Encoder"] });
            }
            var train = new List<string> { "train", dataset, "--output", s["RunDirectory"], "--config", s["Config"] };
            var epochs = EpochOptions(s, resume);
            train.AddRange(new[] { "--epochs", epochs[0].ToString(CultureInfo.InvariantCulture), "--judge-epochs", epochs[1].ToString(CultureInfo.InvariantCulture) });
            if (preference) train.AddRange(new[] { "--stage", "bridge", "--split-manifest", split });
            if (videoFeedback) train.AddRange(new[] { "--video-feedback-root", evidenceRoot });
            if (resume) train.Add("--resume");
            result.Add(train.ToArray());
            if (preference)
            {
                result.Add(new[] { "verify", "--bridge", Path.Combine(s["RunDirectory"], "bridge.safetensors"), "--bridge-only", "--report", Path.Combine(s["RunDirectory"], "weights_verification.json") });
                if (experimental)
                    result.Add(new[] { "evaluate", dataset, "--run", s["RunDirectory"], "--split", "validation", "--alpha", Convert.ToString(config["alpha_min"], CultureInfo.InvariantCulture), "--magnitude-mode", "per_token", "--output", Path.Combine(s["RunDirectory"], transport ? "validation_monotone_fixed.json" : "validation_aligned_fixed.json") });
                else
                    foreach (string mode in new[] { "none", "per_token" })
                        result.Add(new[] { "evaluate", dataset, "--run", s["RunDirectory"], "--split", "validation", "--alpha", "0", "0.05", "0.08", "0.12", "0.2", "--magnitude-mode", mode, "--output", Path.Combine(s["RunDirectory"], "validation_" + mode + ".json") });
            }
            else
            {
                result.Add(new[] { "verify", "--bridge", Path.Combine(s["RunDirectory"], "bridge.safetensors"), "--judge", Path.Combine(s["RunDirectory"], "judge.safetensors"), "--report", Path.Combine(s["RunDirectory"], "weights_verification.json") });
                result.Add(new[] { "evaluate", s["Dataset"], "--run", s["RunDirectory"] });
            }
            return result;
        }

        void Validate(Dictionary<string, string> s, bool resume, bool checkOnly)
        {
            if (s["TrainingMode"] == ExplicitMode) { ValidateExplicit(s, resume); return; }
            if (!File.Exists(Path.Combine(root, "tools", "trainer.py")))
                throw new Exception("请将 EXE 放在训练器根目录，与 tools 文件夹放在一起。");
            foreach (string key in (checkOnly ? new[] { "Python", "Pairs", "Config" } : new[] { "Python", "Encoder", "Pairs", "Config" }))
                if (!File.Exists(s[key])) throw new Exception("找不到文件（" + key + "）：" + s[key]);
            bool aligned = s["TrainingMode"] == "aligned_token_identity_v1" || s["TrainingMode"] == "monotone_token_identity_v1";
            CheckGuiObjective(s["Config"], s["TrainingMode"]);
            if (!Directory.Exists(Path.Combine(s["ComfyRoot"], "comfy")))
                throw new Exception("ComfyUI 目录应包含 comfy 文件夹。");
            foreach (string key in new[] { "Dataset", "RunDirectory" })
                if (String.IsNullOrWhiteSpace(s[key]) || !Path.IsPathRooted(s[key]))
                    throw new Exception(key + " 必须是完整路径。");
            if (!s["Dataset"].EndsWith(".npz", StringComparison.OrdinalIgnoreCase))
                throw new Exception("编码数据集路径必须以 .npz 结尾。");
            if (s["TrainingMode"] != "numeric_scores" && !File.Exists(s["SplitManifest"]))
                throw new Exception("无评分偏好模式必须选择覆盖全部 group_id 的分组划分 JSON。");
            string evidenceRoot;
            if (s.TryGetValue("VideoFeedbackRoot", out evidenceRoot) && !String.IsNullOrWhiteSpace(evidenceRoot) && !Directory.Exists(evidenceRoot))
                throw new Exception("视频反馈证据目录不存在：" + evidenceRoot);
            if (aligned)
            {
                string prepared = s["PreparedDirectory"];
                if (String.IsNullOrWhiteSpace(prepared) || !Path.IsPathRooted(prepared))
                    throw new Exception("实验对齐数据目录必须是完整路径，与原始输入和训练输出分开保存。");
                string p = Path.GetFullPath(prepared).TrimEnd(Path.DirectorySeparatorChar);
                string run = Path.GetFullPath(s["RunDirectory"]).TrimEnd(Path.DirectorySeparatorChar);
                string modelRoot = Path.GetFullPath(Path.Combine(root, "models"));
                if (p.Equals(run, StringComparison.OrdinalIgnoreCase) || p.StartsWith(run + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) || run.StartsWith(p + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ||
                    p.Equals(modelRoot, StringComparison.OrdinalIgnoreCase) || p.StartsWith(modelRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                    throw new Exception("对齐数据目录须与训练输出目录互不包含，并保留作者 models 目录。");
                foreach (string source in new[] { s["Dataset"], Path.ChangeExtension(s["Dataset"], ".json"), s["Dataset"] + ".receipt.json", s["Pairs"], s["Encoder"], s["SplitManifest"] })
                {
                    string sourcePath = Path.GetFullPath(source).TrimEnd(Path.DirectorySeparatorChar);
                    if (sourcePath.Equals(p, StringComparison.OrdinalIgnoreCase) || sourcePath.StartsWith(p + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                        throw new Exception("源数据、元数据、回执、编码器与分组划分不得位于实验对齐数据目录内，也不能与该目录同名：" + source);
                }
                if (File.Exists(p) || (Directory.Exists(p) && !File.Exists(Path.Combine(p, "preparation_receipt.json"))))
                    throw new Exception("对齐数据目录已存在但没有准备回执。请选择新的目录；已有文件不会覆盖。");
                if (resume && !Directory.Exists(p)) throw new Exception("续训必须选择原实验对齐数据目录；不重新准备数据。");
                if (Directory.Exists(p))
                    foreach (string source in new[] { s["Dataset"], Path.ChangeExtension(s["Dataset"], ".json"), s["Dataset"] + ".receipt.json" })
                        if (!File.Exists(source)) throw new Exception("复用实验准备数据需要保留原始编码数据、元数据与回执：" + source);
            }
            if (checkOnly) return;
            var output = Path.GetFullPath(s["RunDirectory"]).TrimEnd(Path.DirectorySeparatorChar);
            var models = Path.GetFullPath(Path.Combine(root, "models"));
            if (output.Equals(models, StringComparison.OrdinalIgnoreCase) || output.StartsWith(models + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new Exception("请使用独立的 runs 输出目录，保留作者模型。");
            if (resume && !File.Exists(Path.Combine(output, "run.json")))
                throw new Exception("该目录没有 run.json。请选中原训练目录，或使用“开始训练”。");
            if (!resume && Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
                throw new Exception("训练目录已有文件。请使用“新输出目录”开始新训练，或选择“断点续训”。");
        }

        async Task Start(bool resume, bool checkOnly)
        {
            if (busy) return;
            var settings = Snapshot();
            try
            {
                Validate(settings, resume, checkOnly);
                if (settings["TrainingMode"] != ExplicitMode) {
                    var selectedEpochs = EpochOptions(settings, resume);
                    bridgeEpochs.Value = selectedEpochs[0]; judgeEpochs.Value = selectedEpochs[1];
                    settings["BridgeEpochs"] = selectedEpochs[0].ToString(CultureInfo.InvariantCulture);
                    settings["JudgeEpochs"] = selectedEpochs[1].ToString(CultureInfo.InvariantCulture);
                    if (resume) Append("续训沿用原任务轮次：语义桥 " + selectedEpochs[0] + "，JEV " + selectedEpochs[1]);
                }
                SaveSettings(settings);
                string logDir = Path.Combine(root, "local", "launcher_logs"); Directory.CreateDirectory(logDir);
                string logPath = Path.Combine(logDir, DateTime.Now.ToString("yyyyMMdd_HHmmss_fff") + ".log");
                logFile = new StreamWriter(logPath, false, new UTF8Encoding(false)) { AutoFlush = true };
                cancelled = false; busy = true; activeRun = settings["RunDirectory"]; activeMode = settings["TrainingMode"];
                updateButton.Enabled = false;
                foreach (var control in editable) control.Enabled = false;
                stop.Enabled = true;
                Append("\n日志：" + logPath);
                var plan = Plan(settings, resume);
                File.WriteAllText(logPath + ".plan.json", new JavaScriptSerializer().Serialize(new { training_mode = settings["TrainingMode"], check_only = checkOnly, python = settings["Python"], commands = plan }), new UTF8Encoding(false));
                await Task.Run(delegate
                {
                    if (settings["TrainingMode"] == ExplicitMode) { RunExplicit(settings, plan, checkOnly, logPath); return; }
                    Run(settings["Python"], new[] { "-u", "-c", EnvironmentProbe }, "检查 Python / PyTorch / GPU");
                    Run(settings["Python"], new[] { "-u", "-c", "import sys; from wushu_bridge.source_revision import revision; print('Trainer source: '+revision(sys.argv[1]))", root }, "检查训练器版本记录");
                    Run(settings["Python"], new[] { "-u", "-c", "import json,sys; from wushu_bridge.trainer_engine import RunConfig; c=RunConfig(**json.load(open(sys.argv[1],encoding='utf-8'))); c.validate(); print(c)", settings["Config"] }, "检查训练参数");
                    if ((settings["TrainingMode"] == "aligned_token_identity_v1" || settings["TrainingMode"] == "monotone_token_identity_v1") && Directory.Exists(settings["PreparedDirectory"]))
                        Run(settings["Python"], new[] { "-u", "-c", PreparedProbe, settings["PreparedDirectory"], settings["Dataset"], settings["Pairs"], settings["Encoder"], settings["SplitManifest"], settings["ComfyRoot"], settings["TrainingMode"] == "monotone_token_identity_v1" ? "token_id_monotone_transport_v1" : "equal_length_edit_blocks_v1", !String.IsNullOrWhiteSpace(settings["VideoFeedbackRoot"]) ? "all_unique_scene_families" : settings["SelectionPolicy"] }, "CPU 检查实验准备回执与原始输入（复用，不覆盖）");
                    if (checkOnly)
                    {
                        Run(settings["Python"], new[] { "-u", Path.Combine(root, "tools", "trainer.py") }.Concat(plan[0]).ToArray(), "检查训练对与分组划分");
                        Run(settings["Python"], new[] { "-u", "-c", "import sys; sys.path.insert(0,sys.argv[1]); sys.argv=[sys.argv[0],'--cpu']; import comfy.options; comfy.options.enable_args_parsing(); import comfy.sd; print('H3 encoder modules: OK')", settings["ComfyRoot"] }, "检查 H3 编码运行环境");
                    }
                    else
                    {
                        foreach (var command in plan)
                            Run(settings["Python"], new[] { "-u", Path.Combine(root, "tools", "trainer.py") }.Concat(command).ToArray(), settings["TrainingMode"] == "monotone_token_identity_v1" ? "变长 token 实验 · " + (command[0] == "evaluate" ? "固定强度 validation 辅助表征诊断" : command[0]) : settings["TrainingMode"] == "aligned_token_identity_v1" ? "等长 token 实验 · " + (command[0] == "evaluate" ? "固定强度 validation 辅助表征诊断" : command[0]) : settings["TrainingMode"] == "preference_only" ? "偏好模式 · " + command[0] : StageName(command[0]));
                    }
                });
                if (cancelled) throw new OperationCanceledException();
                status.Text = settings["TrainingMode"] == ExplicitMode ? (checkOnly ? "显式目标 CPU 校验通过 · 未创建模型或输出" : "CPU 拟合与重载检查完成 · 未验证视频质量或泛化") : checkOnly ? "环境检查通过 · 编码器与对齐资格在数据阶段验证" : "任务完成 · 权重与辅助表征诊断已保存；未验证视频质量";
                Append(status.Text);
                if (!checkOnly)
                {
                    string summary = InferenceSummary(settings["RunDirectory"]);
                    Append(summary);
                    Append("完整设置见输出目录的 模型使用说明.md 与 inference_settings.json；分发模型时请一并提供。");
                    status.Text = "训练完成 · " + summary;
                }
            }
            catch (OperationCanceledException)
            {
                status.Text = activeMode == ExplicitMode ? "CPU 拟合已停止 · 日志保留；无断点续训，重跑需新目录" : "任务已停止 · 可复用编码缓存；训练从上一完整 epoch 续训"; Append(status.Text);
            }
            catch (Exception ex)
            {
                status.Text = "任务未完成 · 请查看错误与日志"; Append("错误：" + ex.Message);
                MessageBox.Show(this, ex.Message, "任务未完成", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally
            {
                busy = false; stop.Enabled = false;
                updateButton.Enabled = true;
                foreach (var control in editable) control.Enabled = true;
                UpdateTrainingMode(false);
                lock (processLock) { if (logFile != null) { logFile.Dispose(); logFile = null; } }
            }
        }

        internal static string InferenceSummary(string directory)
        {
            string path = Path.Combine(directory, "inference_settings.json");
            if (!File.Exists(path)) return "未找到推理说明，请先核对模型契约，不要猜 alpha";
            var report = ReadMap(path);
            var settings = ObjectMap(report["settings"]);
            bool fixedAlpha = Convert.ToString(report["status"]) == "fixed_contract";
            return (fixedAlpha ? "固定 alpha=" : "试用起点 alpha=")
                + Convert.ToDouble(settings["alpha"]).ToString("G", CultureInfo.InvariantCulture)
                + (fixedAlpha ? "（按权重契约）" : "（未视频校准）")
                + " · magnitude=" + Convert.ToString(settings["magnitude_match"]);
        }

        static string StageName(string stage)
        {
            switch (stage)
            {
                case "inspect": return "1/5 检查训练对";
                case "encode": return "2/5 H3 文本编码（缓存可恢复）";
                case "check-dataset": return "2/5 校验已有数据集身份与哈希";
                case "train": return "3/5 训练语义桥与评分头";
                case "verify": return "4/5 验证导出权重（软件检查）";
                default: return "5/5 独立测试集评估";
            }
        }

        void Run(string python, string[] args, string title)
        {
            if (cancelled) throw new OperationCanceledException();
            Ui(delegate { status.Text = title; }); Append("\n── " + title + " ──");
            Append("命令：" + Quote(python) + " " + String.Join(" ", args.Select(Quote)));
            using (var p = new Process())
            {
                p.StartInfo = ProcessInfo(python, args, root);
                p.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) Append(e.Data); };
                p.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) Append(e.Data); };
                lock (processLock)
                {
                    if (cancelled) throw new OperationCanceledException();
                    p.Start(); activeProcess = p;
                }
                try
                {
                    p.BeginOutputReadLine(); p.BeginErrorReadLine(); p.WaitForExit();
                    if (cancelled) throw new OperationCanceledException();
                    if (p.ExitCode != 0) throw new Exception(title + "失败，退出码 " + p.ExitCode + "。后续阶段未执行，请查看日志。");
                }
                finally { lock (processLock) { activeProcess = null; } }
            }
        }

        internal static ProcessStartInfo ProcessInfo(string program, string[] args, string directory)
        {
            var info = new ProcessStartInfo(program, String.Join(" ", args.Select(Quote)))
            {
                WorkingDirectory = directory, UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
            };
            info.EnvironmentVariables["PYTHONUTF8"] = "1";
            info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            info.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
            return info;
        }

        void RequestStop()
        {
            if (!busy || cancelled) return;
            if (MessageBox.Show(this, activeMode == ExplicitMode ? "停止当前 CPU 拟合？\n日志保留；本模式无断点续训，重跑需要新的输出目录。" : "停止当前编码或训练进程？\n已完成的缓存和 epoch 断点会保留；当前 epoch 下次需要重做。", "停止任务", MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes) return;
            Cancel();
        }

        void Cancel()
        {
            cancelled = true; stop.Enabled = false;
            lock (processLock)
            {
                if (activeProcess != null)
                {
                    try { if (!activeProcess.HasExited) activeProcess.Kill(); }
                    catch (InvalidOperationException) { }
                    catch (Exception ex) { Append("停止进程失败：" + ex.Message); }
                }
            }
            status.Text = "正在停止…";
        }

        void OnClosing(object sender, FormClosingEventArgs e)
        {
            if (installingUpdate) { e.Cancel = true; return; }
            if (busy)
            {
                e.Cancel = true;
                RequestStop();
                return;
            }
            try { SaveSettings(Snapshot()); }
            catch (Exception ex) { MessageBox.Show(this, "设置未保存：" + ex.Message); }
        }

        void Ui(Action action)
        {
            if (IsDisposed || Disposing) return;
            if (InvokeRequired) { try { BeginInvoke(action); } catch (InvalidOperationException) { } }
            else action();
        }

        void Append(string line)
        {
            lock (processLock) { if (logFile != null) logFile.WriteLine(DateTime.Now.ToString("HH:mm:ss ") + line); }
            Ui(delegate
            {
                if (log.TextLength > 180000) { log.Select(0, 60000); log.SelectedText = ""; }
                log.AppendText(line + Environment.NewLine); log.SelectionStart = log.TextLength; log.ScrollToCaret();
            });
        }

        void OpenFolder(string path)
        {
            try { if (!Directory.Exists(path)) { MessageBox.Show(this, "目录尚未创建，训练启动后会自动创建。\n" + path); return; } Process.Start("explorer.exe", Quote(path)); }
            catch (Exception ex) { MessageBox.Show(this, ex.Message); }
        }
        void OpenFile(string path)
        {
            try { Process.Start("notepad.exe", Quote(path)); }
            catch (Exception ex) { MessageBox.Show(this, ex.Message); }
        }

        internal static int SelfTest(string python, string output)
        {
            Directory.CreateDirectory(output);
            var results = new List<string>();
            var values = new[] { "", "中文 路径", "E:\\folder with space\\", "quote\"inside", "a&b;$(literal)", "backslash\\\"quote" };
            var args = new[] { "-c", "import json,sys; print(json.dumps(sys.argv[1:],ensure_ascii=False)); print('stderr 中文',file=sys.stderr)" }.Concat(values).ToArray();
            using (var p = Process.Start(ProcessInfo(python, args, AppDomain.CurrentDomain.BaseDirectory)))
            {
                var stdout = p.StandardOutput.ReadToEnd(); var stderr = p.StandardError.ReadToEnd(); p.WaitForExit();
                var actual = new JavaScriptSerializer().Deserialize<string[]>(stdout);
                if (p.ExitCode != 0 || !values.SequenceEqual(actual) || !stderr.Contains("stderr 中文")) throw new Exception("Argument/UTF8 process test failed");
                results.Add("Windows argv quoting, literal metacharacters, UTF8 stdout/stderr: passed");
            }
            using (var form = new Launcher(AppDomain.CurrentDomain.BaseDirectory, false))
            {
                // Realize child controls without exposing a test window.
                form.ShowInTaskbar = false;
                form.Opacity = 0;
                form.Show();
                Application.DoEvents();
                using (var preview = new Bitmap(form.ClientSize.Width, form.ClientSize.Height)) {
                    form.DrawToBitmap(preview, new Rectangle(Point.Empty, form.ClientSize));
                    preview.Save(Path.Combine(output, "launcher-preview.png"));
                }
                var settings = form.Snapshot();
                if (settings["TrainingMode"] != "preference_only" || settings["Pairs"] != form.PreferencePairs() || settings["SplitManifest"] != form.PreferenceSplit()
                    || !File.Exists(settings["Pairs"]) || !File.Exists(settings["SplitManifest"]) || !File.Exists(settings["Config"]))
                    throw new Exception("Portable first-run preference defaults test failed");
                if (settings["Python"] != Path.Combine(form.root, "python", "python.exe") || settings["ComfyRoot"] != Path.Combine(form.root, "ComfyUI")
                    || !settings["Dataset"].StartsWith(Path.Combine(form.root, "local")) || !settings["RunDirectory"].StartsWith(Path.Combine(form.root, "runs"))
                    || !form.pathHints["Pairs"].Text.Contains("格式示例") || !form.Text.Contains("By T8star"))
                    throw new Exception("Portable paths, example notice or branding failed");
                form.trainingMode.SelectedIndex = 0;
                if (form.paths["Pairs"].Text != form.AuthorPairs() || !form.paths["Config"].Text.EndsWith("author_v1.json") || form.paths["SplitManifest"].Text != "")
                    throw new Exception("Bundled author data mode defaults failed");
                form.trainingMode.SelectedIndex = 1;
                if (form.paths["Pairs"].Text != form.PreferencePairs() || form.paths["SplitManifest"].Text != form.PreferenceSplit())
                    throw new Exception("Preference sample restoration failed");
                results.Add("First-run relative bundle paths, paired sample defaults and notices, author-676/preference mode switching, By T8star branding: passed; software checks only");
                if (form.alignedFields.Any(c => c.Enabled)) throw new Exception("Experimental fields must be disabled in ordinary modes");
                settings["TrainingMode"] = "numeric_scores";
                settings["Dataset"] = Path.Combine(output, "test dataset.npz");
                if (File.Exists(settings["Dataset"])) File.Delete(settings["Dataset"]);
                var plan = Plan(settings, false);
                if (plan[1][0] != "encode" || plan[2].Contains("--resume") || plan.Count != 5) throw new Exception("Fresh plan test failed");
                File.WriteAllText(settings["Dataset"], "plan-test-only");
                plan = Plan(settings, true);
                if (plan[1][0] != "check-dataset" || !plan[2].Contains("--resume")) throw new Exception("Resume plan test failed");
                File.Delete(settings["Dataset"]);
                results.Add("Fresh encoding / cached dataset identity check / resume command routing: passed");
                settings["TrainingMode"] = "preference_only";
                settings["SplitManifest"] = Path.Combine(output, "split manifest.json");
                plan = Plan(settings, true);
                if (plan.Count != 6 || !plan[0].Contains("--require-supervision") || !plan[0].Contains(settings["SplitManifest"]) || !plan[2].Contains("--stage") || !plan[2].Contains("bridge") || !plan[2].Contains("--resume") || !plan[2].Contains(settings["SplitManifest"]) || !plan[3].Contains("--bridge-only") || plan[3].Contains("--judge") || plan.Skip(4).Any(c => !c.Contains("validation") || c.Contains("test"))) throw new Exception("Preference-only routing test failed");
                File.WriteAllText(Path.Combine(output, "preference-plan.synthetic.json"), new JavaScriptSerializer().Serialize(plan), new UTF8Encoding(false));
                results.Add("Preference-only: split preflight, bridge stage/resume, no judge, two validation diagnostics: passed; no training started");
                string savedConfig = settings["Config"];
                settings["Config"] = Path.Combine(output, "aligned-token.synthetic.json");
                const string alignedConfig = "{\"semantic_objective\":\"aligned_token_identity_v1\",\"alpha_min\":0.2,\"alpha_max\":0.2,\"magnitude_mode\":\"per_token\",\"residual_skip\":true,\"anchor_weight\":0,\"max_seq_tokens\":2048}";
                File.WriteAllText(settings["Config"], alignedConfig);
                bool alignedRejected = false;
                try { Plan(settings, false); }
                catch (Exception ex) { alignedRejected = ex.Message.Contains("未开始"); }
                if (!alignedRejected) throw new Exception("Aligned config must require explicit experimental GUI mode");
                settings["TrainingMode"] = "aligned_token_identity_v1";
                settings["PreparedDirectory"] = Path.Combine(output, "prepared synthetic " + Guid.NewGuid().ToString("N"));
                plan = Plan(settings, false);
                string preparedDataset = Path.Combine(settings["PreparedDirectory"], "pairs.npz");
                string preparedSplit = Path.Combine(settings["PreparedDirectory"], "split_manifest.json");
                var alignedTrain = plan.Single(c => c[0] == "train");
                var alignedEval = plan.Single(c => c[0] == "evaluate");
                if (plan.Count != 7 || plan[1][0] != "encode" || plan[2][0] != "prepare-aligned" || !plan[2].Contains("2048") || plan[3][0] != "check-dataset" ||
                    alignedTrain[1] != preparedDataset || !alignedTrain.Contains(preparedSplit) || !alignedTrain.Contains("bridge") || alignedTrain.Contains("--resume") ||
                    alignedEval[1] != preparedDataset || !alignedEval.Contains("validation") || !alignedEval.Contains("0.2") || !alignedEval.Contains("per_token") || alignedEval.Contains("none") || alignedEval.Contains("test") ||
                    !plan.Single(c => c[0] == "verify").Contains("--bridge-only")) throw new Exception("Aligned fresh preparation/train/fixed validation routing failed");
                File.WriteAllText(Path.Combine(output, "aligned-plan.synthetic.json"), new JavaScriptSerializer().Serialize(plan), new UTF8Encoding(false));
                bool missingPreparedRejected = false;
                try { Plan(settings, true); } catch (Exception) { missingPreparedRejected = true; }
                if (!missingPreparedRejected) throw new Exception("Resume must never regenerate prepared inputs");
                Directory.CreateDirectory(settings["PreparedDirectory"]);
                File.WriteAllText(settings["Dataset"], "plan-test-only");
                plan = Plan(settings, true);
                if (plan.Count != 6 || plan[1][0] != "check-dataset" || plan.Any(c => c[0] == "prepare-aligned" || c[0] == "encode") || !plan.Single(c => c[0] == "train").Contains("--resume"))
                    throw new Exception("Aligned resume must reuse prepared dataset and source encoding");
                File.Delete(settings["Dataset"]);
                Directory.Delete(settings["PreparedDirectory"]);
                File.WriteAllText(settings["Config"], alignedConfig.Replace("\"alpha_max\":0.2", "\"alpha_max\":0.3"));
                bool variableAlphaRejected = false;
                try { Plan(settings, false); } catch (Exception) { variableAlphaRejected = true; }
                if (!variableAlphaRejected) throw new Exception("Aligned variable alpha must fail before work");
                File.WriteAllText(settings["Config"], alignedConfig);
                SelfTestPreparedSourceIsolation(output, settings["Config"]);
                results.Add("Actual Validate rejects prepared/source overlap before encoding: absent NPZ, case-insensitive paths, normalized dot segments, equal NPZ/metadata/receipt paths and existing source files; sibling prefix accepted (synthetic paths only)");
                File.WriteAllText(Path.Combine(output, "prepared-probe.synthetic.py"), PreparedProbe, new UTF8Encoding(false));
                results.Add("Experimental aligned routing: source preflight/encoding, independent preparation, derived split, bridge-only, fixed-alpha validation, unchanged-input resume and invalid-mode/alpha rejection: passed; synthetic commands only");
                settings["TrainingMode"] = "monotone_token_identity_v1";
                settings["Config"] = Path.Combine(output, "monotone-token.synthetic.json");
                const string monotoneConfig = "{\"semantic_objective\":\"monotone_token_identity_v1\",\"alpha_min\":0.2,\"alpha_max\":0.2,\"magnitude_mode\":\"per_token\",\"residual_skip\":true,\"anchor_weight\":0,\"max_seq_tokens\":2048}";
                File.WriteAllText(settings["Config"], monotoneConfig);
                settings["PreparedDirectory"] = Path.Combine(output, "transport synthetic " + Guid.NewGuid().ToString("N"));
                plan = Plan(settings, false);
                var transportPrepare = plan.Single(c => c[0] == "prepare-aligned");
                var transportTrain = plan.Single(c => c[0] == "train");
                var transportEval = plan.Single(c => c[0] == "evaluate");
                if (plan.Count != 7 || !transportPrepare.Contains("--method") || !transportPrepare.Contains("token_id_monotone_transport_v1") ||
                    !transportTrain.Contains("bridge") || transportTrain.Contains("judge") ||
                    !transportEval.Contains("validation") || !transportEval.Last().EndsWith("validation_monotone_fixed.json", StringComparison.Ordinal) ||
                    !plan.Single(c => c[0] == "verify").Contains("--bridge-only"))
                    throw new Exception("Monotone fresh preparation/train/fixed validation routing failed");
                if (transportPrepare.Contains("--selection-policy")) throw new Exception("Default preparation must select one row per group");
                settings["SelectionPolicy"] = "all_rows";
                var allRowsPrepare = Plan(settings, false).Single(c => c[0] == "prepare-aligned");
                if (Array.IndexOf(allRowsPrepare, "--selection-policy") < 0 ||
                    allRowsPrepare[Array.IndexOf(allRowsPrepare, "--selection-policy") + 1] != "all_rows")
                    throw new Exception("Opt-in all-rows preparation routing failed");
                settings["SelectionPolicy"] = "one_per_group";
                results.Add("Experimental all-rows option routes explicitly to preparation; default remains one row per group: passed; synthetic commands only");
                settings["VideoFeedbackRoot"] = Path.Combine(output, "private full AV evidence");
                var feedbackPlan = Plan(settings, false);
                foreach (string stage in new[] { "inspect", "encode", "prepare-aligned", "train" })
                {
                    var command = feedbackPlan.Single(c => c[0] == stage);
                    int option = Array.IndexOf(command, "--video-feedback-root");
                    if (option < 0 || option + 1 >= command.Length || command[option + 1] != settings["VideoFeedbackRoot"])
                        throw new Exception("Video feedback evidence root missing from " + stage);
                }
                if (feedbackPlan.Where(c => c[0] == "verify" || c[0] == "evaluate" || c[0] == "check-dataset").Any(c => c.Contains("--video-feedback-root")))
                    throw new Exception("Video feedback root passed to an unsupported command");
                settings["VideoFeedbackRoot"] = "";
                results.Add("Private full-AV evidence root routes through inspect, encode, preparation and train only: passed; synthetic command test");
                File.WriteAllText(Path.Combine(output, "monotone-plan.synthetic.json"), new JavaScriptSerializer().Serialize(plan), new UTF8Encoding(false));
                bool crossedConfigRejected = false;
                settings["Config"] = Path.Combine(output, "aligned-token.synthetic.json");
                try { Plan(settings, false); } catch (Exception ex) { crossedConfigRejected = ex.Message.Contains("未开始"); }
                if (!crossedConfigRejected) throw new Exception("Equal-length config must not run in monotone mode");
                settings["Config"] = Path.Combine(output, "monotone-token.synthetic.json");
                settings["TrainingMode"] = "aligned_token_identity_v1";
                crossedConfigRejected = false;
                try { Plan(settings, false); } catch (Exception ex) { crossedConfigRejected = ex.Message.Contains("未开始"); }
                if (!crossedConfigRejected) throw new Exception("Monotone config must not run in equal-length mode");
                results.Add("Experimental monotone routing: opt-in method, matching config, bridge-only, fixed validation and cross-mode config rejection: passed; synthetic commands only");
                settings["Config"] = savedConfig;
                settings["TrainingMode"] = "preference_only";
                form.trainingMode.SelectedIndex = 1; Application.DoEvents();
                bool rejected = false;
                try { form.Run(python, new[] { "-c", "raise SystemExit(7)" }, "failure test"); }
                catch (Exception ex) { rejected = ex.Message.Contains("7"); }
                if (!rejected) throw new Exception("Failure exit test failed");
                results.Add("Nonzero subprocess exit propagates as pipeline failure: passed");
                Application.DoEvents();
                form.status.Text = "就绪 · 启动器界面与进程调用检查通过";
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                    bitmap.Save(Path.Combine(output, "launcher-preview.png"));
                }
                form.trainingMode.SelectedIndex = 2; Application.DoEvents();
                if (form.Snapshot()["TrainingMode"] != "aligned_token_identity_v1" || !form.paths["Config"].Text.EndsWith("experimental_aligned_token_v1.json", StringComparison.Ordinal) || form.alignedFields.Any(c => !c.Enabled))
                    throw new Exception("Experimental mode selection/default configuration failed");
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                    bitmap.Save(Path.Combine(output, "launcher-aligned-preview.png"));
                }
                form.trainingMode.SelectedIndex = 3; Application.DoEvents();
                if (form.Snapshot()["TrainingMode"] != "monotone_token_identity_v1" || !form.paths["Config"].Text.EndsWith("experimental_monotone_token_v1.json", StringComparison.Ordinal) ||
                    !form.paths["PreparedDirectory"].Text.Contains("transport_") || form.alignedFields.Any(c => !c.Enabled) || !form.allRows.Enabled)
                    throw new Exception("Monotone mode selection/default configuration failed");
                form.allRows.Checked = true;
                if (form.Snapshot()["SelectionPolicy"] != "all_rows") throw new Exception("All-rows GUI selection failed");
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                    bitmap.Save(Path.Combine(output, "launcher-monotone-preview.png"));
                }
                form.Hide();
                results.Add("WinForms construction and layout rendering for legacy, equal-token and monotone modes: passed; no training started");
            }
            string settingsDir = Path.Combine(output, "settings-compat");
            Directory.CreateDirectory(Path.Combine(settingsDir, "local"));
            foreach (var mode in new[] { "aligned_token_identity_v1", "monotone_token_identity_v1" })
            {
                File.WriteAllText(Path.Combine(settingsDir, "local", "launcher.settings.xml"),
                    new XDocument(new XElement("Launcher", new XElement("TrainingMode", mode))).ToString(), new UTF8Encoding(false));
                using (var restored = new Launcher(settingsDir))
                    if (restored.Snapshot()["TrainingMode"] != mode)
                        throw new Exception("Saved training-mode compatibility failed: " + mode);
            }
            results.Add("Saved settings restore both existing equal-length and new monotone mode IDs: passed");
            File.WriteAllText(Path.Combine(settingsDir, "local", "launcher.settings.xml"),
                new XDocument(new XElement("Launcher", new XElement("TrainingMode", "monotone_token_identity_v1"), new XElement("SelectionPolicy", "all_rows"))).ToString(), new UTF8Encoding(false));
            using (var restored = new Launcher(settingsDir))
                if (restored.Snapshot()["SelectionPolicy"] != "all_rows" || !restored.allRows.Enabled)
                    throw new Exception("Saved all-rows option restore failed");
            results.Add("Saved all-rows GUI option restore: passed");
            string updateFixture = Path.Combine(output, "updater-settings-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(updateFixture);
            File.WriteAllText(Path.Combine(updateFixture, "app_version.json"), "{\"version\":\"0.2.1\",\"runtime_epoch\":1}");
            using (var updateForm = new Launcher(updateFixture, false)) {
                if (!updateForm.autoUpdate.Checked || updateForm.ApplicationVersion() != "0.2.1" || updateForm.updateButton.Text != "检查更新")
                    throw new Exception("Updater defaults/version UI failed");
                updateForm.busy = true;
                updateForm.UpdateApplication(true).GetAwaiter().GetResult();
                if (updateForm.updating) throw new Exception("Update installation must be blocked while training");
                updateForm.busy = false;
                updateForm.autoUpdate.Checked = false;
                updateForm.SaveSettings(updateForm.Snapshot());
            }
            using (var updateForm = new Launcher(updateFixture))
                if (updateForm.autoUpdate.Checked) throw new Exception("Auto-check preference restore failed");
            results.Add("Updater UI: current version, default startup check, persistent opt-out, install blocked while training; no network/install in self-test: passed");
            string usageFixture = Path.Combine(output, "usage-summary-synthetic");
            Directory.CreateDirectory(usageFixture);
            string usagePath = Path.Combine(usageFixture, "inference_settings.json");
            File.WriteAllText(usagePath, "{\"status\":\"fixed_contract\",\"settings\":{\"alpha\":1,\"magnitude_match\":\"per_token\"}}");
            if (!InferenceSummary(usageFixture).Contains("固定 alpha=1")) throw new Exception("Fixed alpha summary failed");
            File.WriteAllText(usagePath, "{\"status\":\"unvalidated_starting_point\",\"settings\":{\"alpha\":0.12,\"magnitude_match\":\"none\"}}");
            if (!InferenceSummary(usageFixture).Contains("试用起点 alpha=0.12") || !InferenceSummary(usageFixture).Contains("未视频校准"))
                throw new Exception("Starting alpha must not be presented as calibrated/fixed");
            results.Add("Completion summary distinguishes fixed alpha=1 from uncalibrated starting alpha=0.12; no video-quality inference: passed (synthetic JSON)");
            string fixtureId = Guid.NewGuid().ToString("N");
            string portableFixture = Path.Combine(output, "portable-paths-中文 space-" + fixtureId);
            Directory.CreateDirectory(Path.Combine(portableFixture, "models", "text_encoders"));
            string firstEncoder = Path.Combine(portableFixture, "models", "text_encoders", "fake-path-test-only.safetensors");
            File.WriteAllText(firstEncoder, "Synthetic path discovery fixture; not a model");
            using (var form = new Launcher(portableFixture, false))
                if (form.paths["Encoder"].Text != firstEncoder) throw new Exception("Single encoder discovery failed");
            File.WriteAllText(Path.Combine(portableFixture, "models", "text_encoders", "second-fake-path-only.gguf"), "Synthetic path fixture");
            using (var form = new Launcher(portableFixture, false))
                if (form.paths["Encoder"].Text != "") throw new Exception("Multiple encoders must require user choice");
            using (var form = new Launcher(portableFixture, false))
            {
                form.paths["Pairs"].Text = Path.Combine(portableFixture, "local", "my_dataset", "pairs.jsonl");
                form.paths["SplitManifest"].Text = Path.Combine(portableFixture, "local", "my_dataset", "split_manifest.json");
                form.SaveSettings(form.Snapshot());
            }
            string relocated = Path.Combine(output, "relocated-中文 space-" + fixtureId);
            Directory.CreateDirectory(Path.Combine(relocated, "local"));
            File.Copy(Path.Combine(portableFixture, "local", "launcher.settings.xml"), Path.Combine(relocated, "local", "launcher.settings.xml"), true);
            using (var form = new Launcher(relocated))
                if (form.paths["Pairs"].Text != Path.Combine(relocated, "local", "my_dataset", "pairs.jsonl")
                    || form.paths["SplitManifest"].Text != Path.Combine(relocated, "local", "my_dataset", "split_manifest.json"))
                    throw new Exception("Saved relative paths did not relocate");
            results.Add("Encoder discovery chooses only an unambiguous file; relative saved user data paths survive relocation with Chinese/spaces: passed; fake path fixtures, no model execution");
            SelfTestEpochs(output, results);
            SelfTestExplicit(python, output, results);
            File.WriteAllText(Path.Combine(output, "self-test.json"), new JavaScriptSerializer().Serialize(results), new UTF8Encoding(false));
            return 0;
        }

        static void SelfTestEpochs(string output, List<string> results)
        {
            string fixture = Path.Combine(output, "epoch-options-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path.Combine(fixture, "configs"));
            string config = Path.Combine(fixture, "configs", "preference_v1.json");
            string text = "{\"bridge_epochs\":150,\"judge_epochs\":200}";
            File.WriteAllText(config, text);
            using (var form = new Launcher(fixture, false)) {
                if (form.bridgeEpochs.Value != 150 || form.judgeEpochs.Enabled) throw new Exception("Epoch defaults/preference-only JEV control failed");
                form.bridgeEpochs.Value = 75; form.judgeEpochs.Value = 90;
                var s = form.Snapshot(); var epochs = EpochOptions(s, false);
                if (epochs[0] != 75 || epochs[1] != 90) throw new Exception("Epoch override failed");
                var command = Plan(s, false).Single(c => c[0] == "train");
                if (command[Array.IndexOf(command, "--epochs") + 1] != "75" || command[Array.IndexOf(command, "--judge-epochs") + 1] != "90")
                    throw new Exception("Epoch controls not routed into CLI arguments");
                string run = s["RunDirectory"]; Directory.CreateDirectory(run);
                File.WriteAllText(Path.Combine(run, "run.json"), "{\"config\":{\"bridge_epochs\":150,\"judge_epochs\":200}}");
                epochs = EpochOptions(s, true);
                if (epochs[0] != 150 || epochs[1] != 200) throw new Exception("Resume must use original epochs");
                command = Plan(s, true).Single(c => c[0] == "train");
                if (!command.Contains("--resume") || command[Array.IndexOf(command, "--epochs") + 1] != "150") throw new Exception("Resume epoch CLI routing failed");
                if (File.ReadAllText(config) != text) throw new Exception("Epoch controls modified source config");
                s["BridgeEpochs"] = "0"; bool rejected = false;
                try { EpochOptions(s, false); } catch (Exception) { rejected = true; }
                if (!rejected) throw new Exception("Invalid epochs accepted");
                form.SaveSettings(form.Snapshot());
                form.trainingMode.SelectedIndex = 0;
                if (!form.judgeEpochs.Enabled) throw new Exception("Numeric mode JEV epochs disabled");
                form.trainingMode.SelectedIndex = 4;
                if (form.epochControls.Visible || form.bridgeEpochs.Enabled) throw new Exception("Explicit fixed-step mode exposes epochs");
            }
            using (var restored = new Launcher(fixture))
                if (restored.bridgeEpochs.Value != 75 || restored.judgeEpochs.Value != 90) throw new Exception("Epoch settings persistence failed");
            results.Add("Epoch UI: config defaults, editable bridge/JEV limits, preference disables JEV, persistence, resume original limits, invalid values rejected, source JSON preserved, explicit fixed-step mode hidden: passed; software only");
        }

        internal static int SelfTestExplicitPlan(string settingsPath, string output)
        {
            // Synthetic harness: serializes actual C# argv only; never starts training.
            var s = ReadMap(settingsPath).ToDictionary(p => p.Key, p => Convert.ToString(p.Value));
            if (s["TrainingMode"] != ExplicitMode) throw new Exception("Self-test requires explicit mode.");
            if (File.Exists(output)) throw new Exception("Self-test plan output exists.");
            File.WriteAllText(output, Json().Serialize(Plan(s, false)), new UTF8Encoding(false));
            return 0;
        }

        static void SelfTestExplicit(string python, string output, List<string> results)
        {
            string fixture = Path.Combine(output, "explicit-synthetic-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path.Combine(fixture, "tools"));
            File.WriteAllText(Path.Combine(fixture, "tools", "fit_explicit_targets.py"), "# Synthetic validation path only");
            string manifest = Path.Combine(fixture, "manifest.json"), config = Path.Combine(fixture, "options.json");
            File.WriteAllText(manifest, "{\"kind\":\"explicit_train_target_manifest_v1\",\"evidence_class\":\"train_fit_probe\",\"pairs\":[]}");
            var s = new Dictionary<string, string> { {"Python",python}, {"TrainingMode",ExplicitMode},
                {"TargetManifest",manifest}, {"ExplicitFitConfig",""}, {"RunDirectory",Path.Combine(fixture,"new-run")}, {"ComfyRoot",""} };
            var plan = Plan(s, false);
            if (plan.Count != 2 || plan.Any(c => c[0] != "fit_explicit_targets.py" || c.Contains("--config") || c.Contains("--resume")) ||
                !plan[0].Contains("--dry-run") || plan[0].Contains("--execute") || !plan[1].Contains("--execute") ||
                plan[0][Array.IndexOf(plan[0], "--manifest-sha") + 1] != FileSha(manifest)) throw new Exception("Explicit argv routing failed.");
            bool rejected = false;
            try { Plan(s, true); } catch (Exception) { rejected = true; }
            if (!rejected) throw new Exception("Explicit resume accepted.");
            foreach (string bad in new[] { "{\"steps\":true}", "{\"steps\":1.5}", "{\"hidden\":9,\"heads\":1}", "{\"device\":\"cuda\"}", "{\"alpha\":0}", "{\"x_identity_weight\":1}", "{\"lr\":\"NaN\"}" })
            {
                File.WriteAllText(config, bad); rejected = false;
                try { ExplicitConfig(config); } catch (Exception) { rejected = true; }
                if (!rejected) throw new Exception("Invalid explicit config accepted: " + bad);
            }
            File.WriteAllText(config, "{\"steps\":2,\"hidden\":8,\"layers\":1,\"heads\":2,\"max_tokens\":16}");
            s["ExplicitFitConfig"] = config; plan = Plan(s, false);
            if (plan[0][Array.IndexOf(plan[0], "--steps") + 1] != "2") throw new Exception("Explicit options were not forwarded.");
            using (var form = new Launcher(fixture, false))
            {
                form.Validate(s, false, true);
                Directory.CreateDirectory(s["RunDirectory"]); rejected = false;
                try { form.Validate(s, false, false); } catch (Exception) { rejected = true; }
                if (!rejected) throw new Exception("Existing empty explicit output accepted.");
                s["RunDirectory"] = Path.Combine(fixture, "models", "new"); rejected = false;
                try { form.Validate(s, false, true); } catch (Exception) { rejected = true; }
                if (!rejected) throw new Exception("Explicit model overwrite accepted.");
                string oldConfig = form.paths["Config"].Text, oldPairs = form.paths["Pairs"].Text;
                form.trainingMode.SelectedIndex = 4;
                if (form.Snapshot()["TrainingMode"] != ExplicitMode || form.resumeButton.Enabled || form.paths["Config"].Enabled ||
                    !form.paths["TargetManifest"].Enabled || form.paths["Config"].Text != oldConfig || form.paths["Pairs"].Text != oldPairs)
                    throw new Exception("Explicit UI state or old fields changed.");
                form.Show(); Application.DoEvents();
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(0,0,bitmap.Width,bitmap.Height));
                    bitmap.Save(Path.Combine(output,"launcher-explicit-preview.png"));
                }
                form.Hide();
                form.SaveSettings(form.Snapshot());
                string env = form.RunExplicitCommand(python, new[] {"-c", "import os,json; print(json.dumps({'cuda_hidden':os.environ.get('CUDA_VISIBLE_DEVICES')==''}))"}, "Synthetic CPU environment check");
                if (!Equals(ObjectMap(Json().DeserializeObject(env))["cuda_hidden"], true)) throw new Exception("Explicit child not CPU isolated.");
                form.cancelled = true; rejected = false;
                try { form.RunExplicitCommand(python, new[] {"-c", "raise AssertionError('must not launch')"}, "Synthetic cancelled check"); }
                catch (OperationCanceledException) { rejected = true; }
                if (!rejected) throw new Exception("Cancelled explicit child launched.");
                form.cancelled = false;
                var child = Task.Run(delegate { return form.RunExplicitCommand(python, new[] { "-c", "import time; time.sleep(30)" }, "Synthetic owned cancellation"); });
                var deadline = DateTime.UtcNow.AddSeconds(10);
                while (form.activeProcess == null && !child.IsCompleted && DateTime.UtcNow < deadline)
                { Application.DoEvents(); System.Threading.Thread.Sleep(20); }
                if (form.activeProcess == null) throw new Exception("Synthetic owned child did not start.");
                form.Cancel();
                while (!child.IsCompleted && DateTime.UtcNow < deadline)
                { Application.DoEvents(); System.Threading.Thread.Sleep(20); }
                if (!(child.IsCanceled || (child.IsFaulted && child.Exception.Flatten().InnerExceptions.Any(e => e is OperationCanceledException))) || form.activeProcess != null)
                    throw new Exception("Explicit owned cancellation was not propagated.");
                form.cancelled = false; rejected = false;
                try { form.RunExplicitCommand(python, new[] { "-c", "raise SystemExit(7)" }, "Synthetic explicit failure"); }
                catch (Exception ex) { rejected = ex.Message.Contains("7"); }
                if (!rejected) throw new Exception("Explicit nonzero exit accepted.");
            }
            using (var restored = new Launcher(fixture))
                if (restored.Snapshot()["TrainingMode"] != ExplicitMode) throw new Exception("Explicit settings restore failed.");
            results.Add("Explicit target GUI: CPU-only dry-run/execute argv, strict optional config, no resume, fresh output, model protection, mode persistence, old fields retained, cancelled-before-launch/owned-child stop/nonzero exit and layout: passed; synthetic only.");
        }

        static void SelfTestPreparedSourceIsolation(string output, string config)
        {
            string fixture = Path.Combine(output, "validate-isolation-synthetic-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path.Combine(fixture, "tools"));
            Directory.CreateDirectory(Path.Combine(fixture, "comfy"));
            string sentinel = Path.Combine(fixture, "tools", "trainer.py");
            File.WriteAllText(sentinel, "# synthetic path validation fixture; never executed");
            var settings = new Dictionary<string, string> {
                { "Python", sentinel }, { "Pairs", sentinel }, { "Encoder", sentinel }, { "SplitManifest", sentinel },
                { "Config", config }, { "ComfyRoot", fixture }, { "TrainingMode", "aligned_token_identity_v1" },
                { "RunDirectory", Path.Combine(fixture, "run") }, { "Dataset", Path.Combine(fixture, "source.npz") },
                { "PreparedDirectory", Path.Combine(fixture, "prepared") }
            };
            using (var validator = new Launcher(fixture))
            {
                validator.Validate(settings, false, false); // Separate, not-yet-created outputs are valid.
                string prepared = settings["PreparedDirectory"];
                var cases = new[] {
                    new[] { prepared, Path.Combine(prepared, "source.npz") },
                    new[] { prepared, Path.Combine(prepared.ToUpperInvariant(), "source.npz") },
                    new[] { prepared, Path.Combine(prepared, "unused", "..", "source.npz") },
                    new[] { prepared + Path.DirectorySeparatorChar, Path.Combine(prepared, "source.npz").Replace('\\', '/') },
                    new[] { Path.Combine(fixture, "same.npz"), Path.Combine(fixture, "same.npz") },
                    new[] { Path.Combine(fixture, "same.json"), Path.Combine(fixture, "same.npz") },
                    new[] { Path.Combine(fixture, "same.npz.receipt.json"), Path.Combine(fixture, "same.npz") }
                };
                foreach (var item in cases)
                {
                    settings["PreparedDirectory"] = item[0]; settings["Dataset"] = item[1];
                    bool rejected = false;
                    try { validator.Validate(settings, false, false); }
                    catch (Exception ex) { rejected = ex.Message.Contains("不得位于实验对齐数据目录内"); }
                    if (!rejected || Directory.Exists(item[0]) || File.Exists(item[1])) throw new Exception("Prepared/source isolation failed before encoding: " + item[1]);
                }
                settings["PreparedDirectory"] = prepared;
                settings["Dataset"] = Path.Combine(prepared + "-source", "source.npz");
                validator.Validate(settings, false, false); // A textual prefix alone is not containment.
                Directory.CreateDirectory(prepared);
                string nestedSource = Path.Combine(prepared, "source.fixture");
                File.WriteAllText(nestedSource, "synthetic existing source");
                foreach (string key in new[] { "Pairs", "Encoder", "SplitManifest" })
                {
                    settings[key] = nestedSource;
                    bool rejected = false;
                    try { validator.Validate(settings, false, false); }
                    catch (Exception ex) { rejected = ex.Message.Contains("不得位于实验对齐数据目录内"); }
                    settings[key] = sentinel;
                    if (!rejected) throw new Exception("Prepared existing source isolation failed: " + key);
                }
            }
        }
    }
}
