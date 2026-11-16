use mkr_cli::{args::Args, cmds, init_console, store_cmds, BANNER};
use mkr_core::Config;
use std::path::PathBuf;

const HELP: &str = "USAGE: mkr <command> [options]      (MKR-SPARK-26.exe accepts the same commands)

COMMANDS
  train      Train the neural model from scratch on text (CPU).
               mkr train [--data <file|dir>] [--preset nano|micro|mini|small|base|large | --ram 2|4|8|16]
                         [--steps N] [--batch N] [--ctx N] [--accum N] [--lr X] [--resume] [--out checkpoints]
  chat       Interactive assistant (memory -> stored knowledge -> model -> verified web search).
               mkr chat [--offline] [--no-learn] [--approve-single] [--verbose]
  search     Search, fetch, verify and cite.   mkr search <question> [--hits] [--no-learn] [--approve-single]
  memory     Your memory.    mkr memory <add|list|recall|update|delete|clear|forget-chats|prefs|history>
  knowledge  Verified knowledge base.   mkr knowledge <list|query|show|correct|delete|expire|prune|stats>
  learn      Controlled learning.   mkr learn <stage|queue|approve|reject|finetune|rollback>
  model      mkr model info | quantize [in] [--out f] | generate --prompt \"...\" [--tokens N --temp T]
  data       mkr data seed | prepare <files...> [--out f] | stats <file>
  benchmark  Measure speed and memory on this machine.   mkr benchmark [--preset micro] [--save]
  plan       Show recommended settings for 2/4/8/16 GB RAM.
  config     mkr config [list|get <k>|set <k> <v>|path|init]
  help       This text.

GLOBAL OPTIONS
  --config <file>   configuration file (default config/mkr.conf, or $MKR_CONFIG)
  --model <file>    model file (default from config: models/MKR-SPARK-26.model)
";

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let a = Args::parse(&argv)?;
    let cmd = a.pos.first().map(|s| s.as_str()).unwrap_or("help");
    if cmd == "help" || a.has("help") || cmd == "--help" {
        print!("{BANNER}\n{HELP}");
        return Ok(());
    }
    if cmd == "version" {
        println!("MKR - SPARK 26 {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let cfg_path = a.flag("config").map(PathBuf::from).or_else(|| std::env::var("MKR_CONFIG").ok().map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("config/mkr.conf"));
    let cfg = Config::load(&cfg_path).map_err(|e| format!("{e} (in {})", cfg_path.display()))?;
    if cfg.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(cfg.threads).build_global();
    }
    match cmd {
        "train" => cmds::cmd_train(&a, &cfg),
        "chat" => cmds::cmd_chat(&a, &cfg),
        "search" => cmds::cmd_search(&a, &cfg),
        "memory" => store_cmds::cmd_memory(&a, &cfg),
        "knowledge" => store_cmds::cmd_knowledge(&a, &cfg),
        "learn" => store_cmds::cmd_learn(&a, &cfg),
        "model" => cmds::cmd_model(&a, &cfg),
        "data" => cmds::cmd_data(&a, &cfg),
        "benchmark" => cmds::cmd_benchmark(&a),
        "plan" => cmds::cmd_plan(),
        "config" => cmds::cmd_config(&a, &cfg_path, &cfg),
        other => Err(format!("unknown command '{other}'. Run `mkr help`.")),
    }
}

fn main() {
    init_console();
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
