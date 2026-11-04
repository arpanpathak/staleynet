//! `staleynet`: prompt-to-music with a from-scratch audio language model.
//!
//! ```text
//! staleynet prepare  --song-describer data/raw/song-describer --cache data/cache
//! staleynet tokenize --cache data/cache
//! staleynet train    --cache data/cache --run runs/base --config configs/base.json
//! staleynet sing     --cache data/cache --run runs/base --prompt "..." --out song.wav
//! staleynet reconstruct --cache data/cache --out samples/reconstruction
//! staleynet evaluate --run runs/lyrics --cache data/cache-jamendolyrics
//! staleynet bench    --out benchmarks/results.json
//! ```

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

/// Command-line interface.
#[derive(Parser)]
#[command(name = "staleynet", version, about = "Text-prompted music generation with a Rust + cuda-oxide audio language model")]
struct Cli {
    /// The task to run.
    #[command(subcommand)]
    command: Command,
}

/// Every task, one module each under `commands/`.
#[derive(Subcommand)]
enum Command {
    /// Decode, resample and analyse recordings into the feature cache.
    Prepare(commands::prepare::Arguments),

    /// Train the audio tokenizer and encode every track.
    Tokenize(commands::tokenize::Arguments),

    /// Train the language model.
    Train(commands::train::Arguments),

    /// Generate audio from a text prompt.
    Sing(commands::sing::Arguments),

    /// Render held-out excerpts through the tokenizer and vocoder alone.
    Reconstruct(commands::reconstruct::Arguments),

    /// Measure whether a model reads its lyrics.
    Evaluate(commands::evaluate::Arguments),

    /// Measure kernel and pipeline performance.
    Bench(commands::bench::Arguments),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Prepare(arguments) => commands::prepare::run(arguments),
        Command::Tokenize(arguments) => commands::tokenize::run(arguments),
        Command::Train(arguments) => commands::train::run(arguments),
        Command::Sing(arguments) => commands::sing::run(arguments),
        Command::Reconstruct(arguments) => commands::reconstruct::run(arguments),
        Command::Evaluate(arguments) => commands::evaluate::run(arguments),
        Command::Bench(arguments) => commands::bench::run(arguments),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_commands() {
        let prepare = Cli::try_parse_from(["staleynet", "prepare", "--song-describer", "data/raw/song-describer"]).unwrap();
        let tokenize = Cli::try_parse_from(["staleynet", "tokenize"]).unwrap();
        let train = Cli::try_parse_from(["staleynet", "train", "--run", "runs/base"]).unwrap();

        assert!(matches!(prepare.command, Command::Prepare(_)));
        assert!(matches!(tokenize.command, Command::Tokenize(_)));
        assert!(matches!(train.command, Command::Train(_)));
    }

    #[test]
    fn parses_generation_commands() {
        let sing = Cli::try_parse_from(["staleynet", "sing", "--run", "runs/base", "--prompt", "warm ballad"]).unwrap();
        let reconstruct = Cli::try_parse_from(["staleynet", "reconstruct"]).unwrap();

        assert!(matches!(sing.command, Command::Sing(_)));
        assert!(matches!(reconstruct.command, Command::Reconstruct(_)));
    }

    #[test]
    fn parses_evaluation_and_benchmark_commands() {
        let evaluate = Cli::try_parse_from(["staleynet", "evaluate", "--run", "runs/lyrics"]).unwrap();
        let bench = Cli::try_parse_from(["staleynet", "bench"]).unwrap();

        assert!(matches!(evaluate.command, Command::Evaluate(_)));
        assert!(matches!(bench.command, Command::Bench(_)));
    }

    #[test]
    fn rejects_missing_required_arguments() {
        assert!(Cli::try_parse_from(["staleynet", "prepare"]).is_err());
        assert!(Cli::try_parse_from(["staleynet", "train"]).is_err());
        assert!(Cli::try_parse_from(["staleynet", "sing", "--run", "runs/base"]).is_err());
        assert!(Cli::try_parse_from(["staleynet", "evaluate"]).is_err());
    }
}
