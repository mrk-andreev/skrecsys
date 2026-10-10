//! Open-loop load generator: sends requests at a fixed rate whether or not the service keeps
//! up, and charges each one for the time since it was *due*, not since it was sent.
//!
//! A closed-loop generator (each client waits for its answer before sending the next) slows
//! down together with the service and hides its queueing; this one does not. It is a separate
//! Rust process so that generating load never competes with the Python service for the GIL.

mod stats;

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use clap::Parser;
use reqwest::Client;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior};

use stats::{Outcome, Report};

#[derive(Parser)]
#[command(about)]
struct Cli {
    /// Request URL; `{user}` is replaced by a user id drawn at random
    #[arg(long)]
    url: String,
    /// Users to draw from: ids 0..users
    #[arg(long)]
    users: u64,
    /// Requests per second to send
    #[arg(long)]
    rps: f64,
    /// Measured seconds
    #[arg(long, default_value_t = 10.0)]
    duration: f64,
    /// Seconds sent before measuring starts, for the service to warm up
    #[arg(long, default_value_t = 2.0)]
    warmup: f64,
    /// Request timeout, seconds
    #[arg(long, default_value_t = 5.0)]
    timeout: f64,
    /// Requests allowed in flight; further ones are counted as dropped, not sent
    #[arg(long, default_value_t = 20_000)]
    max_in_flight: usize,
    /// Seed of the user draw
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// `SplitMix64`: a tiny, well-spread generator, enough to pick users.
const fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

async fn request(client: &Client, url: &str) -> Result<()> {
    let response = client.get(url).send().await?.error_for_status()?;
    ensure!(!response.bytes().await?.is_empty(), "empty response");
    Ok(())
}

async fn run(cli: Cli) -> Result<Report> {
    ensure!(cli.url.contains("{user}"), "url must contain {{user}}");
    ensure!(cli.users >= 1, "users must be positive");
    ensure!(cli.rps > 0.0, "rps must be positive");
    ensure!(
        cli.duration > 0.0 && cli.warmup >= 0.0 && cli.timeout > 0.0,
        "invalid times"
    );
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(cli.timeout))
        .no_proxy()
        .pool_max_idle_per_host(cli.max_in_flight)
        .build()?;

    // Fail early, and with a clear message, if the service is not up.
    let probe = cli.url.replace("{user}", "0");
    if let Err(error) = request(&client, &probe).await {
        bail!("the service does not answer {probe}: {error:#}");
    }

    let slots = Arc::new(Semaphore::new(cli.max_in_flight));
    let interval = Duration::from_secs_f64(1.0 / cli.rps);
    let warmup = Duration::from_secs_f64(cli.warmup);
    let total = warmup + Duration::from_secs_f64(cli.duration);
    let begin = Instant::now();
    let mut ticker = tokio::time::interval_at(begin, interval);
    // A late tick is made up for at once: the schedule is the schedule.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Burst);
    let mut state = cli.seed;
    let mut tasks = JoinSet::new();
    let mut outcomes = Vec::new();
    let mut dropped = 0_usize;
    loop {
        let due = ticker.tick().await;
        if due - begin >= total {
            break;
        }
        let measured = due - begin >= warmup;
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            dropped += usize::from(measured);
            continue;
        };
        let user = next(&mut state) % cli.users;
        let url = cli.url.replace("{user}", &user.to_string());
        let client = client.clone();
        tasks.spawn(async move {
            let outcome = request(&client, &url).await;
            drop(permit);
            measured.then(|| Outcome {
                latency_ms: due.elapsed().as_secs_f64() * 1000.0,
                error: outcome.err().map(|error| format!("{error:#}")),
            })
        });
        // Collect what has finished as we go, so a long run does not pile up tasks.
        while let Some(done) = tasks.try_join_next() {
            outcomes.extend(done?);
        }
    }
    while let Some(done) = tasks.join_next().await {
        outcomes.extend(done?);
    }
    Ok(Report::new(&outcomes, dropped, cli.rps, cli.duration))
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(report) => {
            eprintln!("{}", report.summary());
            println!("{}", report.json());
            ExitCode::from(u8::from(!report.clean()))
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(2)
        }
    }
}
