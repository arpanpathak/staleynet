# Building with an older cuda-oxide

The repository builds against the cuda-oxide revision recorded in
`Cargo.lock`, with the toolchain named in `rust-toolchain.toml`. A machine
whose installed `cargo-oxide` and codegen backend belong to an older
cuda-oxide (for example a Jetson Orin NX set up before cuda-oxide moved to its
current repository layout) can still build the project with a few local
settings, none of which is committed.

1. Point the cuda-oxide crates at the older revision's local checkout, in
   `.cargo/config.toml`:

   ```toml
   [patch."https://github.com/NVlabs/cuda-oxide.git"]
   cuda-core = { path = "/home/<user>/.cargo/git/checkouts/cuda-oxide-<hash>/<rev>/crates/cuda-core" }
   cuda-device = { path = "/home/<user>/.cargo/git/checkouts/cuda-oxide-<hash>/<rev>/crates/cuda-device" }
   cuda-host = { path = "/home/<user>/.cargo/git/checkouts/cuda-oxide-<hash>/<rev>/crates/cuda-host" }
   ```

2. Select that revision's toolchain for this directory, for example
   `rustup override set nightly-2026-04-03` for cuda-oxide `b7fe543`.

3. Let Cargo resolve the patched crates, once:

   ```bash
   cargo update -p cuda-core -p cuda-device -p cuda-host
   ```

   This rewrites `Cargo.lock` locally. Do not commit that change; the
   committed lock file records the revision the project is tested with.

Build and test as usual, for example with `cargo oxide build -- --release`.
The kernels and their results do not depend on the cuda-oxide revision.
