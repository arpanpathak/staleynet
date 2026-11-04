CARGO ?= cargo

.PHONY: fmt fmt-check clippy clippy-host test test-gpu coverage deny ci

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

clippy-host:
	$(CARGO) clippy -p staleynet-audio --all-targets -- -D warnings

test:
	$(CARGO) test -p staleynet-audio

test-gpu:
	$(CARGO) oxide test -- --release --workspace

coverage:
	$(CARGO) llvm-cov -p staleynet-audio --summary-only --fail-under-lines 99

deny:
	$(CARGO) deny check

ci: fmt-check clippy test
