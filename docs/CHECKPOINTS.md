# Checkpoint and resume

For a new run, enable checkpoints and use `MCMCMC_RESUME=auto`.
For an intentional continuation, use `MCMCMC_RESUME=require`; a missing
checkpoint must fail rather than silently starting a replacement chain.
`MCMCMC_PRODUCTION_STEPS` is the cumulative per-chain target, not additional
steps. To continue a 100M run to 300M, set the target to 300000000.

Checkpoint cadence is set by `MCMCMC_CHECKPOINT_INTERVAL_STEPS` (for example
2000000); the saved schedule counts completed warmup plus production steps.
A forced final save can therefore have a non-multiple production count.
Read `production_proposed_steps` for work actually done and `production_steps`
for the configured target. Never infer completion from the latter alone.

The saved state includes the complete RNG stream, vertices, lambdas,
adaptation counters and accumulated production statistics. Reading uses
the corrected hexadecimal-token parser. A seed and a step count alone do
not reconstruct the state of the Markov chain.

Keep physics, geometry, precision, table contents, learned proposal and chain
identities unchanged. Reuse the same directory and binary/build where possible.
The historical compatibility gate does not fingerprint every physical input;
an accepted file is not proof that an arbitrary changed parameter file is safe.
Do not edit checkpoint internals or lower an existing target as a generic
operation. Back up params/PBS/logs before a real continuation, verify that no
writer is still active, and only then submit the MCMCMC stage.

The chunked PBS script exports its own `CDET_MCMCMC_RESUME`. Update that
export to `require` as well as params when continuing. Do not regenerate the
entire workflow over an existing run; the migrated generator refuses this.

Local regression:

```sh
make -j 1
python3 tests/test_checkpoint_resume.py
```

This small test checks an unaligned 137-step interruption. Exact equality is
asserted for the same executable and platform, not across different CPUs,
compilers, library versions or floating-point implementations.
