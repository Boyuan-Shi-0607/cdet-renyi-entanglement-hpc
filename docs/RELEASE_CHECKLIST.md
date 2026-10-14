# Before the first public release

This is a local import, with no automatic push, visibility change or HPC job.

- [ ] Author approves the code license and verifies third-party rights/notices.
- [ ] Add the approved LICENSE; update LICENSE_PENDING.md and citation metadata.
- [ ] Review all intended-to-be-public files and the target repository history
      for credentials, private correspondence and unintended data.
- [ ] Review upstream known limitations; either harden them in separately
      tested commits or clearly bound the supported release configuration.
- [ ] Run the fast/sanitizer suites and checkpoint test in a clean Linux
      environment in addition to the local macOS migration checks.
- [ ] Review PBS module setup and validate a small compute-node workflow;
      local script generation is not a scheduler execution test.
- [ ] Decide whether to include publication analysis in this repository or
      provide a separately versioned, persistent data/reproduction archive.
- [ ] Add the verified paper identifier/data DOI, without placeholders.
- [ ] Review the staged diff, commit, and approve a push to the intended remote.
- [ ] Approve public visibility, create a versioned release and record its
      source hashes. Keep the paper baseline distinguishable from later fixes.

Recommended eventual CI: generic-CPU Linux C++ build, small deterministic
tests, resident/chunk consistency, restart regression and sanitizer jobs.
Do not put HPC credentials or large production runs into pull-request CI.
