# Open questions

## Runtime configuration delivery

When distill-run is packaged as an image, decide how an upper-level scheduler
delivers per-job training and Accelerate configuration to the container.

Topics for the follow-up discussion:

- mount a read-only job directory and pass `--config` / `--accelerate-config`;
- optionally support config from stdin or a remote URI;
- package built-in Accelerate presets as Python package data and resolve them
  with `importlib.resources`, rather than relying on repository-relative paths;
- define precedence between built-in presets, mounted config, environment
  expansion, and CLI arguments;
- keep models read-only and outputs writable, with one isolated directory per
  job.
