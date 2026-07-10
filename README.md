# azo-plugin-event-architecture

An Agent Zoo userspace plugin providing two primitives for model-created event-driven workflows:

- `fork_agent(message)` exposes the existing `/fork <message>` behavior as a model tool.
- Command interrupt subscriptions run shell commands in the background and wake the owning session when regex and trigger conditions are met.

Subscriptions are strictly owned by the creating Agent Zoo session ID. Forked sessions do not inherit or adopt their parent's subscriptions; each worker creates the watches for its own role.

## Install

```bash
pixi run azo-plugin install .
```

The plugin is disabled by default, so installation alone exposes no tools or interrupt component. Enable it for a launch by adding:

```bash
azo --config-set event_architecture.enabled=true
```

Use the same flag with your normal `azo` launch arguments. If you change a persistent config file for an already-running session, run `/reload` afterward.

Forks created through `fork_agent` inherit this launch-time enablement through a plugin-private environment marker containing the exact parent session ID. A matching child rotates the marker to its own ID for descendants. This compensates for the current `/fork` launcher not forwarding transient `--config-set` arguments, while unrelated lineages and root sessions remain disabled.

## Tools

- `fork_agent`
- `subscribe_interrupt`
- `list_interrupt_subscriptions`
- `remove_interrupt_subscription`

The built-in `await_interrupts` tool can park a configured agent until a subscription fires.

## Development

```bash
pixi run --environment test test
```

## Example

Ask the agent to create a session-local watcher:

```text
subscribe_interrupt(
  name="open-pr-set",
  command="gh pr list --state open --json number,headRefOid --jq 'sort_by(.number)'",
  match_regex=".+",
  interval_seconds=30,
  trigger="change",
  fire_initial=false
)
```

The first successful observation establishes a baseline. A later matching output change is delivered as a system-generated user interrupt containing the command, match, return code, and bounded stdout/stderr excerpts.

A coordinating agent can create a persistent collaborator with:

```text
fork_agent(
  message="You are the issue agent. Subscribe to new issues, handle that side of the workflow, and use RLMs for bounded tasks."
)
```

## Trigger behavior

- `rising` fires when the regex changes from not matching to matching.
- `change` fires when matching output changes.
- `each` fires on every matching poll.
- `fire_initial=false` establishes a baseline without waking on existing state.
- `wake_on_error=true` wakes once when a new runner error such as a timeout appears.

Commands run asynchronously through `/bin/bash -lc`, with a minimum one-second interval, per-command timeouts, process-group termination, bounded in-memory output tails, a bounded completion queue, and a global concurrency limit of four commands per plugin process. Removing a subscription, switching sessions, disabling the plugin, or shutting down the session terminates its active command groups.

## Session ownership

Each subscription records the creating session ID. A fork may contain cloned subscription records in its saved state, but records owned by another session are inert, hidden from the tools, and never mutated or adopted. The child agent should register its own watches after `fork_agent` gives it its role.

The reactor uses the live run database session ID when available, so hot `/session load` and `/session new` transitions do not delete or reassign subscriptions while runtime attributes are catching up.
