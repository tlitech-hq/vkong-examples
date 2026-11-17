# vLLM stack: session routing and Grafana on two machines

Run vLLM on two RTX 4090 machines behind one VKong URL, keep each chat on the
same machine, and watch the cache per machine in Grafana Cloud.

**Result:** with one header, the prefix cache hit rate went from **25% to 61%**
and follow-up messages got their first token **twice as fast** (0.73 s instead
of 1.47 s).

![Requests go through the VKong URL to two vLLM machines; each machine pushes its metrics to Grafana Cloud](images/architecture.svg)

## Why route by session

In a chat, every new message resends the whole conversation. vLLM keeps the
part it already computed in its **prefix cache** and skips it, but only on the
machine that computed it. If the next message lands on another machine, that
machine computes the whole conversation again.

Add the header `X-Vkong-Session: <conversation id>` and VKong sends every
message of that conversation to the same machine.

## How VKong routes

- **No header:** requests are spread over the machines in turn.
- **With `X-Vkong-Session`:** VKong hashes the value and the hash picks a
  machine. The same value always picks the same machine, so a conversation
  stays where its cache is. Different values spread evenly over the machines.
- **A busy machine is not overloaded:** if the chosen machine has more than
  1.25 times the average number of requests in progress, the request goes to
  the next machine instead, and comes back once it is less busy.
- **Machines come and go:** when a machine is added or stopped, only the
  sessions of that machine move.
- The header is used by VKong only; your App does not receive it.

There is nothing to set in `vkong.yaml` and no router to run.

## Files

| Path | What it is |
|---|---|
| `vkong.yaml` | The App: two machines with one RTX 4090 each |
| `serve.sh` | Starts the metrics agent, then vLLM with prefix caching and an API key |
| `metrics_agent.py` | Sends vLLM's metrics to Grafana Cloud every 15 s, labelled by machine |
| `dashboards/` | Two Grafana dashboards, split per machine |
| `bench/` | The multi-round chat benchmark of vLLM production-stack |

Metrics are sent from each machine because the public URL reaches the App, not
one machine, so Grafana could not read each machine from outside.

## Run it

You need the `vkong` CLI (after `vkong login`), a free Grafana Cloud account
and `uv`.

### 1. Grafana Cloud token

1. In grafana.com, open **Access Policies**, create a policy with only the
   `metrics:write` scope for your stack, and add a token (it is shown once).
2. Open your stack, then **Prometheus → Details**, and note the **Remote Write
   Endpoint** and the **Username / Instance ID**.

### 2. Secrets

In the VKong dashboard, open **Secrets → New secret** and create `grafana`:

| Key | Value |
|---|---|
| `METRICS_WRITE_URL` | Remote Write Endpoint |
| `METRICS_USER` | Username / Instance ID |
| `METRICS_PASSWORD` | the token |

Create the API key your clients will use:

```bash
export VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
vkong secret create vllm-stack VLLM_API_KEY
```

### 3. Deploy

```bash
cd vllm-stack
vkong deploy
```

The first start downloads the model and takes a few minutes. When both
machines serve, the App page in the [VKong dashboard](https://vkong.tli-tech.com/app)
shows **Instances 2/2**:

![The vllm-stack App in the VKong dashboard: healthy, 2/2 instances, one RTX 4090 per machine](images/vkong-app-running.png)

### 4. Dashboards

In Grafana, open **Dashboards → New → Import** and upload the two files in
`dashboards/`:

- **vLLM stack on VKong**: cache hit rate and requests per machine.
- **vLLM (per machine)**: vLLM's own dashboard, one line per machine.

### 5. Benchmark

`bench/multi_round_qa.py` is the
[benchmark](https://github.com/vllm-project/production-stack/tree/main/benchmarks/multi-round-qa)
that vLLM production-stack uses for session routing: users with a long chat
history send message after message. This copy sends `X-Vkong-Session` when
`--request-with-user-id` is set.

```bash
export VKONG_URL=https://<your App URL>
cd bench
bench() {
  uv run --with openai --with pandas --with tqdm python multi_round_qa.py \
    --num-users 15 --num-rounds 10 --qps 0.5 \
    --shared-system-prompt 1000 --user-history-prompt 10000 --answer-len 100 \
    --model Qwen/Qwen3-4B-Instruct-2507 --base-url "$VKONG_URL/v1" \
    --api-key "$VLLM_API_KEY" --time 300 "$@"
}
bench --init-user-id 1000 --output no-header.csv                       # no header
bench --init-user-id 2000 --output session.csv --request-with-user-id  # with X-Vkong-Session
```

Use a different `--init-user-id` for each run so a run never reuses the cache
of the one before.

## Results

Two RTX 4090 machines, one in **Vietnam** and one in the **United States**,
called from a laptop in Vietnam. 15 users, 10 messages each, about 11,000
tokens of conversation per user, 5 minutes per run.

| | No header | `X-Vkong-Session` |
|---|---|---|
| Prefix cache hit rate | 25.1% | **61.2%** |
| Time to first token, follow-up messages, p50 | 1.47 s | **0.73 s** |
| Time to first token, follow-up messages, p90 | 2.42 s | **1.60 s** |
| Time to first token, all messages, p50 | 1.61 s | **1.00 s** |
| Time to first token, first message, mean | 2.07 s | 1.79 s |

The first message of a user is about the same: nothing is cached yet. From the
second message on, the conversation is already cached on its machine.

**About the network:** the two machines are in different regions, and times
are measured on the laptop, so a message served in the United States takes
longer to travel than one served in Vietnam. Both runs used the same two
machines and split requests evenly between them, so the network is the same
for both and the difference comes from the cache. The hit rate is measured
inside vLLM and does not depend on the network.

Without the header, each machine reuses about a quarter of the prompt:

![Grafana without the header: hit rate 25.1%, requests split evenly](images/grafana-no-header.png)

With the header, requests are still split evenly, and the hit rate climbs as
each conversation builds up on its machine:

![Grafana with X-Vkong-Session: hit rate 61.2%, both machines climbing to about 70%](images/grafana-session.png)

Both runs one after the other (the second starts at 00:37). Green is prompt
read from the cache, orange is prompt computed again:

![Grafana, both runs: hit rate per machine, routing split, prompt from cache vs computed](images/grafana-both-runs.png)

## When to use it

- **Use it** for multi-turn chat, agents that resend long histories, and
  questions about the same documents. Send the conversation, user or document
  ID as the value.
- **Skip it** for single, independent requests: there is nothing to reuse.
- **Keep in mind:**
  - The first message of a session gets no benefit.
  - The gain depends on conversations fitting in a machine's KV cache (about
    88,000 tokens per RTX 4090 here).
  - A session stays on its machine wherever the user is; VKong chooses by
    session, not by distance.

```python
from openai import OpenAI

client = OpenAI(base_url=f"{url}/v1", api_key=api_key)
client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=messages,
    extra_headers={"X-Vkong-Session": conversation_id},
)
```

## What is public

The App URL is public. vLLM refuses `/v1/*` requests without `VLLM_API_KEY`.
`/health` and `/metrics` (counts and latencies, never prompts) answer without
it.

## Stop

```bash
vkong app stop vllm-stack
```

This releases both machines and stops billing. Metrics already sent stay in
Grafana Cloud.

## Customize

- Model and vLLM flags: `serve.sh`.
- Number of machines, GPU and price limit: `vkong.yaml`.
- Another Prometheus instead of Grafana Cloud: point the three `METRICS_*` keys
  of the `grafana` secret at its remote-write endpoint.

## Credits

- `bench/`: [vLLM production-stack](https://github.com/vllm-project/production-stack), Apache-2.0.
- `dashboards/vllm.json`: [vLLM](https://github.com/vllm-project/vllm), Apache-2.0, changed as described in `dashboards/NOTICE`.
