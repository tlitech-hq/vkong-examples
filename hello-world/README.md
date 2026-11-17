# Hello World with FastAPI

Run a small HTTP service on the cheapest available GPU machine (`gpu: Any`); the app itself does not use the GPU. The Python image already includes the runtime; setup installs only the app dependencies.

## Run

From this repository's root, after `vkong login`:

```bash
cd hello-world
vkong run
```

Keep the terminal open. Copy the local URL printed by VKong into a second terminal:

```bash
export VKONG_URL=http://127.0.0.1:<local-port>
curl -fsS "$VKONG_URL/"
```

## Run in the background

Use `vkong run --detach` when starting a service that should outlive the terminal.
To publish it at an HTTPS URL, run `vkong deploy` from this directory.
Use that URL as `VKONG_URL` with the same client.

## Stop

```bash
vkong app stop hello-world
```

This releases the machine and stops compute billing.

## Customize

Edit `vkong.yaml` for compute requirements and the hourly price limit.
