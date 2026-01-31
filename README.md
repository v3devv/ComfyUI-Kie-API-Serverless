# ComfyUI KIE API Nodes (Serverless Edition)

## Project overview
A **serverless-compatible** fork of ComfyUI custom nodes that connect to the Kie.ai API for image and video generation workflows.
Thanks to "https://github.com/gateway/ComfyUI-Kie-API" for the original.

**Key difference from original:** API key is passed as a node input field instead of loaded from a config file, making it work in serverless environments like Modal, ComfyDeploy, and RunPod.

## Why this fork exists
The original KIE API nodes load the API key from a config file (`config/kie_key.txt`), which doesn't work in serverless/ephemeral environments. This fork exposes the API key as a mandatory input field on each node, allowing you to:
- Save the key directly in your workflow JSON
- Use it in ComfyDeploy, Modal, RunPod serverless, etc.
- Run alongside the original KIE nodes (different node IDs with `[SL]` suffix)
