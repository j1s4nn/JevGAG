# Security Policy

## Scope

This repository contains research code and audited result files. It does not ship credentials.

## Secrets

- API keys (`TYPESAFE_API_KEY`, `GH_TOKEN`, Kaggle credentials) are read from the environment and are
  never committed. `.env` files and `kaggle.json` are git-ignored.
- If you believe a secret was committed, rotate it immediately and open a private report.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to `jisanahammed1952@gmail.com` rather than opening a
public issue. Include a description, reproduction steps, and the affected commit.

## Supported versions

The `main` branch is the supported line.
