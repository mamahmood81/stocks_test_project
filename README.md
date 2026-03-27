# stocks_test_project

A small Python project for exploring historical stock and ETF price data from CSV files.

## What this project does

This project loads daily market data with columns such as:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

and visualizes:

- adjusted price
- log price
- trading volume
- log returns
- windowed MSD of log price
- local MSD slope as a proxy for changing volatility regimes

## Motivation

I wanted to get familiar with financial time-series data and compare some of its behaviour to concepts from single-particle tracking and diffusion, such as:

- log price as a trajectory-like variable
- log return as a displacement-like variable
- volatility as a diffusion-like quantity
- moving MSD as a way to examine changing regimes over time

## Project structure

```text
stocks_test_project/
├── main.py
├── README.md
└── .gitignore