"""Shared benchmark plumbing: stats, manifests, cached tool runs, result files.

Import as `from common import stats`: a script directly in benchmarks/ has it on sys.path;
one in a subfolder (chains/, agent/) adds benchmarks/ first. stats and manifest need only
numpy, so they also work inside a tool's env (analyze); runner needs the core env.
"""
